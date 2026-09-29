#!/usr/bin/env python3
"""Amendment S.1 gate: real-model smoke of the prompt-protected presses.

No powered prompt-protected row runs until this passes. On three reference
traces at the config's seed (the longest-, median- and shortest-prompt problems
among those with at least ``min_trace_tokens`` trace tokens), over the first
``replay_tokens`` trace tokens:

1. teacher-forced paired replay under each protected press: the press read the
   true prompt length, pre-eviction TV is exactly 0, at least one prune
   happened in every layer, every post-prune cache equals the effective target,
   and the press's live self-check (every prompt position kept, in every layer
   and KV head, at every prune) never raised;
2. identity control: with the protected length forced to what the unprotected
   press already keeps, the protected press's TV profile equals the
   unprotected press's exactly;
3. one free-running generation per press: the press read the tokenized prompt
   length and pruned at least once.

Writes ``<base runs>/pp-smoke-PASS.json`` only if every check holds, otherwise
``pp-smoke-FAIL.json`` with the failures.

    python scripts/pp_smoke.py --config configs/experiments/pp_smoke_s1.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_replay import _load_jsonl, _load_token_records, _reference_key

from kvcache.cli import load_config, plan
from kvcache.datasets import load_benchmark
from kvcache.presses.prompt_protect import (
    effective_budget,
    effective_prunes,
    pp_base_name,
    require_pp_config,
    select_smoke_traces,
    smoke_failures,
)
from kvcache.provenance import identity, write_gate
from kvcache.runmeta import capture_runmeta

MARKER = "pp-smoke"


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    return p


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    plan(cfg, args.config, "ppsmoke")
    require_pp_config(cfg)
    budget = int(cfg["budgets"][0])
    interval = int(cfg["compression_interval"])
    forced = {str(k): int(v) for k, v in cfg["identity_forced_prompt_len"].items()}
    if set(forced) != set(cfg["presses"]):
        raise SystemExit("REFUSED: identity_forced_prompt_len must name every press")
    print(
        json.dumps(
            {
                "budget": budget,
                "replay_tokens": cfg["replay_tokens"],
                "identity_forced_prompt_len": forced,
            }
        )
    )
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")

    import torch

    from kvcache.generation import generate_compressed
    from kvcache.models import load_model_and_tokenizer
    from kvcache.presses import build_press, press_seed
    from kvcache.presses.prompt_protect import build_prompt_protected_press
    from kvcache.replay import TransformersReplayBackend, paired_teacher_forced_replay

    base = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    ref_root = base / cfg["reference_run_name"]
    rows = _load_jsonl(ref_root)
    tokens = _load_token_records(ref_root)
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    problems = {
        p.pid: p
        for p in load_benchmark(
            "math500",
            benchmarks_cfg["math500"],
            subset=cfg["subset"]["math500"],
            allow_download=False,
        )
    }
    seed = int(cfg["seeds"][0])
    model_name = cfg["models"][0]
    model_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"][model_name]
    ident = identity(
        "pp",
        cfg,
        model_revision=str(model_cfg["revision"]),
        reference_run=cfg["reference_run_name"],
    )
    candidates = []
    for r in rows:
        if (
            r["pid"] in problems
            and r["model"] == model_name
            and int(r["decoding"]["seed"]) == seed
        ):
            rec = tokens[_reference_key(r)]
            candidates.append(
                {
                    "pid": r["pid"],
                    "prompt_len": len(rec["prompt_token_ids"]),
                    "trace_len": len(rec["trace_token_ids"]),
                    "key": _reference_key(r),
                }
            )
    chosen = select_smoke_traces(candidates, int(cfg["min_trace_tokens"]))
    model, tokenizer = load_model_and_tokenizer(model_cfg, allow_download=False)

    def replay(press: Any, prompt_ids: list[int], trace_ids: list[int]) -> Any:
        backend = TransformersReplayBackend(model)
        with torch.inference_mode(), press(model):
            return paired_teacher_forced_replay(
                prompt_ids,
                trace_ids,
                cache_factory=backend.cache_factory,
                forward=backend.forward,
                press=press,
            )

    records: list[dict[str, Any]] = []
    for c in chosen:
        rec = tokens[c["key"]]
        prompt_ids = list(rec["prompt_token_ids"])
        trace_ids = list(rec["trace_token_ids"])[: int(cfg["replay_tokens"])]
        for name in cfg["presses"]:
            t0 = time.monotonic()
            kw = dict(
                budget=budget,
                compression_interval=interval,
                seed=press_seed(c["pid"], seed),
                instrumented=True,
            )
            out: dict[str, Any] = {
                "kind": "replay",
                "pid": c["pid"],
                "press": name,
                "prompt_len": len(prompt_ids),
                "n_replayed": len(trace_ids),
                "n_layers": len(model.model.layers),
            }
            try:
                press = build_prompt_protected_press(name, **kw)
                prof = replay(press, prompt_ids, trace_ids)
                pruned = effective_prunes(press.compressions)
                first = min((int(e["step"]) for e in pruned), default=None)
                out.update(
                    protect=press.protect.report(),
                    expected_target=effective_budget(budget, len(prompt_ids))[0],
                    first_effective_prune=first,
                    # a prune during forward k first affects prediction k + 1
                    pre_eviction_max_tv=max(
                        prof.estimate[: (first or 0) + 1], default=0.0
                    ),
                    post_prune_sizes=sorted({int(e["after"]) for e in pruned}),
                    mean_tv=sum(prof.estimate) / len(prof.estimate),
                )
                unprotected = replay(
                    build_press(pp_base_name(name), **kw), prompt_ids, trace_ids
                )
                control = replay(
                    build_prompt_protected_press(
                        name, forced_prompt_len=forced[name], **kw
                    ),
                    prompt_ids,
                    trace_ids,
                )
                out["identity_max_abs_diff"] = max(
                    abs(a - b)
                    for a, b in zip(unprotected.estimate, control.estimate, strict=True)
                )
                out["protection_max_abs_diff"] = max(
                    abs(a - b)
                    for a, b in zip(unprotected.estimate, prof.estimate, strict=True)
                )
            except RuntimeError as exc:
                out["raised"] = str(exc)
            out["wallclock_s"] = time.monotonic() - t0
            records.append(out)
            print(json.dumps(out, sort_keys=True), flush=True)

    gen_case = chosen[len(chosen) // 2]
    for name in cfg["presses"]:
        out = {"kind": "generation", "pid": gen_case["pid"], "press": name}
        try:
            press = build_prompt_protected_press(
                name,
                budget=budget,
                compression_interval=interval,
                seed=press_seed(gen_case["pid"], seed),
            )
            gen = generate_compressed(
                model,
                tokenizer,
                problems[gen_case["pid"]],
                model_cfg,
                sampling_cfg,
                press=press,
                seed=seed,
                max_new_tokens=int(cfg["generation_cap"]),
            )
            out.update(
                protect=press.protect.report(),
                prompt_len=len(gen.prompt_token_ids),
                n_tokens=gen.n_tokens,
            )
        except RuntimeError as exc:
            out["raised"] = str(exc)
        records.append(out)
        print(json.dumps(out, sort_keys=True), flush=True)

    fails = smoke_failures(records)
    report = {
        "gate": "S.1 prompt-protected press smoke",
        "passed": not fails,
        "failures": fails,
        "identity": ident,
        "selected": [
            {k: c[k] for k in ("pid", "prompt_len", "trace_len")} for c in chosen
        ],
        "records": records,
        "runmeta": {
            k: capture_runmeta(cfg["name"], args.config, seed).get(k)
            for k in ("commit", "source_tree_hash")
        },
    }
    target = write_gate(base, f"{MARKER}-PASS.json", f"{MARKER}-FAIL.json", report)
    print(f"{'PASS' if not fails else 'FAIL'}: wrote {target}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""WP4 gate: real-model smoke of the protected-selector families (PREPARED and
HELD 2026-10-05; proposal P2, cap 2 GPU-h).

No WP5 row runs until this passes (``require_protected_config`` accepts only a
PASS record whose identity matches the generation code, model, this config and
the inputs at launch time). On three reference traces at the config's seed (the
longest-, median- and shortest-prompt problems among those with at least
``min_trace_tokens`` trace tokens), over the first ``replay_tokens`` tokens:

1. teacher-forced paired replay under each family: the press read the true
   prompt length; every post-prune cache equals the effective target; every
   layer pruned; pre-eviction TV exactly 0; the live retention check (prompt,
   and the recency buffer where present, kept in every layer and KV head, no
   duplicate positions) never raised;
2. identity control for ``identity_family`` (H2O-style): with nothing
   protected, its selection equals kvpress's own top-k on the same live
   attention scores at every prune, in every head;
3. one free-running generation per family on each of the three traces
   (9 records): prompt length read, at least one prune.

Writes ``<KVCACHE_BASE_RUNS_ROOT>/protected-selectors-smoke-PASS.json`` only if
every check holds, otherwise ``...-FAIL.json`` with the failures (the opposite
marker is retired, never deleted).

    python scripts/protected_selectors_smoke.py \\
        --config configs/experiments/protsel_smoke_wp4.yaml --dry-run
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
    select_smoke_traces,
)
from kvcache.presses.protected_selectors import (
    FAMILIES,
    SMOKE_MARKER,
    family,
    needs_eager_attention,
    smoke_failures,
    smoke_identity,
)
from kvcache.provenance import inputs_sha256, write_gate
from kvcache.runmeta import capture_runmeta

FAIL_MARKER = SMOKE_MARKER.replace("-PASS.json", "-FAIL.json")


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


def validate(cfg: dict[str, Any]) -> None:
    """Config-level refusals (also run in dry-run)."""
    if cfg.get("protected_selector") is not True:
        raise SystemExit("REFUSED: config must set protected_selector: true")
    for name in cfg["presses"]:
        try:
            family(str(name))
        except KeyError as exc:
            raise SystemExit(f"REFUSED: {exc.args[0]}") from None
    ident = str(cfg.get("identity_family", ""))
    if ident not in cfg["presses"] or FAMILIES[ident].fresh_uniform:
        raise SystemExit(
            "REFUSED: identity_family must be a tested, score-based family"
        )


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    plan(cfg, args.config, "protselsmoke")
    validate(cfg)
    budget = int(cfg["budgets"][0])
    interval = int(cfg["compression_interval"])
    print(
        json.dumps(
            {
                "budget": budget,
                "replay_tokens": cfg["replay_tokens"],
                "identity_family": cfg["identity_family"],
                "eager_attention": any(
                    needs_eager_attention(p) for p in cfg["presses"]
                ),
            }
        )
    )
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if cfg.get("held", True):
        raise SystemExit(
            "REFUSED: the smoke config is HELD (proposal P2, not approved)"
        )
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")

    import torch

    from kvcache.generation import generate_compressed
    from kvcache.models import load_model_and_tokenizer
    from kvcache.presses import press_seed
    from kvcache.presses.protected_selectors import build_protected_selector_press
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
    candidates = [
        {
            "pid": r["pid"],
            "prompt_len": len(tokens[_reference_key(r)]["prompt_token_ids"]),
            "trace_len": len(tokens[_reference_key(r)]["trace_token_ids"]),
            "key": _reference_key(r),
        }
        for r in rows
        if r["pid"] in problems
        and r["model"] == model_name
        and int(r["decoding"]["seed"]) == seed
    ]
    chosen = select_smoke_traces(candidates, int(cfg["min_trace_tokens"]))
    eager = any(needs_eager_attention(p) for p in cfg["presses"])
    model, tokenizer = load_model_and_tokenizer(
        model_cfg, allow_download=False, attn_implementation="eager" if eager else None
    )

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
                "inputs_sha256": inputs_sha256(prompt_ids, trace_ids),
                "n_layers": len(model.model.layers),
            }
            t0 = time.monotonic()
            try:
                press = build_protected_selector_press(name, **kw)
                prof = replay(press, prompt_ids, trace_ids)
                pruned = effective_prunes(press.compressions)
                first = min((int(e["step"]) for e in pruned), default=None)
                out.update(
                    protect=press.protect.report(),
                    expected_target=effective_budget(
                        budget,
                        len(prompt_ids),
                        max(64, FAMILIES[name].recent),
                    )[0],
                    first_effective_prune=first,
                    pre_eviction_max_tv=max(
                        prof.estimate[: (first or 0) + 1], default=0.0
                    ),
                    post_prune_sizes=sorted({int(e["after"]) for e in pruned}),
                    mean_tv=sum(prof.estimate) / len(prof.estimate),
                )
            except RuntimeError as exc:
                out["raised"] = str(exc)
            out["wallclock_s"] = time.monotonic() - t0
            records.append(out)
            print(json.dumps(out, sort_keys=True), flush=True)

        name = str(cfg["identity_family"])
        out = {"kind": "identity", "pid": c["pid"], "press": name}
        try:
            press = build_protected_selector_press(
                name,
                budget=budget,
                compression_interval=interval,
                seed=press_seed(c["pid"], seed),
                instrumented=True,
                forced_prompt_len=0,
            )
            replay(press, prompt_ids, trace_ids)
            out.update(
                identity_prunes=press.identity_check["prunes"],
                identity_mismatched_heads=press.identity_check["mismatched_heads"],
            )
        except RuntimeError as exc:
            out["raised"] = str(exc)
        records.append(out)
        print(json.dumps(out, sort_keys=True), flush=True)

    # 2026-10-14 (Meeting 5 review G0.1): free generation on EVERY selected
    # trace (longest, median and shortest prompt), the 21-record design the P2
    # approval described; the first run (dbdf831) covered the median trace only.
    for gen_case, name in [(c, n) for c in chosen for n in cfg["presses"]]:
        out = {"kind": "generation", "pid": gen_case["pid"], "press": name}
        t0 = time.monotonic()
        try:
            press = build_protected_selector_press(
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
        out["wallclock_s"] = time.monotonic() - t0
        records.append(out)
        print(json.dumps(out, sort_keys=True), flush=True)

    fails = smoke_failures(records)
    report = {
        "gate": "WP4 protected-selector smoke",
        "passed": not fails,
        "status": "PASS" if not fails else "FAIL",
        "failures": fails,
        "identity": smoke_identity(cfg),
        "families_tested": sorted(str(p).lower() for p in cfg["presses"]),
        # Review of 6b409001 (R2): the input hash of each replayed trace, so the
        # runtime gate can check the smoke used the frozen reference inputs.
        "selected": [
            {
                **{k: c[k] for k in ("pid", "prompt_len", "trace_len")},
                "inputs_sha256": inputs_sha256(
                    tokens[c["key"]]["prompt_token_ids"],
                    tokens[c["key"]]["trace_token_ids"][: int(cfg["replay_tokens"])],
                ),
            }
            for c in chosen
        ],
        "records": records,
        "runmeta": {
            k: capture_runmeta(cfg["name"], args.config, seed).get(k)
            for k in ("commit", "source_tree_hash")
        },
    }
    target = write_gate(base, SMOKE_MARKER, FAIL_MARKER, report)
    print(f"{report['status']}: wrote {target}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())

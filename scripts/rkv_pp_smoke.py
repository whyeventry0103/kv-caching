#!/usr/bin/env python3
"""Real-model smoke for prompt-protected R-KV, ``rkv_pp_v1`` (Meeting 5, M5-A;
prepared 2026-10-14, HELD -- needs its own approval and GPU-hour cap).

The run adapters accept no rkv_pp_v1 row until this smoke has written a PASS
record whose identity matches the generation code, model, this config and the
inputs at launch time (``kvcache.presses.rkv_prompt_protect.verify_smoke``). On
the three fixed traces (longest, median and shortest prompt with at least
``min_trace_tokens`` trace tokens), it runs:

1. identity control -- base ``rkv`` and ``rkv_pp_v1`` with ``forced_prompt_len=0``
   replay the same trace; every prune's K/V must be byte-identical and the TV
   profiles identical (protection off IS the parity-certified base path);
2. replay -- ``rkv_pp_v1`` with the real prompt: target = max(B, P + 64), every
   post-prune size equal to it, every layer pruned, pre-eviction TV exactly 0,
   and the live check (prompt UNION recent-8 retained, no duplicate) never
   raised;
3. free generation -- the prompt length was read and at least one prune happened.

The checks are ``rkv_prompt_protect.smoke_failures``. The CPU engineering
precursor of (1)-(2) is ``scripts/rkv_pp_cpu_parity.py``.

    python scripts/rkv_pp_smoke.py --config configs/experiments/rkvpp_smoke_m5.yaml \\
        --dry-run
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
from kvcache.presses.rkv_prompt_protect import (
    FAMILY,
    SMOKE_MARKER,
    first_mismatch,
    record_prune_outputs,
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
    # Accepted so the launcher can invoke this like any other stage.
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    return p


def validate(cfg: dict[str, Any]) -> None:
    """Config-level refusals (also run in dry-run)."""
    if cfg.get("rkv_prompt_protected") is not True:
        raise SystemExit("REFUSED: config must set rkv_prompt_protected: true")
    if [str(p).lower() for p in cfg["presses"]] != [FAMILY]:
        raise SystemExit(f"REFUSED: presses must be exactly [{FAMILY}]")
    if len(cfg["budgets"]) != 1 or len(cfg["models"]) != 1:
        raise SystemExit("REFUSED: the smoke tests one model at one budget")


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    plan(cfg, args.config, "rkvppsmoke")
    validate(cfg)
    budget = int(cfg["budgets"][0])
    interval = int(cfg["compression_interval"])
    print(json.dumps({"budget": budget, "replay_tokens": cfg["replay_tokens"]}))
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if cfg.get("held", True):
        raise SystemExit("REFUSED: the rkv-pp smoke config is HELD (not approved)")
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")

    import torch

    from kvcache.generation import generate_compressed
    from kvcache.models import load_model_and_tokenizer
    from kvcache.presses import build_press, press_seed
    from kvcache.presses.rkv_prompt_protect import build_rkv_prompt_protected_press
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

    def emit(out: dict[str, Any], t0: float) -> None:
        out["wallclock_s"] = time.monotonic() - t0
        records.append(out)
        print(json.dumps(out, sort_keys=True), flush=True)

    for c in chosen:
        rec = tokens[c["key"]]
        prompt_ids = list(rec["prompt_token_ids"])
        trace_ids = list(rec["trace_token_ids"])[: int(cfg["replay_tokens"])]
        kw = dict(
            budget=budget,
            compression_interval=interval,
            seed=press_seed(c["pid"], seed),
            instrumented=True,
        )

        out: dict[str, Any] = {"kind": "identity", "pid": c["pid"], "press": FAMILY}
        t0 = time.monotonic()
        try:
            base_press = build_press("rkv", **kw)
            off = build_rkv_prompt_protected_press(FAMILY, forced_prompt_len=0, **kw)
            log_b, log_p = record_prune_outputs(base_press), record_prune_outputs(off)
            prof_b = replay(base_press, prompt_ids, trace_ids)
            prof_p = replay(off, prompt_ids, trace_ids)
            out.update(
                n_prunes_base=len(log_b),
                n_prunes_protected_off=len(log_p),
                first_kv_mismatch=first_mismatch(log_b, log_p),
                tv_identical=prof_b.estimate == prof_p.estimate,
            )
        except RuntimeError as exc:
            out["raised"] = str(exc)
        emit(out, t0)

        out = {
            "kind": "replay",
            "pid": c["pid"],
            "press": FAMILY,
            "prompt_len": len(prompt_ids),
            "n_replayed": len(trace_ids),
            "inputs_sha256": inputs_sha256(prompt_ids, trace_ids),
            "n_layers": len(model.model.layers),
        }
        t0 = time.monotonic()
        try:
            press = build_rkv_prompt_protected_press(FAMILY, **kw)
            prof = replay(press, prompt_ids, trace_ids)
            pruned = effective_prunes(press.compressions)
            first = min((int(e["step"]) for e in pruned), default=None)
            out.update(
                protect=press.protect.report(),
                expected_target=effective_budget(
                    budget, len(prompt_ids), press.protect.min_trace_slots
                )[0],
                first_effective_prune=first,
                pre_eviction_max_tv=max(prof.estimate[: (first or 0) + 1], default=0.0),
                post_prune_sizes=sorted({int(e["after"]) for e in pruned}),
                mean_tv=sum(prof.estimate) / len(prof.estimate),
            )
        except RuntimeError as exc:
            out["raised"] = str(exc)
        emit(out, t0)

        out = {"kind": "generation", "pid": c["pid"], "press": FAMILY}
        t0 = time.monotonic()
        try:
            press = build_rkv_prompt_protected_press(
                FAMILY,
                budget=budget,
                compression_interval=interval,
                seed=press_seed(c["pid"], seed),
            )
            gen = generate_compressed(
                model,
                tokenizer,
                problems[c["pid"]],
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
        emit(out, t0)

    fails = smoke_failures(records)
    report = {
        "gate": "M5-A protected R-KV smoke",
        "passed": not fails,
        "status": "PASS" if not fails else "FAIL",
        "failures": fails,
        "identity": smoke_identity(cfg),
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

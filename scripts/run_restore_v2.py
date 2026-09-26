#!/usr/bin/env python3
"""Amendment R.17.2.3 restoration, redesigned (``kvcache.restore_v2``).

A copy of ``scripts/run_exact_lag.py`` (left untouched: its bytes are part of
the finished exact-lag identity) calling ``run_restore_v2``: trigger at the
first prune at or after step 1024, the R.10 cohort arms at every lag, the full
self-restoration arm, and M/Delta in the none and full-restoration arms.
One ``restore-v2`` row per (trace, policy); resumable.

The v1 description, unchanged in substance:

Amendment R.5 exact-lag restoration (replaces the unexecuted Q.5 sweep).

For each FC reference trace of the Q.2 subset and each policy: the identity-
repaired stash, the cohort (first effective prune at or after step 130), and
the none / matched / random / relaxed arms at lags 1, 4, 16, 64, 256, each
read over 200 forwards from the reinsertion step (``kvcache.exact_lag``).
One ``exact-lag-v1`` row per (trace, policy) in
``<run>/math500/<model>/<press>-<budget>-s<seed>.jsonl``; resumable.

    python scripts/run_exact_lag.py \\
        --config configs/experiments/exact_lag_math500_r5.yaml --dry-run
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

from run_replay import _heal_jsonl, _load_jsonl, _load_token_records, _reference_key

from kvcache.cli import load_config, plan
from kvcache.provenance import (
    check_inputs,
    identity,
    inputs_sha256,
    load_input_manifest,
    same_identity,
)
from kvcache.datasets import load_benchmark
from kvcache.runmeta import capture_runmeta

SCHEMA = "restore-v2"


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
    plan(cfg, args.config, "restore2")
    budget = int(cfg["budgets"][0])
    n_reinsert = int(float(cfg["reinsert_fraction"]) * budget)
    lags = [int(x) for x in cfg["lags"]]
    print(
        json.dumps(
            {
                "lags": lags,
                "n_reinsert": n_reinsert,
                "trigger_step": cfg["trigger_step"],
                "window": cfg["window"],
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

    from kvcache.restore_v2 import run_restore_v2
    from kvcache.models import load_model_and_tokenizer
    from kvcache.presses import build_press, press_seed

    base = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    out_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    ref_root = base / cfg["reference_run_name"]
    rows = _load_jsonl(ref_root)
    tokens = _load_token_records(ref_root)
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    pids = {
        p.pid
        for p in load_benchmark(
            "math500",
            benchmarks_cfg["math500"],
            subset=cfg["subset"]["math500"],
            allow_download=False,
        )
    }
    seeds = [int(s) for s in cfg["seeds"]]
    if args.seeds.strip():
        shard = {int(s) for s in args.seeds.split(",")}
        seeds = [s for s in seeds if s in shard]
    model_name = cfg["models"][0]
    eligible = sorted(
        (
            r
            for r in rows
            if r["pid"] in pids
            and int(r["decoding"]["seed"]) in seeds
            and r["model"] == model_name
        ),
        key=lambda r: (int(r["decoding"]["seed"]), r["pid"]),
    )
    if cfg.get("reference_limit") is not None:
        eligible = eligible[: int(cfg["reference_limit"])]
    model_rev = str(
        _yaml(PROJECT_ROOT / "configs/models.yaml")["models"][model_name]["revision"]
    )
    ident = identity(
        "restore_v2",
        cfg,
        model_revision=model_rev,
        reference_run=cfg["reference_run_name"],
    )
    if not args.resume and any((out_root / "math500").rglob("*.jsonl")):
        raise SystemExit("REFUSED: run namespace not empty; pass --resume")
    manifest = load_input_manifest()
    done = set()
    if args.resume and (out_root / "math500").exists():
        for path in sorted((out_root / "math500").rglob("*.jsonl")):
            for rec in _heal_jsonl(path):
                if not same_identity(rec.get("provenance") or {}, ident):
                    raise SystemExit(
                        f"REFUSED: {path} holds a row from a different experiment "
                        "identity; start a new run namespace"
                    )
                check_inputs(
                    manifest,
                    "restore_v2",
                    rec["pid"],
                    int(rec["seed"]),
                    (rec.get("provenance") or {}).get("inputs_sha256"),
                )
                done.add((rec["pid"], int(rec["seed"]), rec["press"]))
    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    model_cfg = models_cfg[model_name]
    model, _tok = load_model_and_tokenizer(model_cfg, allow_download=False)
    for r in eligible:
        seed = int(r["decoding"]["seed"])
        rec = tokens[_reference_key(r)]
        check_inputs(
            manifest,
            "restore_v2",
            r["pid"],
            seed,
            inputs_sha256(rec["prompt_token_ids"], rec["trace_token_ids"]),
        )
        for press_name in cfg["presses"]:
            if (r["pid"], seed, press_name) in done:
                continue
            t0 = time.monotonic()
            press = build_press(
                press_name,
                budget=budget,
                compression_interval=int(cfg["compression_interval"]),
                seed=press_seed(r["pid"], seed),
                instrumented=True,
            )
            out = run_restore_v2(
                model,
                press,
                rec["prompt_token_ids"],
                rec["trace_token_ids"],
                n_reinsert=n_reinsert,
                rng_key=f"{r['pid']}|{seed}|{press_name}",
                trigger_step=int(cfg["trigger_step"]),
                window=int(cfg["window"]),
                lags=lags,
                every=int(cfg["measure_every"]),
            )
            runmeta = capture_runmeta(cfg["name"], args.config, seed)
            row = {
                "schema_version": SCHEMA,
                "pid": r["pid"],
                "seed": seed,
                "press": press_name,
                "budget": budget,
                "reference_key": list(_reference_key(r)),
                "wallclock_s": time.monotonic() - t0,
                "runmeta": {k: runmeta.get(k) for k in ("commit", "source_tree_hash")},
                "provenance": {
                    **ident,
                    "inputs_sha256": inputs_sha256(
                        rec["prompt_token_ids"], rec["trace_token_ids"]
                    ),
                },
                **out,
            }
            path = (
                out_root
                / "math500"
                / model_name
                / f"{press_name}-{budget}-s{seed}.jsonl"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                f"{r['pid']} s{seed} {press_name}: {out['status']} "
                f"e={out['cohort_step']} {row['wallclock_s']:.0f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

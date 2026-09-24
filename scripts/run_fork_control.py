#!/usr/bin/env python3
"""Amendment R.2 / R.11: the Q.1 direct functional control (mask now, remove later).

For each FC reference trace in scope and each configured interval: a paired
replay with ``kvcache.fork_control.DeferredRemovalPress`` (selection on the
active set, immediate mask, physical removal ``delay`` forwards later) against
the full cache, forking before every physical removal (``run_fork_control``).
One ``fork-control-v1`` row per (trace, policy, interval); resumable.

    python scripts/run_fork_control.py \\
        --config configs/experiments/fork_control_random_r11.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
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
from kvcache.provenance import identity, inputs_sha256, same_identity
from kvcache.datasets import load_benchmark
from kvcache.runmeta import capture_runmeta

SCHEMA = "fork-control-v1"


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _r(x: Any) -> Any:
    if isinstance(x, float):
        return float(f"{x:.6g}") if math.isfinite(x) else None
    if isinstance(x, list):
        return [_r(v) for v in x]
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    return x


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
    plan(cfg, args.config, "forkcontrol")
    intervals = [int(i) for i in cfg["intervals"]]
    print(json.dumps({"intervals": intervals, "delay": cfg["delay"]}))
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")

    from kvcache.fork_control import DeferredRemovalPress, run_fork_control
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
    budget = int(cfg["budgets"][0])
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
        "fork", cfg, model_revision=model_rev, reference_run=cfg["reference_run_name"]
    )
    if not args.resume and any((out_root / "math500").rglob("*.jsonl")):
        raise SystemExit("REFUSED: run namespace not empty; pass --resume")
    done = set()
    if args.resume and (out_root / "math500").exists():
        for path in sorted((out_root / "math500").rglob("*.jsonl")):
            for rec in _heal_jsonl(path):
                if not same_identity(rec.get("provenance") or {}, ident):
                    raise SystemExit(
                        f"REFUSED: {path} holds a row from a different experiment "
                        "identity; start a new run namespace"
                    )
                done.add((rec["pid"], int(rec["seed"]), rec["press"], rec["interval"]))
    model_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"][model_name]
    model, _tok = load_model_and_tokenizer(model_cfg, allow_download=False)
    for r in eligible:
        seed = int(r["decoding"]["seed"])
        rec = tokens[_reference_key(r)]
        for press_name in cfg["presses"]:
            for interval in intervals:
                if (r["pid"], seed, press_name, interval) in done:
                    continue
                t0 = time.monotonic()
                base_press = build_press(
                    press_name,
                    budget=budget,
                    compression_interval=interval,
                    seed=press_seed(r["pid"], seed),
                ).base_press
                press = DeferredRemovalPress(
                    base_press,
                    budget=budget,
                    interval=interval,
                    delay=int(cfg["delay"]),
                )
                out = run_fork_control(
                    model,
                    press,
                    rec["prompt_token_ids"],
                    rec["trace_token_ids"],
                    n_deterministic=int(cfg.get("n_deterministic", 3)),
                )
                meta = capture_runmeta(cfg["name"], args.config, seed)
                row = {
                    "schema_version": SCHEMA,
                    "pid": r["pid"],
                    "seed": seed,
                    "press": press_name,
                    "budget": budget,
                    "interval": interval,
                    "delay": int(cfg["delay"]),
                    "reference_key": list(_reference_key(r)),
                    "wallclock_s": time.monotonic() - t0,
                    "runmeta": {k: meta.get(k) for k in ("commit", "source_tree_hash")},
                    "provenance": {
                        **ident,
                        "inputs_sha256": inputs_sha256(
                            rec["prompt_token_ids"], rec["trace_token_ids"]
                        ),
                    },
                    **_r(out),
                }
                path = (
                    out_root
                    / "math500"
                    / model_name
                    / f"{press_name}-{budget}-i{interval}-s{seed}.jsonl"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    f"{r['pid']} s{seed} {press_name} I={interval}: forks="
                    f"{out['n_forks']} maxdiff={out['max_abs_logit']} "
                    f"det={out['deterministic_control_max_abs_logit']} "
                    f"agree={out['selection_agreement']} "
                    f"{row['wallclock_s']:.0f}s",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Amendment R.17.0.5: fp32 diagnosis of the fork gap, and the precision floor.

Two measurements, one job, one model at a time (a 7B model in fp32 plus one in
bf16 does not fit one 48 GB GPU):

1. bf16 pass: the first-fork diagnosis on the registered trace, then the
   teacher-forced full-cache logits of the floor blocks on the floor traces
   (kept on the host in the dtype produced -- lossless for bf16 logits).
2. fp32 pass (same weights, ``dtype: float32``): the same diagnosis, then the
   floor comparison against the stored bf16 logits.

Writes ``fork_precision.json`` under the run directory, atomically, as the
single output and queue marker.

    python scripts/run_fork_precision.py \\
        --config configs/experiments/fork_precision_r17.yaml --dry-run
"""

from __future__ import annotations

import argparse
import gc
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
from kvcache.provenance import (
    check_inputs,
    identity,
    inputs_sha256,
    load_input_manifest,
    write_json_atomic,
)
from kvcache.runmeta import capture_runmeta

OUTPUT = "fork_precision.json"


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


def _frozen_pids() -> list[str]:
    path = PROJECT_ROOT / "configs/meeting3/q2_subset_pids.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["pids"])


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    plan(cfg, args.config, "forkprecision")
    diag, floor = cfg["diagnosis"], cfg["floor"]
    pids = _frozen_pids()
    diag_pid = pids[int(diag["pid_rank"])]
    floor_pids = pids[: int(floor["n_problems"])]
    print(
        json.dumps(
            {
                "diagnosis_trace": [diag_pid, int(diag["seed"])],
                "floor_traces": [[p, int(floor["seed"])] for p in floor_pids],
                "dtypes": cfg["dtypes"],
                "block_starts": floor["block_starts"],
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

    from kvcache import fork_precision as fp
    from kvcache.fork_control import DeferredRemovalPress
    from kvcache.models import load_model_and_tokenizer
    from kvcache.presses import build_press, press_seed

    base = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    out_dir = Path(os.environ["REMOTE_RUNS_ROOT"])
    out_path = out_dir / OUTPUT
    if out_path.exists():
        print(f"already complete: {out_path}")
        return 0
    ref_root = base / cfg["reference_run_name"]
    rows = _load_jsonl(ref_root)
    tokens = _load_token_records(ref_root)
    model_name = cfg["models"][0]
    by_key = {
        (r["pid"], int(r["decoding"]["seed"])): r
        for r in rows
        if r["model"] == model_name
    }
    manifest = load_input_manifest()

    def trace(pid: str, seed: int) -> tuple[list[int], list[int], str]:
        rec = tokens[_reference_key(by_key[(pid, seed)])]
        sha = inputs_sha256(rec["prompt_token_ids"], rec["trace_token_ids"])
        check_inputs(manifest, "fork_precision", pid, seed, sha)
        return list(rec["prompt_token_ids"]), list(rec["trace_token_ids"]), sha

    model_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"][model_name]
    model_rev = str(model_cfg["revision"])
    ident = identity(
        "fork_precision",
        cfg,
        model_revision=model_rev,
        reference_run=cfg["reference_run_name"],
    )
    budget = int(cfg["budgets"][0])
    d_prompt, d_trace, d_sha = trace(diag_pid, int(diag["seed"]))
    floor_traces = {p: trace(p, int(floor["seed"])) for p in floor_pids}
    starts, length = [int(s) for s in floor["block_starts"]], int(floor["block_length"])

    result: dict[str, Any] = {
        "schema_version": "fork-precision-v1",
        "amendment": "R.17.0.5",
        "diagnosis_trace": {
            "pid": diag_pid,
            "seed": int(diag["seed"]),
            "press": diag["press"],
            "interval": int(diag["interval"]),
            "delay": int(diag["delay"]),
            "budget": budget,
            "inputs_sha256": d_sha,
        },
        "floor_traces": [
            {"pid": p, "seed": int(floor["seed"]), "inputs_sha256": v[2]}
            for p, v in floor_traces.items()
        ],
        "floor_blocks": {"starts": starts, "length": length},
        "provenance": ident,
        "diagnosis": {},
        "floor": {},
    }
    stored: dict[str, dict[int, Any]] = {}
    t0 = time.monotonic()
    for dtype in cfg["dtypes"]:
        mc = dict(model_cfg, dtype=dtype)
        model, _tok = load_model_and_tokenizer(mc, allow_download=False)
        press = DeferredRemovalPress(
            build_press(
                diag["press"],
                budget=budget,
                compression_interval=int(diag["interval"]),
                seed=press_seed(diag_pid, int(diag["seed"])),
            ).base_press,
            budget=budget,
            interval=int(diag["interval"]),
            delay=int(diag["delay"]),
        )
        result["diagnosis"][dtype] = fp.first_fork_diagnosis(
            model, press, d_prompt, d_trace
        )
        print(
            f"[{dtype}] diagnosis: fork step "
            f"{result['diagnosis'][dtype]['fork_step']}, "
            f"{result['diagnosis'][dtype]['n_compared']} steps, "
            f"{time.monotonic() - t0:.0f}s",
            flush=True,
        )
        if dtype == cfg["dtypes"][0]:
            for p, (pr, tr, _sha) in floor_traces.items():
                stored[p] = fp.collect_block_logits(
                    model, pr, tr, starts=starts, length=length
                )
                print(f"[{dtype}] floor ref {p}: {len(stored[p])} steps", flush=True)
        else:
            for p, (pr, tr, _sha) in floor_traces.items():
                cmp = fp.compare_block_logits(model, pr, tr, stored[p])
                result["floor"][p] = {str(j): v for j, v in cmp.items()}
                print(f"[{dtype}] floor cmp {p}: {len(cmp)} steps", flush=True)
        del model, press
        gc.collect()
        torch.cuda.empty_cache()
    meta = capture_runmeta(cfg["name"], args.config, int(diag["seed"]))
    result["runmeta"] = {k: meta.get(k) for k in ("commit", "source_tree_hash")}
    result["wallclock_s"] = time.monotonic() - t0
    result["passed"] = True
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, result)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

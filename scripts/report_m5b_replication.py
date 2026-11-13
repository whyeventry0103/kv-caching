#!/usr/bin/env python3
"""Read one dataset of the M5-B multi-dataset replication (frozen 2026-11-13,
before any confirmation outcome). All logic is ``kvcache.m5b_reads``.

Every cell the queue lists for the dataset (the full-cache confirmation bank,
the compressed-run cells and the forward readouts) is validated against its
own config's frozen design; each is FINAL, PARTIAL, INVALID or ABSENT and is
listed as such. Only FINAL cells enter the read; an arm whose cell is not
FINAL is reported missing, never read from a partial cell. Golds come from
the pinned dataset (offline, on the host) and must match the committed
confirmation resolution record.

    python scripts/report_m5b_replication.py --bench gsm8k --dry-run
    KVCACHE_REMOTE=1 python scripts/report_m5b_replication.py --bench gsm8k \\
        --runs-root "$KVCACHE_BASE_RUNS_ROOT" \\
        --out results/diag/meeting5/replication/gsm8k_replication_read.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import nightly_autorun as scheduler
from run_bank_manifest import load_split, resolve_design

from kvcache.cli import load_config
from kvcache.m5b_reads import (
    cell_status,
    fc_bank,
    grid_arms,
    read_dataset,
    readout_arms,
)
from kvcache.manifest_grid import resolve_grid_design
from kvcache.provenance import write_json_atomic
from kvcache.read_validation import retire_stale

READ_SPEC = PROJECT_ROOT / "configs/meeting5/m5a_read.yaml"
RESOLUTION = PROJECT_ROOT / "results/diag/meeting5/replication"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", required=True, choices=["gsm8k", "bbh"])
    p.add_argument("--runs-root", type=Path)
    p.add_argument("--resolution", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p


def cells(bench: str) -> list[Any]:
    """The dataset's replication jobs, in queue order."""
    _, jobs = scheduler.load_queue(scheduler.QUEUE_DEFAULT)
    return [j for j in jobs if j.name.endswith("-m5b") and f"-{bench}-" in j.name]


def golds_for(bank_design: dict[str, Any], resolution: Path) -> dict[str, str]:
    """Pinned golds, agreeing with the committed confirmation record."""
    rec = json.loads(resolution.read_text(encoding="utf-8"))
    want = {p["pid"]: p for p in rec["problems"]}
    if list(want) != bank_design["pids"]:
        raise ValueError("resolution record IDs differ from the frozen split")
    golds = {}
    for p in load_split(bank_design):
        w = want[p.pid]
        q = hashlib.sha256(p.question.encode()).hexdigest()
        if q != w["question_sha256"] or str(p.gold_answer) != str(w["gold"]):
            raise ValueError(f"{p.pid}: dataset differs from the resolution record")
        golds[p.pid] = str(p.gold_answer)
    return golds


def main() -> int:
    import yaml

    args = _parser().parse_args()
    jobs = cells(args.bench)
    bank_job = next(j for j in jobs if j.command == "bank_manifest_m5b")
    bank_design = resolve_design(load_config(PROJECT_ROOT / bank_job.config))
    resolution = args.resolution or (
        RESOLUTION / f"{args.bench}_confirmation_resolution.json"
    )
    plan = {
        "bench": bank_design["benchmark"],
        "cells": [(j.name, j.command) for j in jobs],
        "n_conditions": len(bank_design["pids"]) * len(bank_design["seeds"]),
        "resolution": str(resolution),
    }
    print(json.dumps(plan))
    if args.dry_run:
        print("DRY RUN: validated only; no rows or dataset read")
        return 0
    if args.runs_root is None:
        raise SystemExit("REFUSED: --runs-root is required for a read")
    spec = yaml.safe_load(READ_SPEC.read_text(encoding="utf-8"))
    golds = golds_for(bank_design, resolution)
    expected = {(p, s) for p in bank_design["pids"] for s in bank_design["seeds"]}
    status: dict[str, dict[str, Any]] = {}
    fc, arms, readouts = None, {}, {}
    for job in jobs:
        run = args.runs_root / job.run_name
        exc, got = None, None
        if run.is_dir():
            try:
                if job.command == "bank_manifest_m5b":
                    got = fc_bank(run, bank_design, golds)
                else:
                    kind = "readout" if job.command == "readout_manifest" else "grid"
                    design = resolve_grid_design(
                        load_config(PROJECT_ROOT / job.config), kind=kind
                    )
                    reader = readout_arms if kind == "readout" else grid_arms
                    got = reader(run, design, golds)
            except ValueError as e:
                exc = e
        st = cell_status(exc, exists=run.is_dir())
        status[job.name] = {"status": st, "detail": str(exc)[:400] if exc else None}
        if st != "FINAL":
            continue
        if job.command == "bank_manifest_m5b":
            fc = got
        elif job.command == "readout_manifest":
            readouts.update(got)
        else:
            arms.update(got)
    out: dict[str, Any] = {
        "schema": "m5b-replication-read-v1",
        "bench": bank_design["benchmark"],
        "cells": status,
        "complete_arms": sorted(f"{p}{'+P' if b else '-P'}" for p, b in arms),
        "complete_readouts": sorted(f"{p}{'+P' if b else '-P'}" for p, b in readouts),
    }
    if fc is None or ("random", True) not in arms and ("random", False) not in arms:
        out["status"] = "PARTIAL: no full-cache bank or no Random arm is FINAL"
    else:
        out["read"] = read_dataset(
            fc,
            arms,
            readouts,
            expected=expected,
            budget=256,
            reference=spec["reference"],
            pairings=tuple(spec["pairings"]),
            descriptive=tuple(spec["descriptive"]),
            equivalence=spec["equivalence"],
        )
        incomplete = [n for n, s in status.items() if s["status"] != "FINAL"]
        out["status"] = "FINAL" if not incomplete else f"PARTIAL: {incomplete}"
    if args.out:
        retire_stale(args.out)
        write_json_atomic(args.out, out)
        print(f"wrote {args.out}")
    print(json.dumps({"status": out["status"], "cells": status}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

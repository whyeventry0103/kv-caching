#!/usr/bin/env python3
"""Read the GSM8K full-cache timing pilot (held; nothing to read until it runs).

Validates all expected keys (frozen calibration split x design seeds) as
complete, matching row + token-ID pairs, then scores the STORED traces with the
frozen GSM8K grader against the pinned gold answers and reports truncation,
trace lengths, seconds per token and whether B=256 / B=1024 would bind
(``kvcache.gsm8k_timing``). The generation row's ``correct`` field is never
the score.

Gold answers come from the pinned dataset, loaded offline on the host, and
must agree with the committed resolution record
(``results/diag/meeting5/gsm8k_timing/split_resolution.json``: same IDs, same
order, same golds, same question hashes) or the read refuses.

    python scripts/read_gsm8k_timing.py --dry-run
    python scripts/read_gsm8k_timing.py --runs-root "$REMOTE_RUNS_ROOT/<run>"
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

from run_bank_manifest import load_split, resolve_design

from kvcache.cli import load_config
from kvcache.gsm8k_timing import expected_keys, read_timing
from kvcache.provenance import write_json_atomic
from kvcache.read_validation import retire_stale

CONFIG = PROJECT_ROOT / "configs/experiments/bank_gsm8k_timing_m5.yaml"
OUT_DIR = PROJECT_ROOT / "results/diag/meeting5/gsm8k_timing"
RESOLUTION = OUT_DIR / "split_resolution.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--runs-root", type=Path)
    p.add_argument("--resolution", type=Path, default=RESOLUTION)
    p.add_argument("--out", type=Path, default=OUT_DIR / "timing_read.json")
    p.add_argument("--dry-run", action="store_true")
    return p


def resolution_golds(design: dict[str, Any], resolution: Path) -> dict[str, dict]:
    """The committed per-ID record, checked against the design's IDs."""
    if not resolution.is_file():
        # results/ is never part of the source sync: on the host the committed
        # record must be copied next to the code before a read.
        raise ValueError(
            f"resolution record {resolution} is missing; copy the committed "
            "results/diag/meeting5/gsm8k_timing/split_resolution.json here"
        )
    rec = json.loads(resolution.read_text(encoding="utf-8"))
    got = [p["pid"] for p in rec["problems"]]
    if got != design["pids"]:
        raise ValueError("resolution record IDs differ from the frozen split")
    if rec["design"]["revision"] != design["revision"]:
        raise ValueError("resolution record revision differs from the pin")
    return {p["pid"]: p for p in rec["problems"]}


def dataset_golds(design: dict[str, Any], committed: dict[str, dict]) -> dict:
    """Golds from the pinned dataset (offline), agreeing with the record."""
    golds = {}
    for p in load_split(design):
        want = committed[p.pid]
        q = hashlib.sha256(p.question.encode()).hexdigest()
        if q != want["question_sha256"] or str(p.gold_answer) != str(want["gold"]):
            raise ValueError(f"{p.pid}: dataset differs from the resolution record")
        golds[p.pid] = str(p.gold_answer)
    return golds


def main() -> int:
    args = _parser().parse_args()
    try:
        design = resolve_design(load_config(args.config))
        committed = resolution_golds(design, args.resolution)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    split_tag = {"split": design["split"], "revision": design["revision"]}
    plan = {
        "bench": design["benchmark"],
        "expected_keys": len(expected_keys(design["pids"], design["seeds"])),
        "golds_in_record": len(committed),
        "manifest_split": split_tag,
    }
    print(json.dumps(plan, sort_keys=True))
    if args.dry_run:
        print("DRY RUN: validated only; no rows, dataset or model read")
        return 0
    if args.runs_root is None:
        raise SystemExit("REFUSED: --runs-root is required for a read")
    try:
        golds = dataset_golds(design, committed)
        read = read_timing(
            args.runs_root,
            bench=design["benchmark"],
            pids=design["pids"],
            seeds=design["seeds"],
            manifest_split=split_tag,
            golds=golds,
        )
    except ValueError as exc:
        retire_stale(args.out)
        raise SystemExit(f"REFUSED: {exc}") from exc
    read["config"] = str(args.config)
    read["runs_root"] = str(args.runs_root)
    retire_stale(args.out)
    write_json_atomic(args.out, read)
    keys = ("accuracy", "truncation_rate", "seconds_per_token", "budget_would_bind")
    print(json.dumps({k: read[k] for k in keys}, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

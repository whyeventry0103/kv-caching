#!/usr/bin/env python3
"""Emit the paper's compute statement as a machine-readable artifact.

The included set is the free-running / bank generation that produced reported
results. Excluded, each for a stated reason:

* ``math500-fc-powered-g0`` / ``-g1`` -- the two shards of the canonical bank,
  superseded by ``math500-fc-powered-merged``; counting all three would double
  the largest single item.
* ``replay-*`` -- teacher-forced, which the paper's compute sentence already
  excludes ("excluding model-load and teacher-forcing overhead").
* ``readout-*`` / ``reverse-readout-*`` -- teacher-forced; these rows carry no
  ``wallclock_s`` at all, so they contribute nothing either way.
* pilots, smokes and probes -- superseded by the corrected pipeline
  (``grid-math500-dissociation-v1`` in particular is pre-double-BOS-fix data,
  invalidated wholesale by commit 67e1194).

    python scripts/report_compute.py --runs-root <runs> --out compute.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.compute_budget import summarize_wallclock

INCLUDED = (
    "math500-fc-powered-merged",
    "fc-aime24",
    "fc-llama50",
    "fc-q14b",
    "fc-q15b",
    "grid-diss256-tok",
    "grid-diss256-h2o",
    "grid-diss256-rkv",
    "grid-diss256-llama",
    "grid-diss256-q15b",
    "grid-diss256-q14b",
    "grid-diss2048-tok",
    "grid-diss2048-h2o",
    "grid-aime24-2048",
    # Added 2026-08-05. INCLUDED is an allowlist, so every cell run after it was
    # written fell to "excluded" by default and the paper's compute sentence
    # silently stopped covering the work it reports. Found by
    # scripts/rederive_artifacts.py, which re-runs this report against the rows
    # and diffs it -- the snapshot had drifted 9 directories behind.
    "grid-diss512",  # resolution confirmatory test, section 4.1
    "grid-diss1024",  # resolution confirmatory test, section 4.1
    "bank-hottcap",  # HOTT-cap cell, section 2 (Amendment N.1)
    "grid-hottcap",  # HOTT-cap StreamingLLM arm (Amendment N.1)
    "fc-aime24-llama8b",  # 16384-cap bank; gate-failed, reported in Amendment O
    "fc-aime24-llama8b-cap32768",  # the bank section 5's R-KV cell reads against
    "rkvpub-g0",  # R-KV published configuration, section 5 (Amendment N.2)
    "rkvpub-g1",
    # The orchestrator's duplicate launch of the cell above. Its rows are NOT
    # part of any cell (that would double-count problems); they back the
    # determinism check in Amendment N.2's result. The GPU time was really
    # spent, so a compute statement that omitted it would understate the cost.
    "grid-aime24-rkv-published-cap32768",
)


def collect(runs_root: Path) -> dict[str, list[float]]:
    """Wall-clock values per run directory, from rows that record one."""
    out: dict[str, list[float]] = {}
    for name in sorted(os.listdir(runs_root)):
        directory = runs_root / name
        if not directory.is_dir():
            continue
        values: list[float] = []
        pattern = str(directory / "**" / "*.jsonl")
        for path in glob.glob(pattern, recursive=True):
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    wallclock = row.get("wallclock_s")
                    if wallclock is not None:
                        values.append(float(wallclock))
        if values:
            out[name] = values
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    per_directory = collect(args.runs_root)
    summary = summarize_wallclock(per_directory, included=INCLUDED)
    summary["note"] = (
        "Summed wall-clock over free-running/bank generation rows. Excludes "
        "teacher-forced work (readout, reverse readout, replay), the two "
        "shards superseded by the merged bank, and superseded pilots. Every "
        "directory is listed with its hours and an 'included' flag, so any "
        "exclusion can be re-added and its effect seen."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"{summary['gpu_hours']:.1f} GPU-hours over {summary['generations']} "
        f"generations ({summary['n_directories_included']} directories; "
        f"{summary['excluded_hours']:.1f} h excluded)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

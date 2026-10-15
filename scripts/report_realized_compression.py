#!/usr/bin/env python3
"""Realized compression of the S.1 grid cells (2026-10-15).

Summarizes each cell's per-row ``compression_ratio`` (``kvcache.compression_
summary``) for the protected and unprotected grids at B = 256 and B = 1024, and
writes results/diag/amendment-s/realized_compression.json atomically (the
previous output is retired first). Coverage is not re-validated here; the S.1
protection reads validate the same rows.

    python scripts/report_realized_compression.py --runs-root <runs>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.compression_summary import summarize
from kvcache.provenance import write_json_atomic
from kvcache.read_validation import retire_stale, tree_files

OUT = PROJECT_ROOT / "results/diag/amendment-s/realized_compression.json"
CELLS = {
    "b256": {
        "grid-pp256": ("random_pp", "streamingllm_pp"),
        "grid-diss256-tok": ("random", "streamingllm"),
    },
    "b1024": {
        "grid-pp1024": ("random_pp", "streamingllm_pp"),
        "grid-diss1024": ("random", "streamingllm"),
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"cells": CELLS, "out": str(OUT)}))
        print("DRY RUN: no run directory read")
        return 0
    if a.runs_root is None:
        ap.error("--runs-root is required")
    retire_stale(OUT)
    out: dict = {"read": "realized compression of the S.1 grid cells", "cells": {}}
    for budget, runs in CELLS.items():
        for run, presses in runs.items():
            rows = [
                json.loads(line)
                for f in tree_files(a.runs_root / run, "math500")
                for line in f.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for p in presses:
                sel = [r for r in rows if r["press"] == p]
                out["cells"].setdefault(budget, {})[p] = {
                    "run": run,
                    **summarize(sel),
                }
    write_json_atomic(OUT, out)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

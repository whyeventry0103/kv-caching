#!/usr/bin/env python3
"""Read an information-probe run (Meeting 5, M5-C; prepared 2026-10-14).

Validates every readout row against the FROZEN design (policies, context ids,
eviction seeds, queries -- never inferred from the rows) and prints the read:
FINAL with output-channel MI, log loss, correct/OTHER probability and greedy
accuracy per policy and query, plus paired MI contrasts against the reference
policy; PARTIAL with coverage only; INVALID failing closed (exit 2). All logic
is ``kvcache.task_information.read_information_probe``.

    python scripts/report_information_probe.py --rows <dir|file.jsonl> \\
        --design <design.json> [--out read.json]
    python scripts/report_information_probe.py --self-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.task_information import (
    INVALID,
    read_information_probe,
    reference_table,
)


def _rows(path: Path) -> list[dict]:
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    return [
        json.loads(line)
        for f in files
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path)
    p.add_argument("--design", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--self-check",
        action="store_true",
        help="print the analytic reference table (review section 10.3) and exit",
    )
    args = p.parse_args()
    if args.self_check:
        print(json.dumps(reference_table(), indent=2))
        return 0
    if args.rows is None or args.design is None:
        p.error("--rows and --design are required")
    design = json.loads(args.design.read_text(encoding="utf-8"))
    report = read_information_probe(_rows(args.rows), design)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 2 if report["status"] == INVALID else 0


if __name__ == "__main__":
    raise SystemExit(main())

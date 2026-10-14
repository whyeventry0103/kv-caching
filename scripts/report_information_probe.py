#!/usr/bin/env python3
"""Read an information-probe run (Meeting 5, M5-C; prepared 2026-10-14,
hardened after the review of 6b409001, R3).

Validates every readout row against the FROZEN design (mode, policies, context
ids, eviction seeds, queries, expected provenance -- never inferred from the
rows). ``mode: production`` rows must carry schema ``info-readout-v1``, the
design's provenance (model/tokenizer revision, answer-alphabet, task-layout
and design hashes, implementation id) and passed controls (full/full identity,
clone agreement, no readout eviction); ``mode: reference`` is reserved for
analytic fixtures. Integer keys must be integers; a declared greedy symbol must
be consistent with the stored distribution.

Output contract as every reader since R.14: the previous output is retired
BEFORE parsing; the new one is written atomically (INVALID evidence on any
failure, including a torn file); exit 0 FINAL, 3 PARTIAL (0 with --partial),
2 INVALID. All logic is ``kvcache.task_information.read_information_probe``.

    python scripts/report_information_probe.py --rows <dir|file.jsonl> \\
        --design <design.json> --out read.json
    python scripts/report_information_probe.py --self-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.provenance import write_json_atomic
from kvcache.read_validation import exit_code, files_sha256, retire_stale
from kvcache.task_information import INVALID, read_information_probe, reference_table


def _files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]


def _rows(files: list[Path]) -> list[dict]:
    out = []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    raise ValueError(f"{f.name} line {i}: malformed JSON") from None
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path)
    p.add_argument("--design", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--partial", action="store_true", help="exit 0 on PARTIAL")
    p.add_argument(
        "--self-check",
        action="store_true",
        help="print the analytic reference table (review section 10.3) and exit",
    )
    args = p.parse_args()
    if args.self_check:
        print(json.dumps(reference_table(), indent=2))
        return 0
    if args.rows is None or args.design is None or args.out is None:
        p.error("--rows, --design and --out are required")
    retired = retire_stale(args.out)
    try:
        design = json.loads(args.design.read_text(encoding="utf-8"))
        files = _files(args.rows)
        report = read_information_probe(_rows(files), design)
        report["inputs_sha256"] = files_sha256([*files, args.design])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"status": INVALID, "error": str(exc)}
        print(f"INVALID: {exc}", file=sys.stderr)
    report["retired_previous_output"] = retired.name if retired else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out, report)
    code = exit_code(report["status"], partial_ok=args.partial)
    print(f"wrote {args.out} [{report['status']}] exit {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

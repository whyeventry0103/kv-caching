#!/usr/bin/env python3
"""Read-time primary scores for GSM8K / BBH rows (Meeting 5 M5-B; review of
6b409001, R4). All logic is ``kvcache.task_scoring.rescore``.

Re-scores EVERY row of every given run directory (full cache and every
compressed arm) from its stored raw trace with the frozen grader of the
benchmark's frozen revision, and writes the scores with the grader version,
grader-source hash and per-row trace hash. The previous output is retired
first; the new one is written atomically (INVALID evidence on any failure);
exit 0 FINAL, 2 INVALID. MATH-500 is refused (its in-row grader is frozen).

    python scripts/rescore_task_rows.py --bench gsm8k --revision <sha> \\
        --gold gold.json --runs <run-dir> [<run-dir> ...] --out scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.provenance import write_json_atomic
from kvcache.read_validation import exit_code, retire_stale
from kvcache.task_scoring import INVALID, rescore


def _rows(run: Path) -> list[dict]:
    out = []
    for f in sorted(run.rglob("*.jsonl")):
        if "token_ids" in f.parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    raise ValueError(f"{f.name} line {i}: malformed JSON") from None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--gold", type=Path, help="JSON {pid: gold answer}")
    ap.add_argument("--runs", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(
            json.dumps({"bench": a.bench, "revision": a.revision, "runs": len(a.runs)})
        )
        print("DRY RUN: arguments parsed; no row read")
        return 0
    if a.out is None or a.gold is None or not a.runs:
        ap.error("--out, --gold and at least one --runs directory are required")
    retired = retire_stale(a.out)
    try:
        gold = json.loads(a.gold.read_text(encoding="utf-8"))
        rows = [r for run in a.runs for r in _rows(run)]
        payload = rescore(rows, gold, bench=a.bench, revision=a.revision)
        payload["runs"] = [str(r) for r in a.runs]
    except (OSError, ValueError, KeyError) as exc:
        payload = {"status": INVALID, "error": str(exc)}
        print(f"INVALID: {exc}", file=sys.stderr)
    payload["retired_previous_output"] = retired.name if retired else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(a.out, payload)
    code = exit_code(payload["status"], partial_ok=False)
    print(f"wrote {a.out} [{payload['status']}] exit {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

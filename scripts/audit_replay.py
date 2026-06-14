#!/usr/bin/env python3
"""Audit replay rows and scalar-only profile artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.io_schema import validate_row

FORBIDDEN_PROFILE_KEYS = {"logits", "probabilities", "keys", "values", "kv_cache"}


def _forbidden_keys(value):
    if isinstance(value, dict):
        found = FORBIDDEN_PROFILE_KEYS & value.keys()
        for child in value.values():
            found |= _forbidden_keys(child)
        return found
    if isinstance(value, list):
        found = set()
        for child in value:
            found |= _forbidden_keys(child)
        return found
    return set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.run_root.rglob("*.jsonl")):
        if "readout" in path.parts:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            row = json.loads(line)
            try:
                validate_row(row, replay=True)
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_number}: {exc}") from exc
            profile_path = args.run_root / row["tv_profile_path"]
            if not profile_path.is_file():
                raise SystemExit(f"missing TV profile: {profile_path}")
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            forbidden = _forbidden_keys(profile)
            if forbidden:
                raise SystemExit(f"forbidden profile payloads: {sorted(forbidden)}")
            lengths = {
                len(profile["estimate"]),
                len(profile["lower"]),
                len(profile["upper"]),
                len(profile["reference_token_ids"]),
            }
            if len(lengths) != 1:
                raise SystemExit(f"misaligned profile arrays: {profile_path}")
            rows.append(row)
    if len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(rows)}")
    print(json.dumps({"rows": len(rows), "status": "ok"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

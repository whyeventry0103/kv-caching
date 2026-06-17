#!/usr/bin/env python3
"""Audit a completed result bank without loading models or datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.io_schema import validate_row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument(
        "--allow-mixed-provenance",
        action="store_true",
        help="warn (not fail) on multiple source/config hashes, e.g. a merged bank "
        "from parallel halves; package versions must still be consistent",
    )
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.run_root.rglob("*.jsonl")
        if "token_ids" not in path.relative_to(args.run_root).parts
    )
    if not paths:
        raise SystemExit(f"no JSONL files found under {args.run_root}")

    rows = []
    keys = Counter()
    by_benchmark = defaultdict(list)
    versions = set()
    source_hashes = set()
    config_hashes = set()
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            row = json.loads(line)
            try:
                validate_row(row)
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_number}: {exc}") from exc
            key = (
                row["bench"],
                row["pid"],
                row["model"],
                row["press"],
                row["budget"],
                row["decoding"]["seed"],
            )
            keys[key] += 1
            rows.append(row)
            by_benchmark[row["bench"]].append(row)
            versions.add(tuple(sorted(row["versions"].items())))
            source_hashes.add(row["runmeta"].get("source_tree_hash", ""))
            config_hashes.add(row["runmeta"]["config_hash"])

    duplicates = [key for key, count in keys.items() if count != 1]
    if duplicates:
        raise SystemExit(f"duplicate result keys: {duplicates[:5]}")
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {len(rows)}")
    if len(versions) != 1:
        raise SystemExit(f"inconsistent package versions: {versions}")
    if "" in source_hashes:
        raise SystemExit("missing source_tree_hash on some rows")
    if len(source_hashes) != 1 and not args.allow_mixed_provenance:
        raise SystemExit(f"inconsistent source hashes: {source_hashes}")
    if len(source_hashes) != 1:
        print(f"WARNING: {len(source_hashes)} source hashes (mixed provenance allowed)")
    if len(config_hashes) != 1 and args.allow_mixed_provenance:
        print(f"WARNING: {len(config_hashes)} config hashes (mixed provenance allowed)")
    elif len(config_hashes) != 1:
        raise SystemExit(f"inconsistent config hashes: {config_hashes}")

    summary = {
        "files": len(paths),
        "rows": len(rows),
        "versions": dict(next(iter(versions))),
        "source_tree_hash": next(
            iter(source_hashes),
        ),
        "config_hash": next(iter(config_hashes)),
        "benchmarks": {},
    }
    for benchmark, benchmark_rows in sorted(by_benchmark.items()):
        summary["benchmarks"][benchmark] = {
            "rows": len(benchmark_rows),
            "correct": sum(bool(row["correct"]) for row in benchmark_rows),
            "accuracy": sum(bool(row["correct"]) for row in benchmark_rows)
            / len(benchmark_rows),
            "truncated": sum(bool(row["truncated"]) for row in benchmark_rows),
            "truncation_rate": sum(bool(row["truncated"]) for row in benchmark_rows)
            / len(benchmark_rows),
            "mean_tokens": sum(row["n_tokens"] for row in benchmark_rows)
            / len(benchmark_rows),
        }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

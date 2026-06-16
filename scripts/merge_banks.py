#!/usr/bin/env python3
"""Merge split bank run dirs (e.g. the two powered FC-bank halves) into one.

The halves hold disjoint seed files (g0: s11.. , g1: s131..), so merging is a copy
with collision checks. Validates that every (bench, pid, seed) key is unique across
sources and that result rows and token-ID sidecars stay 1:1, then copies result rows
and token_ids/ into the output dir. Pure file ops; no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def _keys(root: Path, subdir: str) -> Counter:
    keys: Counter = Counter()
    base = root / subdir
    for path in base.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seed = row.get("decoding", {}).get("seed", row.get("seed"))
            keys[(row["bench"], row["pid"], seed)] += 1
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bench", default="math500")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")

    result_keys: Counter = Counter()
    token_keys: Counter = Counter()
    for src in args.sources:
        if not src.is_dir():
            raise SystemExit(f"missing source: {src}")
        result_keys += _keys(src, args.bench)
        token_keys += _keys(src, "token_ids")

    dup = [k for k, c in result_keys.items() if c != 1]
    if dup:
        raise SystemExit(f"duplicate (bench,pid,seed) across sources: {dup[:5]}")
    if set(result_keys) != set(token_keys):
        raise SystemExit("result rows and token-ID sidecars are not 1:1 across sources")

    # Copy result rows and token_ids; disjoint seed filenames => no clobber.
    for src in args.sources:
        for subdir in (args.bench, "token_ids"):
            for path in (src / subdir).rglob("*.jsonl"):
                rel = path.relative_to(src)
                dest = args.out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    raise SystemExit(f"file collision on merge: {dest}")
                shutil.copy2(path, dest)

    print(
        json.dumps(
            {
                "merged_rows": sum(result_keys.values()),
                "token_records": sum(token_keys.values()),
                "sources": [str(s) for s in args.sources],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

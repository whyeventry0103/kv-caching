#!/usr/bin/env python3
"""Two-full-cache negative control: prove the ADD instrument does not false-positive.

Split ONE full-cache bank's n samples per problem into two disjoint seed halves and run
ADD between them. Both halves are draws from the SAME full-cache distribution, so a
valid divergence instrument must read ~null. Compare that floor to the real compression
cells (ADD ~0.56-0.75): if the floor is far below, ADD's compression signal is real and
not a finite-sample artifact. A committed credibility artifact (the figure proving the
instruments don't false-positive); needs no GPU -- it reuses the existing FC bank.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.add import add_tv, coarsen_answer, excess_add_permutation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullcache-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pattern = str(args.fullcache_root / "math500" / "**" / "*.jsonl")
    rows = [
        json.loads(line)
        for path in glob.glob(pattern, recursive=True)
        for line in open(path, encoding="utf-8")
        if line.strip()
    ]
    seeds = sorted({r["decoding"]["seed"] for r in rows})
    half = len(seeds) // 2
    group_a, group_b = set(seeds[:half]), set(seeds[half:])

    by_pid: dict[str, tuple[list, list]] = collections.defaultdict(lambda: ([], []))
    for r in rows:
        answer = coarsen_answer(r.get("answer_raw") or None, truncated=r["truncated"])
        which = 0 if r["decoding"]["seed"] in group_a else 1
        by_pid[r["pid"]][which].append(answer)

    adds, pvals = [], []
    for a, b in by_pid.values():
        if a and b:
            adds.append(add_tv(a, b))
            _, p = excess_add_permutation(a, b, seed=0)
            pvals.append(p)

    summary = {
        "n_problems": len(adds),
        "seeds_a": sorted(group_a),
        "seeds_b": sorted(group_b),
        "mean_add": statistics.fmean(adds),
        "median_add": statistics.median(adds),
        "mean_perm_p": statistics.fmean(pvals),
        "frac_p_below_05": sum(p < 0.05 for p in pvals) / len(pvals),
        "interpretation": (
            "Two draws from the SAME full-cache distribution. mean_add is the ADD "
            "noise floor; compression cells (~0.56-0.75) sit far above it. "
            "frac_p_below_05 should be ~0.05 under the null."
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

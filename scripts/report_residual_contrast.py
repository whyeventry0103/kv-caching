#!/usr/bin/env python3
"""Emit the paired cross-policy residual contrast as an artifact.

The paper's method-invariance sentence sets a cross-policy residual spread
(0.008) against CI half-widths of about 0.07 and concludes the residuals are
alike. Those two numbers are not comparable: the spread is a paired quantity
measured on one set of problems, the half-widths are unpaired marginal
intervals. This computes the contrast the claim actually needs -- per-problem,
paired on the problem, with its own interval.

The per-problem residual is ``free - readout`` on that problem, because the
full-cache reference cancels from ``(free - Acc_FC) - (readout - Acc_FC)``.
That is what makes this contrast independent of the reference-population choice
Amendment K corrected.

A problem contributes only where BOTH its free-run and its readout accuracy are
defined for the press in question; a readout row with no answer boundary has no
residual to contribute. Presses are then intersected on problems, so every pair
is compared on identical problems. Both counts are reported.

    python scripts/report_residual_contrast.py --runs-root <runs> \\
        --cell random:grid-diss256-tok:readout-diss256 \\
        --cell streamingllm:grid-diss256-tok:readout-diss256 \\
        --budget 256 --out residual_contrast.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.stats import paired_residual_contrast


def _jsonl(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        if "token_ids" in Path(path).parts:
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def free_by_problem(
    runs_root: Path, run: str, bench: str, press: str, budget: str
) -> dict[str, list[bool]]:
    # A grid row carries press/budget at the top level; a readout row carries
    # them under "condition". Two schemas, one concept -- matching the wrong one
    # silently yields an empty cell rather than an error, so both are asserted
    # by the caller via the non-empty check.
    out: dict[str, list[bool]] = collections.defaultdict(list)
    for row in _jsonl(str(runs_root / run / bench / "**" / "*.jsonl")):
        if str(row.get("press")) != press or str(row.get("budget")) != budget:
            continue
        out[row["pid"]].append(bool(row["correct"]))
    return out


def readout_by_problem(
    runs_root: Path, runs: str, press: str, budget: str
) -> dict[str, list[bool]]:
    """Readout accuracy per problem, over rows with a defined answer boundary.

    ``runs`` may be several run names joined by "+": the budget-256 readout for
    the base cell was sharded across two GPUs (``-g0``/``-g1``) and the shards
    together are one cell.
    """
    out: dict[str, list[bool]] = collections.defaultdict(list)
    for run in runs.split("+"):
        for row in _jsonl(str(runs_root / run / "readout" / "**" / "*.jsonl")):
            cond = row.get("condition") or {}
            if str(cond.get("press")) != press or str(cond.get("budget")) != budget:
                continue
            if row.get("readout_correct") is None:
                continue
            out[str(row["reference_key"][1])].append(bool(row["readout_correct"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        dest="cells",
        metavar="PRESS:GRID_RUN:READOUT_RUN",
    )
    parser.add_argument("--budget", required=True)
    parser.add_argument("--bench", default="math500")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    residuals: dict[str, dict[str, float]] = {}
    coverage: dict[str, Any] = {}
    for spec in args.cells:
        press, grid_run, readout_run = spec.split(":")
        free = free_by_problem(args.runs_root, grid_run, args.bench, press, args.budget)
        readout = readout_by_problem(args.runs_root, readout_run, press, args.budget)
        if not free:
            raise SystemExit(
                f"no free-run rows for {press} @{args.budget} in {grid_run}"
            )
        if not readout:
            raise SystemExit(
                f"no readout rows for {press} @{args.budget} in {readout_run}"
            )
        common = sorted(set(free) & set(readout))
        residuals[press] = {
            pid: statistics.fmean(float(v) for v in free[pid])
            - statistics.fmean(float(v) for v in readout[pid])
            for pid in common
        }
        coverage[press] = {
            "n_problems_free": len(free),
            "n_problems_readout_defined": len(readout),
            "n_problems_used": len(common),
        }

    result = paired_residual_contrast(
        residuals, n_resamples=args.n_resamples, seed=args.seed
    )
    result["budget"] = args.budget
    result["benchmark"] = args.bench
    result["coverage"] = coverage
    result["note"] = (
        "Per-problem residual = free - readout; the full-cache reference cancels, "
        "so this contrast is independent of the Amendment K reference choice. "
        "Report a pair as differing only if its paired CI excludes zero -- the "
        "marginal accuracy CIs are not the right comparison for a paired quantity."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(
        f"  mean |pairwise diff| = {result['mean_abs_pairwise_difference']:.4f} "
        f"CI {[round(x, 4) for x in result['mean_abs_pairwise_difference_ci95']]}"
    )
    for name, pair in sorted(result["pairs"].items()):
        flag = "EXCLUDES ZERO" if pair["excludes_zero"] else "spans zero"
        print(
            f"  {name:34} {pair['mean_difference']:+.4f} "
            f"[{pair['ci95'][0]:+.4f}, {pair['ci95'][1]:+.4f}]  {flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

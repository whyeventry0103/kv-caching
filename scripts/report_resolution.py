#!/usr/bin/env python3
"""Emit the resolution-collapse comparison for one regime as an artifact.

POST-HOC AND EXPLORATORY (PREREGISTRATION Amendment L). The pattern was noticed
after all cells were read; this script exists so the noticing is measured rather
than eyeballed, and so the claim carries an interval.

The question: how much of the policy separation visible under free-running
survives held-trace readout? Both spreads are accuracy-scale quantities on the
same problems, so this compares readout to readout -- the off-policy caveat that
applies to the residual does not apply here.

    python scripts/report_resolution.py --runs-root <runs> \\
        --cell random:grid-diss256-tok:readout-diss256-g0+readout-diss256-g1 \\
        --budget 256 --label math500-256 --out resolution_256.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.stats import resolution_spread

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from report_residual_contrast import free_by_problem, readout_by_problem


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
    parser.add_argument("--label", required=True)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    free: dict[str, dict[str, float]] = {}
    readout: dict[str, dict[str, float]] = {}
    coverage: dict[str, dict[str, int]] = {}
    for spec in args.cells:
        press, grid_run, readout_run = spec.split(":")
        f = free_by_problem(args.runs_root, grid_run, args.bench, press, args.budget)
        r = readout_by_problem(args.runs_root, readout_run, press, args.budget)
        if not f:
            raise SystemExit(f"no free-run rows for {press} @{args.budget}")
        if not r:
            raise SystemExit(f"no readout rows for {press} @{args.budget}")
        free[press] = {p: statistics.fmean(float(v) for v in vs) for p, vs in f.items()}
        readout[press] = {
            p: statistics.fmean(float(v) for v in vs) for p, vs in r.items()
        }
        coverage[press] = {
            "n_problems_free": len(f),
            "n_problems_readout_defined": len(r),
        }

    result = resolution_spread(
        free, readout, n_resamples=args.n_resamples, seed=args.seed
    )
    result["label"] = args.label
    result["budget"] = args.budget
    result["benchmark"] = args.bench
    result["coverage"] = coverage
    result["note"] = (
        "Readout spread vs free-run spread, mean absolute pairwise difference, "
        "paired on the problem. The DIFFERENCE carries the CI; the ratio is "
        "descriptive only, because at the saturated end the denominator "
        "approaches zero. Post-hoc: label exploratory wherever cited."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(
        f"  {args.label}: free-run spread {result['free_run_spread']:.4f}, "
        f"readout spread {result['readout_spread']:.4f}"
    )
    verdict = (
        "EXCLUDES ZERO" if result["spread_difference_excludes_zero"] else "spans zero"
    )
    print(
        f"  difference {result['spread_difference']:+.4f} "
        f"CI [{result['spread_difference_ci95'][0]:+.4f}, "
        f"{result['spread_difference_ci95'][1]:+.4f}] {verdict}"
    )
    ratio = result["spread_ratio_descriptive"]
    print(
        f"  ratio (descriptive) {ratio:.4f}"
        if ratio is not None
        else "  ratio undefined"
    )
    pairs = result["n_press_pairs"]
    print(f"  rank agreement {result['rank_agreement']:.3f} over {pairs} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

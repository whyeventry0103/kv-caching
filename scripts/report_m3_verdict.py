#!/usr/bin/env python3
"""Amendment M.3: does the divergence staircase survive at a FIXED ratio?

Compares the ratio-matched cell against the fixed-budget cell on the SAME
problems, over the SAME step range, and reports the drift statistic that the
verdict turns on.

Two restrictions, both required and both decided before the data landed:

* **Above the floor only.** A `min_budget` of 32 at ratio 0.038 keeps binding
  until the logical context reaches ~842 tokens. Below that the ratio-matched
  press is a fixed-32 budget whose realized ratio WORSENS as context grows --
  the very confound M.3 removes -- and that period sits at the START of every
  trace, where it would dominate a first-quarter/last-quarter drift ratio.
  Each trace is therefore trimmed to begin where its own press left the floor.
* **The same window in both cells.** Trimming only the ratio-matched cell would
  compare its late trace against the fixed cell's whole trace, and divergence
  varies over a trace. Both are cut at the same absolute step.

The drift ratio is (mean of the last quarter) / (mean of the first quarter) of
the retained window. >1 means divergence rises across the trace (the staircase);
~1 or below means it does not.

    python scripts/report_m3_verdict.py --runs-root <runs> \\
        --matched replay-ratio-matched-m3-v2 --fixed replay-ext-diss256 \\
        --out m3_verdict.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.profile_shape import (  # noqa: E402
    drift_normalized_curve,
    floor_exit_step,
)
from kvcache.replay import is_profile_file  # noqa: E402
from kvcache.stats import cluster_bootstrap_mean  # noqa: E402

CELL = re.compile(r"^(?P<press>[a-z0-9]+)-(?P<budget>\d+)-s(?P<seed>\d+)$")
MIN_WINDOW = 200  # a drift ratio over fewer steps than this is not reported


# The windowing lives in the package so this reader and the P.2-v2 phase
# re-analysis cannot drift apart -- their verdicts are only comparable if the
# window is literally the same code.
_floor_exit_step = floor_exit_step


def _drift(series: list[float]) -> float | None:
    """Last-quarter mean over first-quarter mean; None if either is degenerate."""
    q = len(series) // 4
    if q < 1:
        return None
    first = statistics.fmean(series[:q])
    last = statistics.fmean(series[-q:])
    if first <= 0:
        return None
    return last / first


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--matched", required=True, help="ratio-matched run name")
    p.add_argument("--fixed", required=True, help="fixed-budget run name")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--curves-out",
        type=Path,
        default=None,
        help=(
            "optional SECOND artifact holding the per-arm drift-normalized mean "
            "curves for the figure. Additive: --out is byte-identical whether or "
            "not this is passed, so the committed verdict never moves."
        ),
    )
    p.add_argument("--curve-bins", type=int, default=40)
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    matched_root = a.runs_root / a.matched / "profiles"
    fixed_root = a.runs_root / a.fixed / "profiles"

    # press -> pid -> list of (matched, fixed) drift pairs, one per SEED.
    # Keying by pid alone would let each seed overwrite the last and silently
    # discard three quarters of the data; the pid is the clustering unit, not
    # the observation.
    paired: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    curves: dict[str, dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    never_left_floor = 0
    too_short = 0
    unpaired = 0
    coverage: list[float] = []

    for path in sorted(matched_root.rglob("*.json")):
        if not is_profile_file(path):
            continue
        m = CELL.match(path.parent.name)
        if not m:
            continue
        trace_path = path.with_name(path.stem + ".ratio_trace.json")
        if not trace_path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        interval = int(payload.get("compression_interval") or 64)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))["trace"]
        start = _floor_exit_step(trace, interval)
        if start is None:
            never_left_floor += 1
            continue

        counterpart = fixed_root / path.relative_to(matched_root)
        if not counterpart.exists():
            unpaired += 1
            continue
        fixed_payload = json.loads(counterpart.read_text(encoding="utf-8"))

        mine = payload["estimate"][start:]
        theirs = fixed_payload["estimate"][start:]
        if len(mine) < MIN_WINDOW or len(theirs) < MIN_WINDOW:
            too_short += 1
            continue
        coverage.append(len(mine) / len(payload["estimate"]))

        d_m, d_f = _drift(mine), _drift(theirs)
        if d_m is None or d_f is None:
            too_short += 1
            continue
        paired[m.group("press")][path.stem].append((d_m, d_f))
        c_m = drift_normalized_curve(mine, a.curve_bins)
        c_f = drift_normalized_curve(theirs, a.curve_bins)
        if c_m is not None and c_f is not None:
            curves[m.group("press")]["matched"].append(c_m)
            curves[m.group("press")]["fixed"].append(c_f)

    if not paired:
        raise SystemExit("no paired profiles survived the floor and length filters")

    out: dict[str, Any] = {
        "amendment": "M.3",
        "matched_run": a.matched,
        "fixed_run": a.fixed,
        "restriction": (
            "each trace trimmed to begin where its own ratio-matched press left "
            "min_budget; the fixed-budget counterpart is cut at the SAME step"
        ),
        "min_window": MIN_WINDOW,
        "excluded": {
            "never_left_floor": never_left_floor,
            "window_too_short": too_short,
            "no_fixed_counterpart": unpaired,
        },
        "retained_fraction_of_trace_median": (
            round(statistics.median(coverage), 3) if coverage else None
        ),
        "per_press": {},
    }

    for press, per_pid in sorted(paired.items()):
        pids = sorted(per_pid)
        # Problem-clustered bootstrap on the PAIRED difference, the repo's
        # standing convention (2000 resamples, seed 0, percentile CI).
        by_pid_matched = {k: [m for m, _ in per_pid[k]] for k in pids}
        by_pid_fixed = {k: [f for _, f in per_pid[k]] for k in pids}
        by_pid_diff = {k: [m - f for m, f in per_pid[k]] for k in pids}
        stat_m = cluster_bootstrap_mean(
            by_pid_matched, n_resamples=a.n_resamples, seed=a.seed
        )
        stat_f = cluster_bootstrap_mean(
            by_pid_fixed, n_resamples=a.n_resamples, seed=a.seed
        )
        stat_d = cluster_bootstrap_mean(
            by_pid_diff, n_resamples=a.n_resamples, seed=a.seed
        )
        matched_rises = stat_m["ci95"][0] > 1.0
        fixed_rises = stat_f["ci95"][0] > 1.0
        if fixed_rises and not matched_rises:
            verdict = "STAIRCASE FLATTENS AT FIXED RATIO -> ratio effect"
        elif fixed_rises and matched_rises:
            verdict = "STAIRCASE SURVIVES AT FIXED RATIO -> horizon reading stands"
        elif not fixed_rises:
            verdict = "NO STAIRCASE IN THE FIXED CELL OVER THIS WINDOW -> not readable"
        else:
            verdict = "INDETERMINATE"
        out["per_press"][press] = {
            "n_problems": len(pids),
            "n_observations": sum(len(per_pid[k]) for k in pids),
            "drift_matched": stat_m,
            "drift_fixed": stat_f,
            "drift_difference_matched_minus_fixed": stat_d,
            "fixed_cell_rises": fixed_rises,
            "matched_cell_rises": matched_rises,
            "verdict": verdict,
        }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")

    if a.curves_out is not None:
        # A SEPARATE artifact on purpose. Adding a field to m3_verdict.json
        # would move a committed result's hash for a presentational reason,
        # and the verdict is cited by the paper; this file is additive and the
        # verdict stays byte-identical.
        curve_doc: dict[str, Any] = {
            "amendment": "M.3 (figure support)",
            "derived_from": a.out.name,
            "matched_run": a.matched,
            "fixed_run": a.fixed,
            "restriction": out["restriction"],
            "min_window": MIN_WINDOW,
            "n_bins": a.curve_bins,
            "quantity": (
                "per-trace per-step TV binned over the retained window and "
                "divided by that trace's own first-quarter mean, then averaged "
                "across traces. Bin values are DRIFT-RELATIVE, not TV levels: "
                "the two arms sit at different levels because the ratio-matched "
                "press compresses far harder over this window, and no "
                "measurement in this project licenses a level comparison "
                "between them."
            ),
            "per_press": {},
        }
        for press in sorted(curves):
            entry: dict[str, Any] = {}
            for arm in ("fixed", "matched"):
                rows = curves[press][arm]
                entry[arm] = {
                    "n_traces": len(rows),
                    "mean_curve": [
                        statistics.fmean([r[b] for r in rows])
                        for b in range(a.curve_bins)
                    ],
                }
            curve_doc["per_press"][press] = entry
        a.curves_out.parent.mkdir(parents=True, exist_ok=True)
        a.curves_out.write_text(
            json.dumps(curve_doc, indent=2, sort_keys=True) + "\n", "utf-8"
        )
        print(f"wrote {a.curves_out}")

    print(f"retained median {out['retained_fraction_of_trace_median']} of each trace")
    print(f"excluded: {out['excluded']}")
    for press, s in out["per_press"].items():
        m, f = s["drift_matched"], s["drift_fixed"]
        print(
            f"{press:<14} pids={s['n_problems']:<3} obs={s['n_observations']:<4} "
            f"drift_fixed={f['mean']:.3f}{[round(x, 3) for x in f['ci95']]} "
            f"drift_matched={m['mean']:.3f}{[round(x, 3) for x in m['ci95']]}"
        )
        print(f"{'':14} -> {s['verdict']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

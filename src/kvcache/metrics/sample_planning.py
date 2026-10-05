"""Sample-size planning for new confirmation designs (execution plan §9.2).

Scientific rationale. The inferential unit is the problem: seeds are nested in
problems and averaged first, so four seeds do not make 100 problems into 400
units. Planning therefore resamples PROBLEMS from discovery data (the S.1
conditions), keeps each problem's own seed-level values, subsamples seeds to the
planned count, recentres the per-condition values on a target effect, and
measures, over simulated studies, the equal-problem estimator's 95% half-width
and the share of studies whose interval excludes zero.

These are planning approximations: they borrow the discovery data's spread
between and within problems. A new model or task can have a different spread,
and the numbers are not a promise of power there. The discovery data are used
for variance only; the effect is set by hand (0, 0.03, 0.05).
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def per_problem_values(
    values: Mapping[tuple[str, int], float],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for (pid, _seed), v in sorted(values.items()):
        out.setdefault(pid, []).append(float(v))
    return out


def problem_sd(per_problem: Mapping[str, Sequence[float]]) -> float:
    """SD across problems of the per-problem mean (the planning s of §9.2)."""
    means = [statistics.fmean(v) for v in per_problem.values()]
    return statistics.stdev(means)


def normal_n(sd: float, half_width: float) -> int:
    """Rough problems needed for a 95% half-width: (1.96 s / h)^2."""
    return math.ceil((1.96 * sd / half_width) ** 2)


def simulate(
    per_problem: Mapping[str, Sequence[float]],
    *,
    n_problems: int,
    n_seeds: int,
    effect: float,
    reps: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulated equal-problem studies of ``n_problems`` x ``n_seeds``.

    Each study draws problems with replacement, then ``n_seeds`` of that
    problem's observed seed values without replacement (all of them if it has
    fewer), and shifts every value by ``effect - observed mean`` so the study's
    true mean is ``effect``. The interval is the normal approximation of the
    equal-problem mean's sampling error, used here for speed.
    """
    pids = sorted(per_problem)
    if not pids:
        raise ValueError("no discovery data")
    centre = statistics.fmean(statistics.fmean(v) for v in per_problem.values())
    shift = effect - centre
    rng = random.Random(seed)
    halfs, hits = [], 0
    for _ in range(reps):
        means = []
        for _ in range(n_problems):
            vals = list(per_problem[pids[rng.randrange(len(pids))]])
            take = vals if len(vals) <= n_seeds else rng.sample(vals, n_seeds)
            means.append(statistics.fmean(take) + shift)
        m = statistics.fmean(means)
        half = 1.96 * statistics.stdev(means) / math.sqrt(n_problems)
        halfs.append(half)
        hits += (m - half > 0) or (m + half < 0)
    return {
        "n_problems": n_problems,
        "n_seeds": n_seeds,
        "effect": effect,
        "mean_half_width": statistics.fmean(halfs),
        "share_excluding_zero": hits / reps,
        "reps": reps,
    }

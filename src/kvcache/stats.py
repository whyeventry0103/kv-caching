"""Statistical utilities for preregistered analyses."""

from __future__ import annotations

import collections
import itertools
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def bootstrap(
    pairs: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if not pairs:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    values = sorted(
        statistic([pairs[rng.randrange(len(pairs))] for _ in pairs])
        for _ in range(n_resamples)
    )
    return (
        values[int(0.025 * (n_resamples - 1))],
        values[int(0.975 * (n_resamples - 1))],
    )


def exact_permutation_test(
    a: Sequence[T],
    b: Sequence[T],
    statistic: Callable[[Sequence[T], Sequence[T]], float],
    *,
    max_exact: int = 100_000,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return observed statistic and one-sided per-problem permutation p-value."""
    pooled = list(a) + list(b)
    n_a = len(a)
    observed = statistic(a, b)
    combinations = math.comb(len(pooled), n_a)
    is_exact = combinations <= max_exact
    if is_exact:
        assignments = itertools.combinations(range(len(pooled)), n_a)
    else:
        rng = random.Random(seed)
        assignments = (
            tuple(sorted(rng.sample(range(len(pooled)), n_a)))
            for _ in range(n_resamples)
        )
    exceed = 0
    count = 0
    all_indices = set(range(len(pooled)))
    for selected_tuple in assignments:
        selected = set(selected_tuple)
        perm_a = [pooled[i] for i in selected]
        perm_b = [pooled[i] for i in all_indices - selected]
        exceed += statistic(perm_a, perm_b) >= observed - 1e-15
        count += 1
    p_value = exceed / count if is_exact else (exceed + 1) / (count + 1)
    return observed, p_value


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=p_values.__getitem__)
    adjusted = [0.0] * m
    running = 1.0
    for rank_from_end, index in enumerate(reversed(order), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = running
    return adjusted


def negative_control_ok(
    excess_add: float,
    tdp_percentile: float,
    *,
    add_tolerance: float = 0.05,
    tdp_floor: float = 0.95,
) -> bool:
    return abs(excess_add) <= add_tolerance and tdp_percentile >= tdp_floor


def paired_policy_contrast(
    a_by_problem: Mapping[str, Sequence[bool]],
    b_by_problem: Mapping[str, Sequence[bool]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired per-problem accuracy contrast between two policies.

    Supports the paper's random-beats-recency claim, which is stated in three
    parts that must agree: the mean difference, its problem-clustered CI, and
    the per-problem win/tie/loss breakdown. The breakdown exists because
    "non-negative on 94/100" is the honest phrasing of that result -- it is not
    94 wins (PAPER_FINDINGS §5), and reporting it as such is a standing rule.

    Both policies must cover the same problems; the contrast is paired on the
    problem, which is the sampling unit used everywhere else in this repo.

    The returned ``seed``/``n_resamples`` and ``difference_histogram`` exist for
    provenance, and they are load-bearing: the percentile CI is a Monte-Carlo
    quantity, so an unpinned seed makes it irreproducible. The published
    ``[0.130, 0.242]`` could not be reproduced exactly for that reason -- over
    200 seeds the upper endpoint spans [0.2375, 0.2450] -- and the histogram is
    a sufficient statistic for the whole bootstrap, so any reader can recompute
    the interval from the artifact alone.
    """
    problems = sorted(set(a_by_problem) & set(b_by_problem))
    if not problems:
        raise ValueError("no problems in common between the two policies")
    missing = (set(a_by_problem) | set(b_by_problem)) - set(problems)
    per_problem = {
        pid: statistics.fmean(float(bool(v)) for v in a_by_problem[pid])
        - statistics.fmean(float(bool(v)) for v in b_by_problem[pid])
        for pid in problems
    }
    diffs = [per_problem[pid] for pid in problems]
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    losses = sum(1 for d in diffs if d < 0)

    rng = random.Random(seed)
    k = len(problems)
    means = sorted(
        statistics.fmean(rng.choice(diffs) for _ in range(k))
        for _ in range(n_resamples)
    )
    lo = means[int(0.025 * (n_resamples - 1))]
    hi = means[int(0.975 * (n_resamples - 1))]
    histogram = collections.Counter(diffs)
    return {
        "n_problems": k,
        "mean_difference": statistics.fmean(diffs),
        "ci95": [round(lo, 4), round(hi, 4)],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "non_negative": wins + ties,
        "n_problems_dropped": len(missing),
        "seed": seed,
        "n_resamples": n_resamples,
        "difference_histogram": [
            [value, histogram[value]] for value in sorted(histogram)
        ],
    }

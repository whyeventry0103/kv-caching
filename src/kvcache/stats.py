"""Statistical utilities for preregistered analyses."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence
from typing import TypeVar

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

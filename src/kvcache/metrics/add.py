"""Answer Distribution Distortion metrics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from kvcache.stats import exact_permutation_test


def answer_distribution(samples: Sequence[str]) -> dict[str, float]:
    if not samples:
        raise ValueError("answer samples cannot be empty")
    counts = Counter(samples)
    return {answer: count / len(samples) for answer, count in counts.items()}


def distribution_tv(a: dict[str, float], b: dict[str, float]) -> float:
    return 0.5 * sum(abs(a.get(x, 0.0) - b.get(x, 0.0)) for x in a.keys() | b.keys())


def add_tv(full: Sequence[str], compressed: Sequence[str]) -> float:
    return distribution_tv(answer_distribution(full), answer_distribution(compressed))


def greedy_flip_rate(full: Sequence[str], compressed: Sequence[str]) -> float:
    """Greedy decoding uses paired answer flips, not sampled-inference ADD."""
    if len(full) != len(compressed):
        raise ValueError("paired greedy answers must have equal length")
    return (
        sum(a != b for a, b in zip(full, compressed, strict=True)) / len(full)
        if full
        else 0.0
    )


def excess_add_permutation(
    full: Sequence[str],
    compressed: Sequence[str],
    *,
    seed: int = 0,
) -> tuple[float, float]:
    return exact_permutation_test(full, compressed, add_tv, seed=seed)


def entropy_miller_madow(samples: Sequence[str]) -> float:
    distribution = answer_distribution(samples)
    entropy = -sum(p * math.log(p) for p in distribution.values() if p)
    return entropy + (len(distribution) - 1) / (2 * len(samples))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Codex pass@k estimator."""
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

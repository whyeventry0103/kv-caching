"""Answer Distribution Distortion metrics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from kvcache.stats import exact_permutation_test

OTHER_ANSWER = "__OTHER__"
_INVALID_ANSWERS = {"", "none", "null", "nan", "invalid", "unknown"}


def coarsen_answer(
    answer: str | None,
    *,
    valid_support: set[str] | None = None,
    truncated: bool = False,
) -> str:
    """Map missing, truncated, or out-of-support answers to one frozen bucket."""
    if truncated or answer is None:
        return OTHER_ANSWER
    normalized = answer.strip()
    if normalized.lower() in _INVALID_ANSWERS:
        return OTHER_ANSWER
    if valid_support is not None and normalized not in valid_support:
        return OTHER_ANSWER
    return normalized


def answer_distribution(
    samples: Sequence[str | None],
    *,
    valid_support: set[str] | None = None,
) -> dict[str, float]:
    if not samples:
        raise ValueError("answer samples cannot be empty")
    counts = Counter(
        coarsen_answer(answer, valid_support=valid_support) for answer in samples
    )
    return {answer: count / len(samples) for answer, count in counts.items()}


def distribution_tv(a: dict[str, float], b: dict[str, float]) -> float:
    return 0.5 * sum(abs(a.get(x, 0.0) - b.get(x, 0.0)) for x in a.keys() | b.keys())


def add_tv(
    full: Sequence[str | None],
    compressed: Sequence[str | None],
    *,
    valid_support: set[str] | None = None,
) -> float:
    return distribution_tv(
        answer_distribution(full, valid_support=valid_support),
        answer_distribution(compressed, valid_support=valid_support),
    )


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


def truncation_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the retained-row truncation rate for a result cell."""
    if not rows:
        return 0.0
    return sum(bool(row["truncated"]) for row in rows) / len(rows)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Codex pass@k estimator."""
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

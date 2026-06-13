"""Paired teacher-forced replay with stepwise total variation.

Full-vocabulary distributions are consumed one step at a time and never stored.
For the top-k fallback, probabilities are queried on the union of both top-k
supports and the remaining mass is collapsed into an unknown tail. Its TV
contribution is between `abs(tail_p-tail_q)/2` and `(tail_p+tail_q)/2`. The
midpoint estimate is returned with those lower and upper bounds.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundedTV:
    estimate: float
    lower: float
    upper: float


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("distributions must have equal support")
    if abs(sum(p) - 1.0) > 1e-6 or abs(sum(q) - 1.0) > 1e-6:
        raise ValueError("distributions must sum to one")
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


def topk_tv_bounds(
    p: dict[int, float],
    q: dict[int, float],
    *,
    p_tail: float,
    q_tail: float,
) -> BoundedTV:
    if p.keys() != q.keys():
        raise ValueError("query both distributions on the union top-k support")
    if min(p_tail, q_tail, *p.values(), *q.values()) < 0:
        raise ValueError("probabilities cannot be negative")
    if abs(sum(p.values()) + p_tail - 1.0) > 1e-6:
        raise ValueError("p visible probabilities and tail must sum to one")
    if abs(sum(q.values()) + q_tail - 1.0) > 1e-6:
        raise ValueError("q visible probabilities and tail must sum to one")
    visible = 0.5 * sum(abs(p[i] - q[i]) for i in p)
    tail_lower = 0.5 * abs(p_tail - q_tail)
    tail_upper = 0.5 * (p_tail + q_tail)
    lower = visible + tail_lower
    upper = min(1.0, visible + tail_upper)
    return BoundedTV(estimate=(lower + upper) / 2, lower=lower, upper=upper)


def replay_tv_profile(
    full_steps: Iterable[Sequence[float]],
    compressed_steps: Iterable[Sequence[float]],
) -> list[float]:
    """Compute and retain only one scalar TV per paired replay step."""
    profile = []
    for full, compressed in zip(full_steps, compressed_steps, strict=True):
        profile.append(total_variation(full, compressed))
    return profile

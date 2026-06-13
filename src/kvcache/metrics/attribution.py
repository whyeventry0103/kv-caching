"""Evicted-anchor attribution and damage-latency summaries."""

from __future__ import annotations

from collections.abc import Sequence


def damage_latency(eviction_step: int, divergence_step: int) -> int:
    if divergence_step < eviction_step:
        raise ValueError("divergence cannot precede the attributed eviction")
    return divergence_step - eviction_step


def top_evicted_anchors(scores: Sequence[tuple[int, float]], k: int = 10) -> list[int]:
    return [
        index for index, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:k]
    ]

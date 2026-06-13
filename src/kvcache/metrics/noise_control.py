"""TV-matched full-cache noise control calibration."""

from __future__ import annotations

from collections.abc import Sequence


def calibration_error(target: Sequence[float], achieved: Sequence[float]) -> float:
    if len(target) != len(achieved):
        raise ValueError("TV profiles must have equal length")
    return (
        sum(abs(a - b) for a, b in zip(target, achieved, strict=True)) / len(target)
        if target
        else 0.0
    )

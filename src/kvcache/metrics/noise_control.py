"""TV-matched full-cache noise control: the on-policy top rung of the causal ladder.

CONTRACT (mandatory, not optional). During full-cache free-running generation,
inject calibrated per-step perturbations ``(1 - eps) * p + eps * q`` whose induced
per-step TV is matched to the profile measured for method ``M`` under replay. The
test:

* if matched *generic* noise reproduces ``M``'s accuracy change, then the
  magnitude of readout distortion suffices and any compounding is generic;
* if it does not, ``M``'s damage is *structured* (e.g. anchor deletion, not
  noise) — the mechanism claim, proven interventionally, on-policy, with the
  generation feedback loop intact.

This simultaneously answers the off-policy objection to readout/TDP and gives the
ladder its strongest rung; it is treated as required, not a nice-to-have. Like
every instrument here, its accuracy effect is reported with signed flip-mass so a
reproduced *improvement* is not mislabeled as damage.

This module currently exposes only the calibration objective (matching achieved
TV to the target profile). The injection/generation engine is a GPU/M7 deliverable.
"""

from __future__ import annotations

from collections.abc import Sequence


def calibration_error(target: Sequence[float], achieved: Sequence[float]) -> float:
    """Mean absolute per-step gap between target and achieved TV profiles."""
    if len(target) != len(achieved):
        raise ValueError("TV profiles must have equal length")
    return (
        sum(abs(a - b) for a, b in zip(target, achieved, strict=True)) / len(target)
        if target
        else 0.0
    )

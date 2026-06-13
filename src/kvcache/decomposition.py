"""Honest accuracy decomposition.

`residual = total accuracy change - on-reference readout change` is reported as
"accuracy change unexplained by on-reference readout." It is not a measured
cause and must not be relabeled as causal trajectory distortion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Decomposition:
    total_change: float
    readout_change: float
    unexplained_residual: float


def decompose(total_change: float, readout_change: float) -> Decomposition:
    return Decomposition(
        total_change=total_change,
        readout_change=readout_change,
        unexplained_residual=total_change - readout_change,
    )

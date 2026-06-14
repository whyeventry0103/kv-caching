"""TV divergence profiles and localization summaries."""

from __future__ import annotations

import re
from collections.abc import Sequence

THRESHOLDS = (0.05, 0.10, 0.15, 0.20)
CONTENT_TOKEN = re.compile(r"(?:\d|[+\-*/=]|\\boxed)")


def mean_tv(profile: Sequence[float]) -> float:
    """Length-normalized divergence mass."""
    return sum(profile) / len(profile) if profile else 0.0


def total_tv(profile: Sequence[float]) -> float:
    """Unnormalized divergence mass; report trace length beside comparisons."""
    return sum(profile)


def auc_tv(profile: Sequence[float]) -> float:
    """Backward-compatible name for normalized AUC (mean TV)."""
    return mean_tv(profile)


def fraction_above(profile: Sequence[float], tau: float) -> float:
    return sum(value > tau for value in profile) / len(profile) if profile else 0.0


def tdp(profile: Sequence[float], tau: float) -> int | None:
    """Return the first threshold crossing; use AUC/fraction as primaries."""
    return next((index for index, value in enumerate(profile) if value > tau), None)


def tdp_percentile(profile: Sequence[float], tau: float) -> float:
    """Normalize TDP for figures; no crossing is the endpoint 1.0."""
    crossing = tdp(profile, tau)
    if crossing is None or not profile:
        return 1.0
    return crossing / len(profile)


def summarize_profile(profile: Sequence[float]) -> dict[str, object]:
    return {
        "auc_tv": auc_tv(profile),
        "mean_tv": mean_tv(profile),
        "total_tv": total_tv(profile),
        "n_steps": len(profile),
        "frac_above": {
            f"p{int(tau * 100):02d}": fraction_above(profile, tau) for tau in THRESHOLDS
        },
        "tdp": {f"p{int(tau * 100):02d}": tdp(profile, tau) for tau in THRESHOLDS},
    }


def onset_aligned(profile: Sequence[float], onset_step: int) -> dict[int, float]:
    if not 0 <= onset_step <= len(profile):
        raise ValueError("onset step is outside the profile")
    return {step - onset_step: value for step, value in enumerate(profile)}


def content_token_profile(
    tokens: Sequence[str],
    profile: Sequence[float],
    *,
    answer_span: range | None = None,
) -> list[float]:
    if len(tokens) != len(profile):
        raise ValueError("tokens and TV profile must have equal length")
    answer_indices = set(answer_span or ())
    return [
        value
        for index, (token, value) in enumerate(zip(tokens, profile, strict=True))
        if index in answer_indices or CONTENT_TOKEN.search(token)
    ]


def segment_aggregate(
    profile: Sequence[float], segments: Sequence[tuple[str, int, int]]
) -> dict[str, float]:
    result = {}
    for name, start, stop in segments:
        if not (0 <= start <= stop <= len(profile)):
            raise ValueError(f"invalid segment bounds for {name}")
        result[name] = auc_tv(profile[start:stop])
    return result

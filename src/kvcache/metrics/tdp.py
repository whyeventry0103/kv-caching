"""TV divergence profiles and localization summaries."""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

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


WINDOW_NAMES = ("before", "first50", "later")


def onset_window(offset: int) -> str:
    """Which onset-aligned window a step falls in.

    ``before`` is pre-eviction and doubles as the replay harness's self-control:
    it MUST come out at exactly 0.0000, because before the first prune the two
    caches are identical. A non-zero value there is an instrument bug, not a
    finding.
    """
    if offset < 0:
        return "before"
    return "first50" if offset < 50 else "later"


def windowed_tv(
    profiles: Iterable[tuple[Sequence[float], int | None]],
) -> dict[str, Any]:
    """Onset-aligned windowed TV means over a set of (profile, onset) pairs.

    Full horizon -- no +/-400 cap, unlike the rendered figure curve. Two
    estimators, because they answer different questions and the paper reports
    the first:

    * ``per_profile`` -- mean of per-profile window means, so each trace counts
      once. The trace is the sampling unit, matching the problem-clustered
      convention used everywhere else.
    * ``pooled`` -- mean over every step, so long traces weigh more.

    Extracted from ``scripts/make_paper_figures.py`` so that the figure path and
    the artifact path cannot drift: these numbers are quoted in the mechanism
    section and had no machine-readable source before.

    ``median_onset`` is reported because the paper quotes it twice ("median
    onset at step 65, one step after the first prune at the compression
    interval of 64"), and it was itself an unsourced literal.
    """
    pooled: dict[str, list[float]] = {name: [] for name in WINDOW_NAMES}
    per_profile: dict[str, list[float]] = {name: [] for name in WINDOW_NAMES}
    onsets: list[int] = []
    n_profiles = 0
    for estimate, onset in profiles:
        if not estimate or onset is None:
            continue
        n_profiles += 1
        onsets.append(int(onset))
        local: dict[str, list[float]] = {name: [] for name in WINDOW_NAMES}
        for index, value in enumerate(estimate):
            name = onset_window(index - onset)
            pooled[name].append(float(value))
            local[name].append(float(value))
        for name, values in local.items():
            if values:
                per_profile[name].append(statistics.fmean(values))
    out: dict[str, Any] = {
        "n_profiles": n_profiles,
        "median_onset": statistics.median(onsets) if onsets else None,
        "windows": {},
    }
    for name in WINDOW_NAMES:
        out["windows"][name] = {
            "per_profile": (
                statistics.fmean(per_profile[name]) if per_profile[name] else None
            ),
            "pooled": statistics.fmean(pooled[name]) if pooled[name] else None,
            "n_profiles": len(per_profile[name]),
            "n_steps": len(pooled[name]),
        }
    return out

"""The restoration test (Amendment P.5).

At a fixed step in a compressed teacher-forced replay, re-insert the most
recently evicted K/V entries and measure whether per-step divergence against the
full cache falls.

**Four arms, not three.** The reply to Meeting #2 added the one that makes the
result falsifiable:

* ``none`` -- no intervention at all. **The temporal null.**
* ``relaxed`` -- reinsert; the cache grows. Upper bound, confounded with budget.
* ``matched`` -- reinsert AND evict an equal number of others under the same
  policy. **The causal test.**
* ``random`` -- reinsert an equal number of randomly chosen previously-evicted
  entries, budget-matched. Separates "these tokens were load-bearing" from "any
  perturbation helps".

Without ``none`` the sentence "divergence dropped after reinsertion" is
unfalsifiable: divergence at the trigger step has a trajectory of its own, and
some of it would have decayed anyway. Every effect below is therefore a
*difference against the no-intervention arm on the same traces*, never a
before/after within one arm.

The same discipline retired a statistic in Amendment P.2: a shape claim needs a
shuffled null before it is reported. ``random`` is that null for *which* tokens,
``none`` is that null for *time*.

Only ``matched`` answers the question. ``relaxed`` is a reference point and must
never be reported as the result.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

ARMS = ("none", "relaxed", "matched", "random")

# Pre-committed in Amendment P.5, before any run.
TRIGGER_STEP = 130  # 2x the median onset (65)
REINSERT_FRACTION = 0.25  # of the budget
READ_WINDOW = 200  # steps after the trigger that the read covers


def plan_reinsertion(
    *,
    evicted_order: Sequence[int],
    retained: Sequence[int],
    n_reinsert: int,
    arm: str,
    rng: random.Random,
) -> dict[str, list[int]]:
    """Which cache positions to put back, and which to drop to stay budget-matched.

    ``evicted_order`` is every previously evicted position, MOST RECENT LAST --
    the order matters because "most recently evicted" is the intervention and
    "randomly chosen from the same pool" is its control.

    The budget-matched arms evict from ``retained`` to make room. Positions that
    are being reinserted are never chosen for eviction, which would make the
    intervention a no-op that still looked like one.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm == "none":
        return {"reinsert": [], "evict": []}
    if n_reinsert <= 0:
        raise ValueError("n_reinsert must be positive")
    if n_reinsert > len(evicted_order):
        raise ValueError(
            f"cannot reinsert {n_reinsert} of {len(evicted_order)} evicted entries"
        )

    if arm == "random":
        reinsert = rng.sample(list(evicted_order), n_reinsert)
    else:  # relaxed and matched both reinsert the MOST RECENT evictions
        reinsert = list(evicted_order)[-n_reinsert:]

    if arm == "relaxed":
        return {"reinsert": sorted(reinsert), "evict": []}

    pool = [p for p in retained if p not in set(reinsert)]
    if len(pool) < n_reinsert:
        raise ValueError("not enough retained entries to stay budget-matched")
    return {"reinsert": sorted(reinsert), "evict": sorted(rng.sample(pool, n_reinsert))}


def restoration_read(
    arms: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    trigger_step: int = TRIGGER_STEP,
    window: int = READ_WINDOW,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Effect of each arm against the no-intervention arm, paired on the trace.

    ``arms`` maps arm name -> {trace_id: per-step divergence}. Every arm must
    cover the same traces: the comparison is paired, and an unpaired difference
    would confound the arm with which traces happened to be in it.

    The reported effect is ``none`` minus ``arm`` over the read window, so a
    POSITIVE effect means the intervention reduced divergence.
    """
    if "none" not in arms:
        raise ValueError(
            "the no-intervention arm is required: without it a post-trigger "
            "decline cannot be distinguished from what would have happened anyway"
        )
    traces = set(arms["none"])
    for name, series in arms.items():
        if set(series) != traces:
            raise ValueError(f"arm {name!r} does not cover the same traces as 'none'")

    def window_mean(values: Sequence[float]) -> float | None:
        segment = list(values)[trigger_step : trigger_step + window]
        return statistics.fmean(segment) if segment else None

    ordered = sorted(traces)
    baseline = {t: window_mean(arms["none"][t]) for t in ordered}
    usable = [t for t in ordered if baseline[t] is not None]
    if not usable:
        raise ValueError("no trace is long enough to reach the read window")

    out: dict[str, Any] = {
        "trigger_step": trigger_step,
        "window": window,
        "n_traces": len(usable),
        "n_resamples": n_resamples,
        "seed": seed,
        "baseline_mean_divergence": statistics.fmean(baseline[t] for t in usable),
        "arms": {},
    }
    for name, series in arms.items():
        if name == "none":
            continue
        deltas = []
        for t in usable:
            treated = window_mean(series[t])
            if treated is not None:
                deltas.append(baseline[t] - treated)
        if not deltas:
            continue
        rng = random.Random(seed)
        draws = sorted(
            statistics.fmean(rng.choice(deltas) for _ in deltas)
            for _ in range(n_resamples)
        )
        lo = draws[int(0.025 * (n_resamples - 1))]
        hi = draws[int(0.975 * (n_resamples - 1))]
        out["arms"][name] = {
            "effect_vs_no_intervention": statistics.fmean(deltas),
            "effect_ci95": [lo, hi],
            "excludes_zero": (lo > 0.0) or (hi < 0.0),
            "n_traces": len(deltas),
        }
    out["note"] = (
        "Positive effect = divergence reduced relative to no intervention. Only "
        "the 'matched' arm answers the causal question; 'relaxed' is confounded "
        "with cache size and is a reference point only. A 'matched' effect that "
        "'random' reproduces shows that any perturbation helps, not that the "
        "reinserted tokens were load-bearing."
    )
    return out


def classify_restoration(read: Mapping[str, Any]) -> str:
    """The Amendment P.5 branch, selected from the numbers rather than by eye."""
    arms = read.get("arms", {})
    matched = arms.get("matched")
    control = arms.get("random")
    if matched is None:
        return "NOT RUN"
    if not matched["excludes_zero"] or matched["effect_vs_no_intervention"] <= 0:
        return "NO RESTORATION"
    if control is not None and control["effect_vs_no_intervention"] >= (
        matched["effect_vs_no_intervention"]
    ):
        return "NON-SPECIFIC"
    return "RESTORES"


def restoration_effect(tv_before: float, tv_after_reinsertion: float) -> float:
    """Retained for the original call sites. Prefer :func:`restoration_read`.

    A bare before/after difference has no temporal null and cannot support a
    causal reading on its own.
    """
    return tv_before - tv_after_reinsertion

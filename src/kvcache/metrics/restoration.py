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
    applied_steps: Mapping[str, int] | None = None,
    trigger_step: int = TRIGGER_STEP,
    window: int = READ_WINDOW,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Effect of each arm against the no-intervention arm, paired on the trace.

    ``arms`` maps arm name -> {trace_id: per-step divergence}. Every arm must
    cover the same traces: the comparison is paired, and an unpaired difference
    would confound the arm with which traces happened to be in it.

    ``applied_steps`` maps trace_id -> the step the intervention ACTUALLY landed
    on, and the window is anchored there rather than at ``trigger_step``. The
    two differ: the press can only intervene during a compression event that
    already has a full stash, so a nominal trigger of 130 lands at 256 in the
    smoke, and the offset varies with prompt length. Anchoring on the nominal
    step would put a trace-dependent stretch of *pre*-intervention steps inside
    the window. Those steps contribute exactly zero to the paired difference --
    every arm is identical before the intervention -- so the effect would be
    diluted by a factor that differs per trace, which is a heterogeneous
    estimand rather than merely a noisy one. The same anchor is applied to the
    ``none`` arm, which is what keeps the pair comparable.

    Falls back to ``trigger_step`` when ``applied_steps`` is absent, and records
    which anchoring was used.
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
    if applied_steps is not None:
        missing = traces - set(applied_steps)
        if missing:
            raise ValueError(
                f"no applied step for {len(missing)} trace(s); an unanchored trace "
                "cannot be pooled with anchored ones"
            )

    def anchor(trace: str) -> int:
        return trigger_step if applied_steps is None else int(applied_steps[trace])

    def window_mean(values: Sequence[float], start: int) -> float | None:
        segment = list(values)[start : start + window]
        return statistics.fmean(segment) if segment else None

    ordered = sorted(traces)
    baseline = {t: window_mean(arms["none"][t], anchor(t)) for t in ordered}
    usable = [t for t in ordered if baseline[t] is not None]
    if not usable:
        raise ValueError("no trace is long enough to reach the read window")

    out: dict[str, Any] = {
        "trigger_step": trigger_step,
        "anchored_on": "applied_step" if applied_steps is not None else "trigger_step",
        "applied_step_range": (
            [
                min(applied_steps[t] for t in usable),
                max(applied_steps[t] for t in usable),
            ]
            if applied_steps is not None
            else None
        ),
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
            treated = window_mean(series[t], anchor(t))
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


# --------------------------------------------------------------------------
# Cache surgery. Kept as a pure function over tensors so the index bookkeeping
# is testable without a model: the arithmetic that decides WHICH entries end up
# in the cache is where a restoration test goes silently wrong.
# --------------------------------------------------------------------------


def apply_reinsertion(
    keys: Any,
    values: Any,
    *,
    evicted_keys: Any,
    evicted_values: Any,
    reinsert_local: Sequence[int],
    evict_local: Sequence[int],
) -> tuple[Any, Any]:
    """Put stashed entries back and drop ``evict_local`` from the live cache.

    ``keys``/``values`` are (batch, heads, positions, dim). ``reinsert_local``
    indexes ``evicted_keys``; ``evict_local`` indexes ``keys``. Returns the new
    pair. For the budget-matched arm the two lists have equal length, so the
    cache size is unchanged -- the whole point of that arm.

    Reinserted entries are appended rather than restored to their original
    slots: kvpress caches are position-indexed by order, and re-threading them
    into the middle would silently reorder every later entry's RoPE phase.
    Appending is the honest operation and is what the retained-set gate checks.
    """
    import torch

    keep = [i for i in range(keys.shape[2]) if i not in set(evict_local)]
    kept_k = keys[:, :, keep, :]
    kept_v = values[:, :, keep, :]
    if not len(reinsert_local):
        return kept_k, kept_v
    idx = torch.as_tensor(list(reinsert_local), device=keys.device, dtype=torch.long)
    add_k = evicted_keys.index_select(2, idx)
    add_v = evicted_values.index_select(2, idx)
    return (
        torch.cat([kept_k, add_k], dim=2),
        torch.cat([kept_v, add_v], dim=2),
    )


def retained_set_report(
    new_keys: Any, expected_keys: Any, *, n_reinserted: int
) -> dict[str, Any]:
    """Instrument gate: are the reinserted entries present and BITWISE identical?

    Amendment P.5 refuses to read any divergence until this passes, in the
    spirit of the R-KV Jaccard parity gate. A reinsertion that silently dropped,
    reordered or cast its entries would still produce a divergence curve, and
    that curve would mean nothing.
    """
    import torch

    if n_reinserted == 0:
        return {"passed": True, "n_reinserted": 0, "note": "no-intervention arm"}
    tail = new_keys[:, :, -n_reinserted:, :]
    exact = bool(torch.equal(tail, expected_keys))
    max_abs = float((tail - expected_keys).abs().max().item())
    return {
        "passed": exact,
        "n_reinserted": int(n_reinserted),
        "bitwise_identical": exact,
        "max_abs_difference": max_abs,
        "note": (
            "Reinserted entries occupy the tail of the cache and must match the "
            "stashed tensors exactly; a non-zero difference means the surgery "
            "changed the values it claimed only to move."
        ),
    }

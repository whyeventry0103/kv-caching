"""Capacity feasibility and predicted eviction exposure for the frozen-ID
replication splits (M5-B; prepared 2026-11-13).

Scientific rationale. Meeting 5 plan §8.2 fixes the capacity rule for a
prompt-evictable versus prompt-retained comparison: with P prompt entries and
nominal budget B, both protection conditions and every policy use the same
``B_eff = max(B, P + 64)`` (the S.1 ``effective_budget``, imported, never
re-implemented), and the primary fixed-B comparison is the strict-fit cohort
``P + 64 <= B``. Whether a budget is informative on a new dataset therefore
depends on two pre-outcome quantities this module summarizes:

* the prompt-length distribution of the frozen split (how many problems fit
  at B, how many trace slots remain) -- a property of the inputs alone; and
* from a FULL-CACHE calibration bank, how often the trace would outgrow
  ``B_eff`` at all (``P + T > B_eff``), i.e. whether a compressed run would
  ever reach the eviction schedule (plan §9.2: "how often a run actually
  reaches eviction"; a budget rarely reached is a low-exposure regime).

Both are read before any compressed outcome exists; neither looks at which
policy wins. Exposure here is the nominal prediction from full-cache lengths;
the realized exposure of a compressed run is recorded by the press itself.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from kvcache.presses.prompt_protect import MIN_TRACE_SLOTS, effective_budget


def _quantiles(values: Sequence[int]) -> dict[str, float]:
    v = sorted(values)
    if not v:
        return {}

    def q(p: float) -> float:
        # nearest-rank, so every reported value is an observed length
        return float(v[min(len(v) - 1, max(0, int(-(-p * len(v) // 1)) - 1))])

    return {
        "min": float(v[0]),
        "p25": q(0.25),
        "median": float(statistics.median(v)),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": float(v[-1]),
    }


def prompt_fit(
    prompt_lens: Mapping[str, int],
    budget: int,
    *,
    min_trace_slots: int = MIN_TRACE_SLOTS,
) -> dict[str, Any]:
    """Strict-fit count and the B_eff / trace-slot distributions at ``budget``."""
    if not prompt_lens:
        raise ValueError("no prompts")
    effs = {
        pid: effective_budget(budget, p, min_trace_slots)
        for pid, p in prompt_lens.items()
    }
    relaxed = sorted(pid for pid, (_, r) in effs.items() if r)
    slots = [effs[pid][0] - p for pid, p in prompt_lens.items()]
    fit_slots = [b - prompt_lens[pid] for pid, (b, r) in effs.items() if not r]
    return {
        "budget": int(budget),
        "min_trace_slots": int(min_trace_slots),
        "n": len(prompt_lens),
        "n_strict_fit": len(prompt_lens) - len(relaxed),
        "n_relaxed": len(relaxed),
        "relaxed_pids": relaxed,
        "b_eff": _quantiles([b for b, _ in effs.values()]),
        "trace_slots_all": _quantiles(slots),
        "trace_slots_strict_fit": _quantiles(fit_slots),
    }


def predicted_exposure(
    prompt_lens: Mapping[str, int],
    traces: Sequence[Mapping[str, Any]],
    budget: int,
    *,
    min_trace_slots: int = MIN_TRACE_SLOTS,
) -> dict[str, Any]:
    """From full-cache traces ({pid, n_tokens}): how many would outgrow B_eff.

    ``reached`` means the cache would exceed its target at some step
    (``P + T > B_eff``); ``excess`` is how many generated tokens lie beyond
    capacity. Reported for all traces and for the strict-fit cohort.
    """
    out: dict[str, Any] = {"budget": int(budget)}
    for label, keep_relaxed in (("all", True), ("strict_fit", False)):
        reached, excess, n = 0, [], 0
        for t in traces:
            p = int(prompt_lens[str(t["pid"])])
            b_eff, relaxed = effective_budget(budget, p, min_trace_slots)
            if relaxed and not keep_relaxed:
                continue
            n += 1
            over = p + int(t["n_tokens"]) - b_eff
            if over > 0:
                reached += 1
                excess.append(over)
        out[label] = {
            "n": n,
            "reached": reached,
            "reached_rate": reached / n if n else None,
            "excess_tokens": _quantiles(excess),
        }
    return out

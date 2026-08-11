"""The additivity (interaction) test, Convention 2.

Convention 2, binding from Meeting #2::

    Delta_tot       = Acc_free - Acc_fc
    readout_error   = Acc_ro   - Acc_fc            (measured directly)
    residual_fwd    = Delta_tot - readout_error    = Acc_free - Acc_ro
    reverse_readout = Acc_free - Acc_rev
    residual_rev    = Delta_tot - reverse_readout  = Acc_rev - Acc_fc

``readout_error`` and ``residual_rev`` are measured independently of each other
and of ``Delta_tot``, so testing whether they sum to ``Delta_tot`` is a real
test rather than an identity. The shortfall is the interaction.

**The population rule is the whole difficulty.** Amendment K fixes each term's
reference to that term's own population: readout is defined only where the
reference trace reaches an answer boundary, reverse readout only where the
*compressed* trace does, and the free-run over every condition. Those are three
different denominators. Summing them and calling the shortfall an interaction
conflates a real interaction with a population mismatch -- the exact defect
Amendment K exists for.

So the primary estimand here is computed on the COMMON population: conditions
where all four protocols are defined. Amendment P.1 pre-registers this, and the
per-term (Amendment-K) version is returned alongside for disclosure, because it
is what the stored artifacts give directly and the difference between the two is
itself worth reporting.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

Key = tuple[str, int]


def _acc(keys: Sequence[Key], table: Mapping[Key, bool]) -> float:
    return statistics.fmean(float(table[k]) for k in keys)


def interaction_on(
    keys: Sequence[Key],
    fc: Mapping[Key, bool],
    free: Mapping[Key, bool],
    readout: Mapping[Key, bool],
    reverse: Mapping[Key, bool],
) -> dict[str, float]:
    """Convention-2 terms and the interaction, all on the SAME condition set."""
    acc_fc = _acc(keys, fc)
    acc_free = _acc(keys, free)
    acc_ro = _acc(keys, readout)
    acc_rev = _acc(keys, reverse)
    delta_tot = acc_free - acc_fc
    readout_error = acc_ro - acc_fc
    residual_rev = acc_rev - acc_fc
    return {
        "acc_fc": acc_fc,
        "acc_free": acc_free,
        "acc_ro": acc_ro,
        "acc_rev": acc_rev,
        "delta_tot": delta_tot,
        "readout_error": readout_error,
        "residual_fwd": acc_free - acc_ro,
        "reverse_readout": acc_free - acc_rev,
        "residual_rev": residual_rev,
        "sum_of_terms": readout_error + residual_rev,
        "interaction": delta_tot - (readout_error + residual_rev),
    }


def additivity_test(
    fc: Mapping[Key, bool],
    free: Mapping[Key, bool],
    readout: Mapping[Key, bool],
    reverse: Mapping[Key, bool],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Interaction on the common population, with a problem-clustered CI.

    Problems are the resampling unit, matching every other interval in this
    repo. Resampling is joint across the four protocols -- the same problem is
    drawn for all of them at once -- which is what makes the interaction's
    interval meaningful: the four accuracies are not independent samples, they
    are four measurements of the same problems.
    """
    common = sorted(set(fc) & set(free) & set(readout) & set(reverse))
    if not common:
        raise ValueError("no conditions where all four protocols are defined")

    by_problem: dict[str, list[Key]] = {}
    for key in common:
        by_problem.setdefault(key[0], []).append(key)
    problems = sorted(by_problem)

    point = interaction_on(common, fc, free, readout, reverse)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_resamples):
        keys: list[Key] = []
        for _ in range(len(problems)):
            keys.extend(by_problem[rng.choice(problems)])
        draws.append(interaction_on(keys, fc, free, readout, reverse)["interaction"])
    draws.sort()
    lo = draws[int(0.025 * (n_resamples - 1))]
    hi = draws[int(0.975 * (n_resamples - 1))]

    # Amendment K layer-1, applied to the SELECTION this estimand rests on.
    # The common population drops every condition where any protocol is
    # undefined -- disproportionately non-terminating traces -- so it is an
    # easier subset. Scoring the dropped conditions both ways bounds how much
    # that selection could be worth, and makes it visible instead of absorbed.
    dropped = sorted((set(fc) & set(free)) - set(common))
    bounds: list[float] | None = None
    if dropped:
        lo_tab = {k: False for k in dropped}
        hi_tab = {k: True for k in dropped}
        ends = []
        for imputed in (lo_tab, hi_tab):
            keys = list(common) + dropped
            ends.append(
                interaction_on(
                    keys,
                    fc,
                    {**free, **{k: free[k] for k in dropped}},
                    {**readout, **imputed},
                    {**reverse, **imputed},
                )["interaction"]
            )
        bounds = [min(ends), max(ends)]

    return {
        **point,
        "n_conditions": len(common),
        "n_dropped_by_selection": len(dropped),
        "primary_bound_interaction": bounds,
        "estimand_scope": (
            "interaction among conditions where ALL protocols terminated; the "
            "common population excludes non-terminating traces and is therefore "
            "an easier subset than the full cell"
        ),
        "n_problems": len(problems),
        "interaction_ci95": [lo, hi],
        "interaction_excludes_zero": (lo > 0.0) or (hi < 0.0),
        "interaction_sign": ("negative" if point["interaction"] < 0 else "positive"),
        "interaction_pct_of_total": (
            abs(point["interaction"] / point["delta_tot"]) * 100.0
            if point["delta_tot"]
            else None
        ),
        "n_resamples": n_resamples,
        "seed": seed,
        "population": "common: all four protocols defined",
        "convention": "2",
    }

"""Statistical utilities for preregistered analyses."""

from __future__ import annotations

import collections
import itertools
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def bootstrap(
    pairs: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if not pairs:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    values = sorted(
        statistic([pairs[rng.randrange(len(pairs))] for _ in pairs])
        for _ in range(n_resamples)
    )
    return (
        values[int(0.025 * (n_resamples - 1))],
        values[int(0.975 * (n_resamples - 1))],
    )


def exact_permutation_test(
    a: Sequence[T],
    b: Sequence[T],
    statistic: Callable[[Sequence[T], Sequence[T]], float],
    *,
    max_exact: int = 100_000,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return observed statistic and one-sided per-problem permutation p-value."""
    pooled = list(a) + list(b)
    n_a = len(a)
    observed = statistic(a, b)
    combinations = math.comb(len(pooled), n_a)
    is_exact = combinations <= max_exact
    if is_exact:
        assignments = itertools.combinations(range(len(pooled)), n_a)
    else:
        rng = random.Random(seed)
        assignments = (
            tuple(sorted(rng.sample(range(len(pooled)), n_a)))
            for _ in range(n_resamples)
        )
    exceed = 0
    count = 0
    all_indices = set(range(len(pooled)))
    for selected_tuple in assignments:
        selected = set(selected_tuple)
        perm_a = [pooled[i] for i in selected]
        perm_b = [pooled[i] for i in all_indices - selected]
        exceed += statistic(perm_a, perm_b) >= observed - 1e-15
        count += 1
    p_value = exceed / count if is_exact else (exceed + 1) / (count + 1)
    return observed, p_value


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=p_values.__getitem__)
    adjusted = [0.0] * m
    running = 1.0
    for rank_from_end, index in enumerate(reversed(order), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = running
    return adjusted


def negative_control_ok(
    excess_add: float,
    tdp_percentile: float,
    *,
    add_tolerance: float = 0.05,
    tdp_floor: float = 0.95,
) -> bool:
    return abs(excess_add) <= add_tolerance and tdp_percentile >= tdp_floor


def percentile_ci_mc_error(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    n_seeds: int = 200,
) -> dict[str, Any]:
    """Monte-Carlo error of a percentile bootstrap CI's endpoints.

    A percentile interval is itself a random variable: at a fixed resample
    count its endpoints move with the RNG seed. This repeats the standard
    2000-resample percentile bootstrap over ``n_seeds`` seeds and reports how
    far the endpoints travel, which is the quantity needed to say whether an
    "excludes zero" claim rests on a real margin or on a lucky seed.

    Deliberately does NOT raise the resample count: 2000 with 2.5/97.5 indices
    is the stated method throughout this repo, and changing it would be a
    protocol deviation touching every interval in the paper. The fix for
    irreproducibility is to pin and record the seed (see
    ``paired_policy_contrast``), not to round the interval -- a rounded
    interval is still seed-dependent, it merely hides the dependence.

    ``margin_sd`` is reported per endpoint: the distance from zero in units of
    the endpoint's own MC standard deviation. A margin under ~3 means the sign
    of the endpoint is not safely established at this resample count.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    k = len(values)
    los: list[float] = []
    his: list[float] = []
    for seed in range(n_seeds):
        rng = random.Random(seed)
        means = sorted(
            statistics.fmean(rng.choice(values) for _ in range(k))
            for _ in range(n_resamples)
        )
        los.append(means[int(0.025 * (n_resamples - 1))])
        his.append(means[int(0.975 * (n_resamples - 1))])

    def _describe(samples: list[float]) -> dict[str, Any]:
        sd = statistics.stdev(samples) if len(samples) > 1 else 0.0
        mean = statistics.fmean(samples)
        return {
            "mean": mean,
            "sd": sd,
            "min": min(samples),
            "max": max(samples),
            "margin_sd": (abs(mean) / sd) if sd else None,
            "crosses_zero": sum(1 for v in samples if v >= 0) if mean < 0 else None,
        }

    return {
        "n_seeds": n_seeds,
        "n_resamples": n_resamples,
        "n_units": k,
        "lower": _describe(los),
        "upper": _describe(his),
    }


def paired_policy_contrast(
    a_by_problem: Mapping[str, Sequence[bool]],
    b_by_problem: Mapping[str, Sequence[bool]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired per-problem accuracy contrast between two policies.

    Supports the paper's random-beats-recency claim, which is stated in three
    parts that must agree: the mean difference, its problem-clustered CI, and
    the per-problem win/tie/loss breakdown. The breakdown exists because
    "non-negative on 94/100" is the honest phrasing of that result -- it is not
    94 wins (PAPER_FINDINGS §5), and reporting it as such is a standing rule.

    Both policies must cover the same problems; the contrast is paired on the
    problem, which is the sampling unit used everywhere else in this repo.

    The returned ``seed``/``n_resamples`` and ``difference_histogram`` exist for
    provenance, and they are load-bearing: the percentile CI is a Monte-Carlo
    quantity, so an unpinned seed makes it irreproducible. The published
    ``[0.130, 0.242]`` could not be reproduced exactly for that reason -- over
    200 seeds the upper endpoint spans [0.2375, 0.2450] -- and the histogram is
    a sufficient statistic for the whole bootstrap, so any reader can recompute
    the interval from the artifact alone.
    """
    problems = sorted(set(a_by_problem) & set(b_by_problem))
    if not problems:
        raise ValueError("no problems in common between the two policies")
    missing = (set(a_by_problem) | set(b_by_problem)) - set(problems)
    per_problem = {
        pid: statistics.fmean(float(bool(v)) for v in a_by_problem[pid])
        - statistics.fmean(float(bool(v)) for v in b_by_problem[pid])
        for pid in problems
    }
    diffs = [per_problem[pid] for pid in problems]
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    losses = sum(1 for d in diffs if d < 0)

    rng = random.Random(seed)
    k = len(problems)
    means = sorted(
        statistics.fmean(rng.choice(diffs) for _ in range(k))
        for _ in range(n_resamples)
    )
    lo = means[int(0.025 * (n_resamples - 1))]
    hi = means[int(0.975 * (n_resamples - 1))]
    histogram = collections.Counter(diffs)
    return {
        "n_problems": k,
        "mean_difference": statistics.fmean(diffs),
        "ci95": [round(lo, 4), round(hi, 4)],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "non_negative": wins + ties,
        "n_problems_dropped": len(missing),
        "seed": seed,
        "n_resamples": n_resamples,
        "difference_histogram": [
            [value, histogram[value]] for value in sorted(histogram)
        ],
    }


def paired_residual_contrast(
    residuals_by_press: Mapping[str, Mapping[str, float]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired per-problem contrast of the trajectory residual across presses.

    The paper reports that the forward residual is of similar magnitude for
    every policy tested, and supports it with a cross-policy *spread* (0.008)
    set against CI half-widths of roughly 0.07. That comparison is not
    admissible: the spread is a difference between policies measured on the
    same 100 problems, while the half-widths are unpaired accuracy intervals.
    A difference of paired quantities has far lower variance than the
    difference of their marginal intervals suggests, so the spread could be
    either much better or much worse supported than it looks. Pairing on the
    problem is what makes the question answerable at all.

    The per-problem residual is ``free - readout`` on that problem. The
    full-cache reference cancels out of it exactly -- residual is
    ``(free - Acc_FC) - (readout - Acc_FC)`` -- which is why this contrast is
    insensitive to the reference-population choice that Amendment K corrected,
    and therefore a clean way to ask whether the policies really differ.

    Returns the pairwise differences, each with a problem-clustered percentile
    CI, plus the mean absolute pairwise difference as a single summary. Seed,
    resample count and the per-pair difference histogram travel with the result
    for the same reason as in :func:`paired_policy_contrast`.
    """
    presses = sorted(residuals_by_press)
    if len(presses) < 2:
        raise ValueError("need at least two presses to contrast")
    common = set.intersection(*(set(residuals_by_press[p]) for p in presses))
    if not common:
        raise ValueError("no problems in common across the presses")
    problems = sorted(common)
    rng = random.Random(seed)
    k = len(problems)
    # One resampling of problems drives every pair, so the pairs are compared
    # on the same bootstrap draws rather than on independent ones.
    draws = [[rng.randrange(k) for _ in range(k)] for _ in range(n_resamples)]

    pairs: dict[str, Any] = {}
    for i, a in enumerate(presses):
        for b in presses[i + 1 :]:
            diffs = [
                residuals_by_press[a][pid] - residuals_by_press[b][pid]
                for pid in problems
            ]
            means = sorted(statistics.fmean(diffs[j] for j in draw) for draw in draws)
            lo = means[int(0.025 * (n_resamples - 1))]
            hi = means[int(0.975 * (n_resamples - 1))]
            histogram = collections.Counter(round(d, 6) for d in diffs)
            pairs[f"{a}_vs_{b}"] = {
                "mean_difference": statistics.fmean(diffs),
                "ci95": [lo, hi],
                "excludes_zero": (lo > 0.0) or (hi < 0.0),
                "difference_histogram": [
                    [value, histogram[value]] for value in sorted(histogram)
                ],
            }

    # Mean absolute pairwise difference: stable at 3-4 presses in a way max-min
    # is not, since max-min is driven by the two most extreme cells alone.
    def _mapd(index: Sequence[int]) -> float:
        vals = []
        for i, a in enumerate(presses):
            for b in presses[i + 1 :]:
                vals.append(
                    abs(
                        statistics.fmean(
                            residuals_by_press[a][problems[j]]
                            - residuals_by_press[b][problems[j]]
                            for j in index
                        )
                    )
                )
        return statistics.fmean(vals)

    observed = _mapd(list(range(k)))
    spread_means = sorted(_mapd(draw) for draw in draws)
    return {
        "presses": presses,
        "n_problems": k,
        "mean_abs_pairwise_difference": observed,
        "mean_abs_pairwise_difference_ci95": [
            spread_means[int(0.025 * (n_resamples - 1))],
            spread_means[int(0.975 * (n_resamples - 1))],
        ],
        "pairs": pairs,
        "n_pairs_excluding_zero": sum(1 for p in pairs.values() if p["excludes_zero"]),
        "seed": seed,
        "n_resamples": n_resamples,
    }


def resolution_spread(
    free_by_press: Mapping[str, Mapping[str, float]],
    readout_by_press: Mapping[str, Mapping[str, float]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """How much policy separation survives held-trace evaluation, per regime.

    POST-HOC. This analysis was formed after every cell had been read; it is
    exploratory and labelled as such wherever it appears (PREREGISTRATION
    Amendment L). Stating that plainly is the price of running it at all.

    The question is whether readout-style evaluation can *rank* policies, and
    the honest form of it is a comparison of spreads: how far apart the presses
    sit under free-running versus under readout, on the same problems. Both are
    accuracy-scale quantities measured on the same conditions, so this compares
    readout to readout and free-run to free-run -- no off-policy caveat applies.

    Design choices that matter:

    * The spread is the **mean absolute pairwise difference**, not max minus
      min. With three or four presses, max-min is a function of the two extreme
      cells alone and moves discontinuously when they swap.
    * The reported quantity is the **difference** of the two spreads, with a
      percentile CI. The ratio is descriptive only: at the saturated end the
      readout spread approaches zero, and a ratio CI on a near-zero denominator
      is not an honest interval.
    * Bootstrap resamples problems (the sampling unit used throughout) and
      drives both spreads from the same draw, so the difference is paired.

    ``rank_agreement`` is the fraction of press pairs ordered the same way by
    both protocols, which distinguishes "readout ranks them differently" from
    "readout cannot separate them at all".
    """
    presses = sorted(set(free_by_press) & set(readout_by_press))
    if len(presses) < 2:
        raise ValueError("need at least two presses")
    common = set.intersection(
        *(
            [set(free_by_press[p]) for p in presses]
            + [set(readout_by_press[p]) for p in presses]
        )
    )
    if not common:
        raise ValueError("no problems in common")
    problems = sorted(common)
    k = len(problems)

    def _spread(
        table: Mapping[str, Mapping[str, float]], index: Sequence[int]
    ) -> float:
        vals = []
        for i, a in enumerate(presses):
            for b in presses[i + 1 :]:
                vals.append(
                    abs(
                        statistics.fmean(
                            table[a][problems[j]] - table[b][problems[j]] for j in index
                        )
                    )
                )
        return statistics.fmean(vals)

    whole = list(range(k))
    free_spread = _spread(free_by_press, whole)
    readout_spread = _spread(readout_by_press, whole)

    rng = random.Random(seed)
    draws = [[rng.randrange(k) for _ in range(k)] for _ in range(n_resamples)]
    diffs = sorted(
        _spread(free_by_press, draw) - _spread(readout_by_press, draw) for draw in draws
    )
    lo = diffs[int(0.025 * (n_resamples - 1))]
    hi = diffs[int(0.975 * (n_resamples - 1))]

    # Rank agreement over press pairs, ties counted as agreement only when both
    # protocols tie -- a readout tie against a free-run ordering is exactly the
    # failure this analysis is about.
    agree = 0
    total = 0
    for i, a in enumerate(presses):
        for b in presses[i + 1 :]:
            fa = statistics.fmean(free_by_press[a][p] for p in problems)
            fb = statistics.fmean(free_by_press[b][p] for p in problems)
            ra = statistics.fmean(readout_by_press[a][p] for p in problems)
            rb = statistics.fmean(readout_by_press[b][p] for p in problems)
            total += 1
            if ((fa > fb) and (ra > rb)) or ((fa < fb) and (ra < rb)):
                agree += 1
            elif (fa == fb) and (ra == rb):
                agree += 1

    return {
        "presses": presses,
        "n_problems": k,
        "free_run_spread": free_spread,
        "readout_spread": readout_spread,
        "spread_difference": free_spread - readout_spread,
        "spread_difference_ci95": [lo, hi],
        "spread_difference_excludes_zero": (lo > 0.0) or (hi < 0.0),
        "spread_ratio_descriptive": (
            readout_spread / free_spread if free_spread > 0 else None
        ),
        "rank_agreement": agree / total if total else None,
        "n_press_pairs": total,
        "seed": seed,
        "n_resamples": n_resamples,
        "analysis_status": "post-hoc, exploratory (PREREGISTRATION Amendment L)",
    }


def published_config_branch(
    ratio: float, difference_ci: Sequence[float]
) -> dict[str, Any]:
    """Select PREREGISTRATION Amendment N.2's branch from the numbers alone.

    N.2 pre-commits three branches for the R-KV published-configuration cell,
    keyed on ``r = Acc_cell / Acc_fullcache`` and the problem-clustered CI on
    the accuracy difference (cell minus full cache):

    * REPLICATES -- ``r >= 1.0`` and the difference CI includes or exceeds zero;
    * DOES NOT REPLICATE -- ``r < 1.0`` and the difference CI excludes zero;
    * INCONCLUSIVE -- the CI spans zero without the first branch applying.

    This exists as code, and is called before the numbers are looked at, because
    a read chosen after seeing the data is a rationalisation rather than a
    pre-registration. The branch is therefore a function of the artifact, not of
    whoever writes the paragraph.
    """
    lo, hi = float(difference_ci[0]), float(difference_ci[1])
    spans_zero = lo <= 0.0 <= hi
    if ratio >= 1.0 and hi >= 0.0:
        branch = "REPLICATES"
    elif ratio < 1.0 and not spans_zero:
        branch = "DOES NOT REPLICATE"
    else:
        branch = "INCONCLUSIVE"
    return {
        "branch": branch,
        "ratio": ratio,
        "difference_ci95": [lo, hi],
        "difference_ci_spans_zero": spans_zero,
        "rule": "PREREGISTRATION Amendment N.2",
    }


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, with midranks for ties.

    Amendment P.3 makes this the PRIMARY statistic for the AUC-vs-residual test,
    not Pearson: "proportional" is a strictly stronger claim than "correlated",
    and a rank correlation must never be described as proportionality. Linearity
    through the origin is tested separately, and only after.
    """
    if len(x) != len(y):
        raise ValueError("spearman needs paired samples")
    if len(x) < 3:
        raise ValueError("spearman on fewer than 3 points is not a correlation")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            midrank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = midrank
            i = j + 1
        return out

    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        raise ValueError("spearman undefined when one variable is constant")
    return num / (dx * dy)


def correlation_with_cluster_bootstrap(
    cells: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Spearman between two per-cell quantities, with a problem-clustered CI.

    ``cells`` maps cell name -> {"auc": {problem: value}, "residual": {problem:
    value}}. Both cell-level quantities are re-derived from the SAME resampled
    problems on every draw, so the interval reflects that the two are measured
    on shared problems rather than treating them as independent.

    n is the number of CELLS, and it is reported with every coefficient: a
    three-point "correlation" is not one, and Amendment P.3 forbids presenting
    it as such.
    """
    names = sorted(cells)
    if len(names) < 3:
        raise ValueError(f"{len(names)} cells is not a correlation")

    def point(sampler=None) -> float:
        xs, ys = [], []
        for name in names:
            auc, res = cells[name]["auc"], cells[name]["residual"]
            problems = (
                sampler(sorted(set(auc) & set(res)))
                if sampler
                else sorted(set(auc) & set(res))
            )
            xs.append(statistics.fmean(auc[p] for p in problems))
            ys.append(statistics.fmean(res[p] for p in problems))
        return spearman_rho(xs, ys)

    observed = point()
    rng = random.Random(seed)
    draws = []
    for _ in range(n_resamples):
        try:
            draws.append(point(lambda ps: [rng.choice(ps) for _ in ps]))
        except ValueError:
            continue  # a resample made a variable constant; skip, do not clamp
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return {
        "n_cells": len(names),
        "cells": names,
        "spearman_rho": observed,
        "rho_ci95": [lo, hi],
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
        "n_resamples": n_resamples,
        "n_usable_draws": len(draws),
        "seed": seed,
        "discipline": (
            "Rank correlation. NEVER describe as proportionality; that is a "
            "separate, stronger test (linearity through the origin)."
        ),
    }

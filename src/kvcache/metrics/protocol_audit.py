"""Evaluation-protocol audit: the operational 2 x 2 and its practical reading.

Scientific rationale (docs/paper/EXECUTION_PLAN.md Sections 4 and 9, adopted
2026-10-05). For one compressed policy C and the full cache F, on the same
(problem, seed) conditions:

* ``ff`` -- full-cache free run (the bank),            A_FF
* ``fc`` -- full-cache trace read out under C,           A_FC (forward readout)
* ``cc`` -- compressed free run,                         A_CC
* ``cf`` -- compressed trace read out under F,           A_CF (reverse readout)

``D = A_CC - A_FF`` splits two ways: readout first, ``(A_FC - A_FF) + (A_CC -
A_FC)`` (forward), or trajectory first, ``(A_CF - A_FF) + (A_CC - A_CF)``
(reverse). ``A_CC - A_FC`` and ``A_CF - A_FF`` are the same trajectory contrast
measured under the compressed and the full-cache readout; ``I`` is their
difference (the 2 x 2 interaction), i.e. how far the two replay directions
disagree about the part held-trace readout does not account for. These are
operational contrasts, not bounds on a causal "trajectory share": opposite
signs or crossed endpoints are valid outcomes, reported as such.

Readouts are defined only where the source trace has an answer boundary, a
post-treatment property (the reverse source is the compressed trace, so its
support depends on the policy). Three layers are therefore reported: the free
run on the whole population; each contrast on its own declared support; the
full 2 x 2 on the common support, plus worst-case bounds on the whole
population with every undefined readout imputed anywhere in [0, 1].

The practical layer (Section 4.3) is a separately named sensitivity set after
the B=256 result was seen: margins 1/3/5 accuracy points, a Bonferroni pair of
one-sided bounds for "readout looks fine while free generation is not", and a
TOST-style 90% interval for equivalence. None of it relabels a registered read.

Every interval is a problem-cluster percentile bootstrap of an equal-problem
mean (seeds within problem, then problems), drawn JOINTLY for all quantities of
one read so derived contrasts and ratios keep their pairing. With one series
the draws are identical to ``kvcache.coverage.equal_problem_bootstrap``.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

Key = tuple[str, int]

PRACTICAL_MARGINS = (0.01, 0.03, 0.05)
PRIMARY_MARGIN = 0.03
# Linear contrasts over the four arm accuracies (coefficients on A_arm). Two
# factors: the trajectory (reference F or compressed C) and the readout policy.
# ``residual_forward`` is the registered residual (Amendment K complete case);
# ``trajectory_under_full_readout`` is Amendment H's reverse endpoint.
CONTRASTS: dict[str, dict[str, int]] = {
    "D_total": {"cc": 1, "ff": -1},
    "readout_on_reference_trace": {"fc": 1, "ff": -1},
    "residual_forward": {"cc": 1, "fc": -1},
    "trajectory_under_full_readout": {"cf": 1, "ff": -1},
    "readout_on_compressed_trace": {"cc": 1, "cf": -1},
    "interaction": {"cc": 1, "fc": -1, "cf": -1, "ff": 1},
}
ARMS = ("ff", "fc", "cc", "cf")


def _quantile(sorted_draws: Sequence[float], p: float) -> float:
    """The repository's percentile convention (index int(p * (n - 1)))."""
    return sorted_draws[int(p * (len(sorted_draws) - 1))]


def _problem_means(values: Mapping[Key, float]) -> dict[str, float]:
    per: dict[str, list[float]] = defaultdict(list)
    for (pid, _s), v in values.items():
        per[pid].append(float(v))
    return {pid: statistics.fmean(v) for pid, v in per.items()}


def joint_draws(
    series: Mapping[str, Mapping[Key, float]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Equal-problem means of every series and their joint bootstrap replicates.

    All series must cover the same (problem, seed) keys: the problem draw of a
    replicate is shared, so differences and ratios computed per replicate are
    paired. Returns (point estimates, one dict of means per replicate).
    """
    names = list(series)
    if not names:
        raise ValueError("no series")
    keyset = set(series[names[0]])
    for n in names[1:]:
        if set(series[n]) != keyset:
            raise ValueError(f"series {n!r} covers different conditions")
    if not keyset:
        raise ValueError("series are empty")
    pm = {n: _problem_means(series[n]) for n in names}
    pids = sorted(pm[names[0]])
    point = {n: statistics.fmean(pm[n].values()) for n in names}
    rng = random.Random(seed)
    reps: list[dict[str, float]] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(len(pids)) for _ in pids]
        reps.append({n: statistics.fmean(pm[n][pids[i]] for i in idx) for n in names})
    return point, reps


def summarize(
    point: float, draws: Sequence[float], *, n_resamples: int, seed: int
) -> dict[str, Any]:
    s = sorted(draws)
    lo, hi = _quantile(s, 0.025), _quantile(s, 0.975)
    return {
        "mean": point,
        "ci95": [lo, hi],
        "ci90": [_quantile(s, 0.05), _quantile(s, 0.95)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "estimator": "equal-problem, joint problem-cluster bootstrap",
        "n_resamples": n_resamples,
        "seed": seed,
    }


def _contrast_value(means: Mapping[str, float], coefs: Mapping[str, int]) -> float:
    return sum(c * means[a] for a, c in coefs.items())


def _support_block(
    arms: Mapping[str, Mapping[Key, bool]],
    keys: Collection[Key],
    contrasts: Mapping[str, Mapping[str, int]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    used = sorted({a for c in contrasts.values() for a in c})
    series = {a: {k: float(arms[a][k]) for k in keys} for a in used}
    point, reps = joint_draws(series, n_resamples=n_resamples, seed=seed)
    kw = {"n_resamples": n_resamples, "seed": seed}
    out: dict[str, Any] = {
        "n_conditions": len(keys),
        "n_problems": len({k[0] for k in keys}),
        "accuracy": {a: summarize(point[a], [r[a] for r in reps], **kw) for a in used},
    }
    for name, coefs in contrasts.items():
        out[name] = summarize(
            _contrast_value(point, coefs),
            [_contrast_value(r, coefs) for r in reps],
            **kw,
        )
    return out


def worst_case_bounds(
    arms: Mapping[str, Mapping[Key, bool]],
    population: Collection[Key],
    coefs: Mapping[str, int],
) -> list[float]:
    """[min, max] of an equal-problem contrast over every completion of the
    undefined readouts in [0, 1]. Linear in each unknown, so the extremes put
    each unknown at 0 or 1 according to the sign of its coefficient."""
    out = []
    for direction in (-1, 1):
        means = {}
        for a in coefs:
            fill = 1.0 if coefs[a] * direction > 0 else 0.0
            vals = {k: float(arms[a][k]) if k in arms[a] else fill for k in population}
            means[a] = statistics.fmean(_problem_means(vals).values())
        out.append(_contrast_value(means, coefs))
    return out


def two_by_two(
    *,
    ff: Mapping[Key, bool],
    fc: Mapping[Key, bool],
    cc: Mapping[Key, bool],
    cf: Mapping[Key, bool],
    population: Collection[Key],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The operational 2 x 2 for one policy at one budget.

    ``ff`` and ``cc`` must cover ``population`` (free runs are always scored;
    truncations count as incorrect). ``fc`` and ``cf`` hold only the defined
    readouts and must lie inside it.
    """
    pop = set(population)
    for name, arm in (("ff", ff), ("cc", cc)):
        if not pop <= set(arm):
            raise ValueError(f"{name} does not cover the population")
    for name, arm in (("fc", fc), ("cf", cf)):
        if not set(arm) <= pop:
            raise ValueError(f"{name} has conditions outside the population")
    for name, arm in (("ff", ff), ("fc", fc), ("cc", cc), ("cf", cf)):
        bad = [k for k in arm if not isinstance(arm[k], bool)]
        if bad:
            raise ValueError(f"{name} has non-bool correctness at {bad[:3]}")
    arms = {"ff": ff, "fc": fc, "cc": cc, "cf": cf}
    kw = {"n_resamples": n_resamples, "seed": seed}
    fwd = sorted(set(fc))
    rev = sorted(set(cf))
    common = sorted(set(fc) & set(cf))
    sub = {
        n: CONTRASTS[n]
        for n in ("readout_on_reference_trace", "residual_forward", "D_total")
    }
    rsub = {
        n: CONTRASTS[n]
        for n in (
            "trajectory_under_full_readout",
            "readout_on_compressed_trace",
            "D_total",
        )
    }
    result: dict[str, Any] = {
        "population": {
            "n_conditions": len(pop),
            "n_forward_defined": len(fwd),
            "n_reverse_defined": len(rev),
            "n_common_support": len(common),
            "forward_only": [list(k) for k in sorted(set(fc) - set(cf))],
            "reverse_only": [list(k) for k in sorted(set(cf) - set(fc))],
            "neither": [list(k) for k in sorted(pop - set(fc) - set(cf))],
        },
        "free_run_full_population": _support_block(
            arms, sorted(pop), {"D_total": CONTRASTS["D_total"]}, **kw
        ),
        "forward_support": _support_block(arms, fwd, sub, **kw) if fwd else None,
        "reverse_support": _support_block(arms, rev, rsub, **kw) if rev else None,
        "common_support": (
            _support_block(arms, common, CONTRASTS, **kw) if common else None
        ),
        "bounds_full_population": {
            name: worst_case_bounds(arms, sorted(pop), coefs)
            for name, coefs in CONTRASTS.items()
        },
        "interpretation": (
            "operational replay comparisons: residual_forward is the trajectory "
            "contrast under compressed readout, trajectory_under_full_readout the "
            "same contrast under full-cache readout; interaction is their "
            "difference. Neither is a bound on a causal trajectory share without "
            "further assumptions"
        ),
    }
    if common:
        cs = result["common_support"]
        result["endpoints_cross"] = bool(
            cs["residual_forward"]["mean"] * cs["trajectory_under_full_readout"]["mean"]
            < 0
        )
    return result


def protection_by_direction(
    protected: Mapping[str, Mapping[Key, bool]],
    base: Mapping[str, Mapping[Key, bool]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """How prompt protection changes each 2 x 2 contrast, paired.

    ``protected`` and ``base`` map arm name -> defined correctness for the two
    cells. The support is every condition with all four arms defined in BOTH
    cells (the readouts' supports differ by cell, so this is the only set on
    which every contrast is pairable). Each contrast gets the shrinkage ratio
    and the paired absolute change from one joint draw per contrast.
    """
    support = set(protected["ff"])
    for cell in (protected, base):
        for arm in ARMS:
            support &= set(cell[arm])
    keys = sorted(support)
    if not keys:
        raise ValueError("no condition has all four arms defined in both cells")
    out: dict[str, Any] = {
        "support": "all four arms defined in both cells",
        "n_conditions": len(keys),
        "n_problems": len({k[0] for k in keys}),
    }
    for name in ("residual_forward", "trajectory_under_full_readout", "interaction"):
        coefs = CONTRASTS[name]

        def per_key(
            cell: Mapping[str, Mapping[Key, bool]], coefs: Mapping[str, int] = coefs
        ) -> dict[Key, float]:
            return {
                k: sum(c * float(cell[a][k]) for a, c in coefs.items()) for k in keys
            }

        out[name] = shrinkage_ratio(
            per_key(protected), per_key(base), n_resamples=n_resamples, seed=seed
        )
    # Review of fbf869f: percentages use different baseline denominators, so
    # "protection works mainly in one direction" needs the paired difference of
    # the ABSOLUTE improvements: (fwd_pp - fwd_base) - (rev_pp - rev_base).
    # Algebraically this equals interaction_pp - interaction_base.
    f, r = CONTRASTS["residual_forward"], CONTRASTS["trajectory_under_full_readout"]

    def val(cell: Mapping[str, Mapping[Key, bool]], coefs: Mapping[str, int], k: Key):
        return sum(c * float(cell[a][k]) for a, c in coefs.items())

    series = {
        "d": {
            k: (val(protected, f, k) - val(base, f, k))
            - (val(protected, r, k) - val(base, r, k))
            for k in keys
        }
    }
    point, reps = joint_draws(series, n_resamples=n_resamples, seed=seed)
    out["difference_of_improvements"] = {
        **summarize(
            point["d"], [x["d"] for x in reps], n_resamples=n_resamples, seed=seed
        ),
        "definition": "(forward improvement) - (reverse improvement), paired, on "
        "the four-arm support; equals interaction(protected) - interaction(base)",
    }
    return out


def practical_reading(
    *,
    ff: Mapping[Key, bool],
    cc: Mapping[Key, bool],
    fc: Mapping[Key, bool],
    margins: Sequence[float] = PRACTICAL_MARGINS,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Section 4.3 on the forward-readout support (where all three are defined).

    F_loss = A_FF - A_CC, R_loss = A_FF - A_FC, residual = A_CC - A_FC. For each
    margin m:

    * ``false_reassurance``: R_loss's upper one-sided 97.5% bound <= m AND
      F_loss's lower one-sided 97.5% bound > m (Bonferroni pair, joint draws).
    * ``readout_already_lossy``: R_loss's lower bound > m -- the readout itself
      shows the loss, so any gap is underestimation, not false certification.
    * ``residual_equivalent``: the residual's 90% interval inside (-m, m)
      (TOST at alpha 0.05 each side). Not equivalent is not "different".
    """
    keys = sorted(set(fc))
    if not keys:
        raise ValueError("no defined forward readouts")
    if not (set(keys) <= set(ff) and set(keys) <= set(cc)):
        raise ValueError("free runs do not cover the readout support")
    series = {
        "F_loss": {k: float(ff[k]) - float(cc[k]) for k in keys},
        "R_loss": {k: float(ff[k]) - float(fc[k]) for k in keys},
        "residual": {k: float(cc[k]) - float(fc[k]) for k in keys},
    }
    point, reps = joint_draws(series, n_resamples=n_resamples, seed=seed)
    draws = {n: sorted(r[n] for r in reps) for n in series}
    kw = {"n_resamples": n_resamples, "seed": seed}
    out: dict[str, Any] = {
        "support": "forward-readout defined",
        "n_conditions": len(keys),
        "n_problems": len({k[0] for k in keys}),
        **{n: summarize(point[n], draws[n], **kw) for n in series},
        "margins": {},
        "margin_status": (
            "SENSITIVITY: margins proposed 2026-10-05 after the B=256 result "
            "was seen; 0.03 is the declared primary for NEW designs only"
        ),
    }
    r_hi = _quantile(draws["R_loss"], 0.975)
    r_lo = _quantile(draws["R_loss"], 0.025)
    f_lo = _quantile(draws["F_loss"], 0.025)
    res90 = (_quantile(draws["residual"], 0.05), _quantile(draws["residual"], 0.95))
    for m in margins:
        out["margins"][f"{m:.2f}"] = {
            "false_reassurance": bool(r_hi <= m and f_lo > m),
            "readout_already_lossy": bool(r_lo > m),
            "free_run_loss_exceeds_margin": bool(f_lo > m),
            "residual_equivalent_within_margin": bool(-m < res90[0] and res90[1] < m),
        }
    return out


def shrinkage_ratio(
    residual_protected: Mapping[Key, float],
    residual_base: Mapping[Key, float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """1 - |R_protected| / |R_base| with a paired joint bootstrap interval.

    Denominator stability is reported alongside: the share of replicates whose
    base residual is within 0.05 of zero (an unstable ratio) and the share in
    which the protected residual has the opposite sign to the base one.
    """
    if set(residual_protected) != set(residual_base):
        raise ValueError("residuals cover different conditions")
    point, reps = joint_draws(
        {"pp": residual_protected, "base": residual_base},
        n_resamples=n_resamples,
        seed=seed,
    )

    def ratio(m: Mapping[str, float]) -> float | None:
        return None if m["base"] == 0 else 1 - abs(m["pp"]) / abs(m["base"])

    vals = [ratio(r) for r in reps]
    defined = sorted(v for v in vals if v is not None)
    return {
        "ratio": ratio(point),
        "ci95": (
            [_quantile(defined, 0.025), _quantile(defined, 0.975)] if defined else None
        ),
        "absolute_change": summarize(
            point["pp"] - point["base"],
            [r["pp"] - r["base"] for r in reps],
            n_resamples=n_resamples,
            seed=seed,
        ),
        "share_base_near_zero": sum(abs(r["base"]) < 0.05 for r in reps) / len(reps),
        "share_sign_flipped": sum(r["pp"] * r["base"] < 0 for r in reps) / len(reps),
        "n_undefined_replicates": len(vals) - len(defined),
        "estimand": "ratio of equal-problem point estimates; descriptive, not a "
        "causal share",
    }


def _sign(lo: float, hi: float) -> int:
    """+1 / -1 when the interval excludes zero on that side, else 0."""
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def joint_label(*, acc_sign: int, met_sign: int) -> str:
    """Accuracy difference (a - b) against loss-metric difference (a - b).

    The metric is loss-type, so a metric that ranks policies correctly has
    OPPOSITE signs (more loss, lower accuracy). Same signs are a mismatch in
    either orientation. Fixed 2026-10-05 (review of fbf869f, R4): the old table
    called two negative differences "aligned".
    """
    if acc_sign == 0 or met_sign == 0:
        return "NOT ESTABLISHED"
    first = (
        f"first policy: {'higher' if acc_sign > 0 else 'lower'} accuracy, "
        f"{'more' if met_sign > 0 else 'less'} loss"
    )
    kind = "MISMATCH" if acc_sign == met_sign else "ALIGNED"
    return f"{kind} ESTABLISHED ({first})"


def policy_contrasts(
    accuracy: Mapping[str, Mapping[Key, float]],
    metric: Mapping[str, Mapping[Key, float]] | None,
    pairs: Sequence[tuple[str, str]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Matched-key paired policy contrasts, accuracy and (optionally) a metric,
    from ONE joint problem draw so each pair's two contrasts share replicates.

    ``metric`` is a loss-type quantity (higher = more perturbation, e.g. lost
    attention mass M). Every series must cover the same keys. The joint reading
    for a pair (a, b) is "a ranks above b on accuracy while b ranks above a
    on the metric" only when both FAMILY-ADJUSTED intervals (Bonferroni over
    the pairs: two-sided level 1 - 0.05 / len(pairs)) exclude zero in those
    directions; otherwise it is reported as not established. The unadjusted
    95% intervals are kept as descriptive.
    """
    series: dict[str, Mapping[Key, float]] = {
        f"acc:{p}": v for p, v in accuracy.items()
    }
    if metric:
        series.update({f"met:{p}": v for p, v in metric.items()})
    point, reps = joint_draws(series, n_resamples=n_resamples, seed=seed)
    kw = {"n_resamples": n_resamples, "seed": seed}
    out: dict[str, Any] = {}

    tail = 0.025 / len(pairs)

    def diff(name_a: str, name_b: str) -> dict[str, Any]:
        draws = sorted(r[name_a] - r[name_b] for r in reps)
        out_ = summarize(point[name_a] - point[name_b], draws, **kw)
        out_["ci_family"] = [_quantile(draws, tail), _quantile(draws, 1 - tail)]
        out_["family"] = f"Bonferroni over {len(pairs)} pairs"
        return out_

    for a, b in pairs:
        block: dict[str, Any] = {"accuracy": diff(f"acc:{a}", f"acc:{b}")}
        if metric and a in metric and b in metric:
            block["metric"] = diff(f"met:{a}", f"met:{b}")
            acc_lo, acc_hi = block["accuracy"]["ci_family"]
            met_lo, met_hi = block["metric"]["ci_family"]
            block["joint"] = joint_label(
                acc_sign=_sign(acc_lo, acc_hi), met_sign=_sign(met_lo, met_hi)
            )
        out[f"{a} - {b}"] = block
    return out


def by_key(
    rows: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], Key],
    value: Callable[[Mapping[str, Any]], bool | None],
) -> dict[Key, bool]:
    """Defined (bool) values by key; raises on a duplicate key."""
    out: dict[Key, bool] = {}
    for r in rows:
        k = key(r)
        if k in out:
            raise ValueError(f"duplicate key {k}")
        v = value(r)
        if v is not None:
            out[k] = bool(v)
    return out

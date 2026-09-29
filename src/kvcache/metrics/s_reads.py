"""Amendment S reads: pure functions over rows already on disk.

Scientific rationale. Both reads were registered with their branches before any
row existed (PREREGISTRATION Amendment S), so the labels below are lookups on
intervals, not judgments. S.1 asks whether the trajectory residual survives
when the prompt is never evicted. It is read on the same (problem, seed)
conditions as the unprotected cell, and paired, because the two cells share
their reference traces and seeds. S.2 asks which evicted entries carry the part
of the divergence that full restoration recovers, as a fraction of it.

Conventions match Amendment K and R.17: the residual is the accuracy change
unexplained by on-reference readout, never a measured causal effect; truncated
generations stay in every denominator, scored incorrect; every interval is a
problem-cluster percentile bootstrap of an equal-problem mean.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from kvcache.coverage import equal_problem_bootstrap, paired_equal_problem

Key = tuple[str, int]

RHO_CARRIES = 0.50
RHO_MINOR = 0.20
DETERMINISM_TOL = 1e-6
ORIGIN_ARMS = ("frozen", "prompt", "trace", "full")


# ------------------------------------------------------------------ S.1
def residual_label(ci: Sequence[float]) -> str:
    """S.1: does the protected residual's CI exclude 0?"""
    lo, hi = ci
    if hi < 0:
        return "RESIDUAL PERSISTS"
    if lo > 0:
        return "EXCLUDES ZERO, POSITIVE (fits no registered branch)"
    return "RESIDUAL NOT DETECTED"


def protection_effect_label(ci: Sequence[float]) -> str:
    """S.1: protected minus unprotected residual (the residual is negative)."""
    lo, hi = ci
    if lo > 0:
        return "SHRINKS"
    if hi < 0:
        return "GROWS"
    return "UNCHANGED"


def _restrict(table: Mapping[Key, Any], drop: Collection[str]) -> dict[Key, Any]:
    return {k: v for k, v in table.items() if k[0] not in drop}


def protection_contrasts(
    free_pp: Mapping[Key, bool],
    free_base: Mapping[Key, bool],
    readout_pp: Mapping[Key, bool],
    readout_base: Mapping[Key, bool],
    *,
    exclude_pids: Collection[str] = (),
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """One press and budget: the protected residual and the paired contrasts.

    ``free_*`` cover the cell's whole population. ``readout_*`` hold only the
    conditions whose reference trace has an answer boundary, which is a
    property of the reference and so the same set in both cells. The residual
    is taken per condition on that set (free minus readout), as ``report_m4``'s
    complete-case layer does, which is what makes it pairable.
    """
    fp, fb = _restrict(free_pp, exclude_pids), _restrict(free_base, exclude_pids)
    rp, rb = _restrict(readout_pp, exclude_pids), _restrict(readout_base, exclude_pids)
    if set(fp) != set(fb):
        raise ValueError("protected and unprotected cells cover different conditions")
    if set(rp) != set(rb):
        raise ValueError(
            "readout is defined on different conditions in the two cells; the "
            "answer boundary is a property of the shared reference trace"
        )
    if not set(rp) <= set(fp):
        raise ValueError("readout rows outside the cell population")
    kw = {"n_resamples": n_resamples, "seed": seed}
    res_p = {k: float(fp[k]) - float(rp[k]) for k in rp}
    res_b = {k: float(fb[k]) - float(rb[k]) for k in rb}
    residual_pp = equal_problem_bootstrap(res_p, **kw)
    d_res = paired_equal_problem(res_p, res_b, **kw)
    return {
        "n_conditions": len(fp),
        "n_readout_defined": len(rp),
        "excluded_problems": sorted(set(exclude_pids)),
        "free_accuracy_pp": equal_problem_bootstrap(
            {k: float(v) for k, v in fp.items()}, **kw
        ),
        "free_accuracy_base": equal_problem_bootstrap(
            {k: float(v) for k, v in fb.items()}, **kw
        ),
        "readout_accuracy_pp": equal_problem_bootstrap(
            {k: float(v) for k, v in rp.items()}, **kw
        ),
        "readout_accuracy_base": equal_problem_bootstrap(
            {k: float(v) for k, v in rb.items()}, **kw
        ),
        "residual_pp": residual_pp,
        "residual_base": equal_problem_bootstrap(res_b, **kw),
        "residual_label": residual_label(residual_pp["ci95"]),
        "delta_free": paired_equal_problem(
            {k: float(v) for k, v in fp.items()},
            {k: float(v) for k, v in fb.items()},
            **kw,
        ),
        "delta_readout": paired_equal_problem(
            {k: float(v) for k, v in rp.items()},
            {k: float(v) for k, v in rb.items()},
            **kw,
        ),
        "delta_residual": d_res,
        "protection_effect": protection_effect_label(d_res["ci95"]),
    }


def flip_masses(
    reference: Mapping[Key, bool], cell: Mapping[Key, bool]
) -> dict[str, float]:
    """Harmful and beneficial flip mass of a cell against the full cache."""
    keys = sorted(cell)
    if not keys or not set(keys) <= set(reference):
        raise ValueError("cell conditions absent from the reference")
    n = len(keys)
    return {
        "harmful": sum(1 for k in keys if reference[k] and not cell[k]) / n,
        "beneficial": sum(1 for k in keys if not reference[k] and cell[k]) / n,
    }


def paper_consequence(labels: Mapping[str, Mapping[int, str]]) -> str:
    """The S.1 paper-level branch from per-press, per-budget residual labels."""
    per_press = {}
    for press, by_budget in labels.items():
        if set(by_budget) != {256, 1024}:
            return "INCOMPLETE: both budgets are needed"
        if any(v.startswith("EXCLUDES ZERO, POSITIVE") for v in by_budget.values()):
            return "FITS NO BRANCH: reported plainly"
        p256 = by_budget[256] == "RESIDUAL PERSISTS"
        p1024 = by_budget[1024] == "RESIDUAL PERSISTS"
        per_press[press] = (
            "A"
            if p256 and p1024
            else "B" if p256 else "C" if not p1024 else "FITS NO BRANCH"
        )
    found = set(per_press.values())
    if len(found) != 1:
        return f"PRESSES DISAGREE: {dict(sorted(per_press.items()))}; no pooled claim"
    return {
        "A": "A: PERSISTS at both budgets; the headline stands",
        "B": "B: PERSISTS at 256 only; the 4x regime is not established under "
        "prompt protection",
        "C": "C: NOT DETECTED at both budgets; attributable to prompt loss at "
        "this scope",
        "FITS NO BRANCH": "FITS NO BRANCH: reported plainly",
    }[found.pop()]


# ------------------------------------------------------------------ S.2
def rho_label(ci: Sequence[float]) -> str:
    lo, hi = ci
    if lo >= RHO_CARRIES:
        return "PROMPT CARRIES IT"
    if hi <= RHO_MINOR:
        return "PROMPT MINOR"
    if lo > RHO_MINOR and hi < RHO_CARRIES:
        return "PARTIAL"
    return "INDETERMINATE"


def _problems(keys: Collection[Key]) -> dict[str, list[Key]]:
    per: dict[str, list[Key]] = defaultdict(list)
    for k in sorted(keys):
        per[k[0]].append(k)
    return per


def _eqm(vals: Mapping[Key, float], per: Mapping[str, list[Key]], ps: Sequence[str]):
    return statistics.fmean(statistics.fmean(vals[k] for k in per[p]) for p in ps)


def origin_ratios(
    tv: Mapping[str, Mapping[Key, float]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """S.2: rho_prompt, rho_trace and their difference, on one resampling.

    rho_arm = (TV_frozen - TV_arm) / (TV_frozen - TV_full), a ratio of
    equal-problem means. The difference is bootstrapped on the SAME problem
    draws as the two ratios, so it is a paired interval.
    """
    if set(tv) != set(ORIGIN_ARMS):
        raise ValueError(f"need exactly the arms {ORIGIN_ARMS}")
    keys = set(tv["frozen"])
    if any(set(tv[a]) != keys for a in ORIGIN_ARMS) or not keys:
        raise ValueError("the four arms must cover the same, non-empty conditions")
    per = _problems(keys)
    pids = sorted(per)
    gap = {k: tv["frozen"][k] - tv["full"][k] for k in keys}
    num = {
        a: {k: tv["frozen"][k] - tv[a][k] for k in keys} for a in ("prompt", "trace")
    }
    denominator = equal_problem_bootstrap(gap, n_resamples=n_resamples, seed=seed)
    lo, hi = denominator["ci95"]
    if not lo > 0:
        return {
            "label": "UNDEFINED: nothing to apportion",
            "restorable_gap": denominator,
            "note": (
                "the CI of TV_frozen - TV_full does not lie above 0"
                if hi > 0
                else "full restoration did not reduce divergence"
            ),
        }
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {"prompt": [], "trace": [], "diff": []}
    for _ in range(n_resamples):
        ps = [pids[rng.randrange(len(pids))] for _ in pids]
        d = _eqm(gap, per, ps)
        if d <= 0:
            continue
        rp, rt = _eqm(num["prompt"], per, ps) / d, _eqm(num["trace"], per, ps) / d
        draws["prompt"].append(rp)
        draws["trace"].append(rt)
        draws["diff"].append(rp - rt)

    def ci(xs: list[float]) -> list[float]:
        xs = sorted(xs)
        return [xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]]

    d0 = _eqm(gap, per, pids)
    point = {a: _eqm(num[a], per, pids) / d0 for a in ("prompt", "trace")}
    diff_ci = ci(draws["diff"])
    return {
        "restorable_gap": denominator,
        "rho_prompt": {"ratio": point["prompt"], "ci95": ci(draws["prompt"])},
        "rho_trace": {"ratio": point["trace"], "ci95": ci(draws["trace"])},
        "rho_prompt_minus_trace": {
            "mean": point["prompt"] - point["trace"],
            "ci95": diff_ci,
        },
        "label": rho_label(ci(draws["prompt"])),
        "specificity": "PROMPT-SPECIFIC" if diff_ci[0] > 0 else "NON-SPECIFIC",
        "n_traces": len(keys),
        "n_problems": len(pids),
        "n_resamples_used": len(draws["diff"]),
        "window_mean_tv": {
            a: equal_problem_bootstrap(dict(tv[a]), n_resamples=n_resamples, seed=seed)
            for a in ORIGIN_ARMS
        },
    }


def determinism_control(
    v3: Mapping[Key, Mapping[str, float]],
    v2: Mapping[Key, Mapping[str, float]],
    tol: float = DETERMINISM_TOL,
) -> dict[str, Any]:
    """S.2 control: restore-v3's ``full`` and ``none`` equal restore-v2's.

    Each map is key -> {"full": .., "none": ..}. A READ restore-v3 condition
    with no restore-v2 counterpart fails the control: it cannot be checked.
    """
    missing = sorted(set(v3) - set(v2))
    worst = {"full": 0.0, "none": 0.0}
    for k in set(v3) & set(v2):
        for f in worst:
            worst[f] = max(worst[f], abs(float(v3[k][f]) - float(v2[k][f])))
    passed = not missing and bool(v3) and all(v <= tol for v in worst.values())
    return {
        "passed": passed,
        "tolerance": tol,
        "max_abs_diff": worst,
        "n_compared": len(set(v3) & set(v2)),
        "missing_in_restore_v2": [list(k) for k in missing],
        "on_failure": "STOP: an instrument bug, not a result",
    }

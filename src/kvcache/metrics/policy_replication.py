"""Protection-by-policy accuracy read for Meeting 5 M5-A (prepared 2026-10-14,
before any protected H2O-style or R-KV row exists).

Scientific rationale. The review's primary M5-A endpoint is the paired
free-running accuracy contrast between Random and each scored policy j, with
and without prompt protection, and its interaction

    I_j = (A_{R,+P} - A_{j,+P}) - (A_{R,-P} - A_{j,-P}),

computed per (problem, seed) on matched keys, averaged over seeds within a
problem and then over problems. The registered inference is the B=1024 one
(``protocol_audit.problem_level_t_tests``): a two-sided problem-level t test,
Holm over the scientific pairings. A protected contrast "vanishes" only when
both one-sided tests reject inside the frozen margin (TOST, protocol 4.3);
otherwise it is "inconclusive" or "outside the margin" -- a non-significant
difference is never called equivalence.

Coverage. Expected keys come from the frozen cohort and seeds (never from
rows). Every arm must hold every expected key exactly once; otherwise the read
is PARTIAL and carries no number. Truncated generations are already scored
incorrect in the rows and stay in every denominator.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from kvcache.metrics.protocol_audit import Key, problem_level_t_tests

MARGIN = 0.05  # protocol 4.3, frozen 2026-10-14
TOST_ALPHA = 0.05
FINAL, PARTIAL, INVALID = "FINAL", "PARTIAL", "INVALID"

Arm = tuple[str, bool]  # (policy, protected)


def arm_coverage(
    rows: Sequence[Mapping[str, Any]], expected: set[Key]
) -> tuple[dict[Key, float] | None, dict[str, int]]:
    """Per-key correctness of one arm, or None unless coverage is exact."""
    vals: dict[Key, float] = {}
    dup = foreign = 0
    for r in rows:
        k = (str(r["pid"]), int(r["decoding"]["seed"]))
        if k not in expected:
            foreign += 1
            continue
        if k in vals:
            dup += 1
            continue
        vals[k] = float(bool(r["correct"]))
    info = {
        "n_rows": len(rows),
        "n_covered": len(vals),
        "n_expected": len(expected),
        "n_duplicates": dup,
        "n_foreign": foreign,
    }
    ok = len(vals) == len(expected) and dup == 0
    return (vals if ok else None), info


def tost(problem_values: Sequence[float], margin: float = MARGIN) -> dict[str, Any]:
    """Two one-sided t tests of |mean| < margin over problem-level values."""
    from scipy import stats

    n = len(problem_values)
    if n < 2:
        return {"label": "NOT_TESTABLE", "n_problems": n}
    mean = statistics.fmean(problem_values)
    se = statistics.stdev(problem_values) / math.sqrt(n)
    if se == 0.0:
        inside = abs(mean) < margin
        return {
            "label": "within margin" if inside else "outside margin",
            "mean": mean,
            "se": 0.0,
            "n_problems": n,
        }
    t_crit = float(stats.t.ppf(1 - TOST_ALPHA, df=n - 1))
    lo, hi = mean - t_crit * se, mean + t_crit * se
    if -margin < lo and hi < margin:
        label = "within margin"
    elif hi <= -margin or lo >= margin:
        label = "outside margin"
    else:
        label = "inconclusive"
    return {
        "label": label,
        "mean": mean,
        "se": se,
        "ci90": [lo, hi],
        "margin": margin,
        "n_problems": n,
    }


def _problem_means(values: Mapping[Key, float]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for (pid, _), v in values.items():
        by.setdefault(pid, []).append(v)
    return {p: statistics.fmean(v) for p, v in by.items()}


def replication_read(
    arms: Mapping[Arm, Sequence[Mapping[str, Any]] | None],
    *,
    expected: set[Key],
    reference: str,
    pairings: Sequence[str],
    descriptive: Sequence[str] = (),
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The M5-A read over one cohort; ``None`` for an arm means no rows yet."""
    from kvcache.stats import cluster_bootstrap_mean

    coverage: dict[str, Any] = {}
    values: dict[Arm, dict[Key, float]] = {}
    for arm, rows in arms.items():
        tag = f"{arm[0]}{'+P' if arm[1] else '-P'}"
        if rows is None:
            coverage[tag] = {"status": "absent"}
            continue
        vals, info = arm_coverage(rows, expected)
        coverage[tag] = info
        if vals is not None:
            values[arm] = vals
    needed = {
        (p, prot)
        for p in (reference, *pairings, *descriptive)
        for prot in (False, True)
    }
    missing = sorted(f"{p}{'+P' if b else '-P'}" for p, b in needed - set(values))
    report: dict[str, Any] = {
        "coverage": coverage,
        "n_expected_keys": len(expected),
        "margin": MARGIN,
    }
    if missing:
        return {**report, "status": PARTIAL, "missing_arms": missing}

    def contrast(j: str, protected: bool) -> dict[Key, float]:
        a, b = values[(reference, protected)], values[(j, protected)]
        return {k: a[k] - b[k] for k in expected}

    def ci(vals: Mapping[Key, float]) -> dict[str, Any]:
        pm = _problem_means(vals)
        bs = cluster_bootstrap_mean(
            {p: [v] for p, v in pm.items()}, n_resamples=n_resamples, seed=seed
        )
        return {"mean": bs["mean"], "ci95": bs["ci95"], "n_problems": len(pm)}

    family: dict[str, dict[Key, float]] = {}
    out: dict[str, Any] = {}
    for j in (*pairings, *descriptive):
        prot, unprot = contrast(j, True), contrast(j, False)
        inter = {k: prot[k] - unprot[k] for k in expected}
        if j in pairings:
            family[f"{reference}-{j}"] = inter
        out[f"{reference}-{j}"] = {
            "role": "scientific" if j in pairings else "descriptive",
            "protected": {
                **ci(prot),
                "tost": tost(list(_problem_means(prot).values())),
            },
            "unprotected": ci(unprot),
            "interaction": ci(inter),
        }
    accuracy = {
        f"{p}{'+P' if b else '-P'}": ci(v) for (p, b), v in sorted(values.items())
    }
    return {
        **report,
        "status": FINAL,
        "accuracy": accuracy,
        "contrasts": out,
        "registered_inference": problem_level_t_tests(family),
        "bootstrap": {"unit": "problem", "n_resamples": n_resamples, "seed": seed},
    }

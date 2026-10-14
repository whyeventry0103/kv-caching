"""Protection-by-policy accuracy read for Meeting 5 M5-A (prepared 2026-10-14,
before any protected H2O-style or R-KV row exists; validation rebuilt after the
review of 6b409001, R1).

Scientific rationale. The review's primary M5-A endpoint is the paired
free-running accuracy contrast between Random and each scored policy j, with
and without prompt protection, and its interaction

    I_j = (A_{R,+P} - A_{j,+P}) - (A_{R,-P} - A_{j,-P}),

computed per (problem, seed) on matched keys, averaged over seeds within a
problem and then over problems. Inference is the registered B=1024 one
(``protocol_audit.problem_level_t_tests``: problem-level t test, Holm over the
two scientific pairings).

Validation goes through the shared boundary (``read_checks.validate_rows``:
exact coverage, required fields, EXACT booleans checked before any
conversion -- "False" and NaN are not booleans). Each SOURCE run is validated
in full against its own config's design before the cohort is selected, so a
foreign seed or a duplicate is INVALID even when it lies outside the cohort;
missing expected keys are PARTIAL. On top of the shared checks (the same
"malformed" channel): model and benchmark must be the declared ones, a
truncated row may not be scored correct, a LEGACY source row may not carry an
identity block it never had (and none is fabricated), and an M5 source row
must carry the expected identity and frozen prompt hash
(``kvcache.m5_runs.row_identity_problems``).

Equivalence (protocol 4.3; owner decision 2026-10-14, frozen in the read spec):
a protected contrast is "within margin" only if both one-sided tests reject at
alpha (TOST) with the PRIMARY margin; the sensitivity margin is reported
beside it. Per-comparison equivalence claims are Holm-adjusted over the
scientific pairings; the joint claim "every scientific contrast is within
margin" is an intersection-union test (largest TOST p <= alpha). A degenerate
sampling estimate (zero standard error) is NOT TESTABLE -- its observed
difference is still reported descriptively. A non-significant difference is
never called equivalence.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from kvcache.metrics.protocol_audit import NOT_TESTABLE, Key, problem_level_t_tests

FINAL, PARTIAL, INVALID = "FINAL", "PARTIAL", "INVALID"
GRID_REQUIRED = ("pid", "press", "budget", "decoding.seed", "trace", "model", "bench")
GRID_BOOLS = ("correct", "truncated")

Arm = tuple[str, bool]  # (policy, protected)


def grid_key(r: Mapping[str, Any]) -> tuple:
    return (
        str(r["press"]),
        str(r["budget"]),
        str(r["pid"]),
        int(r["decoding"]["seed"]),
    )


def row_value_problems(
    row: Mapping[str, Any],
    *,
    model: str,
    bench: str,
    provenance: str,
    identity_check: Any = None,
) -> list[str]:
    """M5-A checks beyond the shared boundary (empty list = well formed)."""
    out = []
    if row.get("model") != model:
        out.append(f"model {row.get('model')!r} is not {model!r}")
    if row.get("bench") != bench:
        out.append(f"bench {row.get('bench')!r} is not {bench!r}")
    if row.get("truncated") is True and row.get("correct") is not False:
        out.append("truncated row is not scored incorrect")
    if provenance == "legacy":
        if "provenance" in row:
            out.append("legacy-source row carries an identity block it never had")
    elif provenance == "m5":
        out += identity_check(row) if identity_check else ["no identity check"]
    else:
        out.append(f"unknown provenance contract {provenance!r}")
    return out


def validate_source(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected: set[tuple],
    model: str,
    bench: str,
    provenance: str,
    identity_check: Any = None,
) -> dict[str, Any]:
    """The shared validation block for one source run, plus M5-A value checks."""
    from kvcache.read_checks import validate_rows

    block = validate_rows(
        rows,
        key=grid_key,
        expected=expected,
        required=GRID_REQUIRED,
        bool_fields=GRID_BOOLS,
    )
    for i, r in enumerate(rows):
        probs = row_value_problems(
            r,
            model=model,
            bench=bench,
            provenance=provenance,
            identity_check=identity_check,
        )
        if probs:
            try:
                k = repr(grid_key(r))
            except (KeyError, TypeError, ValueError):
                k = f"<row {i}>"
            block["malformed"][k] = block["malformed"].get(k, []) + probs
            block["status"] = INVALID
    return block


def tost(
    problem_values: Sequence[float], *, margin: float, alpha: float
) -> dict[str, Any]:
    """Two one-sided t tests of |mean| < margin over problem-level values."""
    from scipy import stats

    n = len(problem_values)
    if n < 2:
        return {"label": NOT_TESTABLE, "n_problems": n, "margin": margin}
    mean = statistics.fmean(problem_values)
    se = statistics.stdev(problem_values) / math.sqrt(n)
    if se == 0.0 or not math.isfinite(se):
        return {
            "label": NOT_TESTABLE,
            "reason": "degenerate sampling estimate (zero standard error)",
            "observed_mean": mean,
            "n_problems": n,
            "margin": margin,
        }
    df = n - 1
    p_lower = float(stats.t.sf((mean + margin) / se, df))  # H0: mean <= -margin
    p_upper = float(stats.t.cdf((mean - margin) / se, df))  # H0: mean >= +margin
    p = max(p_lower, p_upper)
    t_crit = float(stats.t.ppf(1 - alpha, df))
    return {
        "label": "within margin" if p <= alpha else "not shown within margin",
        "mean": mean,
        "se": se,
        "df": df,
        "ci_1_minus_2alpha": [mean - t_crit * se, mean + t_crit * se],
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p,
        "margin": margin,
        "n_problems": n,
    }


def equivalence_family(
    members: Mapping[str, Sequence[float]], *, margin: float, alpha: float
) -> dict[str, Any]:
    """Per-comparison TOST with Holm over the family, and the joint IUT claim."""
    rows = {k: tost(v, margin=margin, alpha=alpha) for k, v in members.items()}
    m = len(rows)
    tested = sorted((r["p_tost"], k) for k, r in rows.items() if "p_tost" in r)
    running = 0.0
    for rank, (p, k) in enumerate(tested):
        running = max(running, min(1.0, (m - rank) * p))
        rows[k]["p_tost_holm"] = running
        rows[k]["equivalent_per_comparison"] = bool(running <= alpha)
    joint = (
        bool(len(tested) == m and max(p for p, _ in tested) <= alpha) if m else False
    )
    return {
        "margin": margin,
        "alpha": alpha,
        "members": rows,
        "per_comparison_rule": "TOST, Holm over the scientific pairings",
        "joint_rule": "intersection-union: every member's TOST p <= alpha",
        "joint_within_margin": joint,
    }


def _problem_means(values: Mapping[Key, float]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for (pid, _), v in values.items():
        by.setdefault(pid, []).append(v)
    return {p: statistics.fmean(v) for p, v in by.items()}


def replication_read(
    values: Mapping[Arm, Mapping[Key, float]],
    *,
    expected: set[Key],
    reference: str,
    pairings: Sequence[str],
    descriptive: Sequence[str] = (),
    equivalence: Mapping[str, Any],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimates from VALIDATED arm values (every arm covers ``expected``)."""
    from kvcache.stats import cluster_bootstrap_mean

    for arm, v in values.items():
        if set(v) != expected:
            raise ValueError(f"arm {arm} does not cover the cohort exactly")

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
    protected_pm: dict[str, list[float]] = {}
    out: dict[str, Any] = {}
    for j in (*pairings, *descriptive):
        prot, unprot = contrast(j, True), contrast(j, False)
        inter = {k: prot[k] - unprot[k] for k in expected}
        name = f"{reference}-{j}"
        if j in pairings:
            family[name] = inter
            protected_pm[name] = list(_problem_means(prot).values())
        out[name] = {
            "role": "scientific" if j in pairings else "descriptive",
            "protected": ci(prot),
            "unprotected": ci(unprot),
            "interaction": ci(inter),
        }
    alpha = float(equivalence["alpha"])
    eq = {
        "primary": equivalence_family(
            protected_pm, margin=float(equivalence["primary_margin"]), alpha=alpha
        ),
        "sensitivity": equivalence_family(
            protected_pm, margin=float(equivalence["sensitivity_margin"]), alpha=alpha
        ),
        "declared": dict(equivalence),
    }
    accuracy = {
        f"{p}{'+P' if b else '-P'}": ci(v) for (p, b), v in sorted(values.items())
    }
    return {
        "accuracy": accuracy,
        "contrasts": out,
        "registered_inference": problem_level_t_tests(family),
        "equivalence_protected_contrasts": eq,
        "bootstrap": {"unit": "problem", "n_resamples": n_resamples, "seed": seed},
    }

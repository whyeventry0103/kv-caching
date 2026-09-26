"""Amendment R.17 reads over data already on disk (registered 2026-09-27).

Pure functions only: every row loading and file writing lives in
``scripts/report_r17.py``. Each read here is the computation PREREGISTRATION
R.17 fixes, with the branch labels spelled exactly as registered, so a read can
never be relabelled after its numbers are seen.

The scientific point of keeping them together: these are the reads that decide
whether Meeting #3's divergence results are licensed to enter the paper at all
(R.17.1.2 can withdraw the onset effect; R.17.1.8 asks whether any fine-scale
quantity predicts a free-running outcome), so they share one estimator -- the
equal-problem, problem-clustered bootstrap -- and one set of conventions.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from kvcache.coverage import equal_problem_bootstrap, paired_equal_problem

Key = tuple[str, int]


# ------------------------------------------------------------- R.17 addendum
def eligibility_profile(
    eligible: Mapping[Key, bool],
    fc: Mapping[Key, Mapping[str, Any]],
    trace_tokens: Mapping[Key, int],
    *,
    prompt_tokens: Mapping[Key, int] | None = None,
    budget: int = 256,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Who the restoration read describes: eligible vs ineligible traces.

    With ``prompt_tokens`` the realized ratio is reported per subset as
    ``budget / (mean prompt + mean trace tokens)`` (the project convention).

    ``eligible`` maps (problem, seed) -> whether the trace was read; ``fc`` maps
    the same keys to the full-cache reference row (``correct``, ``truncated``);
    ``trace_tokens`` to the reference trace length. Full-cache accuracy is the
    equal-problem mean with its CI, the estimator every other accuracy uses.
    """
    out: dict[str, Any] = {}
    for label, want in (("eligible", True), ("ineligible", False)):
        keys = [k for k, v in eligible.items() if bool(v) is want]
        missing = [k for k in keys if k not in fc]
        if missing:
            raise ValueError(f"{len(missing)} traces have no full-cache row")
        acc = equal_problem_bootstrap(
            {k: float(bool(fc[k]["correct"])) for k in keys},
            n_resamples=n_resamples,
            seed=seed,
        )
        trunc = equal_problem_bootstrap(
            {k: float(bool(fc[k]["truncated"])) for k in keys},
            n_resamples=n_resamples,
            seed=seed,
        )
        out[label] = {
            "n_traces": len(keys),
            "n_problems": len({k[0] for k in keys}),
            "problems": sorted({k[0] for k in keys}),
            "fc_accuracy": {k: acc[k] for k in ("mean", "ci95", "n_problems")},
            "fc_truncation": {k: trunc[k] for k in ("mean", "ci95")},
            "median_trace_tokens": (
                statistics.median(trace_tokens[k] for k in keys) if keys else None
            ),
        }
        if prompt_tokens is not None and keys:
            mp = statistics.fmean(prompt_tokens[k] for k in keys)
            mt = statistics.fmean(trace_tokens[k] for k in keys)
            out[label]["mean_prompt_tokens"] = mp
            out[label]["mean_trace_tokens"] = mt
            out[label]["realized_ratio"] = budget / (mp + mt)
    return out


# ------------------------------------------------------------------ R.17.1.2
ONSET_PAIRS = (("0", "2048"), ("512", "2048"), ("0", "512"))


def restricted_onset_contrasts(
    comp: Mapping[tuple[str, str, str], Mapping[Key, float]],
    exposed: Mapping[tuple[str, str, str], Mapping[Key, bool]],
    fc_correct: Mapping[Key, float],
    fc_len: Mapping[Key, int],
    *,
    min_len: int = 2304,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """R.17.1.2: onset contrasts restricted on the REFERENCE trace length.

    ``comp[(press, budget, onset)]`` maps (problem, seed) -> compressed
    correctness. The restriction uses ``fc_len`` (the full-cache reference's
    generated tokens), a pre-treatment covariate; the compressed run's own
    length is never consulted. Contrast = Acc(later onset) - Acc(earlier onset)
    = the extra damage from the earlier onset.
    """
    all_keys = set(fc_len)
    kept = {k for k in all_keys if fc_len[k] > min_len}
    out: dict[str, Any] = {
        "min_reference_tokens_exclusive": min_len,
        "n_pairs_total": len(all_keys),
        "n_pairs_retained": len(kept),
        "retained_fraction": len(kept) / len(all_keys) if all_keys else None,
        "retained_fc_accuracy": equal_problem_bootstrap(
            {k: fc_correct[k] for k in kept}, n_resamples=n_resamples, seed=seed
        ),
        "cells": {},
    }
    for press, budget in sorted({(p, b) for p, b, _ in comp}):
        late = exposed.get((press, budget, "2048"), {})
        attr = sum(1 for k in kept if k in late and not late[k])
        cell: dict[str, Any] = {"unexposed_at_t0_2048_within_retained": attr}
        for early, later in ONSET_PAIRS:
            a, b = comp[(press, budget, later)], comp[(press, budget, early)]
            un = paired_equal_problem(a, b, n_resamples=n_resamples, seed=seed)
            re = paired_equal_problem(
                {k: v for k, v in a.items() if k in kept},
                {k: v for k, v in b.items() if k in kept},
                n_resamples=n_resamples,
                seed=seed,
            )
            cell[f"t0_{early}_vs_{later}"] = {
                "unrestricted": _brief(un),
                "restricted": _brief(re),
            }
        r = cell["t0_0_vs_2048"]["restricted"]
        if budget == "256":
            cell["registered_reading"] = (
                "onset effect NOT ESTABLISHED; unrestricted effect attributable "
                "to never-compressed traces"
                if not r["excludes_zero"]
                else (
                    "onset effect holds among long traces"
                    if r["mean"] > 0
                    else "restricted contrast excludes zero with the opposite sign"
                )
            )
        out["cells"][f"{press}-{budget}"] = cell
    return out


def _brief(o: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean": o["mean"],
        "ci95": o["ci95"],
        "excludes_zero": o.get("excludes_zero"),
        "n_pairs": o.get("n_pairs"),
        "n_problems": o.get("n_problems"),
    }


# ------------------------------------------------------------------ R.17.1.5
def spread_read(stats: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Spread across policies and the median CI half-width (H2's two numbers)."""
    means = [float(s["mean"]) for s in stats.values()]
    hws = [(float(s["ci95"][1]) - float(s["ci95"][0])) / 2 for s in stats.values()]
    return {
        "spread": max(means) - min(means),
        "median_half_width": statistics.median(hws),
        "n_policies": len(means),
    }


def h2_label(m: Mapping[str, float], d: Mapping[str, float]) -> str:
    """MIRROR / INVERTED / NEITHER exactly as R.17.1.5 registers them."""
    d_small = d["spread"] < d["median_half_width"]
    m_large = m["spread"] > m["median_half_width"]
    if d_small and m_large:
        return "MIRROR"
    if d["spread"] > d["median_half_width"] and m["spread"] < m["median_half_width"]:
        return "INVERTED"
    return "NEITHER"


def h3_read(
    m_h2o: Mapping[Key, float],
    m_random: Mapping[Key, float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired M(H2O-style) - M(Random); HOLDS iff the CI is entirely negative."""
    o = paired_equal_problem(m_h2o, m_random, n_resamples=n_resamples, seed=seed)
    holds = o["ci95"] is not None and o["ci95"][1] < 0
    return {**_brief(o), "label": "HOLDS" if holds else "FAILS"}


# ------------------------------------------------------------------ R.17.1.7
def per_layer_omega(
    rows: Mapping[Key, Mapping[str, Sequence[float]]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per layer: trace omega = sum_omega / n_finite, then equal-problem mean.

    A trace with no finite omega in a layer is excluded from that layer (the
    R.6 rule), never counted as zero.
    """
    n_layers = len(next(iter(rows.values()))["sum_omega"])
    out = []
    for layer in range(n_layers):
        vals = {
            k: r["sum_omega"][layer] / r["n_finite"][layer]
            for k, r in rows.items()
            if r["n_finite"][layer] > 0
        }
        o = equal_problem_bootstrap(vals, n_resamples=n_resamples, seed=seed)
        out.append(
            {
                "layer": layer,
                "mean": o["mean"],
                "ci95": o["ci95"],
                "n_problems": o["n_problems"],
            }
        )
    return out


# ------------------------------------------------------------------ R.17.1.8
def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """P(score of a positive > score of a negative), ties counted one half.

    ``None`` when either class is empty: an AUROC with no positives is
    undefined, not 0.5.
    """
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def auroc_cluster_bootstrap(
    scores: Mapping[Key, float],
    labels: Mapping[Key, bool],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """AUROC with a problem-cluster percentile CI and a two-sided bootstrap p.

    Problems are resampled with all their seeds. Replicates with one class
    missing are dropped and counted; p = 2 * min(P(A* <= .5), P(A* >= .5)).
    """
    keys = sorted(set(scores) & set(labels))
    by_pid: dict[str, list[Key]] = defaultdict(list)
    for k in keys:
        by_pid[k[0]].append(k)
    pids = sorted(by_pid)
    point = auroc([scores[k] for k in keys], [labels[k] for k in keys])
    rng = random.Random(seed)
    draws = []
    dropped = 0
    for _ in range(n_resamples):
        ks = [k for _ in pids for k in by_pid[pids[rng.randrange(len(pids))]]]
        a = auroc([scores[k] for k in ks], [labels[k] for k in ks])
        if a is None:
            dropped += 1
        else:
            draws.append(a)
    draws.sort()
    n = len(draws)
    ci = [draws[int(0.025 * (n - 1))], draws[int(0.975 * (n - 1))]] if n else None
    below, above = sum(a <= 0.5 for a in draws), sum(a >= 0.5 for a in draws)
    p = min(1.0, 2 * min(below, above) / n) if n else None
    return {
        "auroc": point,
        "ci95": ci,
        "p_two_sided": p,
        "n_traces": len(keys),
        "n_positive": sum(bool(labels[k]) for k in keys),
        "n_problems": len(pids),
        "n_resamples_used": n,
        "n_resamples_dropped_one_class": dropped,
    }


# ------------------------------------------------------------------ R.17.1.1
def window_tv(estimate: Sequence[float], position: int, width: int = 256) -> float:
    """Mean TV over [position, position + width); the window must be full."""
    hi = position + width
    if position < 0 or hi > len(estimate):
        raise ValueError("window does not fit the profile")
    return statistics.fmean(float(v) for v in estimate[position:hi])


def summary_window_tv(summary: Mapping[str, Any], position: int) -> float | None:
    """The same window from a tv-profile-summary-v1 record, or ``None``.

    Summary windows start at multiples of ``window``; a position must be one of
    those starts, and the window must be complete (n_steps >= position + w).
    """
    w = int(summary["window"])
    if position % w:
        raise ValueError(f"position {position} is not a summary window start")
    if int(summary["n_steps"]) < position + w:
        return None
    return float(summary["window_mean_tv"][position // w])


def collapse_label(pairs: Sequence[Mapping[str, Any]], tv_range: float) -> str:
    """COLLAPSE / NO COLLAPSE / MIXED exactly as R.17.1.1 registers them.

    Each pair carries ``mean``, ``ci95`` and ``mismatch_bound``; a CI that
    excludes zero with ``|mean| <= |mismatch_bound|`` counts as including zero.
    """

    def excl(p: Mapping[str, Any]) -> bool:
        lo, hi = p["ci95"]
        return (lo > 0 or hi < 0) and abs(p["mean"]) > abs(p["mismatch_bound"])

    if all(not excl(p) for p in pairs) and all(
        abs(p["mean"]) < 0.25 * tv_range for p in pairs
    ):
        return "COLLAPSE"
    if all(excl(p) for p in pairs):
        signs = {p["mean"] > 0 for p in pairs}
        if len(signs) == 1:
            return (
                "NO COLLAPSE: absolute budget matters"
                if signs.pop()
                else "NO COLLAPSE: horizon matters"
            )
    return "MIXED"

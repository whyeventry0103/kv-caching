"""Per-dataset read of the M5-B multi-dataset replication (frozen 2026-11-13,
before any GSM8K or BBH confirmation outcome exists).

Scientific rationale. The supervisor question is whether Random still beats
StreamingLLM and H2O-style on new datasets, and whether keeping the prompt
changes those gaps. Every estimand here is the Meeting 5 M5-A one, applied to
one dataset at a time and never pooled across datasets (plan 9.4):

* free-running accuracy per arm and its paired loss versus full cache, per
  (problem, seed), averaged over seeds within a problem and then over problems;
* the paired contrasts Random - StreamingLLM and Random - H2O-style (the two
  scientific pairings; registered inference ``protocol_audit.
  problem_level_t_tests``, Holm over the two, separately per protection arm)
  and Random - R-KV (descriptive); problem-cluster bootstrap CIs
  (``stats.cluster_bootstrap_mean``: 2000 resamples, seed 0) are descriptive;
* where both protection arms exist, the protection-by-policy interaction via
  the registered M5-A read (``policy_replication.replication_read``, with the
  M5-A equivalence spec unchanged). A protected-only stage reports no
  interaction and says it cannot estimate the effect of protection;
* forward readout on MATCHED DEFINED keys: a key enters a readout comparison
  only when its boundary is defined for every arm compared; undefined keys are
  counted per arm and never scored. On those keys the residual is the
  accuracy change unexplained by on-reference readout,
  ``(free - FC) - (readout - FC)`` -- never a measured causal effect;
* exposure: the realized budget ratio each row records, the prune count where
  the row carries a protection report (recorded), and the overflow
  ``P + T - B_eff`` from the row's own prompt and trace lengths (derived; the
  unprotected presses record no prune events on this path).

Truncated generations are scored incorrect by the registered graders and stay
in every denominator. Nothing here selects a cohort, a budget or a dataset.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from kvcache.metrics.policy_replication import replication_read
from kvcache.metrics.protocol_audit import problem_level_t_tests
from kvcache.presses.prompt_protect import effective_budget
from kvcache.stats import cluster_bootstrap_mean

Key = tuple[str, int]  # (pid, seed)
Arm = tuple[str, bool]  # (policy, protected)
SIDE = {True: "protected", False: "unprotected"}


def _ci(values: Mapping[Key, float], *, n_resamples: int, seed: int) -> dict:
    by: dict[str, list[float]] = {}
    for (pid, _), v in values.items():
        by.setdefault(pid, []).append(float(v))
    pm = {p: [statistics.fmean(v)] for p, v in by.items()}
    bs = cluster_bootstrap_mean(pm, n_resamples=n_resamples, seed=seed)
    return {
        "mean": bs["mean"],
        "ci95": bs["ci95"],
        "n_problems": len(pm),
        "n_conditions": len(values),
    }


def _quant(v: Sequence[float]) -> dict[str, float] | None:
    if not v:
        return None
    s = sorted(v)
    return {
        "p10": s[int(0.1 * (len(s) - 1))],
        "median": float(statistics.median(s)),
        "p90": s[int(0.9 * (len(s) - 1))],
    }


def exposure(rows: Mapping[Key, Mapping[str, Any]], *, budget: int) -> dict[str, Any]:
    """Realized ratio (recorded), prune count (recorded where present) and the
    derived overflow of one arm's rows."""
    ratios = [float(r["compression_ratio"]) for r in rows.values()]
    prunes = [
        int(r["n_prunes"]) for r in rows.values() if r.get("n_prunes") is not None
    ]
    over = []
    for r in rows.values():
        b_eff, _ = effective_budget(budget, int(r["prompt_tokens"]))
        over.append(int(r["prompt_tokens"]) + int(r["n_tokens"]) - b_eff)
    reached = [o for o in over if o > 0]
    return {
        "realized_ratio": _quant(ratios),
        "n_prunes_recorded": _quant([float(p) for p in prunes]),
        "n_with_prune_record": len(prunes),
        "overflow_reached_rate": len(reached) / len(over),
        "overflow_tokens_when_reached": _quant([float(o) for o in reached]),
        "exposure_source": (
            "ratio recorded; prunes recorded where a protection report exists; "
            "overflow derived from recorded prompt and trace lengths"
        ),
    }


def _matched_defined(
    readouts: Mapping[Arm, Mapping[Key, Mapping[str, Any]]], arms: Sequence[Arm]
) -> list[Key]:
    keys = set.intersection(*(set(readouts[a]) for a in arms))
    return sorted(k for k in keys if all(readouts[a][k]["defined"] for a in arms))


def readout_block(
    fc: Mapping[Key, Mapping[str, Any]],
    arms: Mapping[Arm, Mapping[Key, Mapping[str, Any]]],
    readouts: Mapping[Arm, Mapping[Key, Mapping[str, Any]]],
    *,
    reference: str,
    others: Sequence[str],
    protected: bool,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Readout accuracy, readout change and residual per arm, and the Random -
    j gaps in each, on the keys defined for every arm of this side."""
    side = [(p, protected) for p in (reference, *others) if (p, protected) in readouts]
    if (reference, protected) not in readouts:
        return {"status": "ABSENT"}
    keys = _matched_defined(readouts, side)
    out: dict[str, Any] = {
        "matched_defined_keys": len(keys),
        "undefined_by_arm": {
            a[0]: sum(not r["defined"] for r in readouts[a].values()) for a in side
        },
        "arms": {},
        "contrasts": {},
    }
    if not keys:
        out["status"] = "NO DEFINED KEYS"
        return out

    def vals(f: Any) -> dict[Key, float]:
        return {k: float(f(k)) for k in keys}

    for a in side:
        free = vals(lambda k, a=a: arms[a][k]["correct"])
        ro = vals(lambda k, a=a: readouts[a][k]["correct"])
        ref = vals(lambda k: fc[k]["correct"])
        out["arms"][a[0]] = {
            "free": _ci(free, n_resamples=n_resamples, seed=seed),
            "readout": _ci(ro, n_resamples=n_resamples, seed=seed),
            "fc_same_keys": _ci(ref, n_resamples=n_resamples, seed=seed),
            "readout_change": _ci(
                {k: ro[k] - ref[k] for k in keys}, n_resamples=n_resamples, seed=seed
            ),
            "residual": _ci(
                {k: free[k] - ro[k] for k in keys}, n_resamples=n_resamples, seed=seed
            ),
        }
    r = (reference, protected)
    for j in others:
        if (j, protected) not in readouts:
            continue
        a = (j, protected)
        gap = {}
        for what, src in (("free", arms), ("readout", readouts)):
            gap[what] = _ci(
                {
                    k: float(src[r][k]["correct"]) - float(src[a][k]["correct"])
                    for k in keys
                },
                n_resamples=n_resamples,
                seed=seed,
            )
        gap["residual"] = _ci(
            {
                k: (float(arms[r][k]["correct"]) - float(readouts[r][k]["correct"]))
                - (float(arms[a][k]["correct"]) - float(readouts[a][k]["correct"]))
                for k in keys
            },
            n_resamples=n_resamples,
            seed=seed,
        )
        out["contrasts"][f"{reference}-{j}"] = gap
    out["status"] = "FINAL"
    return out


def read_dataset(
    fc: Mapping[Key, Mapping[str, Any]],
    arms: Mapping[Arm, Mapping[Key, Mapping[str, Any]]],
    readouts: Mapping[Arm, Mapping[Key, Mapping[str, Any]]],
    *,
    expected: set[Key],
    budget: int,
    reference: str,
    pairings: Sequence[str],
    descriptive: Sequence[str],
    equivalence: Mapping[str, Any],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The frozen M5-B read of one dataset from VALIDATED, complete arms."""
    if set(fc) != expected:
        raise ValueError("full-cache bank does not cover the cohort exactly")
    for a, rows in arms.items():
        if set(rows) != expected:
            raise ValueError(f"arm {a} does not cover the cohort exactly")
    for a, recs in readouts.items():
        if set(recs) != expected:
            raise ValueError(f"readout {a} does not cover the cohort exactly")
    kw = {"n_resamples": n_resamples, "seed": seed}
    others = (*pairings, *descriptive)
    fc_v = {k: float(fc[k]["correct"]) for k in expected}
    out: dict[str, Any] = {
        "status": "FINAL",
        "n_problems": len({p for p, _ in expected}),
        "n_conditions": len(expected),
        "full_cache": {
            "accuracy": _ci(fc_v, **kw),
            "truncation_rate": sum(bool(fc[k]["truncated"]) for k in expected)
            / len(expected),
        },
        "arms": {},
        "sides": {},
    }
    for (p, prot), rows in sorted(arms.items()):
        v = {k: float(rows[k]["correct"]) for k in expected}
        out["arms"][f"{p}{'+P' if prot else '-P'}"] = {
            "accuracy": _ci(v, **kw),
            "loss_vs_full_cache": _ci({k: v[k] - fc_v[k] for k in expected}, **kw),
            "truncation_rate": sum(bool(rows[k]["truncated"]) for k in expected)
            / len(expected),
            "exposure": exposure(rows, budget=budget),
        }
    for prot in (True, False):
        r = (reference, prot)
        present = [j for j in others if (j, prot) in arms]
        if r not in arms or not present:
            out["sides"][SIDE[prot]] = {"status": "NOT RUN"}
            continue
        contrasts, family = {}, {}
        for j in present:
            d = {
                k: float(arms[r][k]["correct"]) - float(arms[(j, prot)][k]["correct"])
                for k in expected
            }
            contrasts[f"{reference}-{j}"] = {
                "role": "scientific" if j in pairings else "descriptive",
                **_ci(d, **kw),
            }
            if j in pairings:
                family[f"{reference}-{j}"] = d
        out["sides"][SIDE[prot]] = {
            "status": "FINAL",
            "contrasts": contrasts,
            "registered_inference": problem_level_t_tests(family),
            "readout": readout_block(
                fc,
                arms,
                readouts,
                reference=reference,
                others=present,
                protected=prot,
                **kw,
            ),
        }
    both = all((p, s) in arms for p in (reference, *others) for s in (True, False))
    if both:
        values = {
            a: {k: float(rows[k]["correct"]) for k in expected}
            for a, rows in arms.items()
        }
        out["protection_interaction"] = replication_read(
            values,
            expected=expected,
            reference=reference,
            pairings=pairings,
            descriptive=descriptive,
            equivalence=equivalence,
            n_resamples=n_resamples,
            seed=seed,
        )
    else:
        out["protection_interaction"] = {
            "status": "NOT ESTIMABLE",
            "why": "both protection arms are needed for every policy; a "
            "protected-only stage compares policies under protection but cannot "
            "estimate the effect of protection",
        }
    return out


# --- assembly from validated cells (frozen with the read, 2026-11-13) --------

ARM_OF_PRESS: dict[str, Arm] = {
    "random_pp": ("random", True),
    "streamingllm_pp": ("streamingllm", True),
    "h2o_style_pp_v1": ("h2o", True),
    "rkv_pp_v1": ("rkv", True),
    "random": ("random", False),
    "streamingllm": ("streamingllm", False),
    "h2o": ("h2o", False),
    "rkv": ("rkv", False),
}


def cell_status(exc: Exception | None, *, exists: bool) -> str:
    """ABSENT (no run dir), PARTIAL (only expected keys missing), INVALID
    (anything else: extra, repeated, torn, pending, mismatched), or FINAL."""
    if not exists:
        return "ABSENT"
    if exc is None:
        return "FINAL"
    parts = [p.strip() for p in str(exc).split(";") if p.strip()]
    only_missing = all(
        "expected keys lack" in p or "expected readouts missing" in p for p in parts
    )
    return "PARTIAL" if only_missing else "INVALID"


def grid_arms(
    runs_root: Any,
    design: Mapping[str, Any],
    golds: Mapping[str, str],
) -> dict[Arm, dict[Key, dict[str, Any]]]:
    """Every expected pair of one compressed-run cell, validated and scored with
    the benchmark's registered grader, as per-arm rows; ValueError otherwise."""
    from kvcache.gsm8k_timing import score_rows, validate_pairs
    from kvcache.manifest_grid import design_keys

    split = {"split": design["split"], "revision": design["revision"]}
    pairs = validate_pairs(
        runs_root,
        bench=design["benchmark"],
        keys=design_keys(design),
        manifest_split=split,
    )
    scored = score_rows(
        [r for r, _ in pairs],
        golds,
        revision=design["revision"],
        bench=design["benchmark"],
    )
    out: dict[Arm, dict[Key, dict[str, Any]]] = {}
    for (row, side), s in zip(pairs, scored["scores"], strict=True):
        arm = ARM_OF_PRESS[str(row["press"])]
        protection = row.get("protection") or {}
        out.setdefault(arm, {})[(str(row["pid"]), int(row["decoding"]["seed"]))] = {
            "correct": bool(s["correct"]),
            "truncated": bool(s["truncated"]),
            "compression_ratio": float(row["compression_ratio"]),
            "prompt_tokens": len(side["prompt_token_ids"]),
            "n_tokens": int(row["n_tokens"]),
            "n_prunes": protection.get("n_prunes"),
        }
    return out


def fc_bank(
    runs_root: Any,
    design: Mapping[str, Any],
    golds: Mapping[str, str],
) -> dict[Key, dict[str, Any]]:
    """The full-cache reference bank, validated and scored."""
    from kvcache.gsm8k_timing import score_rows, validate_coverage

    split = {"split": design["split"], "revision": design["revision"]}
    pairs = validate_coverage(
        runs_root,
        bench=design["benchmark"],
        pids=design["pids"],
        seeds=design["seeds"],
        manifest_split=split,
    )
    scored = score_rows(
        [r for r, _ in pairs],
        golds,
        revision=design["revision"],
        bench=design["benchmark"],
    )
    return {
        (str(r["pid"]), int(r["decoding"]["seed"])): {
            "correct": bool(s["correct"]),
            "truncated": bool(s["truncated"]),
            "n_tokens": int(r["n_tokens"]),
            "prompt_tokens": len(side["prompt_token_ids"]),
        }
        for (r, side), s in zip(pairs, scored["scores"], strict=True)
    }


def readout_arms(
    runs_root: Any,
    design: Mapping[str, Any],
    golds: Mapping[str, str],
) -> dict[Arm, dict[Key, dict[str, Any]]]:
    """Every expected readout record of one cell (``manifest_reads``: exactly
    one per key), with its definedness and, if defined, its registered score."""
    from kvcache.manifest_reads import score_readouts

    read = score_readouts(runs_root, design, golds)
    scored = {
        (str(s["press"]), str(s["pid"]), int(s["seed"])): s for s in read["scores"]
    }
    out: dict[Arm, dict[Key, dict[str, Any]]] = {}
    for press in design["presses"]:
        arm = ARM_OF_PRESS[str(press)]
        for seed in design["seeds"]:
            for pid in design["pids"]:
                s = scored.get((str(press), str(pid), int(seed)))
                out.setdefault(arm, {})[(str(pid), int(seed))] = {
                    "defined": s is not None,
                    "correct": bool(s["correct"]) if s is not None else None,
                }
    return out

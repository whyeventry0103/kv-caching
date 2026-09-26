#!/usr/bin/env python3
"""Amendment R.17 reads over rows already on disk (registered 2026-09-27).

Thin orchestration: load rows, call ``kvcache.metrics.r17_reads``, write one
JSON per read under ``results/diag/meeting3/r17/``. Runs where the rows live
(the GPU host) and needs no GPU.

    python scripts/report_r17.py --runs-root <runs> eligibility
    python scripts/report_r17.py --runs-root <runs> dacc-restricted
    python scripts/report_r17.py --runs-root <runs> mdelta
    python scripts/report_r17.py --runs-root <runs> omega-layers
    python scripts/report_r17.py --runs-root <runs> bridge
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.coverage import equal_problem_bootstrap
from kvcache.metrics import r17_reads as r17
from kvcache.provenance import write_json_atomic
from kvcache.stats import benjamini_hochberg

OUT = PROJECT_ROOT / "results" / "diag" / "meeting3" / "r17"
FC_RUN = "math500-fc-powered-merged"
POLICIES = ("random", "streamingllm", "h2o", "rkv")


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for p in sorted((run_dir / "math500").rglob("*.jsonl"))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fc(runs: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for r in _rows(runs / FC_RUN):
        k = (r["pid"], int(r["decoding"]["seed"]))
        if k in out:
            raise ValueError(f"duplicate full-cache row {k}")
        out[k] = r
    return out


def _seed(r: dict[str, Any]) -> int:
    return int(r["seed"] if "seed" in r else r["decoding"]["seed"])


def _write(name: str, payload: dict[str, Any], argv: list[str]) -> Path:
    payload = {
        "amendment": "R.17",
        "reproduce": "python scripts/report_r17.py " + " ".join(argv),
        **payload,
    }
    path = OUT / f"{name}.json"
    write_json_atomic(path, payload)
    print(f"wrote {path}")
    return path


# ------------------------------------------------------------------- reads
def eligibility(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    rows = _rows(runs / "exact-lag-r5")
    fc = _fc(runs)
    by_key: dict[tuple[str, int], set[bool]] = {}
    toks: dict[tuple[str, int], int] = {}
    for r in rows:
        k = (r["pid"], _seed(r))
        by_key.setdefault(k, set()).add(r["status"] == "READ")
        toks[k] = int(r["n_trace_tokens"])
    split = [k for k, v in by_key.items() if len(v) != 1]
    if split:
        raise ValueError(f"eligibility differs across policies for {split[:3]}")
    elig = {k: next(iter(v)) for k, v in by_key.items()}
    prompts = {
        (r["pid"], _seed(r)): int(r["prompt_tokens"])
        for r in _rows(runs / "capture-b256")
    }
    prof = r17.eligibility_profile(
        elig,
        fc,
        toks,
        prompt_tokens=prompts,
        budget=256,
        n_resamples=a.n_resamples,
        seed=a.seed,
    )
    return {
        "read": "R.17 addendum: who the exact-lag restoration read describes",
        "inputs": {"exact_lag_run": "exact-lag-r5", "fc_run": FC_RUN},
        "n_rows": len(rows),
        "rule": "eligible iff stash (= prompt + cohort step - 256) >= 64 (R.10)",
        **prof,
    }


def dacc_restricted(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    tsv = PROJECT_ROOT / "results/diag/meeting3/dacc.conditions.tsv"
    comp: dict[tuple[str, str, str], dict[tuple[str, int], float]] = {}
    exposed: dict[tuple[str, str, str], dict[tuple[str, int], bool]] = {}
    fc_correct: dict[tuple[str, int], float] = {}
    with tsv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            k = (row["pid"], int(row["seed"]))
            cell = (row["press"], row["budget"], row["onset"])
            comp.setdefault(cell, {})[k] = float(row["comp_correct"])
            exposed.setdefault(cell, {})[k] = row["exposed"] == "1"
            fc_correct[k] = float(row["fc_correct"])
    fc = _fc(runs)
    fc_len = {k: int(fc[k]["n_tokens"]) for k in fc_correct}
    out = r17.restricted_onset_contrasts(
        comp,
        exposed,
        fc_correct,
        fc_len,
        min_len=2304,
        n_resamples=a.n_resamples,
        seed=a.seed,
    )
    return {
        "read": "R.17.1.2 D_acc onset contrasts restricted on reference length",
        "inputs": {"dacc_tsv": str(tsv.relative_to(PROJECT_ROOT)), "fc_run": FC_RUN},
        "contrast": "Acc_comp(later onset) - Acc_comp(earlier onset)",
        **out,
    }


def _trace_layer_means(r: dict[str, Any], field: str) -> list[float | None]:
    s = r["sum_by_layer_group"][field]
    n = r["n_measured_by_layer_group"]
    out: list[float | None] = []
    for layer in range(len(s)):
        tot, cnt = sum(s[layer]), sum(n[layer])
        out.append(tot / cnt if cnt else None)
    return out


def mdelta(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    rows = _rows(runs / a.capture_run)
    per: dict[str, dict[tuple[str, int], dict[str, Any]]] = {p: {} for p in POLICIES}
    for r in rows:
        per[r["press"]][(r["pid"], _seed(r))] = r
    stats: dict[str, dict[str, Any]] = {}
    layers: dict[str, dict[str, Any]] = {}
    for f in ("M", "Delta", "M_event"):
        stats[f] = {}
        layers[f] = {}
        for p in POLICIES:
            stats[f][p] = equal_problem_bootstrap(
                {k: r[f"{f}_mean"] for k, r in per[p].items()},
                n_resamples=a.n_resamples,
                seed=a.seed,
            )
            tl = {k: _trace_layer_means(r, f) for k, r in per[p].items()}
            n_layers = len(next(iter(tl.values())))
            layers[f][p] = []
            for layer in range(n_layers):
                o = equal_problem_bootstrap(
                    {k: v[layer] for k, v in tl.items() if v[layer] is not None},
                    n_resamples=a.n_resamples,
                    seed=a.seed,
                )
                layers[f][p].append(
                    {"layer": layer, "mean": o["mean"], "ci95": o["ci95"]}
                )
    m_read = r17.spread_read(stats["M"])
    d_read = r17.spread_read(stats["Delta"])
    h3 = r17.h3_read(
        {k: r["M_mean"] for k, r in per["h2o"].items()},
        {k: r["M_mean"] for k, r in per["random"].items()},
        n_resamples=a.n_resamples,
        seed=a.seed,
    )
    ctrl = {
        "layer0_delta_max": max(float(r["checks"]["layer0_delta_max"]) for r in rows),
        "n_nonfinite": sum(int(r["checks"]["n_nonfinite"]) for r in rows),
        "n_rows": len(rows),
        "registered_tolerance": "layer-0 Delta <= 1e-5 at every event (R.9)",
    }
    brief = {
        f: {
            p: {k: stats[f][p][k] for k in ("mean", "ci95", "n_problems")}
            for p in POLICIES
        }
        for f in stats
    }
    return {
        "read": "R.17.1.5 H2/H3 (b256: exploratory) and the M/Delta report (0.6)",
        "status_on_this_run": (
            "EXPLORATORY: per-policy means were printed before R.17"
            if a.capture_run == "capture-b256"
            else "CONFIRMATORY (R.17.1.5)"
        ),
        "inputs": {"capture_run": a.capture_run, "n_rows": len(rows)},
        "per_policy": brief,
        "per_layer": layers,
        "H2": {"M": m_read, "Delta": d_read, "label": r17.h2_label(m_read, d_read)},
        "H3": {"contrast": "M(H2O-style) - M(Random)", **h3},
        "first_prune_self_control": ctrl,
        "stored": "per-(layer, KV group) sums over events; per-query-head "
        "values are computed and averaged within the group, not stored "
        "(capture-v1)",
    }


def omega_layers(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    rows = _rows(runs / a.capture_run)
    out: dict[str, Any] = {}
    for p in ("random", "streamingllm"):
        lat = {(r["pid"], _seed(r)): r["latent"] for r in rows if r["press"] == p}
        out[p] = r17.per_layer_omega(lat, n_resamples=a.n_resamples, seed=a.seed)
    return {
        "read": "R.17.1.7 per-layer omega (descriptive)",
        "inputs": {"capture_run": a.capture_run},
        "estimator": "trace omega_layer = sum_omega / n_finite; equal-problem mean",
        "k_sensitivity": "not computable: capture-v1 stores per-step per-layer "
        "norms at k = 64, not projection coefficients",
        "per_layer": out,
        "components": {
            "note": "R.17 addendum D: Random - StreamingLLM, paired equal-problem",
            **r17.omega_components(
                {
                    (r["pid"], _seed(r)): r["latent"]
                    for r in rows
                    if r["press"] == "random"
                },
                {
                    (r["pid"], _seed(r)): r["latent"]
                    for r in rows
                    if r["press"] == "streamingllm"
                },
                n_resamples=a.n_resamples,
                seed=a.seed,
            ),
        },
    }


def bridge(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    cap = _rows(runs / a.capture_run)
    grid = _rows(runs / a.grid_run)
    trunc = {
        (r["press"], r["pid"], _seed(r)): bool(r["truncated"])
        for r in grid
        if int(r["budget"]) == 256
    }
    tests = []
    for p in ("random", "streamingllm"):
        rs = [r for r in cap if r["press"] == p]
        labels = {(r["pid"], _seed(r)): trunc[(p, r["pid"], _seed(r))] for r in rs}
        preds = {
            "omega": {(r["pid"], _seed(r)): r["latent"]["omega_trace"] for r in rs},
            "Delta": {(r["pid"], _seed(r)): r["Delta_mean"] for r in rs},
            "M": {(r["pid"], _seed(r)): r["M_mean"] for r in rs},
        }
        for name, sc in preds.items():
            o = r17.auroc_cluster_bootstrap(
                sc, labels, n_resamples=a.n_resamples, seed=a.seed
            )
            tests.append({"policy": p, "predictor": name, **o})
    ps = [t["p_two_sided"] if t["p_two_sided"] is not None else 1.0 for t in tests]
    q = benjamini_hochberg(ps)
    for t, qv in zip(tests, q, strict=True):
        t["q_bh"] = qv
        sig = qv <= 0.05
        few = t["n_positive"] < 5 or (t["n_traces"] - t["n_positive"]) < 5
        if few:
            t["verdict"] = "not established at this n (< 5 in a class)"
        elif t["predictor"] in ("omega", "Delta"):
            t["verdict"] = (
                "CONFIRMED" if (sig and t["auroc"] > 0.5) else "NOT CONFIRMED"
            )
        else:
            t["verdict"] = "CONFIRMED (no prediction)" if not sig else "NOT CONFIRMED"
    return {
        "read": "R.17.1.8 bridge: teacher-forced M/Delta/omega vs free-run truncation",
        "inputs": {"capture_run": a.capture_run, "grid_run": a.grid_run},
        "family": "6 tests, Benjamini-Hochberg q = 0.05",
        "tests": tests,
    }


def _profiles(run_dir: Path) -> list[tuple[dict[str, Any], Path]]:
    from kvcache.replay import is_profile_file

    root = run_dir / "profiles"
    out = []
    for f in sorted(root.rglob("*.json")) if root.exists() else []:
        if is_profile_file(f):
            out.append((json.loads(f.read_text(encoding="utf-8")), f))
    return out


def _pk(d: dict[str, Any]) -> tuple[str, str, int, tuple[str, int]]:
    rk = d["reference_key"]
    c = d["condition"]
    return c["press"], str(c["budget"]), int(rk[5]), (rk[1], int(rk[5]))


def collapse(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    """R.17.1.1: onset-0 window TV at 2304/3328/4352, four budgets, per policy."""
    win: dict[str, dict[tuple[int, int], dict[tuple[str, int], float]]] = {}
    for d, _f in _profiles(runs / "replay-onset-t0"):
        press, budget, _s, key = _pk(d)
        for t in r17.POSITIONS:
            if t + 256 <= len(d["estimate"]):
                win.setdefault(press, {}).setdefault((int(budget), t), {})[key] = (
                    r17.window_tv(d["estimate"], t)
                )
    missing = []
    for b, run in ((512, "t1-tv512"), (1024, "t1-tv1024")):
        profs = _profiles(runs / run)
        if not profs:
            missing.append(run)
        for d, _f in profs:
            press, _b, _s, key = _pk(d)
            for t in r17.POSITIONS:
                v = r17.summary_window_tv(d, t)
                if v is not None:
                    win.setdefault(press, {}).setdefault((b, t), {})[key] = v
    prompts = {
        (r["pid"], _seed(r)): int(r["prompt_tokens"])
        for r in _rows(runs / "capture-b256")
    }
    if missing:
        return {
            "read": "R.17.1.1 collapse test",
            "status": "PENDING",
            "why": f"no profiles yet in {missing}",
        }
    return {
        "read": "R.17.1.1 collapse test (onset 0)",
        "status": "READ",
        "inputs": {
            "b256_b2048": "replay-onset-t0 (tv-profile-v2, full per-step)",
            "b512_b1024": "t1-tv512 / t1-tv1024 (tv-profile-summary-v1 windows)",
            "prompts": "capture-b256 prompt_tokens",
        },
        "ratio_convention": "r = budget / (prompt + position)",
        "per_policy": {
            p: r17.collapse_read(w, prompts, n_resamples=a.n_resamples, seed=a.seed)
            for p, w in sorted(win.items())
        },
    }


def auc_tv(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    """R.17.1.3: per-trace mean TV (AUC) at four budgets, equal-problem CIs."""
    vals: dict[tuple[str, str], dict[tuple[str, int], float]] = {}
    for d, _f in _profiles(runs / "replay-ext-diss256"):
        press, budget, _s, key = _pk(d)
        est = d["estimate"]
        vals.setdefault((press, budget), {})[key] = sum(est) / len(est)
    for run in ("t1-tv512", "t1-tv1024"):
        for d, _f in _profiles(runs / run):
            press, budget, _s, key = _pk(d)
            vals.setdefault((press, budget), {})[key] = float(d["mean_tv"])
    cells = {}
    for (press, budget), v in sorted(vals.items()):
        o = equal_problem_bootstrap(v, n_resamples=a.n_resamples, seed=a.seed)
        cells[f"{press}-{budget}"] = {
            k: o[k] for k in ("mean", "ci95", "n_problems", "n_values")
        }
    return {
        "read": "R.17.1.3 AUC-TV with problem-clustered CIs",
        "inputs": {
            "b256_b2048": "replay-ext-diss256 (the profiles behind "
            "results/diag/amendment-p/auc_resid_tv.json)",
            "b512_b1024": "t1-tv512 / t1-tv1024 summaries (mean_tv)",
        },
        "cells": cells,
    }


def restore(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    """R.17.2.3: eligibility, R_fs with its branch, cohort arms, channels."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from report_exact_lag import read_rows

    rows = _rows(runs / "restore-v2")
    fc = _fc(runs)
    status = {}
    for r in rows:
        status.setdefault(r["press"], []).append(r["status"])
    out: dict[str, Any] = {
        "read": "R.17.2.3 restoration redesigned",
        "n_rows": len(rows),
        "status_counts": {
            p: {s: v.count(s) for s in set(v)} for p, v in sorted(status.items())
        },
        "per_policy": {},
    }
    for press in sorted(status):
        rs = [r for r in rows if r["press"] == press]
        read = [r for r in rs if r["status"] == "READ"]
        elig = {(r["pid"], _seed(r)): r["status"] == "READ" for r in rs}
        toks = {(r["pid"], _seed(r)): int(r["n_trace_tokens"]) for r in rs}
        prof = r17.eligibility_profile(
            elig, fc, toks, n_resamples=a.n_resamples, seed=a.seed
        )
        fs = [r for r in read if (r.get("full_restore") or {}).get("eligible")]
        num = {(r["pid"], _seed(r)): r["full_restore"]["window_mean_tv"] for r in fs}
        den = {
            (r["pid"], _seed(r)): r["full_restore"]["none_window_mean_tv"] for r in fs
        }
        ratio = (
            r17.ratio_equal_problem(num, den, n_resamples=a.n_resamples, seed=a.seed)
            if fs
            else None
        )
        chan: dict[str, Any] = {}
        for arm in ("none", "full_restore"):
            by_off: dict[int, dict[str, dict[tuple[str, int], float]]] = {}
            for r in fs:
                e = r["cohort_step"]
                for s_, v in (r.get("channels") or {}).get(arm, {}).items():
                    off = int(s_) - e
                    for f in ("M", "Delta"):
                        by_off.setdefault(off, {}).setdefault(f, {})[
                            (r["pid"], _seed(r))
                        ] = v[f]
            chan[arm] = {
                str(off): {
                    f: {
                        k: o[k]
                        for k in ("mean", "ci95")
                        for o in [
                            equal_problem_bootstrap(
                                vals, n_resamples=a.n_resamples, seed=a.seed
                            )
                        ]
                    }
                    for f, vals in fs_.items()
                }
                for off, fs_ in sorted(by_off.items())
            }
        out["per_policy"][press] = {
            "eligibility": prof,
            "full_restore": {
                "n_traces": len(fs),
                "R_fs": ratio,
                "branch": r17.restoration_branch(ratio["ci95"]) if ratio else None,
            },
            "cohort_arms": read_rows(read, n_resamples=a.n_resamples, seed=a.seed).get(
                press
            ),
            "channels_by_offset_from_trigger": chan,
        }
    return out


def fork_precision(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    """R.17.0.5: the fork gap against the bf16-vs-fp32 floor; its distribution."""
    from kvcache.fork_precision import fork_p95, gate_read, window_max_p95

    fp_path = runs / "fork-precision" / "fork_precision.json"
    forks: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in ("fork-control-random", "fork-control-slm"):
        for r in _rows(runs / run):
            for f in r.get("forks") or []:
                forks.setdefault((r["press"], int(r["interval"])), []).append(
                    {**f, "n_trace_tokens": r.get("n_forward")}
                )
    dist = {
        f"{p}-i{i}": {
            q: r17.distribution([f[q] for f in fs if f.get(q) is not None])
            for q in ("first_tv", "max_tv", "first_abs_logit", "max_abs_logit")
        }
        for (p, i), fs in sorted(forks.items())
    }
    where = {}
    for (p, i), fs in sorted(forks.items()):
        steps = sorted(f["step"] for f in fs)
        qs = [steps[int(q * (len(steps) - 1))] for q in (0.25, 0.5, 0.75)]
        bins: dict[str, list[float]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        for f in fs:
            b = (
                "Q1"
                if f["step"] <= qs[0]
                else (
                    "Q2"
                    if f["step"] <= qs[1]
                    else ("Q3" if f["step"] <= qs[2] else "Q4")
                )
            )
            bins[b].append(f["max_tv"])
        where[f"{p}-i{i}"] = {
            "step_quartile_edges": qs,
            "max_tv_by_step_quartile": {
                k: r17.distribution(v) for k, v in bins.items()
            },
        }
    out: dict[str, Any] = {
        "read": "R.17.0.5 direct fork control: distribution and precision-floor gate",
        "fork_distribution": dist,
        "fork_max_tv_by_trace_position": where,
        "layers_and_vocabulary": "fork-control-v1 stores per-fork first/max TV and "
        "|logit| only; per-layer and per-vocabulary location come from the "
        "one-trace diagnosis below",
    }
    if not fp_path.exists():
        out["status"] = "PENDING: fork-precision has not run"
        return out
    d = json.loads(fp_path.read_text(encoding="utf-8"))
    floor_blocks: list[list[float]] = []
    for _pid, steps in d["floor"].items():
        js = sorted(int(j) for j in steps)
        block: list[float] = []
        prev = None
        for j in js:
            if prev is not None and j != prev + 1:
                floor_blocks.append(block)
                block = []
            block.append(float(steps[str(j)]["tv"]))
            prev = j
        if block:
            floor_blocks.append(block)
    floor = {i: window_max_p95(floor_blocks, i) for i in (16, 64, 256)}
    fork_stat = {k: fork_p95([f["max_tv"] for f in fs]) for k, fs in forks.items()}
    gate = gate_read(fork_stat, {i: floor[i]["p95"] for i in floor})
    diag = {}
    for dtype, v in d["diagnosis"].items():
        st = v["steps"]
        diag[dtype] = {
            "fork_step": v["fork_step"],
            "n_compared": v["n_compared"],
            "max_tv": max(x["tv"] for x in st),
            "max_abs_logit": max(x["max_abs_logit"] for x in st),
            "median_argmax_rank_in_b": statistics.median(
                x["argmax_token_rank_in_b"] for x in st
            ),
            "min_top10_overlap": min(x["top10_overlap"] for x in st),
            "top1_agree_all": all(x["top1_agree"] for x in st),
        }
    fp32 = diag.get("float32") or {}
    numerics = fp32.get("max_tv", 1) <= 1e-6 and fp32.get("max_abs_logit", 1) <= 1e-3
    out.update(
        {
            "status": "READ",
            "floor": {str(k): v for k, v in floor.items()},
            "floor_all_steps_tv": r17.distribution(
                [x for b in floor_blocks for x in b]
            ),
            "gate": gate,
            "diagnosis": diag,
            "diagnosis_label": "NUMERICS" if numerics else "IMPLEMENTATION",
        }
    )
    return out


READS = {
    "eligibility": ("exact_lag_eligibility", eligibility),
    "dacc-restricted": ("dacc_restricted", dacc_restricted),
    "mdelta": ("mdelta", mdelta),
    "omega-layers": ("omega_layers", omega_layers),
    "bridge": ("bridge", bridge),
    "collapse": ("collapse", collapse),
    "auc-tv": ("auc_tv", auc_tv),
    "restore": ("restore_v2", restore),
    "fork-precision": ("fork_precision", fork_precision),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--capture-run", default="capture-b256")
    ap.add_argument("--grid-run", default="grid-diss256-tok")
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("read", choices=sorted(READS))
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"command": "r17", "read": a.read, "out": str(OUT)}))
        return 0
    stem, fn = READS[a.read]
    name = stem if a.capture_run == "capture-b256" else f"{stem}_{a.capture_run}"
    _write(name, fn(a.runs_root, a), sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

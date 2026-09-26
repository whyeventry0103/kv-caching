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


READS = {
    "eligibility": ("exact_lag_eligibility", eligibility),
    "dacc-restricted": ("dacc_restricted", dacc_restricted),
    "mdelta": ("mdelta", mdelta),
    "omega-layers": ("omega_layers", omega_layers),
    "bridge": ("bridge", bridge),
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

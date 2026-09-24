#!/usr/bin/env python3
"""Amendment R.1: the Task 1 budget curve, assembled with its registered CIs.

Joins, per (policy, budget), the paired problem-clustered contrast computed
here from raw rows (kvcache.metrics.paired) with the readout/residual already
in the committed m4 summaries, the realized ratio from the canonical
prompt-length producer, and AUC-TV where replay profiles exist. Nothing is
recomputed that an artifact already holds; the only new statistics are the
registered paired CIs.

    python scripts/report_budget_curve.py --runs-root <runs> --out t1_budget_curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.paired import paired_flip_contrast  # noqa: E402
from kvcache.stats import cluster_bootstrap_mean  # noqa: E402

FC_RUN = "math500-fc-powered-merged"
GRIDS = {
    256: "grid-diss256-tok",
    512: "grid-diss512",
    1024: "grid-diss1024",
    2048: "grid-diss2048-tok",
}
SUMMARIES = {
    256: "results/diag/amendment-k/m4-powered-diss256-summary.json",
    512: "results/diag/amendment-k/m4-512-summary.json",
    1024: "results/diag/amendment-k/m4-1024-summary.json",
    2048: "results/diag/amendment-k/m4-2048-summary.json",
}
RATIOS = "results/diag/meeting3/prompt_lengths_budget_curve.json"
AUC = "results/diag/amendment-p/auc_resid_tv.json"
PRESSES = ("random", "streamingllm")


def _rows(run_dir: Path):
    for path in sorted((run_dir / "math500").rglob("*.jsonl")):
        if "token_ids" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def _correct_by_condition(run_dir: Path, press: str | None, budget) -> dict:
    out = {}
    for r in _rows(run_dir):
        if press is not None and (
            r.get("press") != press or str(r.get("budget")) != str(budget)
        ):
            continue
        key = (r["pid"], int(r["decoding"]["seed"]))
        if key in out:
            raise SystemExit(f"duplicate condition {key} in {run_dir.name}")
        out[key] = bool(r["correct"])
    return out


def _summary_auc(runs_root: Path, specs: list[str], n_resamples: int, seed: int):
    """Per-(press, budget) AUC-TV from R.12 summary profiles, clustered by problem."""
    out = {}
    for spec in specs:
        budget, _, run = spec.partition("=")
        per: dict[str, dict[str, list[float]]] = {}
        for path in sorted((runs_root / run / "profiles").rglob("*.json")):
            if path.suffixes[-2:] != [".json"]:
                continue
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("schema_version") != "tv-profile-summary-v1":
                raise SystemExit(f"not an R.12 summary profile: {path}")
            per.setdefault(d["condition"]["press"], {}).setdefault(
                d["reference_key"][1], []
            ).append(float(d["mean_tv"]))
        for press, by_pid in per.items():
            # seeds collapse into their problem's mean first (the Amendment P
            # convention), so a problem with more seeds does not weigh more
            means = {pid: [sum(v) / len(v)] for pid, v in by_pid.items()}
            out[(press, int(budget))] = cluster_bootstrap_mean(
                means, n_resamples=n_resamples, seed=seed
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tv-runs",
        nargs="*",
        default=[],
        help="budget=run pairs of R.12 summary-profile TV cells (e.g. 512=t1-tv512)",
    )
    a = ap.parse_args()
    tv_new = _summary_auc(a.runs_root, a.tv_runs, a.n_resamples, a.seed)

    fc = _correct_by_condition(a.runs_root / FC_RUN, None, None)
    ratios = json.loads((PROJECT_ROOT / RATIOS).read_text(encoding="utf-8"))["realized"]
    auc = json.loads((PROJECT_ROOT / AUC).read_text(encoding="utf-8"))[
        "per_cell_summary"
    ]
    cells = []
    for budget, grid in GRIDS.items():
        summ = {
            c["press"]: c
            for c in json.loads(
                (PROJECT_ROOT / SUMMARIES[budget]).read_text(encoding="utf-8")
            )["cells"]
        }
        for press in PRESSES:
            comp = _correct_by_condition(a.runs_root / grid, press, budget)
            pc = paired_flip_contrast(fc, comp, n_resamples=a.n_resamples, seed=a.seed)
            s = summ[press]
            tv = auc.get(f"{press}-{budget}")
            new = tv_new.get((press, budget))
            rr = ratios[f"{press}-{budget}"]
            cells.append(
                {
                    "press": press,
                    "budget": budget,
                    "realized_ratio": rr["ratio"],
                    "realized_ratio_pct": rr["ratio_pct"],
                    "mean_trace_tokens": rr["mean_trace_tokens"],
                    "fc_accuracy_matched": s["full_accuracy"],
                    "free_accuracy": s["compressed_accuracy"],
                    "free_accuracy_ci95": s["compressed_accuracy_ci95"],
                    "readout_accuracy": s["readout_accuracy"],
                    "readout_accuracy_ci95": s["readout_accuracy_ci95"],
                    "readout_change": s["readout_change"],
                    "residual": s["unexplained_residual"],
                    "residual_ci95": s["unexplained_residual_ci95"],
                    "truncation_rate": s["truncation_rate"],
                    "paired": pc,
                    "auc_tv": (
                        new["mean"] if new else (None if tv is None else tv["auc_mean"])
                    ),
                    "auc_tv_ci95": new["ci95"] if new else None,
                    "auc_tv_n_problems": (
                        new["n_clusters"]
                        if new
                        else (None if tv is None else tv["n_problems"])
                    ),
                    "auc_tv_source": (
                        "R.12 summary profiles" if new else (AUC if tv else None)
                    ),
                }
            )
    out = {
        "amendment": "R.1",
        "fc_run": FC_RUN,
        "grids": {str(k): v for k, v in GRIDS.items()},
        "residual_name": "accuracy change unexplained by on-reference readout",
        "ratio_convention": (
            "budget / (mean prompt + mean generated tokens), ratio of means"
        ),
        "tv_note": (
            "AUC-TV = per-trace mean TV, seeds averaged within problem; 256/2048 "
            "from the Amendment P artifact, 512/1024 from R.12 summary profiles "
            "when --tv-runs is given"
        ),
        "cells": cells,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")
    for c in cells:
        t = c["paired"]["total_change"]
        print(
            f"{c['press']:>13} {c['budget']:>5} ratio={c['realized_ratio_pct']:6.2f}% "
            f"total={t['mean']:+.3f} {[round(x, 3) for x in t['ci95']]} "
            f"lost={c['paired']['harmful_flip_mass']['mean']:.3f} "
            f"gained={c['paired']['beneficial_flip_mass']['mean']:.3f} "
            f"n={c['paired']['n_matched']}"
        )
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Amendment R.3 / R.14 D_acc read: free-running accuracy on the (budget, onset) grid.

Readiness (R.14): the rows must cover the EXACT expected grid -- problem x seed
x policy x budget x onset from the frozen design (``--design``) -- with no
duplicated or unexpected key, every arm's rows must share one R.13 identity,
and every condition needs its full-cache comparator. Partial grids are read
per cell for diagnostics, but the overall status stays PARTIAL/INVALID and
``--final`` exits nonzero.

Per (policy, budget, onset) cell: damage ``Acc_FC - Acc_comp`` paired by
(problem, seed) with the equal-problem estimator (seeds within problem, then
problems; the same estimator on every problem-cluster bootstrap replicate),
harmful and beneficial flip masses likewise, both accuracies, truncation rate
and achieved exposure. A per-condition TSV and its sha256 are written beside
the JSON so every number can be recomputed.

Determinism control (Amendment R.8.3): the gated t0 = 0 arm against the legacy
grid rows for the same (pid, seed, press, budget), token for token.

    python scripts/report_dacc.py --runs-root <runs> \\
        --arms 0=dacc-t0 512=dacc-t512 2048=dacc-t2048 \\
        --legacy 256=grid-diss256-tok 2048=grid-diss2048-tok \\
        --design configs/experiments/grid_math500_dacc_t0_r3.yaml \\
        --out dacc.json --final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.coverage import (
    design,
    COMPLETE,
    check_coverage,
    equal_problem_bootstrap,
    expected_keys,
    paired_equal_problem,
)
from kvcache.metrics.paired import trace_identity

FC_RUN = "math500-fc-powered-merged"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--arms", nargs="+", required=True, help="onset=run_name")
    p.add_argument("--legacy", nargs="*", default=[], help="budget=run_name")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--design", type=Path, default=None, help="one D_acc arm config")
    p.add_argument("--final", action="store_true")
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _pairs(specs: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for spec in specs:
        k, _, v = spec.partition("=")
        out[int(k)] = v
    return out


def _jsonl(run_dir: Path, token_ids: bool):
    for path in sorted((run_dir).rglob("*.jsonl")):
        if ("token_ids" in path.parts) != token_ids:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    return list(_jsonl(run_dir, token_ids=False))


def _tokens(run_dir: Path) -> dict[tuple, tuple[int, ...]]:
    return {
        (r["pid"], int(r["seed"]), r["press"], str(r["budget"])): tuple(
            r["trace_token_ids"]
        )
        for r in _jsonl(run_dir, token_ids=True)
    }


def _summ(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    return {
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
    }


def _design(config: Path) -> tuple[list[str], list[int], list[str], list[int]]:
    d = design(config)
    return d["pids"], d["seeds"], d["presses"], d["budgets"]


def _write_table(path: Path, table: list[dict[str, Any]]) -> str:
    cols = [
        "onset",
        "press",
        "budget",
        "pid",
        "seed",
        "fc_correct",
        "comp_correct",
        "truncated",
        "exposed",
        "achieved_exposure",
    ]
    lines = ["\t".join(cols)] + [
        "\t".join(str(e[c]) for c in cols)
        for e in sorted(table, key=lambda e: tuple(str(e[c]) for c in cols[:5]))
    ]
    data = ("\n".join(lines) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_dacc(
    arm_rows: dict[int, list[dict[str, Any]]],
    fc: dict[tuple[str, int], bool],
    *,
    pids: list[str],
    seeds: list[int],
    presses: list[str],
    budgets: list[int],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Exact coverage per onset arm, then the equal-problem paired damage."""
    rows_all = []
    for t0, rows in arm_rows.items():
        for r in rows:
            if (r.get("onset") or {}).get("compression_start_step") != t0:
                raise SystemExit(
                    f"STOP: row for t0={t0} ({r['pid']}) lacks a matching onset "
                    "block: not a gated D_acc row"
                )
            rows_all.append((t0, r))
    expected = expected_keys(
        pids, seeds, presses, [str(b) for b in budgets], list(arm_rows)
    )
    cov = check_coverage(
        [r for _, r in rows_all],
        lambda r: (
            r["pid"],
            int(r["decoding"]["seed"]),
            r["press"],
            str(r["budget"]),
            int(r["onset"]["compression_start_step"]),
        ),
        expected,
    )
    ident_ok = True
    for rows in arm_rows.values():
        ids = {
            json.dumps(
                {
                    k: v
                    for k, v in (r.get("provenance") or {}).items()
                    if k != "inputs_sha256"
                },
                sort_keys=True,
            )
            for r in rows
        }
        if len(ids) > 1 or "{}" in ids:
            ident_ok = False
    missing_fc = sorted({(p, s) for p in pids for s in seeds} - set(fc))
    cells: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    for t0, rows in sorted(arm_rows.items()):
        for press in presses:
            for budget in budgets:
                key = f"{press}-{budget}-t{t0}"
                rs = [
                    r
                    for r in rows
                    if r["press"] == press and str(r["budget"]) == str(budget)
                ]
                comp = {(r["pid"], int(r["decoding"]["seed"])): r for r in rs}
                need = {(p, s) for p in pids for s in seeds}
                if set(comp) != need or len(rs) != len(need) or not need <= set(fc):
                    cells[key] = {"status": "PARTIAL", "n_rows": len(rs)}
                    continue
                fcv = {k: float(fc[k]) for k in need}
                cv = {k: float(bool(comp[k]["correct"])) for k in need}
                lost = {k: float(fc[k] and not comp[k]["correct"]) for k in need}
                gained = {k: float(comp[k]["correct"] and not fc[k]) for k in need}
                zero = dict.fromkeys(need, 0.0)
                for k in need:
                    r = comp[k]
                    table.append(
                        {
                            "onset": t0,
                            "press": press,
                            "budget": budget,
                            "pid": k[0],
                            "seed": k[1],
                            "fc_correct": int(fc[k]),
                            "comp_correct": int(bool(r["correct"])),
                            "truncated": int(bool(r["truncated"])),
                            "exposed": int(bool(r["onset"]["exposed"])),
                            "achieved_exposure": r["onset"]["achieved_exposure"],
                        }
                    )
                ev = [
                    r["onset"]["eviction_onset_step"]
                    for r in rs
                    if r["onset"]["eviction_onset_step"] is not None
                ]
                cells[key] = {
                    "status": "READ",
                    "press": press,
                    "budget": int(budget),
                    "onset": t0,
                    "n_rows": len(rs),
                    "fc_accuracy": equal_problem_bootstrap(
                        fcv, n_resamples=n_resamples, seed=seed
                    )["mean"],
                    "comp_accuracy": equal_problem_bootstrap(
                        cv, n_resamples=n_resamples, seed=seed
                    )["mean"],
                    "damage_fc_minus_comp": paired_equal_problem(
                        fcv, cv, n_resamples=n_resamples, seed=seed
                    ),
                    "harmful_flip_mass": paired_equal_problem(
                        lost, zero, n_resamples=n_resamples, seed=seed
                    ),
                    "beneficial_flip_mass": paired_equal_problem(
                        gained, zero, n_resamples=n_resamples, seed=seed
                    ),
                    "truncation_rate": statistics.fmean(
                        bool(r["truncated"]) for r in rs
                    ),
                    "exposed_fraction": statistics.fmean(
                        bool(r["onset"]["exposed"]) for r in rs
                    ),
                    "n_unexposed": sum(not r["onset"]["exposed"] for r in rs),
                    "achieved_exposure": _summ(
                        [float(r["onset"]["achieved_exposure"]) for r in rs]
                    ),
                    "eviction_onset_step": _summ([float(x) for x in ev]),
                    "mean_n_tokens": statistics.fmean(r["n_tokens"] for r in rs),
                }
    final_ok = cov["status"] == COMPLETE and ident_ok and not missing_fc
    return {
        "status": (
            "FINAL"
            if final_ok
            else (
                "INVALID" if cov["status"] == "INVALID" or not ident_ok else "PARTIAL"
            )
        ),
        "coverage": cov,
        "identity_consistent_within_arms": ident_ok,
        "fc_comparators_missing": [list(k) for k in missing_fc],
        "n_cells_read": sum(c["status"] == "READ" for c in cells.values()),
        "cells": cells,
        "table": table,
    }


def main() -> int:
    a = _parser().parse_args()
    arms, legacy = _pairs(a.arms), _pairs(a.legacy)
    if a.dry_run:
        present = {t0: (a.runs_root / run).is_dir() for t0, run in sorted(arms.items())}
        print(
            json.dumps(
                {
                    "command": "dacc-read",
                    "arms": arms,
                    "legacy": legacy,
                    "arm_dirs_present": present,
                    "gpu_required": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    fc = {
        (r["pid"], int(r["decoding"]["seed"])): bool(r["correct"])
        for r in _rows(a.runs_root / FC_RUN / "math500")
    }
    arm_rows = {}
    for t0, run in sorted(arms.items()):
        arm_dir = a.runs_root / run / "math500"
        arm_rows[t0] = _rows(arm_dir) if arm_dir.is_dir() else []
    if a.design is None:
        raise SystemExit("--design is required: the expected grid comes from it")
    pids, seeds, presses, budgets = _design(a.design)
    res = read_dacc(
        arm_rows,
        fc,
        pids=pids,
        seeds=seeds,
        presses=presses,
        budgets=budgets,
        n_resamples=a.n_resamples,
        seed=a.seed,
    )
    cells = res["cells"]

    control: dict[str, Any] = {"status": "NOT RUN"}
    if 0 in arms and legacy and (a.runs_root / arms[0]).is_dir():
        gated = _tokens(a.runs_root / arms[0])
        per_budget = {}
        for budget, run in sorted(legacy.items()):
            leg = {
                k: v
                for k, v in _tokens(a.runs_root / run).items()
                if k[3] == str(budget)
            }
            mine = {k: v for k, v in gated.items() if k[3] == str(budget)}
            per_budget[str(budget)] = trace_identity(mine, leg)
        control = {
            "status": "RUN",
            "note": "R.8.3 determinism control: gated t0=0 vs legacy rows",
            "per_budget": per_budget,
        }

    out = {
        "amendment": "R.3 (D_acc), R.8, R.14",
        "fc_run": FC_RUN,
        "arms": {str(k): v for k, v in arms.items()},
        "estimand": "damage = Acc_FC - Acc_comp, paired by (problem, seed), "
        "equal-problem mean, problem-cluster bootstrap 95% CI",
        "b2048_exposure_note": (
            "R.8.1: at B=2048 t0=0 and t0=512 reach nearly the same achieved "
            "exposure; exposure readings there are non-identifying"
        ),
        **{k: v for k, v in res.items() if k != "table"},
        "determinism_control": control,
    }
    if res["status"] == "FINAL" and control.get("status") != "RUN":
        out["status"] = "PARTIAL"
        out["why_not_final"] = "determinism control (R.8.3) not run"
    tsv = a.out.with_suffix(".conditions.tsv")
    out["sufficient_statistics_tsv"] = tsv.name
    out["sufficient_statistics_sha256"] = _write_table(tsv, res["table"])
    out["reproduce"] = (
        "python scripts/report_dacc.py --runs-root <runs> --arms "
        + " ".join(f"{k}={v}" for k, v in sorted(arms.items()))
        + f" --design {a.design} --out {a.out} --final"
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"status {out['status']}; coverage {out['coverage']['status']}")
    for key, c in sorted(cells.items()):
        if c["status"] != "READ":
            print(f"{key:>24} {c['status']} {c['n_rows']} rows")
            continue
        d = c["damage_fc_minus_comp"]
        print(
            f"{key:>24} fc={c['fc_accuracy']:.3f} comp={c['comp_accuracy']:.3f} "
            f"damage={d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] "
            f"lost={c['harmful_flip_mass']['mean']:.3f} "
            f"gained={c['beneficial_flip_mass']['mean']:.3f} "
            f"trunc={c['truncation_rate']:.2f} exposed={c['exposed_fraction']:.2f}"
        )
    if control["status"] == "RUN":
        for b, t in control["per_budget"].items():
            print(f"control B={b}: {t['n_identical']}/{t['n_shared']} identical")
    print(f"wrote {a.out}")
    if a.final and out["status"] != "FINAL":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

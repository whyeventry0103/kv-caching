#!/usr/bin/env python3
"""Amendment R.3 D_acc read: free-running accuracy on the (budget, onset) grid.

For every (policy, budget, onset) cell: the paired, problem-clustered contrast
against the 100 matching full-cache rows of the FC bank
(``kvcache.metrics.paired.paired_flip_contrast``: total change, harmful and
beneficial flip mass), both accuracies, truncation rate, and achieved exposure
from each row's ``onset`` block. The registered estimand is damage
``Acc_FC - Acc_comp`` (= -total change), reported with its CI.

Readiness is decided here, not by the reader of the output: a cell with fewer
than its expected 100 rows is reported PARTIAL and is not read.

Determinism control (Amendment R.8.3): the gated t0 = 0 arm against the legacy
grid rows for the same (pid, seed, press, budget), token for token.

    python scripts/report_dacc.py --runs-root <runs> \\
        --arms 0=dacc-t0 512=dacc-t512 2048=dacc-t2048 \\
        --legacy 256=grid-diss256-tok 2048=grid-diss2048-tok --out dacc.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.paired import paired_flip_contrast, trace_identity

FC_RUN = "math500-fc-powered-merged"
EXPECTED_PER_CELL = 100  # 25 problems x 4 seeds


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--arms", nargs="+", required=True, help="onset=run_name")
    p.add_argument("--legacy", nargs="*", default=[], help="budget=run_name")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--expected-per-cell", type=int, default=EXPECTED_PER_CELL)
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
    cells: dict[str, Any] = {}
    for t0, run in sorted(arms.items()):
        arm_dir = a.runs_root / run / "math500"
        rows = _rows(arm_dir) if arm_dir.is_dir() else []
        by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if r.get("onset", {}).get("compression_start_step") != t0:
                raise SystemExit(
                    f"STOP: row in {run} lacks an onset block for t0={t0} "
                    f"({r['pid']}): not a gated D_acc row"
                )
            by_cell[(r["press"], str(r["budget"]))].append(r)
        for press in ("random", "streamingllm"):
            for budget in ("256", "2048"):
                key = f"{press}-{budget}-t{t0}"
                rs = by_cell.get((press, budget), [])
                conds = {(r["pid"], int(r["decoding"]["seed"])) for r in rs}
                if len(conds) != len(rs):
                    raise SystemExit(f"STOP: duplicate conditions in {key}")
                if len(rs) < a.expected_per_cell:
                    cells[key] = {"status": "PARTIAL", "n_rows": len(rs)}
                    continue
                comp = {
                    (r["pid"], int(r["decoding"]["seed"])): bool(r["correct"])
                    for r in rs
                }
                pc = paired_flip_contrast(
                    fc, comp, n_resamples=a.n_resamples, seed=a.seed
                )
                if pc["n_only_comp"]:
                    raise SystemExit(
                        f"STOP: {key} has {pc['n_only_comp']} conditions with no "
                        "FC comparator (R.0.3 says all 100 exist)"
                    )
                tot = pc["total_change"]
                exp = [r["onset"]["achieved_exposure"] for r in rs]
                ev = [
                    r["onset"]["eviction_onset_step"]
                    for r in rs
                    if r["onset"]["eviction_onset_step"] is not None
                ]
                matched = sorted(comp)
                cells[key] = {
                    "status": "READ",
                    "press": press,
                    "budget": int(budget),
                    "onset": t0,
                    "n_rows": len(rs),
                    "fc_accuracy": statistics.fmean(fc[k] for k in matched),
                    "comp_accuracy": statistics.fmean(comp[k] for k in matched),
                    "damage_fc_minus_comp": {
                        "mean": -tot["mean"],
                        "ci95": sorted(-x for x in tot["ci95"]),
                    },
                    "paired": pc,
                    "truncation_rate": statistics.fmean(
                        bool(r["truncated"]) for r in rs
                    ),
                    "exposed_fraction": statistics.fmean(
                        bool(r["onset"]["exposed"]) for r in rs
                    ),
                    "n_unexposed": sum(not r["onset"]["exposed"] for r in rs),
                    "achieved_exposure": _summ([float(x) for x in exp]),
                    "eviction_onset_step": _summ([float(x) for x in ev]),
                    "mean_n_tokens": statistics.fmean(r["n_tokens"] for r in rs),
                }

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

    read = [c for c in cells.values() if c["status"] == "READ"]
    out = {
        "amendment": "R.3 (D_acc), R.8",
        "fc_run": FC_RUN,
        "arms": {str(k): v for k, v in arms.items()},
        "estimand": "damage = Acc_FC - Acc_comp, paired, problem-clustered 95% CI",
        "b2048_exposure_note": (
            "R.8.1: at B=2048 t0=0 and t0=512 reach nearly the same achieved "
            "exposure; exposure readings there are non-identifying"
        ),
        "n_cells_read": len(read),
        "n_cells_partial": len(cells) - len(read),
        "cells": cells,
        "determinism_control": control,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")
    for key, c in sorted(cells.items()):
        if c["status"] != "READ":
            print(f"{key:>24} PARTIAL {c['n_rows']}/{a.expected_per_cell}")
            continue
        d = c["damage_fc_minus_comp"]
        print(
            f"{key:>24} fc={c['fc_accuracy']:.3f} comp={c['comp_accuracy']:.3f} "
            f"damage={d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] "
            f"lost={c['paired']['harmful_flip_mass']['mean']:.3f} "
            f"gained={c['paired']['beneficial_flip_mass']['mean']:.3f} "
            f"trunc={c['truncation_rate']:.2f} exposed={c['exposed_fraction']:.2f}"
        )
    if control["status"] == "RUN":
        for b, t in control["per_budget"].items():
            print(f"control B={b}: {t['n_identical']}/{t['n_shared']} identical")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

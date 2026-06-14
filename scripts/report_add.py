#!/usr/bin/env python3
"""M5 ADD report: answer-distribution divergence with permutation null.

ADD is an answer-level distribution divergence (TV between coarsened, canonicalized
full-cache and compressed answer distributions) — a LOWER BOUND on behavioral change
that only sees changes reaching the answer. Never call it "behavioral change."

Per problem, per cell: observed ADD, a per-problem permutation-test p-value (pool the
n+n samples, relabel), the termination-mass vs within-answered split, entropy shift
(Miller-Madow), and signed flip-mass (gained/lost correct mass via the stored
`correct`). Aggregated across problems with Benjamini-Hochberg.

This is the same pipeline that runs at scale (MATH-500 250xN). On the pilot (n=4 per
cell) it is directional only: per-problem permutation power at n=4 is near zero (the
powered unit is the benchmark via aggregation).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.add import (  # noqa: E402
    add_decomposition,
    aggregate_flip_mass,
    entropy_miller_madow,
    excess_add_permutation,
)
from kvcache.stats import benjamini_hochberg  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_rows(root: Path, subdir: str = "math500") -> list[dict[str, Any]]:
    return [
        row
        for path in sorted((root / subdir).rglob("*.jsonl"))
        if "token_ids" not in path.parts
        for row in _jsonl(path)
    ]


def _by_pid(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row["pid"]].append(row)
    return out


def build_add_report(
    *, fullcache_root: Path, compression_root: Path, perm_seed: int = 0
) -> dict[str, Any]:
    full_rows = _load_rows(fullcache_root)
    comp_rows = _load_rows(compression_root)
    if not full_rows or not comp_rows:
        raise ValueError("ADD report requires full-cache and compressed rows")

    full_by_pid = _by_pid(full_rows)
    fc_canon = {pid: [r["answer_canon"] for r in rs] for pid, rs in full_by_pid.items()}
    fc_correct = {
        pid: [bool(r["correct"]) for r in rs] for pid, rs in full_by_pid.items()
    }

    cells_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comp_rows:
        cells_rows[(row["press"], str(row["budget"]))].append(row)

    cells = []
    for condition in sorted(cells_rows):
        comp_by_pid = _by_pid(cells_rows[condition])
        per_problem = []
        flips = []
        for pid in sorted(comp_by_pid):
            if pid not in fc_canon:
                continue
            fc = fc_canon[pid]
            cc = [r["answer_canon"] for r in comp_by_pid[pid]]
            observed, p_value = excess_add_permutation(fc, cc, seed=perm_seed)
            decomp = add_decomposition(fc, cc)
            flip = statistics.fmean(
                bool(r["correct"]) for r in comp_by_pid[pid]
            ) - statistics.fmean(fc_correct[pid])
            flips.append(flip)
            per_problem.append(
                {
                    "pid": pid,
                    "add": observed,
                    "perm_p": p_value,
                    "termination_shift": decomp.termination_shift,
                    "within_answered_tv": decomp.within_answered_tv,
                    "entropy_shift": entropy_miller_madow(cc)
                    - entropy_miller_madow(fc),
                    "flip_mass": flip,
                }
            )
        bh = benjamini_hochberg([row["perm_p"] for row in per_problem])
        for row, adj in zip(per_problem, bh, strict=True):
            row["perm_p_bh"] = adj
        within = [
            row["within_answered_tv"]
            for row in per_problem
            if row["within_answered_tv"] is not None
        ]
        cells.append(
            {
                "press": condition[0],
                "budget": condition[1],
                "n_problems": len(per_problem),
                "mean_add": statistics.fmean(row["add"] for row in per_problem),
                "mean_termination_shift": statistics.fmean(
                    row["termination_shift"] for row in per_problem
                ),
                "mean_within_answered_tv": (
                    statistics.fmean(within) if within else None
                ),
                "mean_entropy_shift": statistics.fmean(
                    row["entropy_shift"] for row in per_problem
                ),
                # Mean per-problem flip mass: net == compressed_accuracy -
                # fc_accuracy (total_change). Reported as means so it is on the
                # accuracy scale and comparable across cells.
                "signed_flip_mass": {
                    key: value / len(flips)
                    for key, value in aggregate_flip_mass(flips).items()
                },
                "n_significant_bh_0p05": sum(
                    row["perm_p_bh"] < 0.05 for row in per_problem
                ),
                "per_problem": per_problem,
            }
        )

    return {
        "cells": cells,
        "runs": {
            "fullcache": str(fullcache_root),
            "compression": str(compression_root),
        },
        "note": (
            "ADD is answer-level distribution divergence (a lower bound on behavioral "
            "change), reported with signed flip-mass. Per-problem permutation power is "
            "low at small n; the powered unit is the benchmark via aggregation."
        ),
    }


def _write_table(path: Path, cells: list[dict[str, Any]]) -> None:
    lines = [
        "# M5 ADD pilot",
        "",
        "ADD = answer-level distribution divergence (lower bound on behavioral "
        "change). Reported with signed flip-mass; per-problem significance is "
        "underpowered at pilot n.",
        "",
        "| Press | Budget | Mean ADD | Term shift | Within-ans TV | Entropy Δ | "
        "Net flip | Lost | #sig(BH) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        within = c["mean_within_answered_tv"]
        lines.append(
            f"| {c['press']} | {c['budget']} | {c['mean_add']:.3f} | "
            f"{c['mean_termination_shift']:+.3f} | "
            f"{'n/a' if within is None else f'{within:.3f}'} | "
            f"{c['mean_entropy_shift']:+.3f} | "
            f"{c['signed_flip_mass']['net']:+.2f} | "
            f"{c['signed_flip_mass']['lost']:.2f} | "
            f"{c['n_significant_bh_0p05']}/{c['n_problems']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullcache-root", required=True, type=Path)
    parser.add_argument("--compression-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build_add_report(
        fullcache_root=args.fullcache_root,
        compression_root=args.compression_root,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_table(args.out / "table.md", report["cells"])
    print(
        json.dumps(
            {
                "cells": [
                    {
                        k: c[k]
                        for k in (
                            "press",
                            "budget",
                            "mean_add",
                            "mean_termination_shift",
                            "mean_within_answered_tv",
                            "signed_flip_mass",
                        )
                    }
                    for c in report["cells"]
                ]
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sample-size and cost planning for the held paper campaigns (plan §9.2, §10.2).

Thin orchestration over ``kvcache.metrics.sample_planning``. Two outputs under
``results/diag/paper_plan/``:

``sample-size`` -- from the committed S.1 sufficient statistics (variance only):
                   problem-level SD and simulated 95% half-width / share of
                   studies excluding zero for N in {100..500} x seeds {2, 4} x
                   effect {0, 0.03, 0.05}, for the protected forward residual
                   and the protected free-run gap.
``cost``        -- GPU-hours and bytes per proposed manifest from
                   ``configs/paper/cost_inputs.yaml`` (measured rates plus
                   stated assumptions), with a 1.5x storage reserve.

CPU-only; reads only committed files.

    python scripts/plan_paper_sample_size.py sample-size
    python scripts/plan_paper_sample_size.py cost
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

from kvcache.metrics import sample_planning as sp
from kvcache.provenance import write_json_atomic

OUT = PROJECT_ROOT / "results" / "diag" / "paper_plan"
S1_TSV = (
    PROJECT_ROOT / "results" / "diag" / "amendment-s" / "protection_b256.conditions.tsv"
)
COSTS = PROJECT_ROOT / "configs" / "paper" / "cost_inputs.yaml"
N_GRID = (100, 200, 300, 400, 500)
SEEDS_GRID = (2, 4)
EFFECTS = (0.0, 0.03, 0.05)
TARGET_HALF_WIDTH = 0.025


def _b(v: str) -> float | None:
    return {"True": 1.0, "False": 0.0, "1": 1.0, "0": 0.0, "": None}[v]


def sample_size(a: argparse.Namespace) -> dict[str, Any]:
    with S1_TSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out: dict[str, Any] = {
        "read": "sample-size planning (variance from S.1 b256 discovery data)",
        "source": str(S1_TSV.relative_to(PROJECT_ROOT)),
        "status": "PLANNING APPROXIMATION: spread borrowed from Qwen-7B/MATH-500; "
        "a new model or task can differ",
        "target_half_width": TARGET_HALF_WIDTH,
        "quantities": {},
    }
    for press in sorted({r["press"] for r in rows}):
        sub = [r for r in rows if r["press"] == press]
        res, gap = {}, {}
        for r in sub:
            k = (r["pid"], int(r["seed"]))
            fp, rp, fc = _b(r["free_pp"]), _b(r["readout_pp"]), _b(r["fc"])
            gap[k] = fp - fc
            if rp is not None:
                res[k] = fp - rp
        for qname, vals in (
            ("protected_forward_residual", res),
            ("protected_free_gap", gap),
        ):
            pp = sp.per_problem_values(vals)
            sd = sp.problem_sd(pp)
            sims = [
                sp.simulate(pp, n_problems=n, n_seeds=s, effect=e, reps=a.reps, seed=0)
                for n in N_GRID
                for s in SEEDS_GRID
                for e in EFFECTS
            ]
            out["quantities"][f"{press}:{qname}"] = {
                "problem_sd": sd,
                "normal_n_for_half_width_0.025": sp.normal_n(sd, TARGET_HALF_WIDTH),
                "simulations": sims,
            }
    return out


def cost(a: argparse.Namespace) -> dict[str, Any]:
    import yaml

    c = yaml.safe_load(COSTS.read_text(encoding="utf-8"))
    m, asum, b = c["measured"], c["assumed"], c["storage_bytes_per_row"]
    lo, hi = (
        asum["readout_over_free_ratio"]["low"],
        asum["readout_over_free_ratio"]["high"],
    )
    res = c["planning_reserve_multiplier"]

    def block(items: list[tuple[str, int, float, str]]) -> dict[str, Any]:
        """items: (label, conditions, seconds/row, kind in free|readout|capture)."""
        rows, g_lo, g_hi, by = [], 0.0, 0.0, 0.0
        for label, n, sec, kind in items:
            if kind == "readout":
                h = (n * sec * lo / 3600, n * sec * hi / 3600)
                byt = n * b["readout"]
            elif kind == "capture":
                h = (n * sec / 3600,) * 2
                byt = n * b["capture"]
            else:
                h = (n * sec / 3600,) * 2
                byt = n * (b["free"] + b["token_ids"])
            rows.append(
                {
                    "cell": label,
                    "conditions": n,
                    "gpu_hours": [round(x, 1) for x in h],
                    "bytes": int(byt),
                }
            )
            g_lo, g_hi, by = g_lo + h[0], g_hi + h[1], by + byt
        cap = c["capacity"]["window_hours_per_night"] * c["capacity"]["gpus"]
        return {
            "cells": rows,
            "gpu_hours": [round(g_lo, 1), round(g_hi, 1)],
            "nights_at_full_capacity": [round(g_lo / cap, 1), round(g_hi / cap, 1)],
            "bytes": int(by),
            "gib_with_reserve": round(by * res / 2**30, 3),
        }

    f256, fh2o, f1024 = (
        m["free_q7_b256_pp"]["mean"],
        m["free_q7_b256_h2o"]["mean"],
        m["free_q7_b1024"]["mean"],
    )
    fcq, cap = m["fullcache_q7"]["mean"], m["capture_q7_b256"]["mean"]
    q3fc, q3free = m["fullcache_qwen3_4b"]["mean"], m["free_qwen3_4b_b256"]["mean"]
    g_lo, g_hi = (
        asum["gsm8k_over_math500_ratio"]["low"],
        asum["gsm8k_over_math500_ratio"]["high"],
    )

    def confirm(
        n: int, fc_s: float, free256: float, free1024: float, scale: float = 1.0
    ):
        s = 2
        return block(
            [
                ("full-cache bank", n * s, fc_s * scale, "free"),
                ("free b256 x2 policies", 2 * n * s, free256 * scale, "free"),
                ("free b1024 x2 policies", 2 * n * s, free1024 * scale, "free"),
                ("forward b256 x2", 2 * n * s, free256 * scale, "readout"),
                ("forward b1024 x2", 2 * n * s, free1024 * scale, "readout"),
                ("reverse b256 x2", 2 * n * s, fc_s * scale, "readout"),
            ]
        )

    out: dict[str, Any] = {
        "read": "cost estimate for the HELD paper campaigns",
        "status": "ESTIMATE; nothing here is authorized",
        "inputs": str(COSTS.relative_to(PROJECT_ROOT)),
        "assumptions": asum,
        "campaigns": {
            "WP5 protected_selector_q7_math_b256_v1": block(
                [
                    ("h2o_style_pp_v1 free", 400, fh2o, "free"),
                    ("h2o_style_pp_v1 forward", 400, fcq, "readout"),
                    ("h2o_style_pp_v1 reverse", 400, fh2o, "readout"),
                    ("uniform_fresh_pp_v1 free", 400, f256, "free"),
                    ("uniform_fresh_pp_v1 forward", 400, fcq, "readout"),
                    ("uniform_fresh_pp_v1 reverse", 400, f256, "readout"),
                    ("capture x3 policies", 300, cap, "capture"),
                ]
            ),
            "WP6 confirm_qwen3_4b_math500_v1 (N=100, 2 seeds)": confirm(
                100, q3fc, q3free, q3free * f1024 / f256
            ),
            "WP6 confirm_q7_gsm8k_v1 (N=100, 2 seeds), low": confirm(
                100, fcq, f256, f1024, g_lo
            ),
            "WP6 confirm_q7_gsm8k_v1 (N=100, 2 seeds), high": confirm(
                100, fcq, f256, f1024, g_hi
            ),
        },
        "note": "Qwen3 b1024 free cost scaled from Qwen-7B's b1024/b256 ratio "
        "(assumption). Readout cost uses the source trace's free-run rate x "
        "[0.6, 1.0]. Storage counts each row once plus token-ID sidecars for "
        "free runs, times 1.5 reserve.",
    }
    return out


READS = {"sample-size": sample_size, "cost": cost}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("read", choices=sorted(READS))
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"command": "plan_paper_sample_size", "read": a.read}))
        return 0
    payload = {
        "reproduce": "python scripts/plan_paper_sample_size.py "
        + " ".join(sys.argv[1:]),
        **READS[a.read](a),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = "sample_size.json" if a.read == "sample-size" else "cost_estimate.json"
    write_json_atomic(OUT / name, payload)
    print(f"wrote {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

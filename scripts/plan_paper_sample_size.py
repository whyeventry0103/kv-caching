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
REVERSE_TSV = (
    PROJECT_ROOT / "results" / "diag" / "paper_audit" / ("reverse_b256.conditions.tsv")
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
    # The forward/reverse interaction on the common support (review of
    # fbf869f, R3): now central to the proposed contribution, so planned too.
    with REVERSE_TSV.open(encoding="utf-8") as fh:
        rv = list(csv.DictReader(fh, delimiter="\t"))
    for press in ("random_pp", "streamingllm_pp"):
        vals = {
            (r["pid"], int(r["seed"])): float(r["cc"])
            - float(r["fc"])
            - float(r["cf"])
            + float(r["ff"])
            for r in rv
            if r["press"] == press and r["fc"] != "" and r["cf"] != ""
        }
        pp = sp.per_problem_values(vals)
        sd = sp.problem_sd(pp)
        out["quantities"][f"{press}:protected_interaction"] = {
            "problem_sd": sd,
            "normal_n_for_half_width_0.025": sp.normal_n(sd, TARGET_HALF_WIDTH),
            "source": str(REVERSE_TSV.relative_to(PROJECT_ROOT)),
            "simulations": [
                sp.simulate(pp, n_problems=n, n_seeds=s_, effect=e, reps=a.reps, seed=0)
                for n in N_GRID
                for s_ in SEEDS_GRID
                for e in EFFECTS
            ],
        }
    out["endpoint_summary"] = {
        q: {
            "normal_n_for_half_width_0.025": v["normal_n_for_half_width_0.025"],
            "at_300x4": {
                f"effect_{x['effect']}": {
                    "half_width": round(x["mean_half_width"], 4),
                    "share_excluding_zero": x["share_excluding_zero"],
                }
                for x in v["simulations"]
                if x["n_problems"] == 300 and x["n_seeds"] == 4
            },
        }
        for q, v in out["quantities"].items()
    }
    out["reading"] = (
        "Precision is ENDPOINT-SPECIFIC: about 300 problems x 4 seeds reaches a "
        "~0.025 half-width for the protected forward residual only; the free-run "
        "gap needs far more problems than MATH-500 offers for that precision. "
        "Choose the primary endpoint first, then size the design for it."
    )
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
        n: int,
        seeds: int,
        fc_s: float,
        free256: float,
        free1024: float,
        scale: float = 1.0,
        new_family: tuple[float, float] = (1.0, 1.0),
    ) -> dict[str, Any]:
        """Two policies x two budgets; a readout is priced at its SOURCE trace's
        rate (forward reads the full-cache trace, reverse the compressed one).
        Fixed 2026-10-05 (review of fbf869f, R3): the rates were swapped."""
        k = n * seeds
        base = block(
            [
                ("full-cache bank", k, fc_s * scale, "free"),
                ("free b256 x2 policies", 2 * k, free256 * scale, "free"),
                ("free b1024 x2 policies", 2 * k, free1024 * scale, "free"),
                ("forward b256 x2 (reads FC traces)", 2 * k, fc_s * scale, "readout"),
                ("forward b1024 x2 (reads FC traces)", 2 * k, fc_s * scale, "readout"),
                (
                    "reverse b256 x2 (reads b256 traces)",
                    2 * k,
                    free256 * scale,
                    "readout",
                ),
            ]
        )
        return with_overhead(base, n_cells=6, new_family=new_family)

    ov = c["overhead"]

    def with_overhead(
        b: dict[str, Any], *, n_cells: int, new_family: tuple[float, float]
    ) -> dict[str, Any]:
        """Add job-launch setup per cell x shard and the new-family multiplier
        (Python selection + host transfers at each prune; unmeasured)."""
        setup = n_cells * ov["shards_per_cell"] * ov["setup_hours_per_launch"]
        lo, hi = b["gpu_hours"]
        b["gpu_hours_core"] = [lo, hi]
        b["setup_hours"] = round(setup, 1)
        b["new_family_multiplier"] = list(new_family)
        b["gpu_hours"] = [
            round(lo * new_family[0] + setup, 1),
            round(hi * new_family[1] + setup, 1),
        ]
        cap = c["capacity"]["window_hours_per_night"] * c["capacity"]["gpus"]
        b["nights_at_full_capacity"] = [
            round(b["gpu_hours"][0] / cap, 1),
            round(b["gpu_hours"][1] / cap, 1),
        ]
        return b

    nf = (ov["new_family_multiplier"]["low"], ov["new_family_multiplier"]["high"])
    wp5 = with_overhead(
        block(
            [
                ("h2o_style_pp_v1 free", 400, fh2o, "free"),
                ("h2o_style_pp_v1 forward (reads FC traces)", 400, fcq, "readout"),
                ("h2o_style_pp_v1 reverse (reads its traces)", 400, fh2o, "readout"),
                ("uniform_fresh_pp_v1 free", 400, f256, "free"),
                ("uniform_fresh_pp_v1 forward (reads FC traces)", 400, fcq, "readout"),
                (
                    "uniform_fresh_pp_v1 reverse (reads its traces)",
                    400,
                    f256,
                    "readout",
                ),
                ("capture x3 policies", 300, cap, "capture"),
            ]
        ),
        n_cells=7,
        new_family=nf,
    )
    q3_1024 = q3free * f1024 / f256
    campaigns: dict[str, Any] = {"WP5 protected_selector_q7_math_b256_v1": wp5}
    for n, sd in ((100, 2), (300, 2), (300, 4)):
        campaigns[f"WP6 qwen3_4b math500 N={n} x {sd} seeds"] = confirm(
            n, sd, q3fc, q3free, q3_1024, new_family=nf
        )
        for tag, sc in (("low", g_lo), ("high", g_hi)):
            campaigns[f"WP6 q7 gsm8k N={n} x {sd} seeds, {tag}"] = confirm(
                n, sd, fcq, f256, f1024, sc, new_family=nf
            )
    pilot = c["pilot"]
    pilot_rows = pilot["problems"] * pilot["seeds"] * len(pilot["families"])
    out: dict[str, Any] = {
        "read": "cost estimate for the HELD paper campaigns",
        "status": "ESTIMATE from measured historical rates plus stated "
        "assumptions; not a cap; nothing here is authorized",
        "inputs": str(COSTS.relative_to(PROJECT_ROOT)),
        "assumptions": {**asum, "overhead": ov},
        "conditions_scaling_note": "at fixed stages, conditions scale with "
        "problems x seeds: 300x4 is 6x the conditions of 100x2 (1200 vs 200 per "
        "arm), not 3x; no reuse is assumed",
        "engineering_pilot_proposal": {
            "purpose": "measure per-row wallclock of the NEW selector families "
            "(Python selection and host/device transfers at each prune) before "
            "any campaign is costed for approval",
            "design": pilot,
            "rows": pilot_rows * 3,
            "stages": ["free", "forward", "reverse"],
            "cap_gpu_hours": pilot["cap_gpu_hours"],
            "excluded_from_claims": True,
        },
        "wp4_smoke": c["wp4_smoke"],
        "campaigns": campaigns,
        "note": "Qwen3 b1024 free cost scaled from Qwen-7B's b1024/b256 ratio "
        "(assumption). Readout cost uses the SOURCE trace's free-run rate x "
        "[0.6, 1.0]. New-family multiplier applies to every row of a campaign "
        "that uses a new selector (conservative). Storage counts each row once "
        "plus token-ID sidecars for free runs, times 1.5 reserve.",
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

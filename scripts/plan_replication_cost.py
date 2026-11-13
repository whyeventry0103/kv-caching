#!/usr/bin/env python3
"""M5-B multi-dataset replication: measured rates and the priced stage plan
(prepared 2026-11-13). All arithmetic is ``kvcache.replication_cost``.

``--measure-rates`` (GPU host, CPU only, read-only): per-press s/token and
length inflation from the complete MATH-500 Qwen-7B B=256 cells against the
merged full-cache bank, and readout rates from the settled P3 pilot readouts
(one launch each, all 48 records; campaign ledger hours minus one model load).

``--plan``: price every stage of a design file from those rates and each
dataset's own timing read (full-cache lengths and s/token).

    python scripts/plan_replication_cost.py --dry-run
    KVCACHE_REMOTE=1 python scripts/plan_replication_cost.py --measure-rates \\
        --out results/diag/meeting5/replication/math500_b256_rates.json
    python scripts/plan_replication_cost.py --plan \\
        --rates results/diag/meeting5/replication/math500_b256_rates.json \\
        --timing gsm8k=results/diag/meeting5/gsm8k_timing/timing_read.json \\
        --out results/diag/meeting5/replication/cost_plan.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.meeting5_inventory import LOAD_HOURS_PER_LAUNCH, load_rows
from kvcache.replication_cost import press_rates, readout_rate

RUNS = Path("/home/sk10741/kvcache-storage/runs")
LEDGER = PROJECT_ROOT / "logs/campaign_ledger.json"
MODEL = "r1_distill_qwen_7b"
FC_RUN = "math500-fc-powered-merged"
GRID_RUNS = (
    "grid-diss256-tok",  # random, streamingllm
    "grid-diss256-h2o",  # h2o (eager)
    "grid-diss256-rkv",  # rkv
    "grid-pp256",  # random_pp, streamingllm_pp
    "grid-h2opp256-m5",  # h2o_style_pp_v1
    "grid-rkvpp256-m5",  # rkv_pp_v1
)
# P3 protected-selector pilot: one settled launch per stage, 48 records each
READOUT_CAL = ("protsel-pilot-readout", "readout-protsel-pilot", "fullcache")
REVERSE_CAL = ("protsel-pilot-reverse", "reverse-readout-protsel-pilot", "grid")
PILOT_GRID = "grid-protsel-pilot"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--measure-rates", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--rates", type=Path)
    p.add_argument("--timing", action="append", default=[])
    p.add_argument(
        "--design",
        type=Path,
        default=PROJECT_ROOT / "configs/meeting5/replication_design_m5b.json",
    )
    p.add_argument("--out", type=Path)
    return p


def _seed(r: dict[str, Any]) -> int:
    return int((r.get("decoding") or {}).get("seed", r.get("seed")))


def _settled(job: str) -> tuple[float, int]:
    launches = json.loads(LEDGER.read_text(encoding="utf-8"))["launches"]
    hits = [v for v in launches.values() if v["job"] == job and v["rc"] == 0]
    return sum(float(v["settled_hours"]) for v in hits), len(hits)


def measure_rates() -> dict[str, Any]:
    fc_rows = [r for r in load_rows(RUNS / FC_RUN / "math500") if r["model"] == MODEL]
    fc = {(str(r["pid"]), _seed(r)): int(r["n_tokens"]) for r in fc_rows}
    rows = []
    for run in GRID_RUNS:
        for r in load_rows(RUNS / run / "math500"):
            if r.get("model") == MODEL and str(r.get("budget")) == "256":
                rows.append({**r, "seed": _seed(r)})
    grid = {
        (str(r["pid"]), _seed(r), str(r["press"])): int(r["n_tokens"])
        for r in load_rows(RUNS / PILOT_GRID / "math500")
    }
    readouts = {}
    for name, (job, run, ref) in {
        "forward": READOUT_CAL,
        "reverse": REVERSE_CAL,
    }.items():
        recs = load_rows(RUNS / run)
        replayed = []
        for rec in recs:
            k = rec["reference_key"]
            pid, seed = str(k[1]), int(k[5])
            if ref == "fullcache":
                replayed.append(fc[(pid, seed)])
            else:
                # reverse: the reference is the compressed trace (key[3] = its
                # press); the replay condition itself is the full cache
                replayed.append(grid[(pid, seed, str(k[3]))])
        answer = [len(r["generated_token_ids"]) for r in recs]
        hours, n_launch = _settled(job)
        compute = hours - n_launch * LOAD_HOURS_PER_LAUNCH
        readouts[name] = {
            "job": job,
            "settled_gpu_hours": hours,
            "n_launches": n_launch,
            "n_records": len(recs),
            "mean_replayed_tokens": statistics.fmean(replayed),
            "mean_answer_tokens": statistics.fmean(answer),
            "s_per_token": readout_rate(
                compute, len(recs), statistics.fmean(replayed), statistics.fmean(answer)
            ),
        }
    return {
        "schema": "m5b-replication-rates-v1",
        "source": (
            "MATH-500, Qwen-7B (R1-Distill), B=256 complete cells; " "P3 pilot readouts"
        ),
        "fc_bank": {
            "run": FC_RUN,
            "n": len(fc),
            "mean_tokens": statistics.fmean(fc.values()),
        },
        "presses": press_rates(rows, fc),
        "readout": readouts,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.dry_run:
        print(json.dumps({"grid_runs": GRID_RUNS, "readout_cal": READOUT_CAL}))
        print("DRY RUN: nothing read")
        return 0
    if args.measure_rates:
        out = measure_rates()
    elif args.plan:
        from kvcache.replication_cost import plan_stages

        rates = json.loads(args.rates.read_text(encoding="utf-8"))
        timing = {}
        for spec in args.timing:
            bench, path = spec.split("=", 1)
            if path.startswith("capbound:"):
                # bench=capbound:<full-cache s/token>; upper bound only
                timing[bench] = {
                    "cap_bound": True,
                    "seconds_per_token": float(path.split(":", 1)[1]),
                }
                continue
            timing[bench] = json.loads(Path(path).read_text(encoding="utf-8"))
        design = json.loads(args.design.read_text(encoding="utf-8"))
        out = plan_stages(design, rates, timing)
    else:
        raise SystemExit("choose --measure-rates or --plan (or --dry-run)")
    text = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

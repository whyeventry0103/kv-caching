#!/usr/bin/env python3
"""Meeting 5 inventory, row-count ledger and cost model (prepared 2026-10-14).

Two read-only modes; all logic is ``kvcache.meeting5_inventory``.

``--inventory`` (on the GPU host, CPU only): classify every cell of the listed
runs against the frozen 100-problem cohort and seeds, record measured per-row
times, and compute the strict-fit cohort from the full-cache bank's prompt
token ids. Writes the JSON that is archived under results/diag/meeting5/.

``--ledger`` (anywhere): read the manifest (configs/meeting5/m5_manifest.yaml)
and that inventory, check every REUSE cell is complete, count every NEW row,
and price it from the measured comparable cells. Refuses if a reuse cell is
incomplete or a priced cell has no measured comparable.

    python scripts/plan_meeting5_cost.py --dry-run
    python scripts/plan_meeting5_cost.py --inventory --out inventory.json
    python scripts/plan_meeting5_cost.py --aime-inventory --out aime.json
    python scripts/plan_meeting5_cost.py --problem-manifests --out manifests.json
    python scripts/plan_meeting5_cost.py --ledger \\
        --inventory-json results/diag/meeting5/inventory.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.meeting5_inventory import (
    comparable_rates,
    cost_hours,
    hashed_split,
    inventory_readout_run,
    inventory_run,
    load_rows,
    strict_fit,
)

MANIFEST = "configs/meeting5/m5_manifest.yaml"
GRID100 = "configs/meeting3/grid100_pids.json"


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def inventory(manifest: dict[str, Any], runs_root: Path) -> dict[str, Any]:
    cohort = manifest["cohort"]
    pids = json.loads((PROJECT_ROOT / GRID100).read_text())["pids"]
    seeds = [int(s) for s in cohort["seeds"]]
    cells: list[dict[str, Any]] = []
    absent = []
    for run in manifest["inventory_runs"]:
        if not (runs_root / run).is_dir():
            absent.append(run)
            continue
        cells += inventory_run(
            run,
            load_rows(runs_root / run),
            expected_pids=pids,
            expected_seeds=seeds,
        )
    bank = runs_root / cohort["fullcache_run"] / "token_ids"
    lens: dict[str, int] = {}
    for path in sorted(bank.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["pid"] in pids and int(r["seed"]) == seeds[0]:
                lens[r["pid"]] = len(r["prompt_token_ids"])
    if set(lens) != set(pids):
        raise SystemExit(f"bank lacks prompt ids for {sorted(set(pids) - set(lens))}")
    return {
        "schema": "meeting5-inventory-v1",
        "cohort": {**cohort, "n_pids": len(pids), "pid_list": GRID100},
        "absent_runs": absent,
        "cells": cells,
        "strict_fit": {
            str(b): strict_fit(lens, int(b)) for b in cohort["strict_fit_budgets"]
        },
    }


# AIME evidence inventory (review 9.1): run -> (config whose seeds define the
# expected keys). Expected pids are the pinned 30-row AIME-24 set.
AIME_RUNS = {
    "fc-aime24": "configs/experiments/bank_aime24_fc.yaml",
    "grid-aime24-2048": "configs/experiments/grid_aime24_2048.yaml",
    "readout-aime24-2048": "configs/experiments/readout_aime24_2048.yaml",
    "fc-aime24-llama8b-cap32768": "configs/experiments/bank_aime24_llama8b.yaml",
    "grid-aime24-rkv-published-cap32768": (
        "configs/experiments/grid_aime24_rkv_published.yaml"
    ),
}


def aime_inventory(runs_root: Path) -> dict[str, Any]:
    from kvcache.datasets import load_benchmark

    bench = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]["aime24"]
    pids = [p.pid for p in load_benchmark("aime24", bench, allow_download=False)]
    cells: list[dict[str, Any]] = []
    absent = []
    for run, cfg_path in AIME_RUNS.items():
        if not (runs_root / run).is_dir():
            absent.append(run)
            continue
        seeds = [int(x) for x in _yaml(PROJECT_ROOT / cfg_path)["seeds"]]
        reader = inventory_readout_run if run.startswith("readout") else inventory_run
        for c in reader(
            run, load_rows(runs_root / run), expected_pids=pids, expected_seeds=seeds
        ):
            cells.append({**c, "config": cfg_path})
    return {
        "schema": "meeting5-aime-inventory-v1",
        "benchmark": {"name": "aime24", "revision": bench["revision"], "n": len(pids)},
        "absent_runs": absent,
        "cells": cells,
    }


# M5-B problem manifests (review 9.2-9.3): disjoint calibration/confirmation by
# SHA-256 order, salted per dataset; sizes are the planning N of review 9.2.
SPLIT_SALT = "meeting5-2026-10-14"
BBH_N = 250  # review 9.1: the reviewer parsed the file at 9ee07bd (250 examples)
BBH_PROPOSED_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
# pinned by the owner 2026-10-14; the single source of truth is kvcache.bbh
from kvcache.bbh import PINNED_BLOB as BBH_PINNED_BLOB  # noqa: E402
from kvcache.bbh import PINNED_COMMIT as BBH_PINNED_COMMIT  # noqa: E402


def problem_manifests() -> dict[str, Any]:
    from kvcache.datasets import load_benchmark

    benches = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    math = [p.pid for p in load_benchmark("math500", benches["math500"])]
    gsm = [p.pid for p in load_benchmark("gsm8k", benches["gsm8k"])]
    cohort = json.loads((PROJECT_ROOT / GRID100).read_text())["pids"]
    return {
        "schema": "meeting5-problem-manifests-v1",
        "frozen": "2026-10-14, before any M5-B outcome; selection by hash only",
        "math500": {
            "revision": benches["math500"]["revision"],
            "n_total": len(math),
            "role": "confirmation reserve outside the discovery cohort",
            **hashed_split(
                math,
                salt=f"{SPLIT_SALT}|math500",
                n_calibration=0,
                n_confirmation=200,
                exclude=cohort,
            ),
        },
        "gsm8k": {
            "revision": benches["gsm8k"]["revision"],
            "n_total": len(gsm),
            **hashed_split(
                gsm, salt=f"{SPLIT_SALT}|gsm8k", n_calibration=24, n_confirmation=200
            ),
        },
        # metadata amended 2026-10-17 (PREREGISTRATION): the owner pinned the
        # proposed revision on 2026-10-14 (kvcache.bbh); the selection, which
        # depends only on the indices and the salt, is unchanged
        "bbh_logical_deduction_five_objects": {
            "revision": BBH_PINNED_COMMIT,
            "blob": BBH_PINNED_BLOB,
            "n_total": BBH_N,
            "note": "ids are example indices; revision pinned by the owner "
            "2026-10-14 (kvcache.bbh.PINNED_COMMIT/PINNED_BLOB); load_bbh verifies "
            "the blob and the 250-example count. Metadata amended 2026-10-17 "
            "(PREREGISTRATION); the selection is unchanged",
            **hashed_split(
                [f"bbh_lde5/{i}" for i in range(BBH_N)],
                salt=f"{SPLIT_SALT}|bbh_lde5",
                n_calibration=25,
                n_confirmation=200,
            ),
        },
    }


def _find(cells: list[dict[str, Any]], run: str, press: str, budget: Any) -> dict:
    hits = [
        c
        for c in cells
        if c["run"] == run and c["press"] == press and c["budget"] == str(budget)
    ]
    if len(hits) != 1:
        raise SystemExit(f"REFUSED: {run}:{press}@{budget} found {len(hits)} times")
    return hits[0]


def ledger(manifest: dict[str, Any], inv: dict[str, Any]) -> dict[str, Any]:
    cells = inv["cells"]
    cohort = manifest["cohort"]
    reuse = []
    for r in manifest["reuse"]:
        c = _find(cells, r["run"], r["press"], r["budget"])
        if not c["complete"]:
            raise SystemExit(f"REFUSED: reuse cell {r['label']} is incomplete: {c}")
        reuse.append({**r, "n_rows": c["n_rows"], "n_truncated": c["n_truncated"]})
    excluded = set(manifest.get("rate_runs_exclude", []))
    rates = comparable_rates(
        [c for c in cells if c["run"] not in excluded],
        model=cohort["model"],
        bench=cohort["bench"],
        budget=cohort["budget"],
    )
    new = []
    for item in manifest["new"]:
        entry = dict(item)
        if item.get("price") == "comparable":
            entry["gpu_hours"] = cost_hours(
                int(item["rows"]), rates, n_launches=int(item.get("launches", 2))
            )
        elif "measured_gpu_hours" in item:
            h = float(item["measured_gpu_hours"])
            entry["gpu_hours"] = {"lower": h, "central": h, "upper": h}
        else:
            entry["gpu_hours"] = None  # not priceable before a calibration timing
        new.append(entry)
    priced = [e for e in new if e["gpu_hours"]]
    total = {
        k: round(sum(e["gpu_hours"][k] for e in priced), 2)
        for k in ("lower", "central", "upper")
    }
    return {
        "schema": "meeting5-ledger-v1",
        "manifest": MANIFEST,
        "rates_from": rates,
        "reuse": reuse,
        "new": new,
        "total_priced_gpu_hours": total,
        "unpriced": [e["id"] for e in new if not e["gpu_hours"]],
        "strict_fit": {
            b: {k: v for k, v in sf.items() if k != "strict_fit_pids"}
            for b, sf in inv["strict_fit"].items()
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--inventory", action="store_true")
    p.add_argument("--ledger", action="store_true")
    p.add_argument("--aime-inventory", action="store_true")
    p.add_argument("--problem-manifests", action="store_true")
    p.add_argument("--inventory-json", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    manifest = _yaml(PROJECT_ROOT / MANIFEST)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest": manifest["name"],
                    "status": manifest["status"],
                    "inventory_runs": len(manifest["inventory_runs"]),
                    "reuse_cells": len(manifest["reuse"]),
                    "new_items": [n["id"] for n in manifest["new"]],
                },
                indent=2,
            )
        )
        print("DRY RUN: manifest parsed; no run directory read")
        return 0
    modes = (args.inventory, args.ledger, args.aime_inventory, args.problem_manifests)
    if sum(modes) != 1:
        p.error("choose exactly one mode")
    if args.problem_manifests:
        report = problem_manifests()
    elif args.aime_inventory:
        report = aime_inventory(Path(os.environ["KVCACHE_BASE_RUNS_ROOT"]))
    elif args.inventory:
        report = inventory(manifest, Path(os.environ["KVCACHE_BASE_RUNS_ROOT"]))
    else:
        if args.inventory_json is None:
            p.error("--ledger needs --inventory-json")
        report = ledger(manifest, json.loads(args.inventory_json.read_text()))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

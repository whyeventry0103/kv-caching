#!/usr/bin/env python3
"""M5-A protection-by-policy accuracy read (Meeting 5; prepared 2026-10-14).

Reads the arms named in configs/meeting5/m5a_read.yaml from the runs root and
prints FINAL (accuracies, protected/unprotected contrasts, interactions with
problem-cluster CIs, the TOST margin label, and the registered problem-level t
tests with Holm over the two scientific pairings) or PARTIAL (coverage only).
All logic is ``kvcache.metrics.policy_replication``.

``--cohort strict`` (primary): the 97 problems with P + 64 <= 256.
``--cohort whole``: all 100, with the 3 relaxed problems' unprotected rows taken
from the capacity-matched run; PARTIAL until that run exists.

    python scripts/report_policy_replication.py --cohort strict [--out read.json]
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

from kvcache.meeting5_inventory import load_rows
from kvcache.metrics.policy_replication import replication_read

SPEC = "configs/meeting5/m5a_read.yaml"


def build_arms(
    spec: dict[str, Any], runs: Path, pids: list[str], relaxed: dict[str, int]
) -> dict[tuple[str, bool], list[dict] | None]:
    """Rows per (policy, protected); None when a needed run does not exist.

    Protected arms ran every problem under S.1 (relaxed problems at B_eff,
    recorded at the nominal budget). In the whole cohort, the unprotected rows
    of the relaxed problems come from the capacity-matched run at B_eff.
    """
    cache: dict[str, list[dict] | None] = {}

    def rows_of(run: str) -> list[dict] | None:
        if run not in cache:
            cache[run] = load_rows(runs / run) if (runs / run).is_dir() else None
        return cache[run]

    keep = set(pids)
    arms: dict[tuple[str, bool], list[dict] | None] = {}
    for policy, sides in spec["arms"].items():
        for protected in (False, True):
            run, press, budget = sides["protected" if protected else "unprotected"]
            base = rows_of(run)
            if base is None:
                arms[(policy, protected)] = None
                continue
            rows = [
                r
                for r in base
                if str(r["press"]).lower() == press
                and str(r["budget"]) == budget
                and r["pid"] in keep
                and (protected or r["pid"] not in relaxed)
            ]
            if relaxed and not protected:
                cap = rows_of(spec["capmatch_run"])
                if cap is None:
                    arms[(policy, protected)] = None
                    continue
                rows += [
                    r
                    for r in cap
                    if str(r["press"]).lower() == press
                    and r["pid"] in relaxed
                    and int(r["budget"]) == relaxed[r["pid"]]
                ]
            arms[(policy, protected)] = rows
    return arms


def main() -> int:
    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=("strict", "whole"), default="strict")
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    spec = yaml.safe_load((PROJECT_ROOT / SPEC).read_text())
    sf = json.loads((PROJECT_ROOT / spec["cohort_source"]).read_text())
    relaxed = {k: int(v) for k, v in sf["relaxed_effective_budgets"].items()}
    if args.cohort == "strict":
        pids, relaxed_used = list(sf["pids"]), {}
    else:
        pids, relaxed_used = sorted([*sf["pids"], *relaxed]), relaxed
    if args.dry_run:
        print(json.dumps({"cohort": args.cohort, "n_pids": len(pids), "spec": SPEC}))
        print("DRY RUN: spec and cohort parsed; no run directory read")
        return 0
    runs = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    expected = {(pid, int(s)) for pid in pids for s in spec["seeds"]}
    report = replication_read(
        build_arms(spec, runs, pids, relaxed_used),
        expected=expected,
        reference=spec["reference"],
        pairings=spec["pairings"],
        descriptive=spec["descriptive"],
    )
    report["cohort"] = {"name": args.cohort, "n_problems": len(pids)}
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

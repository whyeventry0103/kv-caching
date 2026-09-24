#!/usr/bin/env python3
"""Amendment R.2 / R.11 read of the Q.1 direct functional control.

Tolerance = 10 x the maximum |logit difference| between two identical arm-B
replays (the deterministic control), over every row. Per (policy, interval):
PASS if every compared A-B step is within tolerance, FAIL otherwise. R.11: if
the deterministic control is exactly 0 the tolerance is degenerate; the read is
then EXACT when A-B is also exactly 0, and otherwise TOLERANCE-DEGENERATE with
the raw A-B maxima reported (never PASS, never "equal"). Selection agreement is
reported beside it. The legacy removal-anchored gate (phase excess of arm A's
TV profile anchored at the first physical removal) is computed for the record
and labelled NOT VALID FOR IDENTIFICATION whatever it shows.

    python scripts/report_fork_control.py --runs-root <runs> \\
        --run fork-control-random --run fork-control-slm --out fork.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.profile_shape import phase_amplitude_excess


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def classify(max_ab: float, tol: float) -> str:
    if tol > 0:
        return "PASS" if max_ab <= tol else "FAIL"
    return "EXACT" if max_ab == 0 else "TOLERANCE-DEGENERATE"


def read(rows: list[dict[str, Any]], *, n_resamples: int, seed: int) -> dict:
    det = max((r["deterministic_control_max_abs_logit"] or 0.0) for r in rows)
    tol = 10.0 * det
    cells: dict[str, Any] = {}
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["press"], int(r["interval"]))].append(r)
    for (press, interval), rs in sorted(groups.items()):
        diffs = [f["max_abs_logit"] for r in rs for f in r["forks"]]
        tvs = [f["max_tv"] for r in rs for f in r["forks"]]
        agree = [f["selection_agrees"] for r in rs for f in r["forks"]]
        agree = [a for a in agree if a is not None]
        max_ab = max(diffs) if diffs else 0.0
        legacy = phase_amplitude_excess(
            [
                (r["tv_profile"], r["first_removal_step"], interval)
                for r in rs
                if r["first_removal_step"] is not None
            ],
            n_resamples=n_resamples,
            seed=seed,
        )
        cells[f"{press}-i{interval}"] = {
            "n_traces": len(rs),
            "n_forks": len(diffs),
            "max_abs_logit_A_minus_B": max_ab,
            "max_next_token_tv": max(tvs) if tvs else None,
            "n_forks_outside_tolerance": sum(d > tol for d in diffs),
            "selection_agreement": (sum(agree) / len(agree)) if agree else None,
            "n_selection_compared": len(agree),
            "verdict": classify(max_ab, tol),
            "legacy_removal_anchored_gate": {
                **legacy,
                "label": "NOT VALID FOR IDENTIFICATION (R.2)",
            },
        }
    return {
        "deterministic_control_max_abs_logit": det,
        "tolerance": tol,
        "tolerance_rule": "10 x deterministic control max (R.2); R.11 degenerate rule",
        "cells": cells,
    }


def main() -> int:
    a = _parser().parse_args()
    if a.dry_run:
        print(json.dumps({"command": "fork-control-read", "runs": a.run}))
        return 0
    rows = [
        json.loads(line)
        for run in a.run
        for p in sorted((a.runs_root / run / "math500").rglob("*.jsonl"))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    res = {"amendment": "R.2 / R.11", "runs": a.run}
    res.update(read(rows, n_resamples=a.n_resamples, seed=a.seed))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"tolerance {res['tolerance']}")
    for k, c in res["cells"].items():
        print(k, c["verdict"], c["max_abs_logit_A_minus_B"], c["selection_agreement"])
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

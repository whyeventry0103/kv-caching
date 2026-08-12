#!/usr/bin/env python3
"""Task 2 / Amendment P.5: does reinserting evicted entries reduce divergence?

Reads a restoration run's per-arm profiles and reports each arm's effect against
the NO-INTERVENTION arm, paired on the trace.

Two refusals are built in, and both are the point of the script:

* **The instrument gate.** P.5 refuses to read any divergence until the
  reinserted K/V entries are shown present and bitwise identical to what was
  evicted. A failed or missing gate stops the report rather than footnoting it.
* **A shared anchor.** The window is anchored on the step the intervention
  actually landed on, per trace. If the treated arms did not land on the same
  step for a given trace, they are not comparable on one anchor and the script
  says so instead of silently picking one.

    python scripts/report_restoration.py --runs-root <runs> \\
        --run restoration-diss256 --out restoration.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.restoration import (  # noqa: E402
    ARMS,
    READ_WINDOW,
    classify_restoration,
    restoration_read,
)

# Restoration cells carry the arm in the press field: "random.matched-256-s11".
CELL = re.compile(
    r"^(?P<press>[a-z0-9]+)\.(?P<arm>[a-z]+)-(?P<budget>\d+)-s(?P<seed>\d+)$"
)
MEASURE_KEY = {"tv": "estimate", "jsd": "jsd"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--run", required=True)
    p.add_argument("--measure", default="tv", choices=sorted(MEASURE_KEY))
    p.add_argument("--window", type=int, default=READ_WINDOW)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--allow-failed-gate",
        action="store_true",
        help="read divergence even though the instrument gate did not pass. "
        "Amendment P.5 forbids this for any reported number; it exists only "
        "for diagnosing a failed smoke.",
    )
    a = p.parse_args()

    root = a.runs_root / a.run / "profiles"
    if not root.exists():
        raise SystemExit(f"no profiles under {root}")

    # (press, budget) -> arm -> trace -> series;  and the gate/anchor per trace.
    cells: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    applied: dict[tuple[str, str], dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    gate_failures: list[str] = []
    missing_gates: list[str] = []
    n_profiles = 0

    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".restoration.json"):
            continue
        match = CELL.match(path.parent.name)
        if not match:
            continue
        arm = match.group("arm")
        if arm not in ARMS:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (match.group("press"), match.group("budget"))
        trace = f"{path.stem}-s{match.group('seed')}"
        series = payload.get(MEASURE_KEY[a.measure])
        if not series:
            continue
        n_profiles += 1
        cells[key][arm][trace] = list(series)

        gate_path = path.with_name(path.stem + ".restoration.json")
        if not gate_path.exists():
            missing_gates.append(f"{path.parent.name}/{path.name}")
            continue
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("passed"):
            gate_failures.append(f"{path.parent.name}/{path.name}")
        if arm != "none" and gate.get("applied_at_step") is not None:
            applied[key][trace][arm] = int(gate["applied_at_step"])

    if not cells:
        raise SystemExit(
            f"no restoration cells (<press>.<arm>-<budget>-sN) under {root}"
        )
    if (gate_failures or missing_gates) and not a.allow_failed_gate:
        raise SystemExit(
            f"INSTRUMENT GATE NOT CLEAN: {len(gate_failures)} failed, "
            f"{len(missing_gates)} missing. Amendment P.5 refuses to read "
            f"divergence from an ungated reinsertion. First few: "
            f"{(gate_failures + missing_gates)[:3]}"
        )

    out: dict[str, Any] = {
        "run": a.run,
        "amendment": "P.5",
        "measure": a.measure,
        "window": a.window,
        "n_profiles": n_profiles,
        "gate": {
            "failed": len(gate_failures),
            "missing": len(missing_gates),
            "overridden": bool(a.allow_failed_gate),
        },
        "cells": {},
    }

    for key in sorted(cells):
        press, budget = key
        arms = cells[key]
        # One anchor per trace, and only if every treated arm agrees on it.
        anchors: dict[str, int] = {}
        disagreements: list[str] = []
        for trace, per_arm in applied[key].items():
            steps = set(per_arm.values())
            if len(steps) == 1:
                anchors[trace] = steps.pop()
            else:
                disagreements.append(f"{trace}: {sorted(per_arm.items())}")
        usable = sorted(set(arms.get("none", {})) & set(anchors))
        if not usable:
            out["cells"][f"{press}-{budget}"] = {
                "status": "NOT READABLE",
                "reason": (
                    "no trace has both a no-intervention arm and an agreed "
                    "applied step"
                ),
                "arms_present": sorted(arms),
                "anchor_disagreements": disagreements[:5],
            }
            continue

        paired = {
            arm: {t: series[t] for t in usable if t in series}
            for arm, series in arms.items()
        }
        paired = {
            arm: series for arm, series in paired.items() if len(series) == len(usable)
        }
        if "none" not in paired:
            out["cells"][f"{press}-{budget}"] = {
                "status": "NOT READABLE",
                "reason": "the no-intervention arm does not cover the paired traces",
            }
            continue

        read = restoration_read(
            paired,
            applied_steps={t: anchors[t] for t in usable},
            window=a.window,
        )
        read["verdict"] = classify_restoration(read)
        read["anchor_disagreements"] = disagreements[:5]
        read["n_anchor_disagreements"] = len(disagreements)
        out["cells"][f"{press}-{budget}"] = read

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")

    for name, cell in out["cells"].items():
        if cell.get("status") == "NOT READABLE":
            print(f"{name:<18} NOT READABLE -- {cell['reason']}")
            continue
        bits = " ".join(
            f"{arm}={v['effect_vs_no_intervention']:+.4f}"
            f"[{v['effect_ci95'][0]:+.4f},{v['effect_ci95'][1]:+.4f}]"
            for arm, v in sorted(cell["arms"].items())
        )
        print(
            f"{name:<18} n={cell['n_traces']:<4} anchor={cell['applied_step_range']} "
            f"{bits}  -> {cell['verdict']}"
        )
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the machine-readable Meeting #3 execution manifest.

Reads the declarative cell list in configs/meeting3/cells.yaml and writes
results/diag/meeting3/meeting3_manifest.json with every identity computed
here rather than typed by hand: config-file sha256, the source-tree hash, and
sha256 of every output artifact that exists. A cell whose declared outputs are
missing cannot claim a status above RUNNING, and an unknown status is refused:
a ledger that can say COMPLETE about something that does not exist is worse
than none.

    python scripts/meeting3_manifest.py [--check]

--check exits non-zero on any inconsistency without writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Where results, figures and stage records live: the project tree in
# production; --root points it at an isolated tree (acceptance tests).
OUT_ROOT = PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml  # noqa: E402

from kvcache.source_hash import source_tree_hash  # noqa: E402

STATUSES = (
    "NOT_STARTED",
    "IMPLEMENTED",
    "TESTED",
    "VALIDATED",
    "QUEUED",
    "RUNNING",
    "GENERATED",
    "ANALYZED",
    "COMPLETE",
    "BLOCKED",
    "INVALID",
)
# A status at or above this rank asserts that outputs exist.
NEEDS_OUTPUTS = {"GENERATED", "ANALYZED", "COMPLETE"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence_status(cell: dict, outputs: dict) -> tuple[str | None, str]:
    """R.14: a cell with a ``stage`` takes its status from that stage's record.

    FINAL -> ANALYZED, or COMPLETE when every declared output also exists;
    INVALID -> INVALID; PARTIAL -> GENERATED (the stage runs only after its
    generation jobs completed, so PARTIAL means rows exist but coverage or
    identity is not yet exact). No record -> the declared status stands.
    """
    stage = cell.get("stage")
    if not stage:
        return None, "declared (no stage)"
    rec = OUT_ROOT / "results/diag/meeting3/stages" / f"{stage}.stage.json"
    if not rec.exists():
        return None, f"no record for stage {stage} yet"
    d = json.loads(rec.read_text(encoding="utf-8"))
    st = d.get("status")
    gate = d.get("gate_outcome")
    if st == "FINAL" and gate is not None:
        # analysis complete is not the same as the control passing
        why = f"stage {stage} FINAL; control outcome {gate}"
        return ("COMPLETE" if gate == "PASS" else "ANALYZED"), why
    if st == "FINAL":
        full = all(h is not None for h in outputs.values())
        return ("COMPLETE" if full else "ANALYZED"), f"stage {stage} FINAL"
    if st == "INVALID":
        return "INVALID", f"stage {stage} INVALID"
    return "GENERATED", f"stage {stage} {st}"


def build(cells_path: Path) -> tuple[dict, list[str]]:
    spec = yaml.safe_load(cells_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    out_cells = []
    seen: set[str] = set()
    for cell in spec["cells"]:
        cid = cell["id"]
        if cid in seen:
            problems.append(f"{cid}: duplicate id")
        seen.add(cid)
        status = cell.get("status", "NOT_STARTED")
        if status not in STATUSES:
            problems.append(f"{cid}: unknown status {status!r}")
        configs = {}
        for c in cell.get("configs") or []:
            p = PROJECT_ROOT / c
            if p.exists():
                configs[c] = _sha256(p)
            elif status not in ("NOT_STARTED", "BLOCKED"):
                problems.append(f"{cid}: config {c} missing at status {status}")
            else:
                configs[c] = None
        outputs = {}
        for o in cell.get("outputs") or []:
            p = OUT_ROOT / o
            outputs[o] = _sha256(p) if p.exists() else None
        if status in NEEDS_OUTPUTS:
            missing = [o for o, h in outputs.items() if h is None]
            if missing:
                problems.append(
                    f"{cid}: status {status} but outputs missing: {missing}"
                )
        derived, why = _evidence_status(cell, outputs)
        ladder = STATUSES[: STATUSES.index("COMPLETE") + 1]
        if derived is not None and derived != status:
            if (
                status in ladder
                and derived in ladder
                and ladder.index(status) > ladder.index(derived)
            ):
                problems.append(
                    f"{cid}: declared {status} exceeds evidence ({derived}: {why})"
                )
        out_cells.append(
            {
                **cell,
                "declared_status": status,
                "evidence": why,
                "status": derived or status,
                "config_sha256": configs,
                "output_sha256": outputs,
            }
        )
    manifest = {
        "schema": "meeting3-manifest-v1",
        "model": spec["model"],
        "reference_bank": spec["reference_bank"],
        "runtime_pin": spec["runtime_pin"],
        "source_tree_sha256": source_tree_hash(PROJECT_ROOT),
        "cells": out_cells,
        "status_counts": {
            s: sum(1 for c in out_cells if c["status"] == s) for s in STATUSES
        },
    }
    return manifest, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cells", type=Path, default=PROJECT_ROOT / "configs/meeting3/cells.yaml"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results/diag/meeting3/meeting3_manifest.json",
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", type=Path, default=None)
    a = ap.parse_args()
    if a.root is not None:
        global OUT_ROOT
        OUT_ROOT = a.root
    manifest, problems = build(a.cells)
    for p in problems:
        print(f"PROBLEM: {p}")
    if a.check:
        return 1 if problems else 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(manifest["status_counts"]))
    print(f"wrote {a.out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

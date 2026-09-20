#!/usr/bin/env python3
"""Thin orchestration around ``scripts/report_onset_surface.py`` (Amendment Q.2).

Config-driven so the cell can be queued like every other one and its provenance
sits where the rest do. All science lives in the package and the reporter; this
file only resolves paths and shells out.

CPU-only: it reads stored scalar profiles and touches no model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-understand", action="store_true")
    # The orchestrator appends both to EVERY launch. Accept and ignore, or
    # argparse exits rc=2 before a row is written and the job is marked stuck
    # (landmine 1).
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seeds", default="")
    args = parser.parse_args()
    if args.seeds:
        print(f"note: --seeds {args.seeds} ignored; this analysis is not sharded")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("arms", "run_artifact", "out_artifact"):
        if key not in config:
            raise SystemExit(f"config missing required key: {key}")

    # BASE runs root: this reads OTHER runs' profile trees, and launch.sh scopes
    # REMOTE_RUNS_ROOT to the job's own directory. Same defect that stuck the
    # three shape cells; see run_shape_analysis.py for the full note.
    runs_root = Path(
        os.environ.get("KVCACHE_BASE_RUNS_ROOT")
        or os.environ.get("REMOTE_RUNS_ROOT")
        or (PROJECT_ROOT / "runs")
    )
    out = runs_root / config["run_artifact"]
    archive = PROJECT_ROOT / config["out_artifact"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "report_onset_surface.py"),
        "--runs-root",
        str(runs_root),
        "--out",
        str(out),
        "--arms",
        *[f"{t0}={run}" for t0, run in sorted(config["arms"].items())],
    ]
    plan = {
        "amendment": "Q.2",
        "command": "onset-surface",
        "config": str(args.config),
        "arms": config["arms"],
        "runs_root": str(runs_root),
        "run_artifact": str(out),
        "archived_to": str(archive),
        "gpu_required": False,
        "estimated_generation_count": 0,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        cmd.append("--dry-run")
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    if rc != 0 or args.dry_run:
        return rc
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"archived {out} -> {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

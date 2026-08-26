#!/usr/bin/env python3
"""Amendment P.2-v2: restricted-window phase re-analysis (CPU only, no GPU).

Thin orchestration around ``scripts/report_profile_shape.py``'s restricted mode
so the nightly queue can dispatch it like any other job. All scientific logic
lives in ``kvcache.metrics.profile_shape``; this script only resolves the config
into arguments and records provenance.

    python scripts/run_shape_analysis.py \\
        --config configs/experiments/shape_p2v2_restricted.yaml --dry-run
    python scripts/run_shape_analysis.py --config ... --i-understand
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-understand", action="store_true")
    # The nightly orchestrator appends --resume --seeds <list> to EVERY job it
    # launches (nightly_autorun.build_launch_specs), because every job before
    # this one was a sharded generation run. This analysis has nothing to shard
    # and nothing to resume -- it is one CPU pass over existing profiles -- but
    # it must ACCEPT the flags or argparse exits rc=2 before doing any work.
    # That is exactly how the 2026-08-25 launch failed three times until the
    # orchestrator marked the job stuck.
    parser.add_argument(
        "--resume",
        action="store_true",
        help="accepted for orchestrator compatibility; this job is a single "
        "idempotent pass and always recomputes",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="accepted for orchestrator compatibility; this job reads whole "
        "cells and does not shard by seed",
    )
    args = parser.parse_args()
    if args.seeds:
        print(f"note: --seeds {args.seeds} ignored; this analysis is not sharded")

    config = _yaml(args.config)
    for key in ("source_run", "run_artifact", "out_artifact", "restrict_to_floor_exit"):
        if key not in config:
            raise SystemExit(f"config missing required key: {key}")
    if not config["restrict_to_floor_exit"]:
        raise SystemExit(
            "restrict_to_floor_exit must be true: the unrestricted numbers "
            "already exist and are not the pre-registered test"
        )

    runs_root = Path(os.environ.get("REMOTE_RUNS_ROOT", PROJECT_ROOT / "runs"))
    profiles = runs_root / config["source_run"] / "profiles"
    out = runs_root / config["run_artifact"]
    archive = PROJECT_ROOT / config["out_artifact"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "report_profile_shape.py"),
        "--profiles-dir",
        str(profiles),
        "--label",
        config["name"],
        "--out",
        str(out),
        "--restrict-to-floor-exit",
    ]
    if config.get("budget"):
        cmd += ["--budget", str(config["budget"])]

    plan = {
        "amendment": "P.2-v2",
        "command": "shape",
        "config": str(args.config),
        "name": config["name"],
        "source_run": config["source_run"],
        "profiles_dir": str(profiles),
        "run_artifact": str(out),
        "archived_to": str(archive),
        "gpu_required": False,
        "estimated_generation_count": 0,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))

    if args.dry_run:
        print("DRY RUN: validated only; no profiles read, no artifact written")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: run requires --i-understand")
    if not profiles.exists():
        raise SystemExit(f"missing profiles dir: {profiles}")

    out.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc
    # Archive into the repo only on success, so a failed run cannot leave a
    # half-written artifact where the paper's number provenance reads from.
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, archive)
    print(f"archived {out} -> {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

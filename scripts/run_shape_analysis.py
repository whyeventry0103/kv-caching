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
    # BASE runs root, not the scoped one. launch.sh re-exports
    # REMOTE_RUNS_ROOT="$BASE_RUNS_ROOT/$RUN_NAME" so a generator writes inside
    # its own run directory, and separately exports KVCACHE_BASE_RUNS_ROOT for
    # the unscoped tree. An ANALYSIS job reads ANOTHER run's profiles and its
    # artifact is found by the orchestrator's marker check, which is resolved
    # against the base -- so both of its paths belong to the base root.
    #
    # Using the scoped root produced
    # runs/shape-sawtooth-i16/replay-sawtooth-i16/profiles: the run name
    # appears twice and the source can never be found. Amendment P.2-v2 masked
    # this because its artifact was produced out of band, so the cell was
    # "complete" by marker while its own launch log recorded rc=2.
    runs_root = Path(
        os.environ.get("KVCACHE_BASE_RUNS_ROOT")
        or os.environ.get("REMOTE_RUNS_ROOT")
        or (PROJECT_ROOT / "runs")
    )
    profiles = runs_root / config["source_run"] / "profiles"

    # The floor-exit restriction is mandatory for a RATIO-MATCHED source and
    # optional elsewhere, and the difference is detectable rather than declared:
    # only a ratio-matched press writes .ratio_trace.json sidecars, and only
    # such a run HAS a budget floor to exit. The original guard was written for
    # Amendment P.2-v2, where the unrestricted numbers already existed and were
    # not the pre-registered test. That reasoning is specific to that cell, not
    # a general truth: the Amendment Q.1 sawtooth arms are fixed-budget, have no
    # floor, and their unrestricted phase profile IS the pre-registered test.
    is_ratio_matched = any(profiles.rglob("*.ratio_trace.json"))
    if not config["restrict_to_floor_exit"]:
        if is_ratio_matched:
            raise SystemExit(
                f"{config['source_run']} is a ratio-matched run (it writes "
                "ratio_trace sidecars), so restrict_to_floor_exit must be true: "
                "its floor-bound early trace is the harshest compression in the "
                "run and would dominate any whole-trace statistic. The "
                "unrestricted numbers already exist and are not the "
                "pre-registered test."
            )
        if not config.get("unrestricted_rationale"):
            raise SystemExit(
                "restrict_to_floor_exit is false, so the config must state an "
                "`unrestricted_rationale`. Running unrestricted is a real "
                "choice about which window a verdict is read on, and it is not "
                "allowed to be made silently by omission."
            )
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
    ]
    if config["restrict_to_floor_exit"]:
        cmd.append("--restrict-to-floor-exit")
    if config.get("budget"):
        cmd += ["--budget", str(config["budget"])]

    plan = {
        "amendment": config.get("amendment", "P.2-v2"),
        "restrict_to_floor_exit": bool(config["restrict_to_floor_exit"]),
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
        raise SystemExit(
            f"missing profiles dir: {profiles}\n"
            f"source_run={config['source_run']!r}, runs_root={runs_root}\n"
            "If the run name appears twice in that path, the scoped "
            "REMOTE_RUNS_ROOT was used instead of KVCACHE_BASE_RUNS_ROOT: "
            "launch.sh scopes the former to the job's own run directory, and an "
            "analysis job reads another run's tree."
        )

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

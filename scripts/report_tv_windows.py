#!/usr/bin/env python3
"""Emit the onset-aligned windowed-TV means as a machine-readable artifact.

The mechanism section quotes 0.0000 / 0.0018 / 0.0884 and a step-pooled figure,
and until now those existed only as stdout from
``make_paper_figures.py --print-windows``. A number quoted in the paper with no
artifact behind it is exactly the gap ``scripts/audit_paper_numbers.py`` exists
to catch, so this writes them down.

Computation lives in ``kvcache.metrics.tdp.windowed_tv`` -- shared with the
figure path so the two cannot drift.

    python scripts/report_tv_windows.py --profiles-dir <dir> --out windows.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.tdp import windowed_tv


def load_profiles(profiles_dir: Path) -> list[tuple[list[float], int | None]]:
    out: list[tuple[list[float], int | None]] = []
    for path in sorted(glob.glob(str(profiles_dir / "**/*.json"), recursive=True)):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        out.append((payload.get("estimate") or [], payload.get("onset_step")))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--label", default="", help="cell label recorded in the artifact"
    )
    args = parser.parse_args()

    profiles = load_profiles(args.profiles_dir)
    if not profiles:
        raise SystemExit(f"no TV profiles found under {args.profiles_dir}")
    summary = windowed_tv(profiles)
    summary["label"] = args.label
    summary["profiles_dir"] = str(args.profiles_dir)
    summary["note"] = (
        "Onset-aligned windowed per-step TV, full horizon. 'before' is the "
        "replay harness self-control and MUST be 0.0000; a non-zero value is an "
        "instrument bug, not a finding. The paper reports the per_profile "
        "estimator (the trace is the sampling unit)."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

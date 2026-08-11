#!/usr/bin/env python3
"""Task 4 / Amendment P.4: do JSD or top-1 localise onset earlier than TV?

Reads v2 profiles only. For each measure, finds the first step at which the
onset-aligned series rises above a per-trace baseline threshold, and compares
the distribution of those first-crossings across measures. A measure that
crosses EARLIER localises onset earlier; a measure whose crossing distribution
is TIGHTER localises it more sharply.

Descriptive. These are all TV-scale (distributional) quantities and per
Definition 6 none of them licenses an accuracy-scale claim by adjacency.

    python scripts/report_onset_localization.py --profiles-dir <run>/profiles \\
        --out onset.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CELL = re.compile(r"^(?P<press>[a-z0-9]+)-(?P<budget>\d+)-s(?P<seed>\d+)$")
# A crossing is the first step at or after onset whose value exceeds this
# multiple of the pre-onset baseline spread. Fixed here, before the run.
K_SIGMA = 3.0
MIN_PRE_STEPS = 20


def _first_crossing(series: list[float], onset: int) -> int | None:
    pre = series[:onset]
    if len(pre) < MIN_PRE_STEPS:
        return None
    mu = statistics.fmean(pre)
    sd = statistics.pstdev(pre)
    # A perfectly flat pre-onset baseline (TV is exactly 0 there by construction)
    # has zero spread, so fall back to the smallest positive value as the scale.
    if sd == 0:
        positives = [v for v in series[onset:] if v > 0]
        if not positives:
            return None
        threshold = min(positives)
    else:
        threshold = mu + K_SIGMA * sd
    for offset, value in enumerate(series[onset:]):
        if value > threshold:
            return offset
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    per_press: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    n_v2 = 0
    skipped_v1 = 0
    for path in sorted(a.profiles_dir.rglob("*.json")):
        m = CELL.match(path.parent.name)
        if not m:
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("schema_version") != "tv-profile-v2":
            skipped_v1 += 1
            continue
        onset = d.get("onset_step")
        if onset is None:
            continue
        n_v2 += 1
        cell = f"{m['press']}-{m['budget']}"
        series = {
            "tv": [float(v) for v in d["estimate"]],
            "jsd": [float(v) for v in d.get("jsd", [])],
            # top-1 DISagreement, so that "higher = more diverged" holds for
            # every measure and the crossing rule is the same one.
            "top1_disagree": [0.0 if v else 1.0 for v in d.get("top1_agree", [])],
            "rank_displacement": [float(v) for v in d.get("rank_displacement", [])],
        }
        for name, values in series.items():
            if not values:
                continue
            crossing = _first_crossing(values, int(onset))
            if crossing is not None:
                per_press[cell][name].append(crossing)

    out: dict[str, Any] = {
        "amendment": "P.4",
        "k_sigma": K_SIGMA,
        "n_v2_profiles": n_v2,
        "n_v1_skipped": skipped_v1,
        "per_cell": {},
        "discipline": (
            "Descriptive. All four are distributional quantities; per Definition "
            "6 none licenses an accuracy-scale claim by adjacency."
        ),
    }
    for cell, measures in sorted(per_press.items()):
        out["per_cell"][cell] = {
            name: {
                "n": len(values),
                "median_steps_after_onset": statistics.median(values),
                "mean_steps_after_onset": statistics.fmean(values),
                "iqr": (
                    statistics.quantiles(values, n=4)[2]
                    - statistics.quantiles(values, n=4)[0]
                    if len(values) >= 4
                    else None
                ),
            }
            for name, values in sorted(measures.items())
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")
    for cell, measures in sorted(out["per_cell"].items()):
        parts = " ".join(
            f"{k}={v['median_steps_after_onset']:.0f}"
            for k, v in sorted(measures.items())
        )
        print(f"{cell:<20} median first-crossing (steps after onset): {parts}")
    print(f"wrote {a.out}  (v2={n_v2}, v1 skipped={skipped_v1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

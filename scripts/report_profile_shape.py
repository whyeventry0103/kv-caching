#!/usr/bin/env python3
"""Per-policy shape statistics over stored TV profiles (Amendment P.2).

No new generation: the stored profiles already contain the full horizon per step
(only the rendered figure truncates at +/-400). This splits them BY POLICY --
the published artifact pools Random and StreamingLLM into one number -- and adds
the running integral, the prune-cycle phase profile, and the per-trace shape
mixture.

    python scripts/report_profile_shape.py --profiles-dir <run>/profiles \\
        --label replay-diss256 --out shape_diss256.json
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

from kvcache.metrics.profile_shape import (  # noqa: E402
    MIN_RESTRICTED_WINDOW,
    phase_amplitude_excess,
    restrict_to_floor_exit,
    summarise_profiles,
)
from kvcache.replay import is_profile_file  # noqa: E402

CELL = re.compile(r"^(?P<press>[a-z0-9]+)-(?P<budget>\d+)-s(?P<seed>\d+)$")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles-dir", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--restrict-to-floor-exit",
        action="store_true",
        help="Amendment P.2-v2: trim each trace to the window where its own "
        "ratio-matched press was ABOVE min_budget, using the same windowing "
        "function as the M.3 drift comparison. Requires a sibling "
        "<pid>.ratio_trace.json per profile. Profiles that never left the floor, "
        f"and windows shorter than {MIN_RESTRICTED_WINDOW} steps, are excluded "
        "and counted -- never silently dropped.",
    )
    p.add_argument(
        "--budget",
        default="",
        help="restrict to one budget. Pooling budgets is NOT neutral: a weaker "
        "compression budget dilutes an eviction-locked phase signal, so a "
        "pooled excess is not comparable to a single-budget one.",
    )
    a = p.parse_args()

    by_press: dict[str, list[tuple[list[float], int | None, int]]] = defaultdict(list)
    n_files = 0
    excluded = {"never_left_floor": 0, "window_too_short": 0, "no_ratio_trace": 0}
    retained_fraction: list[float] = []
    pids_kept: dict[str, set] = defaultdict(set)
    for path in sorted(a.profiles_dir.rglob("*.json")):
        if not is_profile_file(path):
            continue  # sidecar (.restoration.json / .ratio_trace.json)
        match = CELL.match(path.parent.name)
        if not match:
            continue
        if a.budget and match.group("budget") != a.budget:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        estimate = payload["estimate"]
        onset = payload.get("onset_step")
        interval = int(payload.get("compression_interval") or 64)

        if a.restrict_to_floor_exit:
            trace_path = path.with_name(path.stem + ".ratio_trace.json")
            if not trace_path.exists():
                excluded["no_ratio_trace"] += 1
                continue
            trace = json.loads(trace_path.read_text(encoding="utf-8"))["trace"]
            restricted = restrict_to_floor_exit(estimate, onset, trace, interval)
            if restricted is None:
                # Distinguish the two exclusion reasons for the report.
                from kvcache.metrics.profile_shape import floor_exit_step

                if floor_exit_step(trace, interval) is None:
                    excluded["never_left_floor"] += 1
                else:
                    excluded["window_too_short"] += 1
                continue
            retained_fraction.append(len(restricted[0]) / len(estimate))
            estimate, onset = restricted[0], restricted[1]

        n_files += 1
        pids_kept[match.group("press")].add(path.stem)
        by_press[match.group("press")].append((estimate, onset, interval))

    if not by_press:
        raise SystemExit(
            f"no profiles matching <press>-<budget>-s<seed> under {a.profiles_dir}"
        )

    out: dict[str, Any] = {
        "label": a.label,
        "amendment": "P.2",
        "budget_filter": a.budget or "ALL BUDGETS POOLED",
        "window": (
            "RESTRICTED to above-floor steps (Amendment P.2-v2; same windowing "
            "as the M.3 drift comparison)"
            if a.restrict_to_floor_exit
            else "WHOLE TRACE (unrestricted)"
        ),
        "excluded": excluded if a.restrict_to_floor_exit else None,
        "retained_fraction_of_trace_median": (
            round(statistics.median(retained_fraction), 3)
            if retained_fraction
            else None
        ),
        "n_problems_per_press": {k: len(v) for k, v in sorted(pids_kept.items())},
        "n_profile_files": n_files,
        "per_policy": {
            press: {
                **summarise_profiles(items),
                # Only the EXCESS over sham licenses an eviction-alignment
                # claim; a raw amplitude is reproduced by shuffled labels.
                "phase_excess": phase_amplitude_excess(items),
            }
            for press, items in sorted(by_press.items())
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")

    def fmt(value: Any, spec: str) -> str:
        # A restricted window can legitimately leave a statistic undefined (a
        # flat trimmed segment has no drift ratio). Printing must not crash on
        # that -- the JSON already records the None, and a crash here would lose
        # a completed analysis to a formatting error.
        return "n/a" if value is None else format(value, spec)

    if a.restrict_to_floor_exit:
        print(
            f"RESTRICTED window; excluded={out['excluded']}; "
            f"retained median {out['retained_fraction_of_trace_median']} of trace; "
            f"problems per press {out['n_problems_per_press']}"
        )
    for press, s in out["per_policy"].items():
        pe = s["phase_excess"]
        print(
            f"{press:<14} n={s['n_profiles']:<4} "
            f"auc={fmt(s['auc_tv_mean'], '.4f')} "
            f"shapes={s['shape_fractions']} "
            f"drift_med={fmt(s['median_drift_ratio'], '.2f')} "
            f"amp={fmt(s['phase_amplitude'], '.4f')} "
            f"sham={fmt(s['phase_amplitude_sham'], '.4f')} "
            f"excess={fmt(pe['excess'], '+.4f')} "
            f"CI={[round(x, 4) for x in pe['excess_ci95']]} "
            f"excl0={pe['excess_excludes_zero']} "
            f"peak={s['phase_peak_index']}"
        )
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

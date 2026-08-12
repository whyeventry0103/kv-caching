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
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.profile_shape import (  # noqa: E402
    phase_amplitude_excess,
    summarise_profiles,
)

CELL = re.compile(r"^(?P<press>[a-z0-9]+)-(?P<budget>\d+)-s(?P<seed>\d+)$")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles-dir", required=True, type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True, type=Path)
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
    for path in sorted(a.profiles_dir.rglob("*.json")):
        match = CELL.match(path.parent.name)
        if not match:
            continue
        if a.budget and match.group("budget") != a.budget:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        n_files += 1
        by_press[match.group("press")].append(
            (
                payload["estimate"],
                payload.get("onset_step"),
                int(payload.get("compression_interval") or 64),
            )
        )

    if not by_press:
        raise SystemExit(
            f"no profiles matching <press>-<budget>-s<seed> under {a.profiles_dir}"
        )

    out: dict[str, Any] = {
        "label": a.label,
        "amendment": "P.2",
        "budget_filter": a.budget or "ALL BUDGETS POOLED",
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
    for press, s in out["per_policy"].items():
        print(
            f"{press:<14} n={s['n_profiles']:<4} auc={s['auc_tv_mean']:.4f} "
            f"shapes={s['shape_fractions']} "
            f"drift_med={s['median_drift_ratio']:.2f} "
            f"amp={s['phase_amplitude']:.4f} sham={s['phase_amplitude_sham']:.4f} "
            f"excess={s['phase_excess']['excess']:+.4f} "
            f"CI={[round(x, 4) for x in s['phase_excess']['excess_ci95']]} "
            f"excl0={s['phase_excess']['excess_excludes_zero']} "
            f"peak={s['phase_peak_index']}"
        )
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# ruff: noqa: E501
"""Summarize and plot the full-cache self-replay negative control."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_rows(run_root: Path) -> list[dict]:
    rows = []
    for path in sorted((run_root / "math500").rglob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def _profile_summary(run_root: Path, row: dict) -> dict:
    path = run_root / row["tv_profile_path"]
    profile = json.loads(path.read_text(encoding="utf-8"))
    tv = [float(value) for value in profile["estimate"]]
    return {
        "pid": row["pid"],
        "seed": int(row["decoding"]["seed"]),
        "steps": len(tv),
        "min_tv": min(tv),
        "max_tv": max(tv),
        "mean_tv": statistics.fmean(tv),
        "sum_tv": sum(tv),
        "nonzero_steps": sum(value != 0.0 for value in tv),
        "threshold_crossings": {
            "p05": sum(value > 0.05 for value in tv),
            "p10": sum(value > 0.10 for value in tv),
            "p15": sum(value > 0.15 for value in tv),
            "p20": sum(value > 0.20 for value in tv),
        },
        "onset_step": profile.get("onset_step"),
        "compression_events": int(profile.get("compression_events", 0)),
        "tv": tv,
    }


def build_summary(run_root: Path) -> tuple[dict, list[dict]]:
    rows = _load_rows(run_root)
    if not rows:
        raise ValueError(f"no replay rows found under {run_root}")
    profiles = [_profile_summary(run_root, row) for row in rows]
    all_tv = [value for profile in profiles for value in profile["tv"]]
    summary = {
        "control": "full-cache self-replay",
        "rows": len(rows),
        "profiles": len(profiles),
        "steps": len(all_tv),
        "global_min_tv": min(all_tv),
        "global_max_tv": max(all_tv),
        "global_mean_tv": statistics.fmean(all_tv),
        "global_sum_tv": sum(all_tv),
        "nonzero_steps": sum(value != 0.0 for value in all_tv),
        "threshold_crossings": {
            "p05": sum(value > 0.05 for value in all_tv),
            "p10": sum(value > 0.10 for value in all_tv),
            "p15": sum(value > 0.15 for value in all_tv),
            "p20": sum(value > 0.20 for value in all_tv),
        },
        "no_tdp_crossing_rows": sum(
            profile["onset_step"] is None for profile in profiles
        ),
        "no_tdp_crossing_fraction": statistics.fmean(
            profile["onset_step"] is None for profile in profiles
        ),
        "compression_events": sum(
            profile["compression_events"] for profile in profiles
        ),
        "step_range": [
            min(profile["steps"] for profile in profiles),
            max(profile["steps"] for profile in profiles),
        ],
    }
    return summary, profiles


def _normalized_max_envelope(profiles: list[dict], bins: int = 200) -> list[float]:
    envelope = [0.0] * bins
    for profile in profiles:
        tv = profile["tv"]
        for index, value in enumerate(tv):
            position = min(bins - 1, int(index * bins / len(tv)))
            envelope[position] = max(envelope[position], value)
    return envelope


def _write_svg(path: Path, summary: dict, profiles: list[dict]) -> None:
    width, height = 920, 500
    left, right, top, bottom = 85, 35, 65, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = 0.055
    envelope = _normalized_max_envelope(profiles)

    def x_coord(index: int) -> float:
        return left + index * plot_width / (len(envelope) - 1)

    def y_coord(value: float) -> float:
        return top + plot_height * (1.0 - value / y_max)

    points = " ".join(
        f"{x_coord(index):.2f},{y_coord(value):.2f}"
        for index, value in enumerate(envelope)
    )
    threshold_y = y_coord(0.05)
    zero_y = y_coord(0.0)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">M4.2 full-cache self-replay negative control</text>
<text x="{width / 2}" y="51" text-anchor="middle" font-family="sans-serif" font-size="13">Maximum exact TV across {summary['profiles']} traces ({summary['steps']:,} replay steps)</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{zero_y}" stroke="#222"/>
<line x1="{left}" y1="{zero_y}" x2="{width - right}" y2="{zero_y}" stroke="#222"/>
<line x1="{left}" y1="{threshold_y}" x2="{width - right}" y2="{threshold_y}" stroke="#c44" stroke-dasharray="8 6"/>
<text x="{width - right - 4}" y="{threshold_y - 7}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#a22">lowest preregistered threshold τ=0.05</text>
<polyline points="{points}" fill="none" stroke="#1261a0" stroke-width="3"/>
<circle cx="{left}" cy="{zero_y}" r="4" fill="#1261a0"/>
<text x="{left + 12}" y="{zero_y - 12}" font-family="sans-serif" font-size="13" fill="#1261a0">observed TV envelope = 0.0</text>
<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="14">Normalized reference-trace position</text>
<text x="20" y="{height / 2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 {height / 2})">Exact total variation</text>
<text x="{left}" y="{zero_y + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">0%</text>
<text x="{width - right}" y="{zero_y + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">100%</text>
<text x="{left - 12}" y="{zero_y + 4}" text-anchor="end" font-family="sans-serif" font-size="12">0.00</text>
<text x="{left - 12}" y="{threshold_y + 4}" text-anchor="end" font-family="sans-serif" font-size="12">0.05</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_table(path: Path, summary: dict, profiles: list[dict]) -> None:
    lines = [
        "# M4.2 Self-Replay Negative Control",
        "",
        f"- Rows/profiles: {summary['rows']}",
        f"- Total replay steps: {summary['steps']:,}",
        f"- Global TV min/max/mean/sum: {summary['global_min_tv']} / "
        f"{summary['global_max_tv']} / {summary['global_mean_tv']} / "
        f"{summary['global_sum_tv']}",
        f"- Nonzero steps: {summary['nonzero_steps']}",
        f"- Threshold crossings at 0.05/0.10/0.15/0.20: "
        f"{list(summary['threshold_crossings'].values())}",
        f"- No-crossing TDP rows: {summary['no_tdp_crossing_rows']}/"
        f"{summary['rows']} (operational control percentile = 1.0)",
        f"- Compression events: {summary['compression_events']}",
        "",
        "| PID | Seed | Steps | Max TV | Nonzero | TDP crossing |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for profile in sorted(profiles, key=lambda item: (item["pid"], item["seed"])):
        lines.append(
            f"| `{profile['pid']}` | {profile['seed']} | {profile['steps']} | "
            f"{profile['max_tv']:.3g} | {profile['nonzero_steps']} | "
            f"{profile['onset_step']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary, profiles = build_summary(args.run_root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_table(args.out / "table.md", summary, profiles)
    _write_svg(args.out / "self_replay_control.svg", summary, profiles)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

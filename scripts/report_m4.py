#!/usr/bin/env python3
"""Build the M4 compressed replay/readout decomposition report."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_rows(root: Path, subdir: str) -> list[dict[str, Any]]:
    return [
        row for path in sorted((root / subdir).rglob("*.jsonl")) for row in _jsonl(path)
    ]


def _run_key(row: dict[str, Any]) -> tuple[str, int]:
    return row["pid"], int(row["decoding"]["seed"])


def _reference_key_from_readout(row: dict[str, Any]) -> tuple[str, int]:
    reference = row["reference_key"]
    return str(reference[1]), int(reference[5])


def _condition_from_readout(row: dict[str, Any]) -> tuple[str, str]:
    condition = row["condition"]
    return str(condition["press"]), str(condition["budget"])


def _accuracy(rows: list[dict[str, Any]], field: str = "correct") -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def _signed_flip_mass(
    reference: dict[tuple[str, int], bool],
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    gained = lost = 0
    for row in rows:
        before = bool(reference[_run_key(row)])
        after = bool(row["correct"])
        if after and not before:
            gained += 1
        elif before and not after:
            lost += 1
    total = len(rows)
    return {
        "gained": gained / total,
        "lost": lost / total,
        "net": (gained - lost) / total,
    }


def build_report(
    *,
    fullcache_root: Path,
    compression_root: Path,
    replay_root: Path,
    readout_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_rows = _load_rows(fullcache_root, "math500")
    compression_rows = _load_rows(compression_root, "math500")
    replay_rows = _load_rows(replay_root, "math500")
    readout_rows = [
        row
        for path in sorted((readout_root / "readout").rglob("*.jsonl"))
        for row in _jsonl(path)
    ]
    if not (full_rows and compression_rows and replay_rows and readout_rows):
        raise ValueError("all M4 inputs must contain rows")

    reference_correct = {_run_key(row): bool(row["correct"]) for row in full_rows}
    full_accuracy = _accuracy(full_rows)

    compression_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    replay_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    readout_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in compression_rows:
        compression_by_cell[(row["press"], str(row["budget"]))].append(row)
    for row in replay_rows:
        replay_by_cell[(row["press"], str(row["budget"]))].append(row)
    for row in readout_rows:
        readout_by_cell[_condition_from_readout(row)].append(row)

    cells = []
    for condition in sorted(compression_by_cell):
        comp = compression_by_cell[condition]
        replay = replay_by_cell[condition]
        readout = readout_by_cell[condition]
        compressed_accuracy = _accuracy(comp)
        readout_accuracy = sum(row["readout_correct"] for row in readout) / len(readout)
        total_change = compressed_accuracy - full_accuracy
        readout_change = readout_accuracy - full_accuracy
        residual = total_change - readout_change
        flip = _signed_flip_mass(reference_correct, comp)
        summed_tv = []
        for row in replay:
            profile = json.loads(
                (replay_root / row["tv_profile_path"]).read_text(encoding="utf-8")
            )
            summed_tv.append(sum(float(value) for value in profile["estimate"]))
        cells.append(
            {
                "press": condition[0],
                "budget": condition[1],
                "rows": len(comp),
                "full_accuracy": full_accuracy,
                "compressed_accuracy": compressed_accuracy,
                "readout_accuracy": readout_accuracy,
                "total_change": total_change,
                "readout_change": readout_change,
                "unexplained_residual": residual,
                "signed_flip_mass": flip,
                "compressed_correct": sum(bool(row["correct"]) for row in comp),
                "readout_correct": sum(bool(row["readout_correct"]) for row in readout),
                "truncated": sum(bool(row["truncated"]) for row in comp),
                "truncation_rate": sum(bool(row["truncated"]) for row in comp)
                / len(comp),
                "mean_free_run_tokens": statistics.fmean(
                    [row["n_tokens"] for row in comp]
                ),
                "mean_readout_tokens": statistics.fmean(
                    len(row["generated_token_ids"]) for row in readout
                ),
                "mean_auc_tv": statistics.fmean(row["auc_tv"] for row in replay),
                "mean_summed_tv": statistics.fmean(summed_tv),
                "mean_frac_above_p05": statistics.fmean(
                    row["frac_above"]["p05"] for row in replay
                ),
                "mean_frac_above_p10": statistics.fmean(
                    row["frac_above"]["p10"] for row in replay
                ),
                "onset_nonnull": sum(row["onset_step"] is not None for row in replay),
            }
        )

    summary = {
        "full_cache": {
            "rows": len(full_rows),
            "correct": sum(bool(row["correct"]) for row in full_rows),
            "accuracy": full_accuracy,
            "truncated": sum(bool(row["truncated"]) for row in full_rows),
        },
        "cells": cells,
        "runs": {
            "fullcache": str(fullcache_root),
            "compression": str(compression_root),
            "replay": str(replay_root),
            "readout": str(readout_root),
        },
        "claim_discipline": (
            "Residual is accuracy change unexplained by on-reference readout, "
            "not a measured cause."
        ),
    }
    return summary, cells


def _write_table(path: Path, cells: list[dict[str, Any]]) -> None:
    lines = [
        "# M4 Decomposition Pilot",
        "",
        "Residual is **accuracy change unexplained by on-reference readout**. "
        "It is not a measured cause.",
        "",
        "| Press | Budget | Free acc | Readout acc | Total Δ | Readout Δ | "
        "Residual | Lost mass | Trunc | Mean AUC-TV | Frac TV>0.05 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell['press']} | {cell['budget']} | "
            f"{cell['compressed_accuracy']:.2f} | "
            f"{cell['readout_accuracy']:.2f} | "
            f"{cell['total_change']:+.2f} | "
            f"{cell['readout_change']:+.2f} | "
            f"{cell['unexplained_residual']:+.2f} | "
            f"{cell['signed_flip_mass']['lost']:.2f} | "
            f"{cell['truncation_rate']:.2f} | "
            f"{cell['mean_auc_tv']:.3f} | "
            f"{cell['mean_frac_above_p05']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_svg(path: Path, cells: list[dict[str, Any]]) -> None:
    width, height = 980, 520
    left = 90
    bar_width = 36
    group_gap = 95
    zero_y = 260
    scale = 190
    labels = [f"{cell['press']}-{cell['budget']}" for cell in cells]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="490" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="20" font-weight="bold">M4 accuracy decomposition</text>',
        '<text x="490" y="52" text-anchor="middle" font-family="sans-serif" '
        'font-size="13">Total Δ = readout Δ + unexplained residual</text>',
        f'<line x1="{left - 25}" y1="{zero_y}" x2="{width - 40}" '
        f'y2="{zero_y}" stroke="#222"/>',
    ]
    for tick in [-1.0, -0.5, 0.0]:
        y = zero_y - tick * scale
        svg.append(
            f'<line x1="{left - 30}" y1="{y:.1f}" x2="{width - 40}" '
            'y2="{y:.1f}" stroke="#ddd"/>'
        )
        svg.append(
            f'<text x="{left - 38}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{tick:+.1f}</text>'
        )
    colors = {
        "total_change": "#355c9a",
        "readout_change": "#d17c2f",
        "unexplained_residual": "#4f8f45",
    }
    for index, cell in enumerate(cells):
        x0 = left + index * (3 * bar_width + group_gap)
        for offset, key in enumerate(
            ["total_change", "readout_change", "unexplained_residual"]
        ):
            value = cell[key]
            x = x0 + offset * bar_width
            height_abs = abs(value) * scale
            if value < 0:
                svg.append(
                    f'<rect x="{x}" y="{zero_y}" width="{bar_width - 5}" '
                    f'height="{height_abs:.1f}" fill="{colors[key]}"/>'
                )
            else:
                svg.append(
                    f'<rect x="{x}" y="{zero_y - height_abs:.1f}" '
                    f'width="{bar_width - 5}" height="{height_abs:.1f}" '
                    f'fill="{colors[key]}"/>'
                )
            svg.append(
                f'<text x="{x + 15}" y="{zero_y + height_abs + 16:.1f}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="11">'
                f"{value:+.2f}</text>"
            )
        svg.append(
            f'<text x="{x0 + bar_width}" y="480" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{labels[index]}</text>'
        )
    legend_x = 675
    for i, (key, color) in enumerate(colors.items()):
        y = 83 + i * 22
        svg.append(
            f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" '
            f'fill="{color}"/>'
        )
        svg.append(
            f'<text x="{legend_x + 22}" y="{y}" font-family="sans-serif" '
            f'font-size="13">{key}</text>'
        )
    svg.append("</svg>\n")
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullcache-root", required=True, type=Path)
    parser.add_argument("--compression-root", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--readout-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary, cells = build_report(
        fullcache_root=args.fullcache_root,
        compression_root=args.compression_root,
        replay_root=args.replay_root,
        readout_root=args.readout_root,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_table(args.out / "table.md", cells)
    _write_svg(args.out / "decomposition.svg", cells)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

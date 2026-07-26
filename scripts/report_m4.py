#!/usr/bin/env python3
"""Build the M4 compressed replay/readout decomposition report."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# A condition: the (problem id, seed) pair that indexes one generation everywhere
# in this pipeline. Amendment K makes it the unit the reference population is
# defined over, so it is named rather than spelled out at every call site.
Key = tuple[str, int]


def _problem_bootstrap_ci(
    rows: list[dict[str, Any]],
    value_fn,
    *,
    pid_fn=lambda row: row["pid"],
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Problem-level 95% bootstrap CI (the powered unit is the problem, not the row).

    At pilot scale (5 problems) this CI is intentionally wide — it makes the
    under-power visible in the numbers, not just in a caveat.
    """
    by_pid: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_pid[pid_fn(row)].append(float(value_fn(row)))
    per_problem = [statistics.fmean(v) for v in by_pid.values()]
    if not per_problem:
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(per_problem)
    means = sorted(
        statistics.fmean(rng.choice(per_problem) for _ in range(k))
        for _ in range(n_resamples)
    )
    lo = means[int(0.025 * (n_resamples - 1))]
    hi = means[int(0.975 * (n_resamples - 1))]
    return (round(lo, 4), round(hi, 4))


def _mean_over(keys: list[Key], table: dict[Key, bool]) -> float | None:
    """Accuracy over exactly ``keys``, skipping keys absent from ``table``."""
    vals = [float(table[k]) for k in keys if k in table]
    return statistics.fmean(vals) if vals else None


def _imputation_bounds(
    cell_keys: list[Key], table: dict[Key, bool]
) -> tuple[float, float]:
    """Accuracy over the FULL cell population under both scorings of the undefined.

    Amendment K layer 1: the population is fixed at every condition in the cell.
    Conditions with no defined answer boundary are scored incorrect for the lower
    endpoint and correct for the upper one, so the reported quantity is an
    interval over a fixed population rather than a point on a selected subset.
    """
    n = len(cell_keys)
    if not n:
        return (0.0, 0.0)
    hits = sum(1 for k in cell_keys if table.get(k) is True)
    undefined = sum(1 for k in cell_keys if k not in table)
    return (hits / n, (hits + undefined) / n)


def _endpoint_cis(
    cell_keys: list[Key],
    comp_by: dict[Key, bool],
    ro_by: dict[Key, bool],
    rev_by: dict[Key, bool],
    fc_by: dict[Key, bool],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """Problem-clustered bootstrap 95% CIs for the two residual-bracket endpoints.

    Endpoints per resample of problems (with replacement, k = n cell problems):
    forward = comp_acc - readout_acc; reverse = reverse_acc - fc_acc.

    Amendment K: each endpoint's two terms are computed over the *same*
    (problem, seed) population — the conditions where that endpoint's readout is
    defined. Previously the point estimates used a global FC mean while these CIs
    already restricted the reference to the cell's problems, so a point estimate
    and its own interval were computed on different populations.
    """
    ro_keys = [k for k in cell_keys if k in ro_by]
    rev_keys = [k for k in cell_keys if k in rev_by]

    def _per_pid(keys: list[Key], table: dict[Key, bool]) -> dict[str, float]:
        by_pid: dict[str, list[float]] = defaultdict(list)
        for key in keys:
            if key in table:
                by_pid[key[0]].append(float(table[key]))
        return {pid: statistics.fmean(vals) for pid, vals in by_pid.items()}

    comp_on_ro = _per_pid(ro_keys, comp_by)
    ro_mean = _per_pid(ro_keys, ro_by)
    rev_mean = _per_pid(rev_keys, rev_by)
    fc_on_rev = _per_pid(rev_keys, fc_by)

    cell_pids = sorted({k[0] for k in cell_keys})
    rng = random.Random(seed)
    k = len(cell_pids)
    forward_samples: list[float] = []
    reverse_samples: list[float] = []
    for _ in range(n_resamples):
        draw = [cell_pids[rng.randrange(k)] for _ in range(k)]
        fwd = [p for p in draw if p in ro_mean]
        if fwd:
            forward_samples.append(
                statistics.fmean(comp_on_ro[p] for p in fwd)
                - statistics.fmean(ro_mean[p] for p in fwd)
            )
        rev = [p for p in draw if p in rev_mean]
        if rev:
            reverse_samples.append(
                statistics.fmean(rev_mean[p] for p in rev)
                - statistics.fmean(fc_on_rev[p] for p in rev)
            )

    def _ci(samples: list[float]) -> tuple[float, float]:
        ordered = sorted(samples)
        n = len(ordered)
        return (
            round(ordered[int(0.025 * (n - 1))], 4),
            round(ordered[int(0.975 * (n - 1))], 4),
        )

    forward_ci = _ci(forward_samples) if forward_samples else (0.0, 0.0)
    reverse_ci = _ci(reverse_samples) if reverse_samples else None
    return forward_ci, reverse_ci


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


def _reference_cell_from_readout(row: dict[str, Any]) -> tuple[str, str]:
    """Cell of a REVERSE readout: the (press, budget) of the *reference* trace it read
    out (the compressed free-run trace), not the readout condition (full cache)."""
    reference = row["reference_key"]
    return str(reference[3]), str(reference[4])


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
    readout_root: Path,
    replay_root: Path | None = None,
    reverse_readout_root: Path | None = None,
    bench: str = "math500",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_rows = _load_rows(fullcache_root, bench)
    compression_rows = _load_rows(compression_root, bench)
    # Replay (per-step TV profiles) is OPTIONAL: the accuracy-scale decomposition
    # (total/readout/residual + CIs, the Amendment-F call) needs only fullcache,
    # compression, and readout. TV columns are added only when a replay run is given.
    replay_rows = _load_rows(replay_root, bench) if replay_root is not None else []
    # Readout may be split across per-shard run dirs (e.g. seed-sharded -g0/-g1); read
    # every supplied root so the cells aggregate all seeds.
    readout_root_list = (
        readout_root if isinstance(readout_root, list) else [readout_root]
    )
    readout_rows = [
        row
        for root in readout_root_list
        for path in sorted((root / "readout").rglob("*.jsonl"))
        for row in _jsonl(path)
    ]
    if not (full_rows and compression_rows and readout_rows):
        raise ValueError("fullcache, compression, and readout inputs must contain rows")

    # Optional reverse readout (M7 first rung): the compressed free-run trace read out
    # under the FULL cache. Forward + reverse readout bracket the on-policy residual.
    reverse_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if reverse_readout_root is not None:
        for path in sorted((reverse_readout_root / "readout").rglob("*.jsonl")):
            for row in _jsonl(path):
                reverse_by_cell[_reference_cell_from_readout(row)].append(row)

    reference_correct = {_run_key(row): bool(row["correct"]) for row in full_rows}
    full_accuracy = _accuracy(full_rows)

    # Pin ONE canonical full-cache reference: every total_change, readout_change,
    # and flip_mass MUST be measured against this exact bank's (pid, seed) set.
    # (Reviewer #4: two FC banks with different accuracies were floating around.)
    ref_keys = set(reference_correct)
    comp_keys = {_run_key(row) for row in compression_rows}
    readout_keys = {_reference_key_from_readout(row) for row in readout_rows}
    if not comp_keys <= ref_keys:
        raise ValueError(
            "compression rows reference (pid,seed) absent from the FC bank: "
            f"{sorted(comp_keys - ref_keys)[:3]}"
        )
    if not readout_keys <= ref_keys:
        raise ValueError(
            "readout rows reference (pid,seed) absent from the FC bank: "
            f"{sorted(readout_keys - ref_keys)[:3]}"
        )

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
        readout_defined = [r for r in readout if r.get("readout_correct") is not None]
        n_readout_undefined = len(readout) - len(readout_defined)
        reverse_rows = reverse_by_cell.get(condition, [])
        reverse = [r for r in reverse_rows if r.get("readout_correct") is not None]

        # --- Amendment K: fix the reference population to the cell's conditions ---
        # The cell population is every (pid, seed) it generated. Readout is defined
        # only where the reference trace has an answer boundary, and reverse readout
        # only where the *compressed* trace has one -- both are selected, non-random
        # subsets (a trace has a boundary iff it terminated). Subtracting a global
        # bank mean from an accuracy measured on such a subset is a population
        # mismatch, not an estimand. Every change below is now computed against a
        # reference on the same population as the quantity it is subtracted from.
        cell_keys = sorted({_run_key(row) for row in comp})
        comp_by = {_run_key(row): bool(row["correct"]) for row in comp}
        ro_by = {
            _reference_key_from_readout(row): bool(row["readout_correct"])
            for row in readout_defined
        }
        rev_by = {
            _reference_key_from_readout(row): bool(row["readout_correct"])
            for row in reverse
        }
        fc_cell_accuracy = _mean_over(cell_keys, reference_correct)

        # Layer 1 (primary): fixed population, imputation bounds on the undefined.
        ro_bounds = _imputation_bounds(cell_keys, ro_by)
        ro_change_bounds = [v - fc_cell_accuracy for v in ro_bounds]
        # residual = total - readout = comp_acc - readout_acc (the FC term cancels),
        # so the wider readout bound gives the more negative residual endpoint.
        residual_bounds = [
            compressed_accuracy - ro_bounds[1],
            compressed_accuracy - ro_bounds[0],
        ]
        if reverse_rows:
            rev_bounds = _imputation_bounds(cell_keys, rev_by)
            rev_change_bounds = [v - fc_cell_accuracy for v in rev_bounds]
            bracket_bounds = [
                min(residual_bounds[0], rev_change_bounds[0]),
                max(residual_bounds[1], rev_change_bounds[1]),
            ]
        else:
            rev_bounds = rev_change_bounds = bracket_bounds = None

        # Layer 2 (secondary, complete-case): the same quantities restricted to the
        # defined subset, with the reference restricted to that identical subset.
        # Post-treatment conditioning -- reported as sensitivity, never as primary.
        ro_keys = [k for k in cell_keys if k in ro_by]
        rev_keys = [k for k in cell_keys if k in rev_by]
        readout_accuracy = _mean_over(ro_keys, ro_by) or 0.0
        fc_on_readout = _mean_over(ro_keys, reference_correct)
        comp_on_readout = _mean_over(ro_keys, comp_by)
        readout_change = (
            readout_accuracy - fc_on_readout if fc_on_readout is not None else None
        )
        residual = (
            comp_on_readout - readout_accuracy if comp_on_readout is not None else None
        )
        reverse_readout_accuracy = _mean_over(rev_keys, rev_by)
        fc_on_reverse = _mean_over(rev_keys, reference_correct)
        if (
            reverse_readout_accuracy is not None
            and fc_on_reverse is not None
            and residual is not None
        ):
            # Reverse endpoint is acc_rev - acc_fc on the reverse-defined conditions:
            # clean readout of the compressed trace estimates the residual on the
            # compressed model's own states (Amendment H).
            reverse_change = reverse_readout_accuracy - fc_on_reverse
            endpoints = (residual, reverse_change)
            residual_bracket = [min(endpoints), max(endpoints)]
        else:
            reverse_change = None
            residual_bracket = None
        total_change = compressed_accuracy - fc_cell_accuracy

        # Problem-clustered bootstrap CIs on the two bracket endpoints, now on the
        # same populations as the complete-case point estimates they accompany.
        forward_endpoint_ci, reverse_endpoint_ci = _endpoint_cis(
            cell_keys, comp_by, ro_by, rev_by, reference_correct
        )
        flip = _signed_flip_mass(reference_correct, comp)
        summed_tv = []
        for row in replay:
            profile = json.loads(
                (replay_root / row["tv_profile_path"]).read_text(encoding="utf-8")
            )
            summed_tv.append(sum(float(value) for value in profile["estimate"]))
        # TV-profile columns are None when no replay run was supplied (deferred).
        mean_auc_tv = statistics.fmean(r["auc_tv"] for r in replay) if replay else None
        mean_summed_tv = statistics.fmean(summed_tv) if summed_tv else None
        mean_frac_p05 = (
            statistics.fmean(r["frac_above"]["p05"] for r in replay) if replay else None
        )
        mean_frac_p10 = (
            statistics.fmean(r["frac_above"]["p10"] for r in replay) if replay else None
        )
        onset_nonnull = (
            sum(r["onset_step"] is not None for r in replay) if replay else None
        )
        cells.append(
            {
                "press": condition[0],
                "budget": condition[1],
                "rows": len(comp),
                # Amendment K: the cell's own reference population. `full_accuracy`
                # is Acc_FC on this cell's (pid, seed) conditions -- NOT the global
                # bank mean, which is kept only for provenance.
                "full_accuracy": fc_cell_accuracy,
                "full_accuracy_all_rows": full_accuracy,
                "reference_population": {
                    "n_conditions": len(cell_keys),
                    "fc_accuracy_cell": fc_cell_accuracy,
                    "fc_accuracy_readout_defined": fc_on_readout,
                    "fc_accuracy_reverse_defined": fc_on_reverse,
                    "n_readout_defined": len(ro_keys),
                    "n_reverse_defined": len(rev_keys),
                },
                # Layer 1 (primary): fixed population, both imputations.
                "primary_bounds": {
                    "readout_accuracy": list(ro_bounds),
                    "readout_change": list(ro_change_bounds),
                    "residual": list(residual_bounds),
                    "reverse_readout_accuracy": (
                        list(rev_bounds) if rev_bounds else None
                    ),
                    "reverse_change": (
                        list(rev_change_bounds) if rev_change_bounds else None
                    ),
                    "residual_bracket": (
                        list(bracket_bounds) if bracket_bounds else None
                    ),
                },
                "compressed_accuracy": compressed_accuracy,
                "compressed_accuracy_ci95": _problem_bootstrap_ci(
                    comp, lambda r: bool(r["correct"])
                ),
                "readout_accuracy": readout_accuracy,
                "readout_accuracy_ci95": _problem_bootstrap_ci(
                    readout_defined,
                    lambda r: bool(r["readout_correct"]),
                    pid_fn=lambda r: r["reference_key"][1],
                ),
                "n_readout_undefined": n_readout_undefined,
                "n_problems": len({r["pid"] for r in comp}),
                "total_change": total_change,
                "readout_change": readout_change,
                "unexplained_residual": residual,
                "reverse_readout_accuracy": reverse_readout_accuracy,
                "reverse_readout_change": reverse_change,
                "residual_bracket": residual_bracket,
                "unexplained_residual_ci95": list(forward_endpoint_ci),
                "reverse_residual_ci95": (
                    list(reverse_endpoint_ci) if reverse_endpoint_ci else None
                ),
                "n_reverse_readout": len(reverse),
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
                "mean_auc_tv": mean_auc_tv,
                "mean_summed_tv": mean_summed_tv,
                "mean_frac_above_p05": mean_frac_p05,
                "mean_frac_above_p10": mean_frac_p10,
                "onset_nonnull": onset_nonnull,
            }
        )

    summary = {
        # Stamped so a pre-Amendment-K artifact can never be silently read as a
        # current one: every number below is on the cell's own reference
        # population, and the two conventions are not comparable.
        "reference_convention": "amendment-k-cell-population",
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
            "replay": str(replay_root) if replay_root else None,
            "readout": [str(r) for r in readout_root_list],
            "reverse_readout": (
                str(reverse_readout_root) if reverse_readout_root else None
            ),
        },
        "claim_discipline": (
            "Residual is accuracy change unexplained by on-reference readout, "
            "not a measured cause. Top-level cell values are COMPLETE-CASE "
            "(Amendment K layer 2): each change is measured against the "
            "reference restricted to the same conditions. Boundary detection is "
            "post-treatment, so the primary estimand is `primary_bounds` "
            "(layer 1: the full cell population, undefined rows imputed both "
            "ways). Never subtract a global bank mean from a subset accuracy."
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
        auc = "n/a" if cell["mean_auc_tv"] is None else f"{cell['mean_auc_tv']:.3f}"
        frac = (
            "n/a"
            if cell["mean_frac_above_p05"] is None
            else f"{cell['mean_frac_above_p05']:.2f}"
        )
        lines.append(
            f"| {cell['press']} | {cell['budget']} | "
            f"{cell['compressed_accuracy']:.2f} | "
            f"{cell['readout_accuracy']:.2f} | "
            f"{cell['total_change']:+.2f} | "
            f"{cell['readout_change']:+.2f} | "
            f"{cell['unexplained_residual']:+.2f} | "
            f"{cell['signed_flip_mass']['lost']:.2f} | "
            f"{cell['truncation_rate']:.2f} | "
            f"{auc} | "
            f"{frac} |"
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
            f'y2="{y:.1f}" stroke="#ddd"/>'
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
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=None,
        help="optional TV-profile replay run; omit for the decomposition-only read "
        "(total/readout/residual + CIs) before the deferred replay has been run",
    )
    parser.add_argument(
        "--readout-root",
        required=True,
        type=Path,
        nargs="+",
        help="one or more readout run dirs (seed-sharded shards are passed together)",
    )
    parser.add_argument(
        "--reverse-readout-root",
        type=Path,
        default=None,
        help="optional reverse-readout run (compressed trace read out under full "
        "cache) to report the residual bracket (M7 first rung)",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--bench",
        default="math500",
        help="benchmark subtree to read rows from (e.g. aime24 for the AIME cells)",
    )
    args = parser.parse_args()
    summary, cells = build_report(
        fullcache_root=args.fullcache_root,
        compression_root=args.compression_root,
        replay_root=args.replay_root,
        readout_root=args.readout_root,
        reverse_readout_root=args.reverse_readout_root,
        bench=args.bench,
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

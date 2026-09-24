#!/usr/bin/env python3
"""Amendment Q.2 read-out grid: window TV per onset arm, and an exposure audit.

``report_onset_surface.py`` owns the pre-registered verdict (slopes on NOMINAL
exposure ``h = t - t0``). This reader adds the two things that verdict needs to
be read correctly, over the same profiles, positions and window:

1. The measured grid behind every slope: window-mean per-step TV for each
   (budget, read position, onset arm), problem-clustered CI, trace and problem
   counts, on the full panel and on the balanced (fixed-population) panel.
2. An occupancy audit. ``compression_start_step`` only ENABLES the press;
   nothing is evicted until prompt + generated tokens exceed the budget at a
   prune step. The ACTUAL eviction onset is measured per trace as the first
   step with nonzero TV (``kvcache.metrics.onset_surface.eviction_onset``), and
   actual exposure ``t - onset`` is reported beside the nominal one. A slope on
   actual exposure is computed as a DIAGNOSTIC (not pre-registered).

It also enforces the instrument invariant: no trace may diverge before its
arm's nominal onset. A violation stops the read.

    python scripts/report_onset_grid.py --runs-root <runs> \
        --arms 0=replay-onset-t0 512=replay-onset-t512 2048=replay-onset-t2048 \
        --out q2_onset_grid.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from report_onset_surface import N_POSITIONS, STEP, WINDOW, _load_arm

from kvcache.metrics.onset_surface import (
    eviction_onset,
    exposure_slope,
    matched_positions,
    window_divergence,
)
from kvcache.stats import cluster_bootstrap_mean

FC_RUN = "math500-fc-powered-merged"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _prompt_lengths(runs_root: Path) -> dict[str, int]:
    """Prompt token count per problem, from the reference bank's sidecars."""
    out: dict[str, int] = {}
    pattern = str(runs_root / FC_RUN / "token_ids" / "**" / "*.jsonl")
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                n = len(r["prompt_token_ids"])
                if out.setdefault(r["pid"], n) != n:
                    raise SystemExit(f"prompt length differs across seeds: {r['pid']}")
    return out


def _summ(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    return {"mean": statistics.fmean(xs), "min": min(xs), "max": max(xs)}


def _panel(
    keys: set,
    loaded: dict[int, dict],
    onsets: list[int],
    onset_of: dict,
    prompt: dict[str, int],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    # tv[(budget, pos)][t0][pid] -> per-trace window means (presses pooled, as
    # in the pre-registered read); by_press splits them for the figure.
    tv: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tv_press: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    act: dict = defaultdict(lambda: defaultdict(list))
    act_slopes: dict = defaultdict(lambda: defaultdict(list))
    prompts: dict = defaultdict(list)
    for key in sorted(keys):
        press, budget, _s, _stem = key
        trace_len = min(len(loaded[t0][key]["estimate"]) for t0 in onsets)
        pid = loaded[onsets[0]][key]["reference_key"][1]
        for pos in matched_positions(
            onsets, WINDOW, trace_len, step=STEP, count=N_POSITIONS
        ):
            ds = [
                window_divergence(loaded[t0][key]["estimate"], pos, WINDOW)
                for t0 in onsets
            ]
            if any(d is None for d in ds):
                continue
            cell = (budget, pos)
            prompts[cell].append(prompt[pid])
            xs_act = []
            for t0, d in zip(onsets, ds, strict=True):
                tv[cell][t0][pid].append(d)
                tv_press[cell][t0][press][pid].append(d)
                h = pos - onset_of[(t0, key)]
                act[cell][t0].append(float(h))
                xs_act.append(float(h))
            s = exposure_slope(xs_act, ds)
            # None = every arm had the same ACTUAL exposure: the design is
            # degenerate for this trace, not a zero slope. Counted, not dropped.
            act_slopes[cell]["__degenerate__" if s is None else pid].append(
                0.0 if s is None else s
            )
    cells: dict[str, Any] = {}
    for (budget, pos), by_arm in sorted(
        tv.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])
    ):
        cell = (budget, pos)
        degenerate = len(act_slopes[cell].pop("__degenerate__", []))
        arms = {}
        for t0 in onsets:
            by_pid = by_arm[t0]
            arms[str(t0)] = {
                "nominal_exposure": pos - t0,
                "actual_exposure": _summ(act[cell][t0]),
                "window_tv": cluster_bootstrap_mean(
                    by_pid, n_resamples=n_resamples, seed=seed
                ),
                "by_press": {
                    press: cluster_bootstrap_mean(
                        pp, n_resamples=n_resamples, seed=seed
                    )
                    for press, pp in sorted(tv_press[cell][t0].items())
                },
            }
        mean_prompt = statistics.fmean(prompts[cell])
        ent: dict[str, Any] = {
            "budget": int(budget),
            "position": int(pos),
            "n_problems": len(by_arm[onsets[0]]),
            "n_traces": sum(len(v) for v in by_arm[onsets[0]].values()),
            "mean_prompt_tokens": mean_prompt,
            "realized_ratio_decode_only": int(budget) / pos,
            "realized_ratio": int(budget) / (mean_prompt + pos),
            "arms": arms,
            "actual_exposure_slope_degenerate_traces": degenerate,
        }
        if act_slopes[cell]:
            st = cluster_bootstrap_mean(
                act_slopes[cell], n_resamples=n_resamples, seed=seed
            )
            lo, hi = st["ci95"]
            ent["actual_exposure_slope"] = st
            ent["actual_exposure_slope_excludes_zero"] = bool(lo > 0 or hi < 0)
        else:
            ent["actual_exposure_slope"] = None
        cells[f"b{budget}@{pos}"] = ent
    return cells


def main() -> int:
    a = _parser().parse_args()
    arms: dict[int, str] = {}
    for spec in a.arms:
        k, _, v = spec.partition("=")
        arms[int(k)] = v
    onsets = sorted(arms)
    loaded = {t0: _load_arm(a.runs_root, run) for t0, run in arms.items()}
    if a.dry_run:
        print(
            json.dumps(
                {
                    "command": "onset-grid",
                    "arms": arms,
                    "profiles_per_arm": {t0: len(v) for t0, v in loaded.items()},
                    "gpu_required": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    shared = set.intersection(*(set(v) for v in loaded.values()))
    prompt = _prompt_lengths(a.runs_root)

    onset_of: dict = {}
    audit: dict = defaultdict(lambda: defaultdict(list))
    violations, never = [], []
    for t0 in onsets:
        for key in sorted(shared):
            payload = loaded[t0][key]
            o = eviction_onset(payload["estimate"])
            if o is None:
                never.append([t0, *key])
                continue
            if o < t0:
                violations.append([t0, *key, o])
            onset_of[(t0, key)] = o
            press, budget = key[0], key[1]
            pid = payload["reference_key"][1]
            audit[f"{press}-{budget}"][str(t0)].append(
                {
                    "eviction_onset": o,
                    "recorded_onset_step": payload.get("onset_step"),
                    "prompt_tokens": prompt[pid],
                }
            )
    if violations:
        raise SystemExit(
            f"STOP: {len(violations)} traces diverge before their nominal onset "
            f"(instrument bug, not a result): {violations[:5]}"
        )
    usable = {k for k in shared if all((t0, k) in onset_of for t0 in onsets)}

    balanced = {
        key
        for key in usable
        if len(
            matched_positions(
                onsets,
                WINDOW,
                min(len(loaded[t0][key]["estimate"]) for t0 in onsets),
                step=STEP,
                count=N_POSITIONS,
            )
        )
        == N_POSITIONS
    }
    occupancy = {}
    for cell, by_t0 in sorted(audit.items()):
        occupancy[cell] = {}
        for t0, rows in sorted(by_t0.items(), key=lambda kv: int(kv[0])):
            ev = [r["eviction_onset"] for r in rows]
            fill = [int(cell.split("-")[1]) - r["prompt_tokens"] for r in rows]
            occupancy[cell][t0] = {
                "n_traces": len(rows),
                "nominal_onset": int(t0),
                "eviction_onset": _summ([float(x) for x in ev]),
                "eviction_minus_nominal": _summ([float(x - int(t0)) for x in ev]),
                "decode_step_cache_reaches_budget": _summ([float(x) for x in fill]),
                "recorded_onset_step": _summ(
                    [float(r["recorded_onset_step"]) for r in rows]
                ),
            }

    out: dict[str, Any] = {
        "amendment": "Q.2 read-out grid + occupancy audit (Amendment R)",
        "arms": {str(k): v for k, v in arms.items()},
        "window": WINDOW,
        "position_step": STEP,
        "n_shared_traces": len(shared),
        "n_usable_traces": len(usable),
        "never_diverged": never,
        "pre_onset_violations": 0,
        "eviction_onset_definition": (
            "first decode step with per-step TV > 0; before any eviction the "
            "compressed and reference passes are identical so TV is exactly 0"
        ),
        "realized_ratio_note": (
            "realized_ratio = budget / (mean prompt + read position), the project "
            "convention; realized_ratio_decode_only = budget / position is what "
            "onset_surface_q2.json reports"
        ),
        "occupancy": occupancy,
        "full_panel": _panel(
            usable, loaded, onsets, onset_of, prompt, a.n_resamples, a.seed
        ),
        "balanced_panel": {
            "n_traces": len(balanced),
            "note": "DIAGNOSTIC ONLY; the pre-registered verdict is the full panel",
            "cells": _panel(
                balanced, loaded, onsets, onset_of, prompt, a.n_resamples, a.seed
            ),
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")
    for cell, by_t0 in occupancy.items():
        for t0, o in by_t0.items():
            print(
                f"{cell:>18} t0={t0:>4}  eviction onset "
                f"mean={o['eviction_onset']['mean']:.0f} "
                f"[{o['eviction_onset']['min']:.0f},{o['eviction_onset']['max']:.0f}]"
                f"  recorded onset_step mean={o['recorded_onset_step']['mean']:.0f}"
            )
    for name, c in out["full_panel"].items():
        parts = []
        for t0, arm in c["arms"].items():
            w = arm["window_tv"]
            parts.append(
                f"t0={t0}: h_nom={arm['nominal_exposure']} "
                f"h_act={arm['actual_exposure']['mean']:.0f} "
                f"tv={w['mean']:.4f}[{w['ci95'][0]:.4f},{w['ci95'][1]:.4f}]"
            )
        print(
            f"{name:>12} pids={c['n_problems']} traces={c['n_traces']} | "
            + " | ".join(parts)
        )
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

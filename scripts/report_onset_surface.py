#!/usr/bin/env python3
"""Amendment Q.2: does divergence depend on EXPOSURE at a matched dose?

``r = B/t`` couples the realised compression ratio to context length, so no
slice of ``(B, t)`` isolates how long compression has been running. Compression
onset is not in that identity: with the press inactive until decode step ``t0``,
two arms read at the same absolute position ``t`` and the same budget ``B`` have
the same realised ratio while differing in exposure ``h = t - t0``.

For each (budget, read position) the three onset arms give three (exposure,
divergence) points on the SAME problems and seeds, and the slope of divergence
on exposure is the pre-registered quantity. Slopes are never pooled across read
positions or budgets, because the ratio is matched only at equal ``t``.

    python scripts/report_onset_surface.py --runs-root <runs> \
        --arms t0=replay-onset-t0 t512=replay-onset-t512 t2048=replay-onset-t2048 \
        --out onset_surface.json
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

from kvcache.metrics.onset_surface import (  # noqa: E402
    exposure_slope,
    matched_positions,
    window_divergence,
)
from kvcache.replay import is_profile_file  # noqa: E402
from kvcache.stats import cluster_bootstrap_mean  # noqa: E402

CELL = re.compile(r"^(?P<press>[a-z0-9]+)-(?P<budget>\d+)-s(?P<seed>\d+)$")
WINDOW = 256  # read-window width, fixed here before the cell is read
STEP = 1024  # spacing between read positions
N_POSITIONS = 3


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument(
        "--arms",
        nargs="+",
        required=True,
        help="onset=run pairs, e.g. 0=replay-onset-t0 512=replay-onset-t512",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    # The orchestrator appends these to every launch; accept and ignore rather
    # than exiting rc=2 before writing a row (landmine 1).
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _load_arm(root: Path, run: str) -> dict[tuple[str, str, str, str], dict]:
    out: dict[tuple[str, str, str, str], dict] = {}
    base = root / run / "profiles"
    if not base.exists():
        return out
    for path in sorted(base.rglob("*.json")):
        if not is_profile_file(path):
            continue
        m = CELL.match(path.parent.name)
        if not m:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("estimate"):
            continue
        key = (m.group("press"), m.group("budget"), m.group("seed"), path.stem)
        out[key] = payload
    return out


def main() -> int:
    a = _parser().parse_args()
    arms: dict[int, str] = {}
    for spec in a.arms:
        k, _, v = spec.partition("=")
        arms[int(k)] = v
    if len(arms) < 2:
        raise SystemExit("need at least two onset arms to estimate a slope")

    loaded = {t0: _load_arm(a.runs_root, run) for t0, run in arms.items()}
    if a.dry_run:
        print(
            json.dumps(
                {
                    "command": "onset-surface",
                    "arms": arms,
                    "profiles_per_arm": {t0: len(v) for t0, v in loaded.items()},
                    "window": WINDOW,
                    "gpu_required": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    # Only traces present in EVERY arm: an unmatched trace would put problem
    # identity into a slope that is supposed to carry exposure alone.
    shared = set.intersection(*(set(v) for v in loaded.values()))
    onsets = sorted(arms)
    # slopes[(budget, position)][pid] -> list of per-(press, seed) slopes
    slopes: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    skipped_short = 0
    for key in sorted(shared):
        press, budget, _seed, pid = key
        trace_len = min(len(loaded[t0][key]["estimate"]) for t0 in onsets)
        positions = matched_positions(
            onsets, WINDOW, trace_len, step=STEP, count=N_POSITIONS
        )
        if not positions:
            skipped_short += 1
            continue
        for pos in positions:
            xs, ys = [], []
            for t0 in onsets:
                d = window_divergence(loaded[t0][key]["estimate"], pos, WINDOW)
                if d is None:
                    break
                xs.append(float(pos - t0))
                ys.append(d)
            else:
                s = exposure_slope(xs, ys)
                if s is not None:
                    slopes[(budget, pos)][pid].append(s)

    out: dict[str, Any] = {
        "amendment": "Q.2",
        "arms": {str(k): v for k, v in arms.items()},
        "window": WINDOW,
        "position_step": STEP,
        "n_shared_traces": len(shared),
        "skipped_trace_too_short": skipped_short,
        "quantity": (
            "OLS slope of window-mean per-step TV on exposure h = t - t0, within "
            "one (budget, read position) cell so the realised ratio r = B/t is "
            "matched by construction. Slopes are NOT pooled across positions or "
            "budgets."
        ),
        "per_cell": {},
    }
    for (budget, pos), by_pid in sorted(slopes.items()):
        stat = cluster_bootstrap_mean(by_pid, n_resamples=a.n_resamples, seed=a.seed)
        lo, hi = stat["ci95"]
        out["per_cell"][f"b{budget}@{pos}"] = {
            "budget": int(budget),
            "position": int(pos),
            "realized_ratio": int(budget) / int(pos),
            "n_problems": len(by_pid),
            "n_observations": sum(len(v) for v in by_pid.values()),
            "slope": stat,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "sign": "positive" if stat["mean"] > 0 else "negative",
        }

    cells = out["per_cell"]
    if not cells:
        out["verdict"] = "NO READABLE CELL"
    else:
        excl = [c for c in cells.values() if c["excludes_zero"]]
        if not excl:
            out["verdict"] = "RATIO-ONLY -> exposure contributes nothing measurable"
        elif len(excl) == len(cells) and len({c["sign"] for c in excl}) == 1:
            out["verdict"] = "HORIZON-PRESENT -> exposure separable from dose"
        else:
            out["verdict"] = "MIXED -> report the interaction, claim no main effect"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")
    for name, c in out["per_cell"].items():
        s = c["slope"]
        print(
            f"{name:>14}  r={c['realized_ratio']:.4f}  pids={c['n_problems']:<3} "
            f"slope={s['mean']:+.3e} {[f'{x:+.2e}' for x in s['ci95']]} "
            f"excl0={c['excludes_zero']}"
        )
    print(f"VERDICT: {out['verdict']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

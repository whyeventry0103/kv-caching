#!/usr/bin/env python3
"""Amendment R.9 read of the capture cells: Q.3 (Task 2) and Q.6 (Task 3).

Registered estimands (R.4, R.6, arithmetic fixed in R.9):

* Q.3: pooled GQA collateral per condition (sum n_collateral / sum n_eligible
  over every (layer, group, event)); paired difference R-KV minus
  ObservedAttentionPress ("h2o") per (problem, seed); problem-clustered 95% CI.
  Negative excluding zero supports the Q.3 ordering; including zero is
  inconclusive; positive excluding zero reverses it.
* Q.6: omega(Random) - omega(StreamingLLM), paired per (problem, seed);
  PREDICTED (positive, CI excludes 0) / NULL / REVERSED. Exploratory.

Descriptive (no test): per-policy M, M_event, Delta, wanted_any_head,
collateral, and the Delta/M ratio at the trigger event (Q.3 cross-instrument
prediction). Rows failing a recorded control (layer-0 Delta above 1e-5,
non-finite values) are counted and excluded; if any exist the output says so
first. A cell with fewer rows than expected is PARTIAL and its contrasts are
reported with their n, never silently.

    python scripts/report_capture.py --runs-root <runs> \\
        --run capture-b256 --out capture_b256.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.capture import LAYER0_DELTA_TOL
from kvcache.stats import cluster_bootstrap_mean

POLICIES = ("random", "streamingllm", "h2o", "rkv")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--expected-per-policy", type=int, default=100)
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((run_dir / "math500").rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def trigger_ratio(row: dict[str, Any]) -> float | None:
    """Delta/M over the trigger event's (layer, group) slice (R.9)."""
    sl = (row.get("slices") or {}).get("trigger")
    if not sl or sl.get("M") is None:
        return None
    ms = [v for layer in sl["M"] if layer for v in layer if v is not None]
    ds = [v for layer in sl["Delta"] if layer for v in layer if v is not None]
    if not ms or not ds or statistics.fmean(ms) <= 0:
        return None
    return statistics.fmean(ds) / statistics.fmean(ms)


def paired_contrast(
    by_policy: dict[str, dict[tuple[str, int], float]],
    a: str,
    b: str,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Mean of (a - b) over shared (problem, seed), clustered by problem."""
    shared = sorted(set(by_policy.get(a, {})) & set(by_policy.get(b, {})))
    per_pid: dict[str, list[float]] = defaultdict(list)
    for key in shared:
        per_pid[key[0]].append(by_policy[a][key] - by_policy[b][key])
    if not per_pid:
        return {"n_pairs": 0, "status": "NO SHARED CONDITIONS"}
    stat = cluster_bootstrap_mean(per_pid, n_resamples=n_resamples, seed=seed)
    lo, hi = stat["ci95"]
    return {
        "contrast": f"{a} - {b}",
        "n_pairs": len(shared),
        "n_problems": len(per_pid),
        **stat,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main() -> int:
    a = _parser().parse_args()
    run_dir = a.runs_root / a.run
    if a.dry_run:
        print(
            json.dumps(
                {"command": "capture-read", "run": a.run, "exists": run_dir.is_dir()}
            )
        )
        return 0
    rows = [r for r in _rows(run_dir) if r["press"] in POLICIES]
    bad = [
        r
        for r in rows
        if r["checks"]["n_nonfinite"]
        or r["checks"]["layer0_delta_max"] > LAYER0_DELTA_TOL
    ]
    good = [r for r in rows if r not in bad]
    counts = {p: sum(r["press"] == p for r in rows) for p in POLICIES}
    budgets = sorted({str(r["budget"]) for r in rows})

    metrics: dict[str, dict[str, dict[tuple[str, int], float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in good:
        key = (r["pid"], int(r["seed"]))
        for name, value in (
            ("collateral", r.get("collateral_pooled")),
            ("wanted_any_head", r.get("wanted_any_head_pooled")),
            ("M", r.get("M_mean")),
            ("M_event", r.get("M_event_mean")),
            ("Delta", r.get("Delta_mean")),
            ("trigger_delta_over_M", trigger_ratio(r)),
            ("omega", (r.get("latent") or {}).get("omega_trace")),
        ):
            if value is not None:
                metrics[name][r["press"]][key] = float(value)

    def summary(name: str, policy: str) -> dict[str, Any] | None:
        vals = metrics[name].get(policy, {})
        if not vals:
            return None
        per_pid: dict[str, list[float]] = defaultdict(list)
        for (pid, _s), v in vals.items():
            per_pid[pid].append(v)
        return cluster_bootstrap_mean(per_pid, n_resamples=a.n_resamples, seed=a.seed)

    q3 = paired_contrast(
        metrics["collateral"], "rkv", "h2o", n_resamples=a.n_resamples, seed=a.seed
    )
    if q3.get("n_pairs"):
        lo, hi = q3["ci95"]
        q3["verdict"] = (
            "SUPPORTS Q.3 ORDERING (R-KV lower)"
            if hi < 0
            else "REVERSED (R-KV higher)" if lo > 0 else "INCONCLUSIVE"
        )
    q6 = paired_contrast(
        metrics["omega"],
        "random",
        "streamingllm",
        n_resamples=a.n_resamples,
        seed=a.seed,
    )
    if q6.get("n_pairs"):
        lo, hi = q6["ci95"]
        q6["verdict"] = "PREDICTED" if lo > 0 else "REVERSED" if hi < 0 else "NULL"
    out = {
        "amendment": "R.4 / R.6 / R.9",
        "run": a.run,
        "budgets": budgets,
        "rows_per_policy": counts,
        "complete": all(counts[p] >= a.expected_per_policy for p in POLICIES),
        "n_rows_failing_controls": len(bad),
        "failing_rows": [
            {"pid": r["pid"], "seed": r["seed"], "press": r["press"], **r["checks"]}
            for r in bad
        ],
        "q3_registered_contrast": q3,
        "q6_registered_contrast": q6,
        "descriptive": {
            name: {p: summary(name, p) for p in POLICIES}
            for name in (
                "collateral",
                "wanted_any_head",
                "M",
                "M_event",
                "Delta",
                "trigger_delta_over_M",
                "omega",
            )
        },
        "q6_note": "exploratory; no accuracy-scale claim (Q.6 standing limit)",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"rows per policy {counts}; failing controls {len(bad)}")
    for label, c in (("Q.3 collateral R-KV - H2O", q3), ("Q.6 omega R - SLM", q6)):
        if c.get("n_pairs"):
            print(
                f"{label}: {c['mean']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] "
                f"n={c['n_pairs']} -> {c['verdict']}"
            )
        else:
            print(f"{label}: {c['status']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

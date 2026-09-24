#!/usr/bin/env python3
"""Amendment R.5 read: the P.5 none-minus-arm effect per lag, then the sweep verdict.

Per policy and lag, ``kvcache.metrics.restoration.restoration_read`` over the
eligible traces (each row already holds its 200-forward window means, anchored
at the reinsertion step), then ``classify_latency_sweep`` over the matched
arm's (lag, effect, CI): DECAY / FLAT-NULL / NON-MONOTONE. Requested and
achieved lags and ineligible counts are reported alongside.

    python scripts/report_exact_lag.py --runs-root <runs> --run exact-lag-r5 \\
        --out exact_lag.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.coverage import COMPLETE, check_coverage, expected_keys, design
from kvcache.metrics.restoration import (
    classify_latency_sweep,
    classify_restoration,
    restoration_read,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--design", type=Path, default=None)
    p.add_argument("--final", action="store_true")
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def read_rows(rows: list[dict[str, Any]], *, n_resamples: int, seed: int) -> dict:
    out: dict[str, Any] = {}
    for press in sorted({r["press"] for r in rows}):
        rs = [r for r in rows if r["press"] == press]
        lags = sorted({int(k) for r in rs for k in r.get("lags", {})})
        per_lag: dict[str, Any] = {}
        points = []
        for lag in lags:
            ok = [
                r
                for r in rs
                if r.get("status") == "READ" and r["lags"][str(lag)].get("eligible")
            ]
            if not ok:
                per_lag[str(lag)] = {"n_traces": 0}
                continue
            arms = {
                arm: {f"{r['pid']}|{r['seed']}": [r["lags"][str(lag)][arm]] for r in ok}
                for arm in ("none", "matched", "random", "relaxed")
            }
            read = restoration_read(
                arms, trigger_step=0, window=1, n_resamples=n_resamples, seed=seed
            )
            read["classification"] = classify_restoration(read)
            read["requested_lag"] = lag
            read["achieved_lags"] = sorted(
                {r["lags"][str(lag)]["achieved_lag"] for r in ok}
            )
            per_lag[str(lag)] = read
            m = read["arms"]["matched"]
            points.append(
                (lag, m["effect_vs_no_intervention"], tuple(m["effect_ci95"]))
            )
        out[press] = {
            "n_rows": len(rs),
            "n_ineligible": sum(r.get("status") != "READ" for r in rs),
            "per_lag": per_lag,
            "sweep": classify_latency_sweep(points),
        }
    return out


def main() -> int:
    a = _parser().parse_args()
    run_dir = a.runs_root / a.run
    if a.dry_run:
        print(json.dumps({"command": "exact-lag-read", "exists": run_dir.is_dir()}))
        return 0
    rows = [
        json.loads(line)
        for p in sorted((run_dir / "math500").rglob("*.jsonl"))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    res = {
        "amendment": "R.5 / R.10 / R.14",
        "run": a.run,
        "note": "recovery-after-delay; does not isolate staleness from position",
        **validate(rows, a.design),
        "policies": read_rows(rows, n_resamples=a.n_resamples, seed=a.seed),
    }
    if res["status"] != "FINAL":
        for v in res["policies"].values():
            v["sweep"] = {**v["sweep"], "verdict_withheld": True}
    lines = ["pid\tseed\tpress\tstatus\tlag\tnone\tmatched\trandom\trelaxed"]
    for r in sorted(rows, key=lambda r: (r["press"], r["pid"], r["seed"])):
        for lag, v in sorted((r.get("lags") or {}).items(), key=lambda kv: int(kv[0])):
            if v.get("eligible"):
                lines.append(
                    "\t".join(
                        str(x)
                        for x in (
                            r["pid"],
                            r["seed"],
                            r["press"],
                            r.get("status"),
                            lag,
                            v["none"],
                            v["matched"],
                            v["random"],
                            v["relaxed"],
                        )
                    )
                )
    data = ("\n".join(lines) + "\n").encode()
    tsv = a.out.with_suffix(".conditions.tsv")
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_bytes(data)
    res["sufficient_statistics_tsv"] = tsv.name
    res["sufficient_statistics_sha256"] = hashlib.sha256(data).hexdigest()
    res["reproduce"] = (
        f"python scripts/report_exact_lag.py --runs-root <runs> --run {a.run} "
        f"--design {a.design} --out {a.out} --final"
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", "utf-8")
    for press, v in res["policies"].items():
        print(press, v["sweep"])
    print(f"status {res['status']}; coverage {res['coverage']['status']}")
    print(f"wrote {a.out}")
    if a.final and res["status"] != "FINAL":
        return 3
    return 0


def validate(rows: list[dict[str, Any]], design: Path | None) -> dict[str, Any]:
    """R.14: exact (problem, seed, policy) coverage and one experiment identity.

    INELIGIBLE rows (no cohort, small stash, short trace) are legitimate
    registered outcomes and count as covered; they are listed, not hidden.
    """
    if design is None:
        return {"status": "PARTIAL", "coverage": {"status": "UNCHECKED"}}
    pids, seeds, presses = _design(design)
    cov = check_coverage(
        rows,
        lambda r: (r["pid"], int(r["seed"]), r["press"]),
        expected_keys(pids, seeds, presses),
    )
    ids = {
        json.dumps(
            {
                k: v
                for k, v in (r.get("provenance") or {}).items()
                if k != "inputs_sha256"
            },
            sort_keys=True,
        )
        for r in rows
    }
    ident_ok = len(ids) == 1 and "{}" not in ids
    return {
        "status": "FINAL" if cov["status"] == COMPLETE and ident_ok else "PARTIAL",
        "coverage": cov,
        "identity_consistent": ident_ok,
        "ineligible_rows": [
            [r["pid"], r["seed"], r["press"], r.get("status")]
            for r in rows
            if r.get("status") != "READ"
        ],
    }


def _design(config: Path) -> tuple[list[str], list[int], list[str]]:
    d = design(config)
    return d["pids"], d["seeds"], d["presses"]


if __name__ == "__main__":
    raise SystemExit(main())

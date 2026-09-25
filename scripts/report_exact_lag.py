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

from kvcache.coverage import COMPLETE, check_coverage, design, expected_keys
from kvcache.gates import expected_run_identity, validate_rows
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
    expected_identity = None
    if a.design is not None:
        expected_identity, _why = expected_run_identity(
            "exact_lag", str(a.design), a.runs_root
        )
    checks = validate(rows, a.design, expected_identity=expected_identity)
    res = {
        "amendment": "R.5 / R.10 / R.14 / R.15",
        "run": a.run,
        "note": "recovery-after-delay; does not isolate staleness from position",
        **checks,
        # no per-lag read at all on INVALID evidence: verdicts are suppressed,
        # and every row reaching read_rows has passed the content checks
        "policies": (
            {}
            if checks["status"] == "INVALID"
            else read_rows(rows, n_resamples=a.n_resamples, seed=a.seed)
        ),
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


REGISTERED_INELIGIBLE = (
    "INELIGIBLE: no eviction at or after the trigger",
    "INELIGIBLE: stash smaller than the reinsertion count",
)
ARMS = ("none", "matched", "random", "relaxed")


def row_problems(
    r: dict[str, Any], *, lags: list[int], window: int, trigger: int
) -> list[str]:
    """Content checks for one exact-lag row (R.15); empty list = valid."""
    import math

    out: list[str] = []
    st = r.get("status")
    if st in REGISTERED_INELIGIBLE:
        return out
    if st != "READ":
        return [f"status {st!r} is neither READ nor a registered ineligibility"]
    e, n = r.get("cohort_step"), r.get("n_trace_tokens")
    if not isinstance(e, int) or e < trigger:
        out.append(f"cohort_step {e!r} invalid (trigger {trigger})")
    if not isinstance(n, int):
        out.append("n_trace_tokens missing")
    sic = r.get("stash_identity_checks")
    if not isinstance(sic, dict) or not (sic.get("n_entries_checked") or 0) > 0:
        out.append("stash identity checks absent or checked nothing")
    got = r.get("lags") or {}
    if {str(x) for x in lags} != set(got):
        out.append(f"lag set {sorted(got)} != configured {lags}")
    if out:
        return out
    for lag in lags:
        v = got[str(lag)]
        start = e + lag
        fits = start + window - 1 <= n - 1
        if not v.get("eligible"):
            if fits:
                out.append(f"lag {lag}: ineligible although the window fits")
            continue
        if not fits:
            out.append(f"lag {lag}: eligible although the window does not fit")
        if not (v.get("requested_lag") == v.get("achieved_lag") == lag):
            out.append(f"lag {lag}: requested/achieved lag mismatch")
        if v.get("applied_step") != start:
            out.append(f"lag {lag}: applied_step {v.get('applied_step')} != {start}")
        for arm in ARMS:
            x = v.get(arm)
            if (
                not isinstance(x, (int, float))
                or not math.isfinite(x)
                or not 0 <= x <= 1
            ):
                out.append(f"lag {lag}: arm {arm} value {x!r} not a finite TV")
        gates = v.get("gates") or {}
        for arm in ARMS[1:]:
            g = gates.get(arm) or {}
            if (
                g.get("passed") is not True
                or "VALUES" not in str(g.get("checked", "")).upper()
            ):
                out.append(
                    f"lag {lag}: arm {arm} K/V restoration gate missing or failed"
                )
    return out


def validate(
    rows: list[dict[str, Any]],
    design: Path | None,
    *,
    expected_identity: dict[str, Any] | None = None,
    check_identity: bool = True,
) -> dict[str, Any]:
    """R.14/R.15: exact coverage, one expected identity, content checks.

    Duplicated / unexpected / malformed / mismatched evidence -> INVALID;
    missing rows -> PARTIAL. Registered ineligibility is a legitimate outcome,
    listed separately; any other non-READ status is INVALID.
    """
    import yaml

    if design is None:
        return {
            "status": "INVALID",
            "coverage": {"status": "UNCHECKED"},
            "why": "no design given",
        }
    pids, seeds, presses = _design(design)
    cfg = yaml.safe_load(Path(design).read_text(encoding="utf-8"))
    lags = [int(x) for x in cfg["lags"]]
    window, trigger = int(cfg["window"]), int(cfg["trigger_step"])
    cov = check_coverage(
        rows,
        lambda r: (r["pid"], int(r["seed"]), r["press"]),
        expected_keys(pids, seeds, presses),
    )
    prov = (
        validate_rows(
            rows,
            kind="exact_lag",
            expected=expected_identity,
            key=lambda r: (r["pid"], int(r["seed"]), r["press"]),
        )
        if check_identity
        else {}
    )
    content = {}
    for r in rows:
        try:
            probs = row_problems(r, lags=lags, window=window, trigger=trigger)
            k = (r["pid"], r["seed"], r["press"])
        except (KeyError, TypeError, ValueError) as e:
            probs, k = [f"malformed row: {e!r}"], (repr(r)[:40],)
        if probs:
            content[k] = probs
    invalid = cov["status"] == "INVALID" or prov or content
    status = (
        "INVALID" if invalid else ("FINAL" if cov["status"] == COMPLETE else "PARTIAL")
    )
    return {
        "status": status,
        "coverage": cov,
        "provenance_failures": [
            {"key": list(k), "problems": v} for k, v in sorted(prov.items(), key=repr)
        ][:50],
        "content_failures": [
            {"key": list(k), "problems": v}
            for k, v in sorted(content.items(), key=repr)
        ][:50],
        "ineligible_rows": [
            [r.get("pid"), r.get("seed"), r.get("press"), r.get("status")]
            for r in rows
            if r.get("status") in REGISTERED_INELIGIBLE
        ],
    }


def _design(config: Path) -> tuple[list[str], list[int], list[str]]:
    d = design(config)
    return d["pids"], d["seeds"], d["presses"]


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Amendment R.2 / R.11 read of the Q.1 direct functional control.

Tolerance = 10 x the maximum |logit difference| between two identical arm-B
replays (the deterministic control), over every row. Per (policy, interval):
PASS if every compared A-B step is within tolerance, FAIL otherwise. R.11: if
the deterministic control is exactly 0 the tolerance is degenerate; the read is
then EXACT when A-B is also exactly 0, and otherwise TOLERANCE-DEGENERATE with
the raw A-B maxima reported (never PASS, never "equal"). Selection agreement is
reported beside it. The legacy removal-anchored gate (phase excess of arm A's
TV profile anchored at the first physical removal) is computed for the record
and labelled NOT VALID FOR IDENTIFICATION whatever it shows.

    python scripts/report_fork_control.py --runs-root <runs> \\
        --run fork-control-random --run fork-control-slm --out fork.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.coverage import COMPLETE, check_coverage, design, expected_keys
from kvcache.gates import expected_run_identity, validate_rows
from kvcache.metrics.profile_shape import phase_amplitude_excess


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--design", action="append", default=[])
    p.add_argument("--final", action="store_true")
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def classify(max_ab: float, tol: float) -> str:
    """R.11's numerical rule, unchanged: now reported as ``logit_gate`` only."""
    if tol > 0:
        return "PASS" if max_ab <= tol else "FAIL"
    return "EXACT" if max_ab == 0 else "TOLERANCE-DEGENERATE"


def selection_gate(agree: list[bool | None]) -> dict[str, Any]:
    """R.14: agreement of the next selection decision, with explicit coverage.

    ``None`` means the trace ended before arm B reached its next selection --
    the only prespecified reason a fork has no comparison. Any disagreement is
    FAIL; no comparison at all is NO-COMPARISONS, never a vacuous PASS.
    """
    compared = [a for a in agree if a is not None]
    n_dis = sum(not a for a in compared)
    status = "FAIL" if n_dis else ("PASS" if compared else "NO-COMPARISONS")
    return {
        "status": status,
        "n_forks": len(agree),
        "n_compared": len(compared),
        "n_ineligible_trace_ended": len(agree) - len(compared),
        "n_disagree": n_dis,
        "agreement": (1 - n_dis / len(compared)) if compared else None,
    }


def overall(logit: str, selection: str, coverage_complete: bool) -> str:
    """Both gates must pass; any failure fails; anything else is INCONCLUSIVE."""
    if logit == "FAIL" or selection == "FAIL":
        return "FAIL"
    if logit in ("PASS", "EXACT") and selection == "PASS" and coverage_complete:
        return "PASS"
    return "INCONCLUSIVE"


def _listing(d: dict) -> list[dict[str, Any]]:
    return [{"key": list(k), "problems": v} for k, v in sorted(d.items(), key=repr)][
        :50
    ]


def row_problems(r: dict[str, Any]) -> list[str]:
    """R.15: the evidence must be well-formed BEFORE any numerical classification.

    A non-finite or negative control magnitude would otherwise turn into an
    infinite tolerance and a vacuous PASS (review of cfa30cd).
    """
    import math

    def fin(x: Any, hi: float | None = None) -> bool:
        return (
            isinstance(x, (int, float))
            and math.isfinite(x)
            and x >= 0
            and (hi is None or x <= hi)
        )

    out = []
    if not fin(r.get("deterministic_control_max_abs_logit")):
        out.append("deterministic control magnitude not finite and >= 0")
    forks = r.get("forks")
    if not isinstance(forks, list):
        return out + ["forks missing"]
    if r.get("n_forks") is not None and r.get("n_forks") != len(forks):
        out.append("n_forks disagrees with the fork records")
    for i, f in enumerate(forks):
        if not fin(f.get("max_abs_logit")):
            out.append(f"fork {i}: max_abs_logit not finite and >= 0")
        if not fin(f.get("max_tv"), 1.0):
            out.append(f"fork {i}: max_tv not a finite TV")
        if f.get("selection_agrees") not in (True, False, None):
            out.append(f"fork {i}: selection_agrees not True/False/None")
    prof = r.get("tv_profile")
    if not isinstance(prof, list) or not all(fin(x, 1.0) for x in prof):
        out.append("tv_profile missing or not finite TV values")
    return out


def read(
    rows: list[dict[str, Any]],
    *,
    n_resamples: int,
    seed: int,
    expected: set[tuple] | None = None,
    expected_identities: dict[str, dict[str, Any]] | None = None,
    check_identity: bool = True,
) -> dict:
    content = {}
    for r in rows:
        try:
            k = (r["pid"], r["seed"], r["press"], r["interval"])
            probs = row_problems(r)
        except (KeyError, TypeError, ValueError) as e:
            k, probs = (repr(r)[:40],), [f"malformed row: {e!r}"]
        if probs:
            content[k] = probs
    prov: dict = {}
    if check_identity:
        by_press: dict[str, list] = defaultdict(list)
        for r in rows:
            by_press[str(r.get("press"))].append(r)
        for press, rs in by_press.items():
            prov.update(
                validate_rows(
                    rs,
                    kind="fork",
                    expected=(expected_identities or {}).get(press),
                    key=lambda r: (r["pid"], int(r["seed"]), r["press"], r["interval"]),
                )
            )
    if content or prov:
        return {
            "status": "INVALID",
            "overall": "INVALID-EVIDENCE",
            "coverage": {"status": "UNCHECKED"},
            "content_failures": [
                {"key": list(k), "problems": v}
                for k, v in sorted(content.items(), key=repr)
            ][:50],
            "provenance_failures": [
                {"key": list(k), "problems": v}
                for k, v in sorted(prov.items(), key=repr)
            ][:50],
            "cells": {},
        }
    cov = (
        check_coverage(
            rows,
            lambda r: (r["pid"], int(r["seed"]), r["press"], int(r["interval"])),
            expected,
        )
        if expected is not None
        else {"status": "UNCHECKED"}
    )
    complete = cov["status"] == COMPLETE
    det = max((r["deterministic_control_max_abs_logit"] or 0.0) for r in rows)
    tol = 10.0 * det
    cells: dict[str, Any] = {}
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["press"], int(r["interval"]))].append(r)
    for (press, interval), rs in sorted(groups.items()):
        diffs = [f["max_abs_logit"] for r in rs for f in r["forks"]]
        tvs = [f["max_tv"] for r in rs for f in r["forks"]]
        sel = selection_gate([f["selection_agrees"] for r in rs for f in r["forks"]])
        max_ab = max(diffs) if diffs else 0.0
        logit = classify(max_ab, tol) if diffs else "NO-FORKS"
        legacy = phase_amplitude_excess(
            [
                (r["tv_profile"], r["first_removal_step"], interval)
                for r in rs
                if r["first_removal_step"] is not None
            ],
            n_resamples=n_resamples,
            seed=seed,
        )
        cells[f"{press}-i{interval}"] = {
            "n_traces": len(rs),
            "n_forks": len(diffs),
            "max_abs_logit_A_minus_B": max_ab,
            "max_next_token_tv": max(tvs) if tvs else None,
            "n_forks_outside_tolerance": sum(d > tol for d in diffs),
            "logit_gate": logit,
            "selection_gate": sel,
            "selection_agreement": sel["agreement"],
            "overall": overall(logit, sel["status"], complete),
            "legacy_removal_anchored_gate": {
                **legacy,
                "label": "NOT VALID FOR IDENTIFICATION (R.2)",
            },
        }
    verdicts = {c["overall"] for c in cells.values()}
    return {
        "status": (
            "INVALID"
            if cov["status"] == "INVALID"
            else ("FINAL" if complete else "PARTIAL")
        ),
        "coverage": cov,
        "deterministic_control_max_abs_logit": det,
        "tolerance": tol,
        "tolerance_rule": "10 x deterministic control max (R.2); R.11 degenerate rule",
        "overall": (
            "FAIL"
            if "FAIL" in verdicts
            else ("PASS" if verdicts == {"PASS"} and complete else "INCONCLUSIVE")
        ),
        "cells": cells,
    }


def main() -> int:
    a = _parser().parse_args()
    if a.dry_run:
        print(json.dumps({"command": "fork-control-read", "runs": a.run}))
        return 0
    rows = [
        json.loads(line)
        for run in a.run
        for p in sorted((a.runs_root / run / "math500").rglob("*.jsonl"))
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = None
    identities: dict[str, dict[str, Any]] = {}
    if a.design:
        expected = set()
        for d in a.design:
            pids, seeds, presses, intervals = _design(Path(d))
            expected |= expected_keys(pids, seeds, presses, intervals)
            ident, _why = expected_run_identity("fork", str(d), a.runs_root)
            for p_ in presses:
                identities[p_] = ident
    res = {"amendment": "R.2 / R.11 / R.14", "runs": a.run}
    res.update(
        read(
            rows,
            n_resamples=a.n_resamples,
            seed=a.seed,
            expected=expected,
            expected_identities=identities,
        )
    )
    cols = (
        "pid seed press interval n_forks max_abs_logit max_tv "
        "n_compared n_disagree det_max"
    )
    lines = ["\t".join(cols.split())]
    for r in sorted(rows, key=lambda r: (r["press"], int(r["interval"]), r["pid"])):
        ag = [f["selection_agrees"] for f in r["forks"]]
        lines.append(
            "\t".join(
                str(x)
                for x in (
                    r["pid"],
                    r["seed"],
                    r["press"],
                    r["interval"],
                    len(r["forks"]),
                    max((f["max_abs_logit"] for f in r["forks"]), default=""),
                    max((f["max_tv"] for f in r["forks"]), default=""),
                    sum(x is not None for x in ag),
                    sum(x is False for x in ag),
                    r["deterministic_control_max_abs_logit"],
                )
            )
        )
    data = ("\n".join(lines) + "\n").encode()
    tsv = a.out.with_suffix(".conditions.tsv")
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.write_bytes(data)
    res["sufficient_statistics_tsv"] = tsv.name
    res["sufficient_statistics_sha256"] = hashlib.sha256(data).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", "utf-8")
    print(
        f"status {res.get('status')}; tolerance {res.get('tolerance')}; "
        f"overall {res.get('overall')}"
    )
    for k, c in res["cells"].items():
        print(k, c["overall"], c["logit_gate"], c["selection_gate"]["status"])
    print(f"wrote {a.out}")
    if a.final and res["status"] != "FINAL":
        return 3
    return 0


def _design(config: Path) -> tuple[list[str], list[int], list[str], list[int]]:
    d = design(config)
    return d["pids"], d["seeds"], d["presses"], d["intervals"]


if __name__ == "__main__":
    raise SystemExit(main())

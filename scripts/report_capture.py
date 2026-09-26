#!/usr/bin/env python3
"""Amendment R.9 / R.14 read of one capture cell: Q.3 (Task 2) and Q.6 (Task 3).

Requirements before any registered verdict is produced (R.14):

* **Coverage** -- rows match the EXACT expected key set (problem x seed x
  policy for the cell's single declared budget); duplicated, missing and
  unexpected keys are listed (``kvcache.coverage.check_coverage``).
* **Identity** -- every row carries the same R.13 provenance (implementation,
  config, model revision, reference run, basis bytes).
* **Controls** -- every row's recorded controls are finite and within
  tolerance (layer-0 Delta <= 1e-5, no non-finite value). Failing rows are
  listed and make the read INVALID, never silently dropped.
* **Eligibility** is separate: an undefined or unbounded latent direction or
  an event with nothing eligible is an excluded VALUE with its reason, not a
  failed instrument.

Registered estimands, equal-problem estimator (seeds within problem, then
problems; the same estimator on every problem-cluster bootstrap replicate):

* Q.3: pooled collateral, R-KV minus ObservedAttentionPress ("h2o"), paired by
  (problem, seed). Negative excluding 0 supports the Q.3 ordering; including 0
  INCONCLUSIVE; positive excluding 0 REVERSED.
* Q.6: omega(Random) - omega(StreamingLLM), paired; PREDICTED / NULL / REVERSED.
  Exploratory; no accuracy-scale claim.

Descriptive only: per-policy M, M_event, Delta, wanted_any_head, collateral,
omega, and Delta/M over the trigger event's slice. The trigger ratio is read at
the first forward after the first eviction at or after step 130; it is NOT a
measurement at the Q.5 reinsertion states, and supports no query-drift causal
claim (R.14).

Also writes a per-condition sufficient-statistics table (TSV) and its sha256,
from which every reported mean and CI can be recomputed.

    python scripts/report_capture.py --runs-root <runs> --run capture-b256 \\
        --config configs/experiments/capture_math500_b256_r9.yaml \\
        --out results/diag/meeting3/capture_b256.json --final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.capture import LAYER0_DELTA_TOL
from kvcache.gates import expected_run_identity, validate_rows
from kvcache.coverage import (
    design,
    COMPLETE,
    check_coverage,
    equal_problem_bootstrap,
    expected_keys,
    paired_equal_problem,
)

POLICIES = ("random", "streamingllm", "h2o", "rkv")
METRICS = (
    "collateral",
    "wanted_any_head",
    "M",
    "M_event",
    "Delta",
    "trigger_delta_over_M",
    "omega",
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--run", required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--final", action="store_true")
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


def control_failures(row: dict[str, Any]) -> list[str]:
    c = row.get("checks") or {}
    out = []
    l0 = c.get("layer0_delta_max")
    if not isinstance(l0, (int, float)) or not math.isfinite(l0):
        out.append("layer0_delta_max missing or non-finite")
    elif l0 > LAYER0_DELTA_TOL:
        out.append(f"layer0_delta_max {l0} > {LAYER0_DELTA_TOL}")
    if c.get("n_nonfinite") != 0:
        out.append(f"n_nonfinite = {c.get('n_nonfinite')}")
    return out


def metric_values(row: dict[str, Any]) -> dict[str, tuple[float | None, str | None]]:
    """(value, exclusion reason) per metric; reason None when eligible."""
    lat = row.get("latent") or {}
    raw = {
        "collateral": (row.get("collateral_pooled"), "no eligible evicted position"),
        "wanted_any_head": (
            row.get("wanted_any_head_pooled"),
            "no eligible evicted position",
        ),
        "M": (row.get("M_mean"), "no measured event"),
        "M_event": (row.get("M_event_mean"), "no measured event"),
        "Delta": (row.get("Delta_mean"), "no measured event"),
        "trigger_delta_over_M": (trigger_ratio(row), "no trigger event or M = 0"),
        "omega": (
            lat.get("omega_trace"),
            "no finite direction (undefined/unbounded only, R.6)",
        ),
    }
    return {
        k: ((float(v), None) if v is not None and math.isfinite(v) else (None, why))
        for k, (v, why) in raw.items()
    }


def collateral_counts(
    rows: list[dict[str, Any]], bad: dict, budget: int
) -> dict[str, dict[str, int]]:
    out = {
        p: {"n_eligible": 0, "n_wanted": 0, "n_collateral": 0, "n_rows": 0}
        for p in POLICIES
    }
    for r in rows:
        try:
            k3 = (r["pid"], int(r["seed"]), r["press"])
            if (
                k3 in bad
                or r["press"] not in POLICIES
                or str(r["budget"]) != str(budget)
            ):
                continue
            cnt = r["counts_by_layer_group"]
            o = out[r["press"]]
            o["n_rows"] += 1
            for src, dst in (
                ("n_eligible", "n_eligible"),
                ("n_wanted", "n_wanted"),
                ("n_collateral", "n_collateral"),
            ):
                o[dst] += int(sum(sum(x) for x in cnt[src]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _config(config: Path) -> dict[str, Any]:
    import yaml

    path = Path(config)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _family(config: Path) -> str:
    """The generation family a config's rows carry (capture-v2 names it, R.17.2.1).

    The expected identity is always derived from committed specs; a config with
    no ``family`` key is capture-v1.
    """
    return _config(config).get("family", "capture")


def read_capture(
    rows: list[dict[str, Any]],
    *,
    pids: list[str],
    seeds: list[int],
    budget: int,
    expected_identity: dict[str, Any] | None,
    n_resamples: int = 2000,
    seed: int = 0,
    kind: str = "capture",
    manifest_path: str | None = None,
    policies: tuple[str, ...] = POLICIES,
) -> dict[str, Any]:
    expected = expected_keys(pids, seeds, policies, [str(budget)])
    cov = check_coverage(
        rows,
        lambda r: (r["pid"], int(r["seed"]), r["press"], str(r["budget"])),
        expected,
    )
    # R.15: every row's provenance must carry the required fields, equal the
    # identity derived from committed specs, and match the frozen input manifest
    prov_bad = validate_rows(
        rows,
        kind=kind,
        expected=expected_identity,
        manifest_path=manifest_path,
        key=lambda r: (r["pid"], int(r["seed"]), r["press"]),
    )
    identity_ok = not prov_bad
    bad = {}
    for r in rows:
        try:
            k3 = (r["pid"], int(r["seed"]), r["press"])
            f = control_failures(r)
        except (KeyError, TypeError, ValueError) as e:
            k3, f = (repr(r)[:40], 0, "?"), [f"malformed row: {e!r}"]
        if f:
            bad[k3] = f
    controls = "VALID" if not bad else "INVALID"
    table: list[dict[str, Any]] = []
    values: dict[str, dict[str, dict[tuple[str, int], float]]] = {
        m: {p: {} for p in POLICIES} for m in METRICS
    }
    exclusions: dict[str, dict[str, dict[str, int]]] = {
        m: {p: {} for p in POLICIES} for m in METRICS
    }
    for r in rows:
        try:
            k3 = (r["pid"], int(r["seed"]), r["press"])
            skip = (
                k3 in bad
                or r["press"] not in POLICIES
                or str(r["budget"]) != str(budget)
            )
            mv = None if skip else metric_values(r)
        except (KeyError, TypeError, ValueError):
            continue  # malformed: already INVALID through coverage/controls
        if mv is None:
            continue
        entry = {"pid": r["pid"], "seed": int(r["seed"]), "press": r["press"]}
        for m, (v, why) in mv.items():
            entry[m] = v
            if v is None:
                ex = exclusions[m][r["press"]]
                ex[why] = ex.get(why, 0) + 1
            else:
                values[m][r["press"]][(r["pid"], int(r["seed"]))] = v
        table.append(entry)
    final_ok = cov["status"] == COMPLETE and identity_ok and controls == "VALID"
    q3 = paired_equal_problem(
        values["collateral"]["rkv"],
        values["collateral"]["h2o"],
        n_resamples=n_resamples,
        seed=seed,
    )
    q6 = paired_equal_problem(
        values["omega"]["random"],
        values["omega"]["streamingllm"],
        n_resamples=n_resamples,
        seed=seed,
    )
    for c, verdicts in (
        (
            q3,
            (
                "SUPPORTS Q.3 ORDERING (R-KV lower)",
                "REVERSED (R-KV higher)",
                "INCONCLUSIVE",
            ),
        ),
        (q6, ("REVERSED", "PREDICTED", "NULL")),
    ):
        if not final_ok or c["ci95"] is None:
            c["verdict"] = None
            continue
        lo, hi = c["ci95"]
        c["verdict"] = verdicts[0] if hi < 0 else verdicts[1] if lo > 0 else verdicts[2]
    return {
        "status": (
            "FINAL"
            if final_ok
            else (
                "INVALID"
                if (bad or not identity_ok or cov["status"] == "INVALID")
                else "PARTIAL"
            )
        ),
        "budget": budget,
        "coverage": cov,
        "identity_consistent": identity_ok,
        "provenance_failures": [
            {"key": list(k), "problems": v}
            for k, v in sorted(prov_bad.items(), key=repr)
        ][:50],
        "n_provenance_failures": len(prov_bad),
        "controls": controls,
        "control_failures": [
            {"pid": k[0], "seed": k[1], "press": k[2], "failures": f}
            for k, f in sorted(bad.items())
        ],
        "q3_registered_contrast": {"contrast": "collateral rkv - h2o", **q3},
        "q6_registered_contrast": {"contrast": "omega random - streamingllm", **q6},
        "descriptive": {
            m: {
                p: equal_problem_bootstrap(
                    values[m][p], n_resamples=n_resamples, seed=seed
                )
                for p in POLICIES
            }
            for m in METRICS
        },
        "eligibility_exclusions": exclusions,
        # R.16: raw totals behind the pooled collateral, per policy, over valid
        # rows (sum over every layer, KV group and event).
        "collateral_counts": collateral_counts(rows, bad, budget),
        "sufficient_statistics": table,
        "notes": [
            "verdicts are withheld unless coverage is COMPLETE, identity is "
            "consistent and every control is VALID (R.14)",
            "trigger Delta/M is descriptive at the first eviction >= step 130; it is "
            "not measured at Q.5 reinsertion states and licenses no query-drift "
            "causal claim",
            "Q.6 is exploratory; no accuracy-scale claim",
        ],
    }


def write_table(path: Path, table: list[dict[str, Any]]) -> str:
    cols = ["pid", "seed", "press", *METRICS]
    lines = ["\t".join(cols)]
    for e in sorted(table, key=lambda x: (x["press"], x["pid"], x["seed"])):
        lines.append("\t".join("" if e.get(c) is None else f"{e[c]}" for c in cols))
    data = ("\n".join(lines) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _design(config: Path) -> tuple[list[str], list[int], int]:
    d = design(config)
    if len(d["budgets"]) != 1:
        raise SystemExit("a capture read declares exactly one budget")
    return d["pids"], d["seeds"], d["budgets"][0]


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
    if a.config is None:
        raise SystemExit("--config is required: the expected design comes from it")
    pids, seeds, budget = _design(a.config)
    kind = _family(a.config)
    expected_identity, why = expected_run_identity(kind, str(a.config), a.runs_root)
    res = read_capture(
        _rows(run_dir),
        pids=pids,
        seeds=seeds,
        budget=budget,
        expected_identity=expected_identity,
        n_resamples=a.n_resamples,
        seed=a.seed,
        kind=kind,
        manifest_path=_config(a.config).get("input_manifest"),
        policies=tuple(_config(a.config).get("presses") or POLICIES),
    )
    table = res.pop("sufficient_statistics")
    tsv = a.out.with_suffix(".conditions.tsv")
    res["sufficient_statistics_tsv"] = str(tsv.name)
    res["sufficient_statistics_sha256"] = write_table(tsv, table)
    res["reproduce"] = (
        f"python scripts/report_capture.py --runs-root <runs> --run {a.run} "
        f"--config {a.config} --out {a.out} --final"
    )
    res["run"] = a.run
    res["expected_identity"] = expected_identity
    res["expected_identity_note"] = why
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", "utf-8")
    cov = res["coverage"]
    print(
        f"status {res['status']}; coverage {cov['status']} "
        f"({cov['n_unique_expected_present']}/{cov['n_expected']}, "
        f"dup {len(cov['duplicated'])}, unexpected {len(cov['unexpected'])}); "
        f"controls {res['controls']}; identity {res['identity_consistent']}"
    )
    for name in ("q3_registered_contrast", "q6_registered_contrast"):
        c = res[name]
        print(
            f"{c['contrast']}: mean={c['mean']} ci={c['ci95']} verdict={c['verdict']}"
        )
    print(f"wrote {a.out}")
    if a.final and res["status"] != "FINAL":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

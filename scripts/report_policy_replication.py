#!/usr/bin/env python3
"""M5-A protection-by-policy accuracy read (Meeting 5; prepared 2026-10-14,
rebuilt on the shared validation boundary after the review of 6b409001, R1).

Every source run named in configs/meeting5/m5a_read.yaml is validated IN FULL
against its own config's design before the cohort is selected (shared
``read_checks.validate_rows`` plus ``policy_replication.validate_source``):
missing keys -> PARTIAL; duplicate, foreign or malformed rows (non-boolean or
NaN ``correct``, wrong model, a truncated row scored correct, an identity block
on a legacy row, a wrong identity on an M5 row, a torn line) -> INVALID.

Output contract (as every reader since R.14): the previous output is retired
BEFORE anything is parsed; the new one is written atomically, including
INVALID evidence on any failure; exit 0 FINAL, 3 PARTIAL (0 with --partial),
2 INVALID; a dependency fingerprint covers every consumed row file, spec,
config and analysis-code file. PARTIAL and INVALID carry no estimate.

``--cohort strict`` (primary): the 97 problems with P + 64 <= 256.
``--cohort whole``: all 100; PARTIAL until the capacity-matched run exists.

    python scripts/report_policy_replication.py --runs-root <runs> --cohort strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.metrics.policy_replication import (
    FINAL,
    INVALID,
    PARTIAL,
    replication_read,
    validate_source,
)
from kvcache.provenance import write_json_atomic
from kvcache.read_validation import (
    dependency_fingerprint,
    design_keys,
    exit_code,
    legacy_provenance_summary,
    retire_stale,
    tree_files,
)

SPEC = "configs/meeting5/m5a_read.yaml"
OUT = PROJECT_ROOT / "results/diag/meeting5/m5a"
ANALYSIS_CODE = (
    "scripts/report_policy_replication.py",
    "src/kvcache/metrics/policy_replication.py",
    "src/kvcache/metrics/protocol_audit.py",
    "src/kvcache/read_checks.py",
    "src/kvcache/read_validation.py",
    "src/kvcache/coverage.py",
    "src/kvcache/m5_runs.py",
    "src/kvcache/stats.py",
)


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rows(files: list[Path]) -> list[dict[str, Any]]:
    """Every row; a torn or non-object line is an error (INVALID), never skipped."""
    out = []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{f.name} line {i}: malformed JSON ({exc})") from None
            if not isinstance(rec, dict):
                raise ValueError(f"{f.name} line {i}: not a JSON object")
            out.append(rec)
    return out


def _summary(block: dict[str, Any]) -> dict[str, Any]:
    keep = ("status", "n_expected", "n_rows", "n_unique_expected_present")
    out = {k: block.get(k) for k in keep}
    for k in ("missing", "duplicated", "unexpected"):
        out[f"n_{k}"] = len(block.get(k) or [])
        out[k] = (block.get(k) or [])[:10]
    out["n_malformed"] = len(block.get("malformed") or {})
    out["malformed"] = dict(list((block.get("malformed") or {}).items())[:10])
    return out


def read(runs: Path, cohort: str) -> tuple[dict[str, Any], str]:
    from kvcache.m5_runs import design_keys as m5_keys
    from kvcache.m5_runs import expected_identity, row_identity_problems
    from kvcache.provenance import load_input_manifest
    from kvcache.read_validation import frozen_pids

    spec = _yaml(PROJECT_ROOT / SPEC)
    sf = json.loads((PROJECT_ROOT / spec["cohort_source"]).read_text())
    relaxed = {k: int(v) for k, v in sf["relaxed_effective_budgets"].items()}
    pids = list(sf["pids"]) if cohort == "strict" else sorted([*sf["pids"], *relaxed])
    use_relaxed = relaxed if cohort == "whole" else {}
    seeds = [int(s) for s in spec["seeds"]]

    sources: dict[str, Any] = {}
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    files_by_run: dict[str, list[Path]] = {}
    controls: dict[str, Any] = {}
    for run, src in spec["sources"].items():
        cfg_path = src["config"]
        _, expected = design_keys(cfg_path, root=PROJECT_ROOT)
        if not (runs / run).is_dir():
            sources[run] = {"status": PARTIAL, "absent": True}
            continue
        files = tree_files(runs / run, spec["bench"])
        files_by_run[run] = files
        rows = _rows(files)
        rows_by_run[run] = rows
        check = None
        if src["provenance"] == "m5":
            cfg = _yaml(PROJECT_ROOT / cfg_path)
            ident = expected_identity(cfg)
            manifest = load_input_manifest(path=cfg.get("input_manifest"))
            keys = m5_keys(
                cfg, frozen_pids(int(cfg["subset"]["math500"]), root=PROJECT_ROOT)
            )

            def check(r, ident=ident, manifest=manifest, keys=keys):
                return row_identity_problems(
                    r, ident=ident, manifest=manifest, keys=keys
                )

        else:
            controls[f"provenance_{run}"] = legacy_provenance_summary(rows)
        sources[run] = _summary(
            validate_source(
                rows,
                expected=expected,
                model=spec["model"],
                bench=spec["bench"],
                provenance=src["provenance"],
                identity_check=check,
            )
        )
    if use_relaxed:
        cap = runs / spec["capmatch_run"]
        sources[spec["capmatch_run"]] = (
            {
                "status": INVALID,
                "reason": "the capacity-matched run has no registered config",
            }
            if cap.is_dir()
            else {"status": PARTIAL, "absent": True}
        )

    statuses = [b["status"] for b in sources.values()]
    if INVALID in statuses or any(not c.get("passed") for c in controls.values()):
        status = INVALID
    elif all(s in ("COMPLETE", FINAL) for s in statuses):
        status = FINAL
    else:
        status = PARTIAL
    deps = dependency_fingerprint(
        {
            **{f"rows_{run}": (runs, fs) for run, fs in files_by_run.items()},
            "specs": (
                PROJECT_ROOT,
                [
                    PROJECT_ROOT / SPEC,
                    PROJECT_ROOT / spec["cohort_source"],
                    *(PROJECT_ROOT / s["config"] for s in spec["sources"].values()),
                ],
            ),
            "analysis_code": (
                PROJECT_ROOT,
                [PROJECT_ROOT / f for f in ANALYSIS_CODE],
            ),
        }
    )
    payload: dict[str, Any] = {
        "read": "M5-A protection-by-policy accuracy",
        "status": status,
        "cohort": {"name": cohort, "n_problems": len(pids), "seeds": seeds},
        "capacity_matched": cohort == "strict",
        "sources": sources,
        "controls": controls,
        "dependencies": deps,
    }
    if status != FINAL:
        return payload, status

    keep = set(pids)
    values: dict[tuple[str, bool], dict[tuple[str, int], float]] = {}
    for policy, sides in spec["arms"].items():
        for protected in (False, True):
            run, press = sides["protected" if protected else "unprotected"]
            vals = {
                (str(r["pid"]), int(r["decoding"]["seed"])): float(r["correct"])
                for r in rows_by_run[run]
                if str(r["press"]) == press
                and r["pid"] in keep
                and (protected or r["pid"] not in use_relaxed)
            }
            values[(policy, protected)] = vals
    expected = {(p, s) for p in pids for s in seeds}
    payload["estimates"] = replication_read(
        values,
        expected=expected,
        reference=spec["reference"],
        pairings=spec["pairings"],
        descriptive=spec["descriptive"],
        equivalence=spec["equivalence"],
    )
    return payload, status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=("strict", "whole"), default="strict")
    ap.add_argument("--runs-root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--partial", action="store_true", help="exit 0 on PARTIAL")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = a.out_dir / f"m5a_read_{a.cohort}.json"
    if a.dry_run:
        print(json.dumps({"cohort": a.cohort, "spec": SPEC, "out": str(out)}))
        print("DRY RUN: spec parsed; no run directory read")
        return 0
    runs = a.runs_root or Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    # Retire the previous output BEFORE anything can fail, so a crash, a parse
    # error or a refusal never leaves an old FINAL standing as current.
    retired = retire_stale(out)
    try:
        payload, status = read(runs, a.cohort)
    except Exception as exc:  # noqa: BLE001 -- evidence, then exit 2
        payload = {"read": "M5-A", "status": INVALID, "error": repr(exc)}
        status = INVALID
        print(f"INVALID: {exc!r}", file=sys.stderr)
    payload["reproduce"] = "python scripts/report_policy_replication.py " + " ".join(
        sys.argv[1:]
    )
    payload["retired_previous_output"] = retired.name if retired else None
    write_json_atomic(out, payload)
    code = exit_code(status, partial_ok=a.partial)
    print(f"wrote {out} [{status}] exit {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

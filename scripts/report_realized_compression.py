#!/usr/bin/env python3
"""Realized (nominal-budget) compression of the S.1 grid cells (2026-10-15;
evidence-bound 2026-10-16 after the review of 5616630, R4).

Summarizes each cell's per-row ``compression_ratio`` (``kvcache.compression_
summary``) for the protected and unprotected grids at B = 256 and B = 1024,
ONLY from rows a FINAL S.1 protection read validated: that read must exist,
its conditions table must match its hash, and its full dependency fingerprint
must equal the one recomputed now (``report_s.protection_dependencies``, as
``report_s.py consequence`` does). Missing evidence -> PARTIAL (exit 3), stale
evidence or malformed rows -> INVALID (exit 2); neither publishes numbers. The
previous output is retired first; the new one is written atomically.

    python scripts/report_realized_compression.py --runs-root <runs> \\
        [--s1-dir results/diag/amendment-s] [--out <json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.compression_summary import (
    DEFINITION,
    RATIO_ONE,
    cell_ratios,
    design_keys,
    prompt_lengths,
    s1_evidence,
)
from kvcache.provenance import write_json_atomic
from kvcache.read_validation import exit_code, retire_stale, tree_files

S1_DIR = PROJECT_ROOT / "results/diag/amendment-s"
OUT = S1_DIR / "realized_compression.json"
BUDGETS = (256, 1024)
SEVERITY = {"FINAL": 0, "PARTIAL": 1, "INVALID": 2}


def _rows(root: Path) -> list[dict]:
    return [
        json.loads(line)
        for f in tree_files(root, "math500")
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path)
    ap.add_argument("--s1-dir", type=Path, default=S1_DIR)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"budgets": BUDGETS, "s1_dir": str(a.s1_dir)}))
        print("DRY RUN: no run directory read")
        return 0
    if a.runs_root is None:
        ap.error("--runs-root is required")
    retired = retire_stale(a.out)
    out: dict = {
        "read": "nominal-budget compression of the S.1 grid cells",
        "definition": DEFINITION,
        "ratio_one": RATIO_ONE,
        "evidence": {},
    }
    status, cells = "FINAL", {}
    try:
        for budget in BUDGETS:
            rp = a.s1_dir / f"protection_b{budget}.json"
            tp = a.s1_dir / f"protection_b{budget}.conditions.tsv"
            report = json.loads(rp.read_text()) if rp.exists() else None
            tsv = tp.read_bytes() if tp.exists() else None
            current = None
            if report is not None:
                import report_s  # the S.1 reader's own fingerprint (same dir)

                current = report_s.protection_dependencies(a.runs_root, budget)[
                    "sha256"
                ]
            st, why = s1_evidence(report, tsv, current)
            out["evidence"][f"b{budget}"] = {
                "status": st,
                "problems": why,
                "s1_read": str(rp.name),
                "s1_dependencies_sha256": current,
            }
            if SEVERITY[st] > SEVERITY[status]:
                status = st
            if st != "FINAL":
                continue
            plen = prompt_lengths(tsv)
            inputs = report["inputs"]
            for run, protected in (
                (inputs["grid_pp"], True),
                (inputs["grid_base"], False),
            ):
                rows = _rows(a.runs_root / run)
                for pp in report["design"]["presses"]:
                    press = pp if protected else pp.removesuffix("_pp")
                    cells.setdefault(f"b{budget}", {})[press] = {
                        "run": run,
                        **cell_ratios(
                            rows,
                            press=press,
                            budget=budget,
                            keys=design_keys(tsv, pp),
                            prompt_len=plen,
                            protected=protected,
                        ),
                    }
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        status = "INVALID"
        out["error"] = str(exc)
    out["status"] = status
    if status == "FINAL":
        out["cells"] = cells
    out["retired_previous_output"] = retired.name if retired else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(a.out, out)
    code = exit_code(status, partial_ok=False)
    print(f"wrote {a.out} [{status}] exit {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

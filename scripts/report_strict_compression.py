#!/usr/bin/env python3
"""Nominal-budget compression of the eight M5-A arms on the 97-problem
strict-fit cohort (2026-11-10). See ``kvcache.strict_compression``.

Writes a SEPARATE artifact from the 100-problem S.1 one. As a reproduction
check it also recomputes the S.1 Random / StreamingLLM medians from the same
rows on the S.1 keys and requires them to equal the published S.1 artifact;
a mismatch makes the read INVALID. Any failure publishes no numbers.

    python scripts/report_strict_compression.py --runs-root <runs> \\
        [--out results/diag/meeting5/m5a/strict_realized_compression.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.compression_summary import DEFINITION, RATIO_ONE, design_keys
from kvcache.provenance import write_json_atomic
from kvcache.read_validation import exit_code, retire_stale, tree_files
from kvcache.strict_compression import (
    ARMS,
    BUDGET,
    arm_ratios,
    strict_keys,
    verified_prompt_lengths,
)

M5A = PROJECT_ROOT / "results/diag/meeting5/m5a"
S1 = PROJECT_ROOT / "results/diag/amendment-s"
OUT = M5A / "strict_realized_compression.json"
TOL = 1e-12


def _rows(root: Path) -> tuple[list[dict], dict[str, str]]:
    files = sorted(tree_files(root, "math500"))
    rows = [
        json.loads(line)
        for f in files
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    digests = {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in files
    }
    return rows, digests


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"arms": [x.label for x in ARMS], "out": str(a.out)}))
        print("DRY RUN: no run directory read")
        return 0
    if a.runs_root is None:
        ap.error("--runs-root is required")
    retired = retire_stale(a.out)
    out: dict = {
        "read": "nominal-budget compression of the eight M5-A arms on the "
        "strict-fit cohort",
        "definition": DEFINITION,
        "ratio_one": RATIO_ONE,
        "budget": BUDGET,
        "population": "97-problem strict-fit cohort x seeds 11/23/47/89 (the "
        "M5-A strict read's own keys); NOT the 100-problem S.1 population",
    }
    status = "FINAL"
    try:
        read = json.loads((M5A / "m5a_read_strict.json").read_text())
        tsv = (M5A / "m5a_read_strict.conditions.tsv").read_bytes()
        keys = strict_keys(read, tsv)
        s1_read = json.loads((S1 / "protection_b256.json").read_text())
        s1_tsv = (S1 / "protection_b256.conditions.tsv").read_bytes()
        plen = verified_prompt_lengths(s1_read, s1_tsv)
        for arm in ARMS:
            src = (read.get("sources") or {}).get(arm.run) or {}
            if src.get("status") != "COMPLETE":
                raise ValueError(f"{arm.run}: not a COMPLETE source of the read")
        cache: dict[str, tuple[list[dict], dict[str, str]]] = {}
        arms = {}
        for arm in ARMS:
            if arm.run not in cache:
                cache[arm.run] = _rows(a.runs_root / arm.run)
            arms[arm.label] = arm_ratios(cache[arm.run][0], arm, keys, plen)
        # reproduction check against the published S.1 artifact (100 problems)
        published = json.loads((S1 / "realized_compression.json").read_text())
        repro = {}
        for arm in ARMS:
            if arm.policy not in ("random", "streamingllm"):
                continue
            s1_press = f"{arm.policy}_pp"
            s1_keys = design_keys(s1_tsv, s1_press)
            from kvcache.compression_summary import cell_ratios

            sel = [r for r in cache[arm.run][0] if r.get("press") == arm.press]
            mine = cell_ratios(
                sel,
                press=arm.press,
                budget=BUDGET,
                keys=s1_keys,
                prompt_len=plen,
                protected=arm.protected,
            )["median"]
            theirs = published["cells"]["b256"][arm.press]["median"]
            if abs(mine - theirs) > TOL:
                raise ValueError(f"{arm.press}: S.1 median {mine} != {theirs}")
            repro[arm.press] = {"recomputed": mine, "published": theirs}
        out["arms"] = arms
        out["reproduces_s1_artifact"] = repro
        out["evidence"] = {
            "strict_read": "m5a_read_strict.json",
            "strict_read_status": read["status"],
            "strict_conditions_sha256": read["sufficient_statistics_sha256"],
            "s1_prompt_lengths_from": "protection_b256.conditions.tsv",
            "s1_conditions_sha256": s1_read["sufficient_statistics_sha256"],
            "row_files_sha256": {run: d for run, (_, d) in sorted(cache.items())},
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        status = "INVALID"
        out.pop("arms", None)
        out["error"] = str(exc)
    out["status"] = status
    out["retired_previous_output"] = retired.name if retired else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(a.out, out)
    code = exit_code(status, partial_ok=False)
    print(f"wrote {a.out} [{status}] exit {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

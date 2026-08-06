#!/usr/bin/env python3
"""Integrate a newly completed cell: harvest, recatalogue, re-gate, and diff.

The point is that a finished run lands WITHOUT a fresh reasoning pass. Every
step here has been done by hand at least once, and each hand pass is where a
number went stale: Amendment K reached nine summaries and missed a tenth, the
figures went on drawing pre-K constants after both the summaries and the .tex
had been regenerated, and two bank-vs-comparator confusions survived a gate that
passed. Doing it in one command with a printed diff removes the step where
someone decides what to check.

    python scripts/integrate_cell.py --unit m4-q14b-256          # remote harvest
    python scripts/integrate_cell.py --unit <u> --local-only     # skip harvest

What it does, in order:

1. harvest the unit on the remote (``harvest_results.py``), which regenerates its
   summary under the Amendment K convention with ``reference_convention``
   declared -- the harvest path is the only producer that sets it;
2. copy the summary back into ``results/diag/amendment-k/``;
3. rebuild ``paper_numbers.json`` and print a BEFORE/AFTER diff of every entry
   whose value moved, plus every entry added or removed;
4. re-run the audit gate (literals, 4-CURRENT, reference-convention);
5. rebuild the figures from the catalogue and the PDF twice.

Steps 3-5 are skipped if the harvest produced nothing new, so running it on an
unfinished unit is free and safe.

It deliberately does NOT edit the .tex. What a result means is a judgement, and
``docs/operations/INTEGRATION_CHECKLIST.md`` holds the pre-registered reads
that make that
judgement mechanical. This script gets the numbers in and tells you what moved.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

REMOTE = "kvcache-gpu"
REMOTE_ROOT = "~/kv-caching"
REMOTE_RUNS = "/home/sk10741/kvcache-storage/runs"
CATALOGUE = PROJECT_ROOT / "paper_numbers.json"
AMENDMENT_K = PROJECT_ROOT / "results" / "diag" / "amendment-k"


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=PROJECT_ROOT, capture_output=True, text=True, **kwargs
    )


def _snapshot() -> dict[str, float]:
    if not CATALOGUE.exists():
        return {}
    payload = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    return {e["key"]: float(e["value"]) for e in payload["entries"]}


def harvest(unit: str) -> str:
    print(f"[1/5] harvesting {unit} on {REMOTE}")
    result = _run(
        [
            "ssh",
            REMOTE,
            f"cd {REMOTE_ROOT} && .venv/bin/python scripts/harvest_results.py "
            f"--force {unit}",
        ],
        timeout=3600,
    )
    print(result.stdout[-1500:] or result.stderr[-800:])
    return result.stdout


def fetch_summary(unit: str) -> Path | None:
    """Copy the harvested summary into the amendment-k tree."""
    for remote_name, local_name in (
        ("summary.json", f"{unit}-summary.json"),
        ("readout_acc.json", "fc-readout-control.json"),
    ):
        probe = _run(
            ["ssh", REMOTE, f"test -f {REMOTE_RUNS}/harvest/{unit}/{remote_name}"],
            timeout=120,
        )
        if probe.returncode != 0:
            continue
        target = AMENDMENT_K / local_name
        got = _run(
            ["ssh", REMOTE, f"cat {REMOTE_RUNS}/harvest/{unit}/{remote_name}"],
            timeout=600,
        )
        if got.returncode != 0 or not got.stdout.strip():
            continue
        payload = json.loads(got.stdout)
        if "reference_convention" not in payload and "cells" in payload:
            raise SystemExit(
                f"REFUSED: {unit}'s summary declares no reference_convention. "
                "The harvest path is the only producer that sets it; a summary "
                "without it has not been through the Amendment K convention."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(got.stdout, encoding="utf-8")
        print(f"[2/5] wrote {target.relative_to(PROJECT_ROOT)}")
        return target
    print(f"[2/5] no summary produced for {unit} (unit not complete?)")
    return None


def recatalogue(before: dict[str, float]) -> bool:
    print("[3/5] rebuilding the catalogue")
    result = _run(
        [sys.executable, "scripts/build_paper_numbers.py", "--out", str(CATALOGUE)]
    )
    print(result.stdout.strip() or result.stderr[-600:])
    after = _snapshot()
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    moved = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    if not (added or removed or moved):
        print("      no catalogue change")
        return False
    print(
        f"\n      CHANGED: {len(moved)} moved, {len(added)} added, "
        f"{len(removed)} removed"
    )
    for key in moved:
        print(f"        ~ {key}: {before[key]:.6g} -> {after[key]:.6g}")
    for key in added:
        print(f"        + {key}: {after[key]:.6g}")
    for key in removed:
        print(f"        - {key} (was {before[key]:.6g})")
    return True


def gate_and_build() -> int:
    print("\n[4/5] audit gate")
    audit = _run([sys.executable, "scripts/audit_paper_numbers.py", "--strict"])
    tail = "\n".join(audit.stdout.strip().splitlines()[-6:])
    print(tail)
    if audit.returncode != 0:
        print("      AUDIT FAILED -- a number in the paper no longer has a source.")
        print("      Fix the prose before rebuilding; do not proceed on a red gate.")
        return audit.returncode

    print("\n[5/5] figures and PDF")
    figs = _run(
        [sys.executable, "scripts/make_paper_figures.py", "--out-dir", "figures"]
    )
    if figs.returncode != 0:
        print(figs.stdout[-600:] or figs.stderr[-600:])
        return figs.returncode
    print("      figures rebuilt from the catalogue")
    if shutil.which("pdflatex"):
        for _ in range(2):
            _run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "kv_readout_vs_trajectory_paper.tex",
                ],
                timeout=900,
            )
        log = PROJECT_ROOT / "kv_readout_vs_trajectory_paper.log"
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="ignore")
            errors = sum(1 for ln in text.splitlines() if ln.startswith("!"))
            pages = [ln for ln in text.splitlines() if "pages" in ln][-1:]
            page_line = pages[0].strip() if pages else "?"
            print(f"      PDF rebuilt: {errors} errors; {page_line}")
    else:
        print("      pdflatex not on PATH; skipped (rebuild before committing)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True, help="harvest unit name")
    parser.add_argument(
        "--local-only", action="store_true", help="skip the remote harvest"
    )
    args = parser.parse_args()

    before = _snapshot()
    if not args.local_only:
        harvest(args.unit)
        if fetch_summary(args.unit) is None:
            print("\nnothing to integrate.")
            return 0
    changed = recatalogue(before)
    if not changed and not args.local_only:
        print("\nnothing moved; skipping rebuild.")
        return 0
    rc = gate_and_build()
    print(
        "\nNEXT: open docs/operations/INTEGRATION_CHECKLIST.md at this unit. "
        "It holds the "
        "pre-registered read and the wording for each outcome. Do not re-derive "
        "the read."
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

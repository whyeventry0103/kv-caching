#!/usr/bin/env python3
"""Emit the Convention-2 additivity (interaction) test for a budget-256 cell.

Amendment P.1. ``readout_error`` and ``residual_rev`` are measured independently
of each other and of ``Delta_tot``, so their sum against ``Delta_tot`` is a real
test. The shortfall is the interaction, and its SIGN is the claim.

Runs on the remote, over raw rows, because the stored summaries give each term
on its own Amendment-K population -- three different denominators -- and summing
across those conflates a real interaction with a population mismatch. The
primary estimand here is the common population; the per-term version is emitted
alongside for disclosure.

    python scripts/report_additivity.py --runs-root <runs> \\
        --fullcache math500-fc-powered-merged --compression grid-diss256-tok \\
        --readout readout-diss256-g0 readout-diss256-g1 \\
        --reverse reverse-readout-diss256 --press random --budget 256 \\
        --out additivity_random_256.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kvcache.additivity import additivity_test, interaction_on  # noqa: E402
from report_m4 import (  # noqa: E402
    _condition_from_readout,
    _reference_cell_from_readout,
    _reference_key_from_readout,
    _run_key,
)

Key = tuple[str, int]


def _rows(root: Path, sub: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted(glob.glob(str(root / sub / "**" / "*.jsonl"), recursive=True)):
        if "token_ids" in Path(path).parts:
            continue
        with open(path, encoding="utf-8") as handle:
            out += [json.loads(ln) for ln in handle if ln.strip()]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--fullcache", required=True)
    p.add_argument("--compression", required=True)
    p.add_argument("--readout", required=True, nargs="+")
    p.add_argument("--reverse", required=True)
    p.add_argument("--press", required=True)
    p.add_argument("--budget", required=True)
    p.add_argument("--bench", default="math500")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    R = a.runs_root

    fc = {_run_key(r): bool(r["correct"]) for r in _rows(R / a.fullcache, a.bench)}
    free = {
        _run_key(r): bool(r["correct"])
        for r in _rows(R / a.compression, a.bench)
        if r["press"] == a.press and str(r["budget"]) == str(a.budget)
    }

    readout_rows = [r for root in a.readout for r in _rows(R / root, "readout")]
    readout = {
        _reference_key_from_readout(r): bool(r["readout_correct"])
        for r in readout_rows
        if r.get("readout_correct") is not None
        and _condition_from_readout(r) == (a.press, str(a.budget))
    }
    # A REVERSE readout is keyed by the cell of the compressed trace it read out,
    # not by its own (full-cache) readout condition -- matching on the latter
    # would silently pair a reverse row with the wrong press.
    reverse = {
        _reference_key_from_readout(r): bool(r["readout_correct"])
        for r in _rows(R / a.reverse, "readout")
        if r.get("readout_correct") is not None
        and _reference_cell_from_readout(r) == (a.press, str(a.budget))
    }

    for name, table in (("free", free), ("readout", readout), ("reverse", reverse)):
        if not table:
            raise SystemExit(f"no {name} rows for press={a.press} budget={a.budget}")

    primary = additivity_test(fc, free, readout, reverse)

    # Disclosure: the same terms on each protocol's OWN population, which is what
    # the stored summaries report. The gap between this and `primary` is exactly
    # the population-mismatch component that must not be read as interaction.
    per_term = {
        "delta_tot": interaction_on(sorted(free), fc, free, free, free)["delta_tot"],
        "readout_error": interaction_on(sorted(readout), fc, readout, readout, readout)[
            "delta_tot"
        ],
        "residual_rev": interaction_on(sorted(reverse), fc, reverse, reverse, reverse)[
            "delta_tot"
        ],
    }
    per_term["sum_of_terms"] = per_term["readout_error"] + per_term["residual_rev"]
    per_term["interaction"] = per_term["delta_tot"] - per_term["sum_of_terms"]
    per_term["n_free"] = len(free)
    per_term["n_readout"] = len(readout)
    per_term["n_reverse"] = len(reverse)
    per_term["note"] = (
        "Each term on its own Amendment-K population (three different "
        "denominators). Reported for disclosure only; NOT the estimand."
    )

    payload = {
        "press": a.press,
        "budget": str(a.budget),
        "benchmark": a.bench,
        "amendment": "P.1",
        "reference_convention": "amendment-k-cell-population",
        "primary_common_population": primary,
        "secondary_per_term_population": per_term,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

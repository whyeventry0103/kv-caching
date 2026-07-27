#!/usr/bin/env python3
"""Gate: every numeric literal in the paper must map to a catalogued source.

Thin orchestration only -- extraction and classification live in
``kvcache.paper_numbers``.

    python scripts/build_paper_numbers.py
    python scripts/audit_paper_numbers.py            # human-readable report
    python scripts/audit_paper_numbers.py --strict   # non-zero exit if unsourced

A literal is *sourced* if its value matches a catalogue entry at any reasonable
display precision, *protocol* if it is an enumerated config constant (budget,
seed, cell size, token cap), *year* if it is plainly a date, and *unsourced*
otherwise. Unsourced is a bug until it is classified -- either the number needs
an artifact, or it belongs on the protocol list with a reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.paper_numbers import Entry, classify, extract_literals, summarise

# Protocol constants that legitimately live in configs rather than in a results
# artifact. Enumerated so that "it's just a config number" is a decision made
# once, on a list, rather than per literal.
PROTOCOL_INTEGERS = {
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,  # counts, enumerations, panel indices
    11,
    23,
    47,
    89,
    131,
    197,
    263,
    347,  # the frozen seed set
    12,
    15,
    16,
    20,
    24,
    25,
    30,
    32,
    40,
    50,  # small counts (problems, windows)
    60,
    64,
    65,
    93,
    95,
    100,
    105,
    120,  # cell sizes, CI level, onset step
    128,
    154,
    180,
    200,
    213,
    240,
    250,
    256,  # budgets, populations
    308,
    320,
    346,
    360,
    372,
    384,
    392,
    400,  # defined-row counts, cell rows
    500,
    512,
    800,
    1024,
    2000,
    2048,  # benchmark size, budgets
    4096,
    8192,
    16384,
    32768,  # token caps
    1500,
    7000,
    8000,
    14000,  # model sizes in millions/labels
}


def load_catalogue(path: Path) -> list[Entry]:
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/build_paper_numbers.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Entry(
            key=e["key"],
            value=float(e["value"]),
            kind=e["kind"],
            source=e["source"],
            description=e.get("description", ""),
            inputs=tuple(e.get("inputs", ())),
        )
        for e in payload["entries"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue", type=Path, default=PROJECT_ROOT / "paper_numbers.json"
    )
    parser.add_argument(
        "--documents",
        type=Path,
        nargs="+",
        default=[
            PROJECT_ROOT / "kv_readout_vs_trajectory_paper.tex",
            PROJECT_ROOT / "PAPER_FINDINGS.md",
        ],
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--show", type=int, default=40, help="unsourced rows to print")
    args = parser.parse_args()

    catalogue = load_catalogue(args.catalogue)
    literals = []
    for doc in args.documents:
        if not doc.exists():
            raise SystemExit(f"missing document: {doc}")
        text = doc.read_text(encoding="utf-8")
        literals.extend(extract_literals(text, doc.name, strip=doc.suffix == ".tex"))
    classify(literals, catalogue, allowed_integers=PROTOCOL_INTEGERS)
    report = summarise(literals)

    print(
        f"catalogue : {len(catalogue)} entries "
        f"({sum(1 for e in catalogue if e.kind == 'derived')} derived)"
    )
    print(f"documents : {', '.join(d.name for d in args.documents)}")
    # State the counting basis in the output. Three mutually inconsistent totals
    # for "the number of literals" were once in circulation at the same time,
    # because occurrences, distinct values, and a sourced-count that also
    # counted the stale subset were all reported as if they were the same
    # quantity. These counts are now part of the paper's provenance story, so
    # the definition travels with them.
    distinct = len({lit.raw for lit in literals})
    print(
        f"basis     : OCCURRENCES of numeric literals in the stripped text "
        f"({distinct} distinct raw forms); buckets below partition the total"
    )
    print(f"literals  : {report['total']}")
    for name, count in report["by_classification"].items():
        print(f"  {name:<12} {count:>5}")
    total_check = sum(report["by_classification"].values())
    if total_check != report["total"]:  # pragma: no cover - arithmetic guard
        raise SystemExit(f"bucket sum {total_check} != total {report['total']}")

    unsourced = report["unsourced"]
    if unsourced:
        print(f"\nUNSOURCED ({len(unsourced)}) -- each is a bug until classified:")
        by_file: dict[str, list[dict]] = {}
        for row in unsourced:
            by_file.setdefault(row["file"], []).append(row)
        for fname, rows in by_file.items():
            print(f"\n  == {fname} ({len(rows)}) ==")
            for row in rows[: args.show]:
                print(
                    f"    L{row['line']:<5} {row['value']:>12}   {row['context'][:88]}"
                )
            if len(rows) > args.show:
                print(f"    ... {len(rows) - args.show} more")

    if args.strict and unsourced:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

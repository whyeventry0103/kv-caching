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

from kvcache.paper_numbers import (
    REFERENCE_RELATIVE_ARTIFACTS,
    Entry,
    check_guarded_section,
    check_reference_conventions,
    classify,
    extract_literals,
    summarise,
)

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
    130,  # restoration trigger step (2x median onset; exact_lag trigger_step)
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
    # Undefined-boundary counts, quoted as raw counts out of a cell. Each is
    # 400 (or 120) minus a catalogued defined-row count, so it is arithmetic on
    # protocol quantities rather than an independent measurement.
    27,
    28,
    46,
    54,
    85,
    92,
    # Bootstrap and permutation parameters. These are method, fixed in
    # PREREGISTRATION, not results: raising the resample count would be a
    # protocol deviation touching every interval in the paper.
    10000,
    # Pilot exchangeability check (19/20 vs 20/20), stated as a limitation.
    19,
    # Repo metadata quoted in README badges and prose: the passing-test count,
    # the re-derived-artifact count, and the pinned-claim count. These describe
    # the repository, not a measurement, and they move when the repo moves.
    236,
    79,
    29,
}

# Protocol constants that are not integers: decoding parameters and the
# percentile indices of the bootstrap. Same status as the integers above --
# enumerated so "it's just a config number" stays a decision on a list.
PROTOCOL_FLOATS = {
    0.6,  # sampling temperature
    0.95,  # top-p
    2.5,
    97.5,  # percentile-interval indices
    # Pre-registered decision thresholds. These are protocol -- the value was
    # fixed in PREREGISTRATION before the data existed -- and must not be
    # sourced to whichever measurement happens to equal them. The 0.5 readout-
    # share threshold was being credited to a budget-256 accuracy CI bound.
    0.5,
    0.75,
    # Python version in the README badge -- a dependency pin, not a result.
    3.11,
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
            PROJECT_ROOT / "docs" / "results" / "PAPER_FINDINGS.md",
            # The README is the repo's front door and is hand-written, which is
            # exactly why it needs the gate: it had drifted to a superseded set
            # of residuals and a superseded resampling effect while every gated
            # document was correct. Its result numbers are now checked too.
            PROJECT_ROOT / "README.md",
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
    classify(
        literals,
        catalogue,
        allowed_integers=PROTOCOL_INTEGERS,
        allowed_floats=PROTOCOL_FLOATS,
    )
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

    # The 4-CURRENT gate. The dated log in docs/results/PAPER_FINDINGS.md is
    # deliberately exempt
    # from the literal check, which leaves a hole: anything wrapped in an
    # archival marker stops being checked. The current-truth section a new
    # session reads as authoritative must not be able to use that hole, so it is
    # checked separately with exemptions refused.
    guarded_problems: list[str] = []
    for doc in args.documents:
        if doc.name != "PAPER_FINDINGS.md":
            continue
        guarded_problems = check_guarded_section(
            doc.read_text(encoding="utf-8"), catalogue
        )
        if guarded_problems:
            print(f"\n4-CURRENT GATE FAILED ({len(guarded_problems)}):")
            for problem in guarded_problems:
                print(f"  {problem}")
        else:
            print(
                "\n4-CURRENT gate: clean (no archival exemption, all literals sourced)"
            )

    # Amendment K enforcement. The defect class is "a quantity compared against
    # the reference", and the fix was once applied to nine cells and stopped
    # there. Every artifact in that class must now declare the convention it was
    # computed under, so an undeclared or stale one fails the gate rather than
    # waiting to be noticed.
    diag = PROJECT_ROOT / "results" / "diag"
    loaded: dict[str, dict] = {}
    for name in REFERENCE_RELATIVE_ARTIFACTS:
        for candidate in (diag / "amendment-k" / name, diag / name):
            if candidate.exists():
                loaded[name] = json.loads(candidate.read_text(encoding="utf-8"))
                break
    convention_problems = check_reference_conventions(loaded)
    if convention_problems:
        print(f"\nREFERENCE-CONVENTION GATE FAILED ({len(convention_problems)}):")
        for problem in convention_problems:
            print(f"  {problem}")
    else:
        print(
            f"reference-convention gate: clean "
            f"({len(REFERENCE_RELATIVE_ARTIFACTS)} artifacts declare "
            f"amendment-k-cell-population)"
        )

    # Attribution strength. "Sourced" means a catalogue entry can produce this
    # display string -- NOT that it is the right entry. The paper's 0.045
    # (0.515 - 0.47) attributes to llama.random.readout_ci1, an unrelated CI
    # bound that happens to equal 0.045, and a derived hottcap.residual_gap entry
    # does not displace it because exact-match precedence prefers the entry whose
    # stored float is exactly 0.045. Value-based matching cannot separate two
    # quantities that share a value. This does not gate -- it reports, so the
    # green tick is not read as more assurance than it carries.
    by_display: dict[str, set[str]] = {}
    for entry in catalogue:
        for form in entry.display_rules():
            by_display.setdefault(form, set()).add(entry.key)
    ambiguous = 0
    for lit in literals:
        raw = lit.raw.lstrip("+-")
        if "." not in raw or len(raw.split(".")[1]) < 3:
            continue  # coarse forms collide by nature; not the risk class
        if len(by_display.get(raw, set())) > 1:
            ambiguous += 1
    print(
        f"attribution strength: {ambiguous} of the result-precision "
        f"(3+ decimal) literal occurrences match MORE THAN ONE catalogue "
        f"entry, so 'sourced' does not pin them to one quantity"
    )

    if args.strict and (unsourced or guarded_problems or convention_problems):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

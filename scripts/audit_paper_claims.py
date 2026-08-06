#!/usr/bin/env python3
"""Gate: each load-bearing number asserts the quantity it is supposed to assert.

`audit_paper_numbers.py` answers "does this literal match SOME catalogue entry".
That is not the question a reader cares about. The paper's $0.045$ passed that
gate while matching an unrelated Llama CI bound, and 360 result-precision
literals match more than one entry, so value-matching alone cannot pin a number
to a quantity.

This gate closes that for the numbers that carry the paper's claims.
`paper_claims.json` names, per claim, the sentence it lives in and the catalogue
key it asserts; a claim passes only if the named key can display that literal.
A number whose catalogue entry has drifted now fails here even though its value
still collides with something else in the catalogue.

    python scripts/audit_paper_claims.py            # report
    python scripts/audit_paper_claims.py --strict   # non-zero exit on failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.paper_numbers import Entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims", type=Path, default=PROJECT_ROOT / "paper_claims.json"
    )
    parser.add_argument(
        "--catalogue", type=Path, default=PROJECT_ROOT / "paper_numbers.json"
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    catalogue = {
        e["key"]: Entry(
            key=e["key"],
            value=float(e["value"]),
            kind=e["kind"],
            source=e["source"],
            description=e.get("description", ""),
        )
        for e in json.loads(args.catalogue.read_text(encoding="utf-8"))["entries"]
    }
    documents = {
        "kv_readout_vs_trajectory_paper.tex": (
            PROJECT_ROOT / "kv_readout_vs_trajectory_paper.tex"
        ).read_text(encoding="utf-8"),
        "docs/results/PAPER_FINDINGS.md": (
            PROJECT_ROOT / "docs" / "results" / "PAPER_FINDINGS.md"
        ).read_text(encoding="utf-8"),
        # The README is the most-read surface in the repo, so its headline
        # table is pinned to intent like the paper's own.
        "README.md": (PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    }

    claims = json.loads(args.claims.read_text(encoding="utf-8"))["claims"]
    problems: list[str] = []
    for i, claim in enumerate(claims):
        doc = claim.get("document", "kv_readout_vs_trajectory_paper.tex")
        text = documents[doc]
        context, literal, key = claim["context"], claim["literal"], claim["key"]
        where = f"claim {i} [{key}]"

        occurrences = text.count(context)
        if occurrences == 0:
            problems.append(f"{where}: context not found in {doc} -- {context[:60]!r}")
            continue
        if occurrences > 1:
            # An ambiguous context could match a different sentence later, which
            # would silently verify the wrong site.
            problems.append(f"{where}: context appears {occurrences}x in {doc}")
            continue
        # LaTeX writes a thousands separator as 18{,}870; the catalogue displays
        # 18870. Strip only that construct -- a plain comma is left alone so an
        # interval like [0.130, 0.240] does not fuse into one meaningless run.
        if literal not in context.replace("{,}", ""):
            problems.append(f"{where}: literal {literal!r} absent from its own context")
            continue
        entry = catalogue.get(key)
        if entry is None:
            problems.append(f"{where}: key not in catalogue")
            continue
        forms = entry.display_rules()
        if literal.lstrip("+-") not in forms:
            problems.append(
                f"{where}: asserts {literal!r} but {key} = {entry.value!r} "
                f"(displays: {sorted(forms)[:6]})"
            )

    # Coverage. A pinned number is verified against its intended quantity; an
    # unpinned one rests on value-matching alone, which cannot distinguish two
    # quantities that round the same way. Reporting the gap is the point --
    # a gate that hides how much it does not cover is worse than no gate.
    from kvcache.paper_numbers import extract_literals

    tex = documents["kv_readout_vs_trajectory_paper.tex"]
    literals = extract_literals(tex, "tex", strip=True)
    by_display: dict[str, set[str]] = {}
    for entry in catalogue.values():
        for form in entry.display_rules():
            by_display.setdefault(form, set()).add(entry.key)
    pinned = {c["literal"].lstrip("+-") for c in claims}
    risky = [
        lit
        for lit in literals
        if "." in lit.raw
        and len(lit.raw.split(".")[1]) >= 3
        and len(by_display.get(lit.raw.lstrip("+-"), set())) > 1
        and lit.raw.lstrip("+-") not in pinned
    ]
    print(f"claims checked : {len(claims)}")
    print(f"catalogue      : {len(catalogue)} entries")
    print(
        f"coverage       : {len(risky)} result-precision literal occurrences in "
        f"the .tex are BOTH ambiguous (match >1 entry) and unpinned by a claim"
    )
    if problems:
        print(f"\nCLAIM GATE FAILED ({len(problems)}):")
        for problem in problems:
            print(f"  {problem}")
    else:
        print(
            "claim gate: clean -- every pinned number displays the value of the "
            "quantity it claims to be"
        )
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())

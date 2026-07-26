"""Provenance for every number that appears in the paper.

Amendment K invalidated a whole class of reported quantities at once, and a
manual pass would have preserved at least one sentence that had become false
(the budget-2048 "nominally above the reference" clause, whose supporting value
changed sign). The lesson generalises: *any* systemic correction invalidates
every number downstream of it, and prose numbers do not announce themselves.

This module supports a catalogue mapping each number in the paper to the
artifact it came from, and an audit that finds numeric literals in the ``.tex``
with no entry in that catalogue. The gate is deliberately blunt: a numeric
literal without a source is a bug until classified.

Two entry kinds:

* **primary** -- read straight out of a report artifact at a JSON path.
* **derived** -- computed from other entries by a named formula. These are the
  dangerous ones (ratios, spreads, factors): they read as prose, they survive a
  visual pass, and regenerating a primary value silently invalidates them. A
  derived entry records its inputs, so recomputing the catalogue recomputes
  them and any drift shows up as a changed value rather than as a stale claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Numeric literal, allowing LaTeX's escaped comma-thousands (16{,}384) and a
# leading sign. Kept deliberately greedy: over-matching is cheap to classify,
# under-matching hides an unsourced number.
_NUMBER = re.compile(r"-?\d+(?:\{,\}\d{3})*(?:\.\d+)?")

# A leading "-" is only a minus sign when it does not follow a word character,
# another dash, or a closing brace: "MATH-500", "128--512" and "Qwen-1.5B" are
# names and ranges, not negative numbers. Getting this wrong invented ~90 false
# positives on the first run and would have buried the real ones.
_HYPHEN_IN_WORD = re.compile(r"[0-9A-Za-z}\-]$")
# Version strings: a number followed by a further ".<digits>" component.
_VERSION_TAIL = re.compile(r"^\.\d")

# Contexts whose numbers are structural rather than claims: lengths, font
# sizes, graphics options, cross-references, and package/version pins.
_STRUCTURAL_UNITS = re.compile(
    r"(?:pt|em|ex|cm|mm|in|bp|sp|\\textwidth|\\linewidth|\\columnwidth|\\baselineskip)\b"
)
_STRIP_COMMANDS = (
    "cite",
    "citep",
    "citet",
    "label",
    "ref",
    "cref",
    "Cref",
    "autoref",
    "eqref",
    "includegraphics",
    "usepackage",
    "documentclass",
    "definecolor",
    "hypersetup",
    "geometry",
    "setlength",
    "vspace",
    "hspace",
    "url",
    "href",
)


@dataclass(frozen=True)
class Entry:
    """One catalogued number."""

    key: str
    value: float
    kind: str  # "primary" | "derived"
    source: str  # artifact path, or formula description for derived
    description: str = ""
    inputs: tuple[str, ...] = ()

    def displays(self, places: tuple[int, ...] = (0, 1, 2, 3, 4)) -> set[str]:
        """Every string form this value could legitimately appear as.

        The repo's display convention (PAPER_FINDINGS §5) is ``f"{v:.3f}"`` of
        the artifact float, but the same value legitimately appears rounded to
        fewer places in prose ("roughly -0.34"). Both signs count, because a
        magnitude is often quoted without its sign ("a residual of 0.34").

        Percentage forms are deliberately NOT generated here. Multiplying every
        entry by 100 made ``0.015`` match the "1.5" in "Qwen-1.5B" and produced
        a whole class of false matches; quantities that are genuinely quoted as
        percentages get their own catalogue entries instead.
        """
        out: set[str] = set()
        for p in places:
            out.add(f"{self.value:.{p}f}")
            out.add(f"{abs(self.value):.{p}f}")
        return {s.lstrip("+") for s in out}


@dataclass
class Literal:
    """A numeric literal found in a document, with enough context to judge it."""

    raw: str
    value: float
    context: str
    line: int
    source_file: str
    classification: str = "unsourced"
    matched_keys: tuple[str, ...] = field(default_factory=tuple)


def strip_latex_noise(text: str) -> str:
    """Remove comments, the preamble, and commands whose numbers are structural.

    Everything removed here is replaced by a space rather than deleted, so line
    numbers and rough column context survive for reporting.
    """

    def blank(match: re.Match[str]) -> str:
        # Preserve newlines, or every line number after the blanked span shifts
        # and the audit report points at the wrong line.
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    # Drop the preamble: claims live between \begin{document} and \end{document}.
    start = text.find(r"\begin{document}")
    if start != -1:
        head = text[:start]
        text = "".join("\n" if ch == "\n" else " " for ch in head) + text[start:]
    # Strip comments (an unescaped % to end of line).
    text = re.sub(r"(?<!\\)%[^\n]*", blank, text)
    # Strip structural commands together with their bracketed/braced arguments.
    for name in _STRIP_COMMANDS:
        pattern = re.compile(
            r"\\" + name + r"\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", re.DOTALL
        )
        text = pattern.sub(blank, text)
    return text


def extract_literals(
    text: str, source_file: str, *, strip: bool = True
) -> list[Literal]:
    """Find every numeric literal that could be a claim."""
    cleaned = strip_latex_noise(text) if strip else text
    lines = cleaned.splitlines()
    out: list[Literal] = []
    for lineno, line in enumerate(lines, start=1):
        for match in _NUMBER.finditer(line):
            raw = match.group(0)
            start = match.start()
            if raw.startswith("-") and _HYPHEN_IN_WORD.search(line[:start] or " "):
                # A hyphen inside a name or range; re-read without the sign.
                raw = raw[1:]
                start += 1
            tail = line[match.end() : match.end() + 14]
            if _STRUCTURAL_UNITS.match(tail.lstrip()) or _VERSION_TAIL.match(tail):
                continue
            try:
                value = float(raw.replace("{,}", ""))
            except ValueError:  # pragma: no cover - regex guarantees parseability
                continue
            lo = max(0, match.start() - 45)
            out.append(
                Literal(
                    raw=raw,
                    value=value,
                    context=line[lo : match.end() + 45].strip(),
                    line=lineno,
                    source_file=source_file,
                )
            )
    return out


def classify(
    literals: list[Literal],
    catalogue: list[Entry],
    *,
    allowed_integers: set[int] | None = None,
) -> list[Literal]:
    """Mark each literal sourced, structural, protocol, or unsourced.

    ``allowed_integers`` are protocol constants that live in configs rather than
    in a results artifact -- budgets, seeds, cell sizes, token caps. They are
    legitimately not in the catalogue, but they are enumerated explicitly so
    that "it's just a config number" is a decision on a list rather than a
    judgement made per literal.
    """
    allowed = allowed_integers or set()
    by_display: dict[str, list[str]] = {}
    for entry in catalogue:
        for form in entry.displays():
            by_display.setdefault(form, []).append(entry.key)

    for lit in literals:
        stripped = lit.raw.replace("{,}", "").lstrip("-")
        keys = by_display.get(stripped) or by_display.get(lit.raw.replace("{,}", ""))
        if keys:
            lit.classification = "sourced"
            lit.matched_keys = tuple(sorted(set(keys)))
            continue
        if lit.value.is_integer() and int(lit.value) in allowed:
            lit.classification = "protocol"
            continue
        if lit.value.is_integer() and 1900 <= lit.value <= 2100:
            lit.classification = "year"
            continue
        lit.classification = "unsourced"
    return literals


def summarise(literals: list[Literal]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for lit in literals:
        counts[lit.classification] = counts.get(lit.classification, 0) + 1
    return {
        "total": len(literals),
        "by_classification": dict(sorted(counts.items())),
        "unsourced": [
            {
                "value": lit.raw,
                "file": lit.source_file,
                "line": lit.line,
                "context": lit.context,
            }
            for lit in literals
            if lit.classification == "unsourced"
        ],
    }

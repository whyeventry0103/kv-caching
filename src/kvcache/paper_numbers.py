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

# Environment arguments that are typesetting parameters, not claims. The
# bibliography's "{99}" is a label-width hint; it was matching a derived ratio
# rounded to zero places (98.74 -> "99") and being counted as *sourced*, which
# is worse than being counted as unsourced -- it inflates the audit's own
# success metric with a number that is not a claim at all.
_STRUCTURAL_ENVIRONMENTS = re.compile(r"\\begin\{thebibliography\}\{[^{}]*\}")

# The whole bibliography body, not just its label width. Every number between
# \begin{thebibliography} and \end{thebibliography} is an arXiv identifier, a
# volume, or a year -- 13 arXiv IDs were being reported as unsourced claims,
# which is the bibliography-label bug again with a wider blast radius. Nothing
# in a reference list is a claim about our results.
_BIBLIOGRAPHY_BODY = re.compile(
    r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL
)

# arXiv identifiers outside a bibliography (PAPER_FINDINGS.md cites them inline).
_ARXIV_ID = re.compile(r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?", re.IGNORECASE)

# Bare arXiv identifiers in a table column, with no "arXiv:" prefix.
_BARE_ARXIV_ID = re.compile(r"\b\d{4}\.\d{5}\b")

# Section references and ISO dates. A cross-reference and a date are addresses,
# not measurements; "§4.1" and "2026-07-17" were reporting 4.1 and 17 as
# unsourced claims. Same class as the arXiv identifiers.
_SECTION_REF = re.compile(r"§\s?\d+(?:\.\d+)*[a-z]?")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# An explicitly marked archival span. PAPER_FINDINGS.md's dated read-in log is a
# HISTORICAL RECORD: each entry is correct as of its date, and several exist
# precisely to show a superseded value beside its correction (Amendment H's
# bracket, the -0.465/-0.466 decimal). Rewriting those to current values would
# delete the honesty trail this project treats as its moat -- the gate's job is
# to stop stale numbers reaching the PAPER, not to launder the log that records
# them. So an archival span is skipped, and the marker makes that a deliberate,
# reviewable declaration in the document rather than a silent exemption here.
_ARCHIVAL_SPAN = re.compile(
    r"<!--\s*AUDIT:ARCHIVAL-BEGIN\s*-->.*?<!--\s*AUDIT:ARCHIVAL-END\s*-->",
    re.DOTALL,
)

# Contexts whose numbers are structural rather than claims: lengths, font
# sizes, graphics options, cross-references, and package/version pins.
#
# ``B`` and ``M`` are here because a model scale is a name, not a measurement:
# "1.5B/7B/14B" and "PG19, streaming to 4M tokens" contributed a dozen phantom
# unsourced literals. A parameter count is not a quantity this paper reports.
_STRUCTURAL_UNITS = re.compile(
    r"(?:pt|em|ex|cm|mm|in|bp|sp|\\textwidth|\\linewidth|\\columnwidth|\\baselineskip)\b"
)

# A model scale or token-count suffix fused to the number: "1.5B", "14B", "4M".
# Checked against the UNSTRIPPED tail so a space breaks it -- "0.343 B" in a
# table column is a number followed by a cell, not a parameter count.
_SCALE_SUFFIX = re.compile(r"[BMk]\b")

# A number glued to the end of a word is part of an identifier, never a claim:
# PG19, R1-Distill, H2O. Claim numbers are always preceded by whitespace or an
# opening delimiter.
_IDENTIFIER_PREFIX = re.compile(r"[A-Za-z]$")
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

        **A rounding that destroys the value is not a display of it.** For
        ``|value| < 1`` the zero-place form is always "0" or "1", so emitting it
        made every bare 0 and 1 in the prose match some residual or accuracy:
        418 occurrences -- 65% of the reported "sourced" count -- were config
        constants credited to unrelated entries. Nobody writes an accuracy of
        0.882 as "1".

        **The zero-place form is generated only for integer-valued entries.**
        An earlier version of this rule dropped it merely for ``|value| < 1``,
        which still let a ratio of 6.7 claim every bare "7" in the prose. The
        honest rule is that rounding a fractional quantity to zero places
        destroys it, whatever its magnitude; counts (wins, problems, profiles,
        the onset step) are genuinely integers and still match their bare form.
        """
        out: set[str] = set()
        if self.value.is_integer():
            out.add(f"{self.value:.0f}")
            out.add(f"{abs(self.value):.0f}")
        for p in places:
            if p == 0:
                continue
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
    # Strip the bibliography: its label width, then its entire body.
    text = _STRUCTURAL_ENVIRONMENTS.sub(blank, text)
    text = _BIBLIOGRAPHY_BODY.sub(blank, text)
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

    # An arXiv identifier is a name. Stripped for both document kinds: the .tex
    # keeps them in the bibliography (already blanked) but PAPER_FINDINGS.md
    # cites them inline, where no LaTeX stripping runs.
    def _blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    cleaned = _ARXIV_ID.sub(_blank, cleaned)
    cleaned = _BARE_ARXIV_ID.sub(_blank, cleaned)
    cleaned = _SECTION_REF.sub(_blank, cleaned)
    cleaned = _ISO_DATE.sub(_blank, cleaned)
    cleaned = _ARCHIVAL_SPAN.sub(_blank, cleaned)
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
            if _SCALE_SUFFIX.match(tail):
                continue
            if _IDENTIFIER_PREFIX.search(line[:start] or " "):
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
    allowed_floats: set[float] | None = None,
) -> list[Literal]:
    """Mark each literal sourced, structural, protocol, or unsourced.

    ``allowed_integers`` are protocol constants that live in configs rather than
    in a results artifact -- budgets, seeds, cell sizes, token caps. They are
    legitimately not in the catalogue, but they are enumerated explicitly so
    that "it's just a config number" is a decision on a list rather than a
    judgement made per literal.

    **Protocol is checked before sourced, and the order is load-bearing.** A
    catalogued percentage rounds to a bare integer at zero places (3.157% ->
    "3", 29.8% -> "30"), so matching sourced first let a plain count in the
    prose claim provenance from an unrelated ratio. That inflates the gate's own
    success metric with numbers that were never in question -- the same failure
    mode as the bibliography label width, one layer up. An enumerated config
    constant is a config constant regardless of what else happens to round to it.
    """
    allowed = allowed_integers or set()
    allowed_f = allowed_floats or set()
    by_display: dict[str, list[str]] = {}
    external_keys = {e.key for e in catalogue if e.kind == "external"}
    exact_value = {e.key: e.value for e in catalogue}
    for entry in catalogue:
        for form in entry.displays():
            by_display.setdefault(form, []).append(entry.key)

    for lit in literals:
        if lit.value.is_integer() and int(lit.value) in allowed:
            lit.classification = "protocol"
            continue
        if lit.value in allowed_f:
            lit.classification = "protocol"
            continue
        stripped = lit.raw.replace("{,}", "").lstrip("-")
        keys = by_display.get(stripped) or by_display.get(lit.raw.replace("{,}", ""))
        if keys:
            # An EXACT value match outranks one that only holds after rounding.
            # Without this, a literal is attributed to whichever catalogue entry
            # happens to round onto it -- provenance by coincidence. Sampling the
            # rendered PDF found two such attributions (0.158 credited to an AIME
            # flip-mass rather than the paired CI bound it actually is), and while
            # the value was right in both cases, a gate that names the wrong
            # source cannot help anyone checking the number.
            unique = sorted(set(keys))
            exact = [k for k in unique if exact_value.get(k) == lit.value]
            candidates = exact or unique
            # A number we measured outranks one we transcribed, and a match that
            # is ONLY external is reported as such: its verification path is
            # different in kind -- a human must open the cited table and check
            # the transcription, which no gate here can do.
            internal = [k for k in candidates if k not in external_keys]
            lit.classification = "sourced" if internal else "external"
            lit.matched_keys = tuple(candidates)
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


# A section whose literals must all be sourced AND inside which no archival
# exemption is permitted. The archival mechanism is necessary for the dated log
# but it is also a hole: anything wrapped in it stops being checked. A
# current-truth section that could quietly wrap its own numbers would be worse
# than no section at all, because it reads as gated while not being gated.
_GUARDED_SECTION_START = re.compile(r"^#{1,6}\s*4-CURRENT\b.*$", re.MULTILINE)
_GUARDED_SECTION_END = re.compile(r"^#{1,6}\s+(?!4-CURRENT)\S.*$", re.MULTILINE)


def guarded_section(text: str) -> str | None:
    """The current-truth span of PAPER_FINDINGS.md, or None if absent.

    Runs from the ``4-CURRENT`` heading to the next heading of any level. The
    span is returned RAW -- archival spans inside it are deliberately not
    stripped, so :func:`check_guarded_section` can refuse them.
    """
    start = _GUARDED_SECTION_START.search(text)
    if start is None:
        return None
    rest = text[start.end() :]
    end = _GUARDED_SECTION_END.search(rest)
    return rest[: end.start()] if end else rest


def check_guarded_section(text: str, catalogue: list[Entry]) -> list[str]:
    """Problems with the current-truth section. Empty list means it is clean.

    Two rules, both hard:

    1. no ``AUDIT:ARCHIVAL`` marker may appear inside it -- superseded values
       belong in the log, not in the section a new session reads as truth;
    2. every numeric literal in it must be sourced or protocol.
    """
    section = guarded_section(text)
    if section is None:
        return ["PAPER_FINDINGS.md has no 4-CURRENT section (the gate needs one)"]
    problems: list[str] = []
    if "AUDIT:ARCHIVAL" in section:
        problems.append(
            "4-CURRENT contains an AUDIT:ARCHIVAL marker: the current-truth "
            "section may not exempt itself from the gate"
        )
    literals = extract_literals(section, "PAPER_FINDINGS.md#4-CURRENT", strip=False)
    classify(literals, catalogue, allowed_integers=set(range(0, 3000)))
    for lit in literals:
        if lit.classification == "unsourced":
            problems.append(
                f"4-CURRENT has an unsourced literal {lit.raw!r}: {lit.context[:70]}"
            )
    return problems


# Artifacts that carry at least one reference-relative quantity -- a value
# formed by comparing something against the full-cache reference. Amendment K
# was a defect in exactly this class, and it was applied to the decomposition
# cells and stopped there, leaving the retired comparison inside
# fc-readout-control for weeks. Enumerating the class and checking it
# mechanically is what converts "we checked" into "it cannot regress".
#
# Membership is by *quantity*, not by producer: an artifact belongs here if any
# number in it is a difference against the reference, however it was computed.
REFERENCE_RELATIVE_ARTIFACTS = (
    "m4-powered-diss256-summary.json",
    "m4-h2o256-full-summary.json",
    "m4-rkv-256-summary.json",
    "m4-2048-summary.json",
    "m4-2048-h2o-summary.json",
    "m4-aime-2048-summary.json",
    "m4-llama-256-summary.json",
    "m4-q15b-256-summary.json",
    "m4-q14b-256-summary.json",
    "add/add-256-randslm.json",
    "add/add-256-h2o.json",
    "add/add-256-rkv.json",
    "add/add-2048.json",
    "fc-readout-control.json",
)

CURRENT_REFERENCE_CONVENTION = "amendment-k-cell-population"

# Values that identify a pre-Amendment-K artifact even if it claims compliance:
# the unmatched full-bank means that the retired convention differenced against.
_PRE_K_MARKERS = ("unmatched", "subset-vs-bank", "pre-amendment-k")


def check_reference_conventions(
    artifacts: dict[str, dict[str, Any]],
) -> list[str]:
    """Verify every reference-relative artifact declares the current convention.

    ``artifacts`` maps a name (as listed in
    :data:`REFERENCE_RELATIVE_ARTIFACTS`) to its loaded payload. A missing
    artifact is a failure, not a skip: silently passing because a file was not
    found is how a class-wide check stops checking anything.

    A retired value may still appear, but only under a key that names it as
    retired -- fc-readout-control deliberately keeps its pre-K figure so the
    stale number stays recognisable, and that must not be mistaken for a
    regression.
    """
    problems: list[str] = []
    for name in REFERENCE_RELATIVE_ARTIFACTS:
        payload = artifacts.get(name)
        if payload is None:
            problems.append(f"{name}: reference-relative artifact missing")
            continue
        declared = payload.get("reference_convention")
        if declared is None:
            problems.append(
                f"{name}: carries a reference-relative quantity but declares no "
                "reference_convention"
            )
        elif declared != CURRENT_REFERENCE_CONVENTION:
            problems.append(
                f"{name}: reference_convention is {declared!r}, expected "
                f"{CURRENT_REFERENCE_CONVENTION!r}"
            )
        for key in payload:
            lowered = key.lower()
            if any(marker in lowered for marker in _PRE_K_MARKERS) and (
                "retired" not in lowered
            ):
                problems.append(
                    f"{name}: key {key!r} looks pre-Amendment-K and is not "
                    "marked RETIRED"
                )
    return problems

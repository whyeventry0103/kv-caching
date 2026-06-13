"""Frozen answer canonicalizers.

The primary canonicalizer uses boxed/option extraction and `sympy.nsimplify`.
`canonicalize_ablation` is intentionally simpler and must be reported only as a
canonicalization sensitivity analysis.
"""

from __future__ import annotations

import re

CANONICALIZER_VERSION = "boxed-option-sympify-v2"
ABLATION_VERSION = "normalized-text-v1"

_OPTION = re.compile(r"(?:answer\s*(?:is|:)\s*)?\(?([A-E])\)?\s*$", re.I)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")
_LATEX_FRAC = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")


def _last_boxed(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    content_start = start + len(r"\boxed{")
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip()
    return None


def extract_answer(text: str) -> str:
    """Extract the final boxed, option, or numeric answer candidate."""
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed
    option = _OPTION.search(text.strip())
    if option:
        return option.group(1).upper()
    fractions = _LATEX_FRAC.findall(text)
    if fractions:
        numerator, denominator = fractions[-1]
        return rf"\frac{{{numerator}}}{{{denominator}}}"
    numbers = _NUMBER.findall(text.replace(",", ""))
    return numbers[-1] if numbers else text.strip()


def _is_answer_expression(text: str) -> bool:
    stripped = text.strip()
    return (
        stripped.startswith(("(", "[", "{"))
        or r"\frac" in stripped
        or r"\dfrac" in stripped
        or r"\pi" in stripped
    )


def _normalize_latex(candidate: str) -> str:
    previous = None
    while previous != candidate:
        previous = candidate
        candidate = _LATEX_FRAC.sub(r"(\1)/(\2)", candidate)
    return (
        candidate.replace(r"\pi", "pi")
        .replace(r"\left", "")
        .replace(r"\right", "")
        .replace(r"\,", "")
    )


def canonicalize(text: str) -> str:
    boxed = _last_boxed(text)
    if boxed is not None:
        candidate = boxed
    elif _is_answer_expression(text):
        candidate = text.strip()
    else:
        candidate = extract_answer(text)
    if len(candidate) == 1 and candidate.upper() in "ABCDE":
        return candidate.upper()
    candidate = _normalize_latex(candidate)
    try:
        import sympy

        return str(sympy.sympify(candidate))
    except (ImportError, TypeError, ValueError, SyntaxError):
        return re.sub(r"\s+", "", candidate).lower()


def canonicalize_ablation(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

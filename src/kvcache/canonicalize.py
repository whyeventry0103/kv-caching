"""Frozen answer canonicalizers.

The primary canonicalizer uses boxed/option extraction and `sympy.nsimplify`.
`canonicalize_ablation` is intentionally simpler and must be reported only as a
canonicalization sensitivity analysis.
"""

from __future__ import annotations

import re

CANONICALIZER_VERSION = "boxed-option-nsimplify-v1"
ABLATION_VERSION = "normalized-text-v1"

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_OPTION = re.compile(r"(?:answer\s*(?:is|:)\s*)?\(?([A-E])\)?\s*$", re.I)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")


def _extract(text: str) -> str:
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    option = _OPTION.search(text.strip())
    if option:
        return option.group(1).upper()
    numbers = _NUMBER.findall(text.replace(",", ""))
    return numbers[-1] if numbers else text.strip()


def canonicalize(text: str) -> str:
    candidate = _extract(text)
    if len(candidate) == 1 and candidate.upper() in "ABCDE":
        return candidate.upper()
    try:
        import sympy

        return str(sympy.nsimplify(candidate))
    except (ImportError, TypeError, ValueError, SyntaxError):
        return re.sub(r"\s+", "", candidate).lower()


def canonicalize_ablation(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

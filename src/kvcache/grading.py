"""Vetted pairwise MATH answer equivalence.

`is_equiv(gold_raw, pred_raw)` is the **correctness authority** for the project.
It is the canonical Hendrycks et al. MATH normalization (the grader the MATH /
MATH-500 line of work uses), reproduced verbatim here so scoring is deterministic
and has **no `eval`/`sympify` execution path on model-generated text**.

`canonicalize()` in :mod:`kvcache.canonicalize` is intentionally NOT used for
correctness any more: a `sympify`-to-string canonical form is brittle (set/tuple
ordering, decimal-vs-fraction, silent `except` fallbacks) and already produced a
silent false-negative on a structured answer during the smoke run. The
`canonicalize` path is retained only as an ADD *bucketing* key and as a reported
canonicalization sensitivity.

Comparability caveat (verify before the grid): confirm the exact grader used by
HOTT (arXiv:2512.12008) / the kvpress eval / MATH-500 and reconcile. Hendrycks
`is_equiv` is the conservative default; `math_verify` is the modern alternative
but is a heavier dependency that may not match the pinned stack.

Source: Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH
Dataset" (arXiv:2103.03874), `math_equivalence.py`.
"""

from __future__ import annotations

GRADER_VERSION = "hendrycks-is-equiv-v1"


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a = int(a_str)
        b = int(b_str)
    except ValueError:
        return string
    if string != f"{a}/{b}":
        return string
    return "\\frac{" + str(a) + "}{" + str(b) + "}"


def _remove_right_units(string: str) -> str:
    # Strip a trailing "\\text{ ...}" unit annotation.
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(gold_raw: str | None, pred_raw: str | None) -> bool:
    """Return whether two raw answer strings are mathematically equivalent."""
    if gold_raw is None and pred_raw is None:
        # Two missing answers are not a correct match for scoring purposes.
        return False
    if gold_raw is None or pred_raw is None:
        return False
    try:
        return _strip_string(gold_raw) == _strip_string(pred_raw)
    except (IndexError, ValueError):
        return gold_raw == pred_raw


CROSSCHECK_VERSION = "sympify-canonical-crosscheck-v1"


def crosscheck_equiv(gold_raw: str | None, pred_raw: str | None) -> bool | None:
    """Second-opinion equivalence via the sympify canonicalizer.

    **Not an authority, and deliberately not one.** ``is_equiv`` remains the
    correctness authority; this exists solely so the grader audit has an oracle
    that does not share ``is_equiv``'s normalization path. Re-running
    ``is_equiv`` over stored answers and finding no disagreement is guaranteed
    by construction -- it tests that scoring replays deterministically, which is
    a replay property, not a grader-validity property. Only a *different*
    normalization can speak to whether ``is_equiv`` under-credits equivalent
    forms, which is the threat the audit exists to close.

    The canonicalizer is applied to the **extracted answer**, never to raw model
    text: the standing prohibition is on an ``eval``/``sympify`` execution path
    over model-generated prose, and ``canonicalize`` already runs in exactly
    this position for ADD bucketing. Returns ``None`` when either side has no
    extractable answer, so "cannot be compared" never masquerades as "agrees".

    Its known weakness is documented and is the reason it is a second opinion
    rather than a replacement: a sympify-to-string canonical form is brittle on
    sets, tuples and decimal-versus-fraction, and produced a silent false
    negative on a structured answer during the smoke run. Disagreements between
    the two graders are therefore *candidates for human review*, not
    corrections -- neither side wins automatically.
    """
    if gold_raw is None or pred_raw is None:
        return None
    from kvcache.canonicalize import canonicalize

    gold_canon = canonicalize(gold_raw)
    pred_canon = canonicalize(pred_raw)
    if not gold_canon or not pred_canon:
        return None
    return gold_canon == pred_canon

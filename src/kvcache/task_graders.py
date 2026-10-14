"""Frozen graders for the Meeting 5 replication tasks (M5-B; prepared
2026-10-14, before any GSM8K or BBH generation exists).

Scientific rationale. The repo's MATH grader (``kvcache.grading.is_equiv``) is
the frozen primary for MATH-500 and stays untouched. Probed on GSM8K-style
answers on 2026-10-14, it marks ``\\boxed{1,250}`` (gold 1250),
``\\boxed{18.00}`` (gold 18) and ``\\boxed{72 clips}`` (gold 72) WRONG -- a
systematic false negative for a numeric-answer task. Re-scoring would be the
wrong fix (it would move frozen MATH rows), and so would wiring a new grader
into generation (generation.py is generation identity for pending cells). So
the GSM8K primary score is this module, applied at READ time to the stored
trace; the in-row ``correct`` field of a GSM8K row is not the GSM8K primary.

GSM8K rule (frozen): a truncated generation is incorrect. The answer region is
the text after the last ``</think>`` (the whole text if there is none). The
final candidates are EVERY ``\\boxed{...}`` in that region plus every
"answer is <number>" phrase; if they name different numbers the answer is
incorrect (``conflicting_final_answers``) -- a last box never overrides a
conflicting final statement (review of 6b409001, R4). With no box and no such
phrase, the candidate is the last number in the region (the prompt is the raw
question, with no instruction to box). The candidate is normalized --
``\\text{}``/``\\$``/``$``/``%``/``\\%``,
thousands commas and trailing unit words removed, ``\\frac{a}{b}`` and
``a/b`` read as fractions -- and compared to the gold number by EXACT
rational equality. Anything unparsable is incorrect with a recorded reason.
The boxed-only variant is a declared sensitivity, never a rescue.

BBH logical_deduction_five_objects rule (frozen; tightened 2026-10-14 after the
review of 6b409001, R4, before any BBH generation): options are ``(A)``-``(E)``.
The final candidates are EVERY ``\\boxed{...}`` in the answer region and every
"answer is ..." statement in it. A box naming several options is incorrect
(``several_options``); candidates naming different options are incorrect
(``conflicting_final_answers``) -- e.g. ``\\boxed{(A)} or \\boxed{(C)}`` and
``\\boxed{(C)}. Actually, the answer is (D).``; no candidate is incorrect
(``no_answer``). There is no correction rule and no LLM judge.
"""

from __future__ import annotations

import re
from fractions import Fraction

_THINK_CLOSE = "</think>"
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_FRAC = re.compile(r"\\[dt]?frac\{\s*(-?[\d.]+)\s*\}\{\s*(-?[\d.]+)\s*\}")
_TEXT = re.compile(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}")
_OPTION = re.compile(r"\(([A-E])\)|\b([A-E])\b")


def answer_region(trace: str) -> str:
    cut = trace.rfind(_THINK_CLOSE)
    return trace[cut + len(_THINK_CLOSE) :] if cut >= 0 else trace


def parse_number(text: str) -> Fraction | None:
    """An exact number from a short answer string, or None if not exactly one."""
    s = _TEXT.sub(r"\1", text)
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("{,}", ",").strip()
    m = _FRAC.fullmatch(s)
    if m:
        den = Fraction(m.group(2))
        return Fraction(m.group(1)) / den if den else None
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators only
    s = re.sub(r"\s*[A-Za-z][A-Za-z .]*$", "", s).strip()  # trailing unit words
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        den = Fraction(m.group(2))
        return Fraction(m.group(1)) / den if den else None
    if re.fullmatch(r"-?\d+(?:\.\d+)?|-?\.\d+", s):
        return Fraction(s)
    return None


GRADER_VERSIONS = {"gsm8k": "gsm8k-numeric-v2", "bbh": "bbh-option-v2"}
_ANSWER_IS = re.compile(r"answer\s+is\s*:?\s*", re.IGNORECASE)


def all_boxed(text: str) -> list[str]:
    """Contents of every ``\\boxed{...}`` in order (balanced braces)."""
    out, start = [], 0
    while True:
        at = text.find(r"\boxed{", start)
        if at < 0:
            return out
        i, depth = at + len(r"\boxed{"), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth:
            return out  # unterminated box: ignored, as the MATH extractor does
        out.append(text[at + len(r"\boxed{") : i - 1].strip())
        start = i


def _answer_is_windows(region: str) -> list[str]:
    return [region[m.end() : m.end() + 40] for m in _ANSWER_IS.finditer(region)]


def gsm8k_grade(
    trace: str, gold: str, *, truncated: bool, boxed_only: bool = False
) -> tuple[bool, str]:
    """(correct, rule) under the frozen GSM8K rule."""
    if truncated:
        return False, "truncated"
    want = Fraction(gold)
    region = answer_region(trace)
    boxes = all_boxed(region)
    stated = [] if boxed_only else _answer_is_windows(region)
    values = []
    for b in boxes:
        v = parse_number(b)
        if v is None:
            return False, "boxed_unparsable"
        values.append(v)
    for w in stated:
        nums = _NUM.findall(w)
        v = parse_number(nums[0]) if nums else None
        if v is not None:
            values.append(v)
    if values:
        if len(set(values)) > 1:
            return False, "conflicting_final_answers"
        return values[0] == want, "boxed" if boxes else "answer_is"
    if boxed_only:
        return False, "no_boxed"
    nums = _NUM.findall(region)
    if not nums:
        return False, "no_number"
    got = parse_number(nums[-1])
    if got is None:
        return False, "last_number_unparsable"
    return got == want, "last_number"


def bbh_option(text: str) -> tuple[str | None, str]:
    """The single option letter named by an answer string, with the rule."""
    letters = {a or b for a, b in _OPTION.findall(_TEXT.sub(r"\1", text))}
    if len(letters) == 1:
        return letters.pop(), "one_option"
    return None, "no_option" if not letters else "several_options"


def bbh_grade(trace: str, gold: str, *, truncated: bool) -> tuple[bool, str]:
    """(correct, rule) under the frozen BBH option rule; gold is like "(C)"."""
    gold_letter, why = bbh_option(gold)
    if gold_letter is None:
        raise ValueError(f"BBH gold is not a single option: {gold!r} ({why})")
    if truncated:
        return False, "truncated"
    region = answer_region(trace)
    letters = []
    for b in all_boxed(region):
        letter, why = bbh_option(b)
        if why == "several_options":
            return False, "boxed:several_options"
        if letter:
            letters.append(letter)
    for w in _answer_is_windows(region):
        m = _OPTION.search(_TEXT.sub(r"\1", w))
        if m:
            letters.append(m.group(1) or m.group(2))
    if not letters:
        return False, "no_answer"
    if len(set(letters)) > 1:
        return False, "conflicting_final_answers"
    return letters[0] == gold_letter, "final_answer"

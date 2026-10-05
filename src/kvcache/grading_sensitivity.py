"""Secondary, format-aware scoring for the Qwen3 bank investigation (S.4).

Scientific rationale. The frozen grader (``kvcache.grading.is_equiv`` on
``kvcache.canonicalize.extract_answer`` of the whole trace) is the PRIMARY
score and stays so: the Qwen3-4B bank scored 169/200 = 0.845 under it, below
the registered band [0.85, 0.99], and that FAIL is kept. This module is a
SECONDARY score, specified 2026-10-05 after the 31 bank errors had been
inspected and before any dependent Qwen cell was read
(docs/protocol/QWEN_SCORING_SENSITIVITY.md). It is a sensitivity analysis, not
a prospectively registered grader, and never replaces the primary.

Three questions are kept separate, as the execution plan (Section 8) requires:

1. Boundary: did the model leave its reasoning (``</think>``) and give a final
   answer after it? An answer boxed inside unclosed thinking is not treated as
   a final answer: such rows are AMBIGUOUS, not rescued.
2. Extraction: the answer is taken from the post-reasoning region only. Two or
   more boxes are joined only when they form one terminal block (nothing but
   separators between them, ending the region); boxes scattered through the
   text are never collected.
3. Equivalence: ``is_equiv`` first; then a fixed list of rules, each with an
   ID, applied identically to correct and incorrect rows of every arm:
   ``TEXT_WRAPPER`` (strip ``\\text{}``-style wrappers, keep their content),
   ``UNORDERED_LIST`` (order-insensitive element matching, only when the
   PROBLEM text asks for an unordered list and the gold is a bare comma list).
   Answers that differ only by a base subscript (``52`` vs ``52_8``) are
   AMBIGUOUS (``BASE_SUBSCRIPT``): whether the bare numeral answers the
   question is a semantic judgement this rule list does not make.

Ambiguous rows keep their primary score in the point estimate and are reported
as sensitivity bounds (all ambiguous incorrect / all correct). No rule looks at
the gold to decide how to extract, and no per-problem exception exists.
"""

from __future__ import annotations

import re
from typing import Any

from kvcache.canonicalize import extract_answer
from kvcache.generation import has_answer_marker
from kvcache.grading import is_equiv

SECONDARY_VERSION = "format-aware-v1"
THINK_CLOSE = "</think>"

_BOX = r"\boxed{"
_WRAPPER = re.compile(r"\\(?:text|textbf|textit|mathrm|mbox|operatorname)\{([^{}]*)\}")
# Separators allowed between boxes of one terminal answer block.
_SEPARATOR = re.compile(
    r"^(?:\s|,|;|&|\$|\\\(|\\\)|\\\[|\\\]|\\\\|\\quad|\\qquad|\\,|\\;|and|or"
    r"|\\text\{\s*(?:and|or|,)?\s*\})*$",
    re.I,
)
# Trailing text allowed after the last box of the region (closing delimiters).
_TAIL = re.compile(r"^(?:\s|\$|\\\)|\\\]|\.|!)*$")
_UNORDERED_ASK = re.compile(
    r"separated by commas|all (?:the )?(?:possible |real |integer |complex )?"
    r"(?:solutions|values|roots|zeros|numbers|integers)|enter all",
    re.I,
)
_ORDERED_ASK = re.compile(
    r"in order|\bordered|increasing|decreasing|from (?:least|smallest)", re.I
)
_BASE_SUB = re.compile(r"^(.*?)_\{?(\d+)\}?$")


def _boxes(text: str) -> list[tuple[int, int, str]]:
    """(start, end, content) of every balanced ``\\boxed{...}``, in order."""
    out = []
    i = text.find(_BOX)
    while i >= 0:
        j, depth = i + len(_BOX), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth:
            break
        out.append((i, j, text[i + len(_BOX) : j - 1].strip()))
        i = text.find(_BOX, j)
    return out


def boundary_status(trace: str, *, hit_cap: bool) -> str:
    """Where the answer sits relative to the reasoning boundary."""
    if THINK_CLOSE in trace:
        region = trace.rsplit(THINK_CLOSE, 1)[1]
        return "closed_with_answer" if has_answer_marker(region) else "closed_no_answer"
    if has_answer_marker(trace):
        return "answer_inside_unclosed_thinking"
    return "truncated_no_answer" if hit_cap else "no_answer"


def secondary_extract(trace: str) -> tuple[str | None, str]:
    """The final answer from the post-reasoning region, with the rule used.

    Returns (None, reason) when there is no usable final region.
    """
    if THINK_CLOSE not in trace:
        return None, "NO_CLOSED_REASONING"
    region = trace.rsplit(THINK_CLOSE, 1)[1]
    boxes = _boxes(region)
    if not boxes:
        if not has_answer_marker(region):
            return None, "NO_FINAL_ANSWER"
        return extract_answer(region), "PRIMARY_EXTRACTOR_ON_REGION"
    block = [boxes[-1]]
    for prev in reversed(boxes[:-1]):
        between = region[prev[1] : block[0][0]]
        if not _SEPARATOR.match(between):
            break
        block.insert(0, prev)
    if len(block) == 1:
        return block[0][2], "LAST_BOX"
    if not _TAIL.match(region[block[-1][1] :].split("\n\n")[0]):
        return block[-1][2], "LAST_BOX"
    return ", ".join(b[2] for b in block), "TERMINAL_BOX_BLOCK"


def _unwrap(s: str) -> str:
    prev = None
    while prev != s:
        prev, s = s, _WRAPPER.sub(r"\1", s)
    return s.strip()


def _bare_list(s: str) -> list[str] | None:
    """Top-level comma elements of a list NOT enclosed in brackets."""
    s = s.strip()
    if not s or s[0] in "([{" and s[-1] in ")]}":
        return None
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}.get(ch, 0)
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts if len(parts) >= 2 and all(parts) else None


def _multiset_equiv(gold: list[str], pred: list[str]) -> bool:
    if len(gold) != len(pred):
        return False
    remaining = list(pred)
    for g in gold:
        hit = next((p for p in remaining if is_equiv(g, p)), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def secondary_equiv(
    gold: str, pred: str, *, problem_text: str
) -> tuple[bool | None, str]:
    """(verdict, rule). Verdict None means AMBIGUOUS under this rule list."""
    if is_equiv(gold, pred):
        return True, "IS_EQUIV"
    g, p = _unwrap(gold), _unwrap(pred)
    if (g, p) != (gold, pred) and is_equiv(g, p):
        return True, "TEXT_WRAPPER"
    if g.strip().lower() == p.strip().lower() and g.strip() and g.isalpha():
        return True, "TEXT_WRAPPER"
    gl, pl = _bare_list(g), _bare_list(p)
    if (
        gl is not None
        and pl is not None
        and _UNORDERED_ASK.search(problem_text)
        and not _ORDERED_ASK.search(problem_text)
        and _multiset_equiv(gl, pl)
    ):
        return True, "UNORDERED_LIST"
    m = _BASE_SUB.match(g.replace(" ", ""))
    if m and is_equiv(m.group(1), p):
        return None, "BASE_SUBSCRIPT"
    return False, "NOT_EQUIVALENT"


def score_row(
    *, trace: str, gold: str, problem_text: str, hit_cap: bool, primary: bool
) -> dict[str, Any]:
    """Primary and secondary scores of one generation, with the reasons."""
    status = boundary_status(trace, hit_cap=hit_cap)
    primary_answer = extract_answer(trace) if has_answer_marker(trace) else ""
    answer, extraction = secondary_extract(trace)
    if status == "answer_inside_unclosed_thinking":
        verdict, rule = None, "UNCLOSED_THINKING"
    elif answer is None:
        verdict, rule = False, extraction
    else:
        verdict, rule = secondary_equiv(gold, answer, problem_text=problem_text)
    point = primary if verdict is None else verdict
    return {
        "secondary_version": SECONDARY_VERSION,
        "boundary_status": status,
        "primary_answer": primary_answer,
        "primary_correct": bool(primary),
        "secondary_answer": answer,
        "secondary_extraction": extraction,
        "secondary_verdict": verdict,
        "secondary_rule": rule,
        "secondary_point": bool(point),
        "transition": (
            "AMBIGUOUS"
            if verdict is None
            else (
                "UNCHANGED"
                if verdict == primary
                else "INCORRECT_TO_CORRECT" if verdict else "CORRECT_TO_INCORRECT"
            )
        ),
    }

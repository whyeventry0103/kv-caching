"""Frozen M5-B graders: the GSM8K numeric rule fixes the shared MATH grader's
false negatives on GSM8K-style answers; BBH needs exactly one option."""

from __future__ import annotations

from fractions import Fraction

import pytest

from kvcache.grading import is_equiv
from kvcache.task_graders import bbh_grade, bbh_option, gsm8k_grade, parse_number


@pytest.mark.parametrize(
    "boxed, gold",
    [("1,250", "1250"), ("18.00", "18"), ("72 clips", "72")],
)
def test_math_grader_false_negatives_that_motivate_the_gsm8k_rule(boxed, gold):
    assert not is_equiv(gold, boxed)  # the reason this module exists
    assert gsm8k_grade(
        f"<think>x</think> \\boxed{{{boxed}}}", gold, truncated=False
    ) == (
        True,
        "boxed",
    )


@pytest.mark.parametrize(
    "text, value",
    [
        ("1,250", Fraction(1250)),
        ("\\$18.50", Fraction(37, 2)),
        ("\\text{72 clips}", Fraction(72)),
        ("10\\%", Fraction(10)),
        ("\\frac{3}{4}", Fraction(3, 4)),
        ("3/4", Fraction(3, 4)),
        ("-3.5", Fraction(-7, 2)),
        ("1{,}000", Fraction(1000)),
        ("12,34", None),  # not a thousands separator
        ("x + 2", None),
        ("\\frac{1}{0}", None),
    ],
)
def test_parse_number(text, value):
    assert parse_number(text) == value


def test_gsm8k_rule_order_and_reasons():
    assert gsm8k_grade("<think>\\boxed{5}</think> so 72", "72", truncated=False) == (
        True,
        "last_number",
    )  # boxes inside the reasoning are not the answer
    assert gsm8k_grade("... answer 72", "72", truncated=False, boxed_only=True) == (
        False,
        "no_boxed",
    )
    assert gsm8k_grade("\\boxed{72}", "72", truncated=True) == (False, "truncated")
    assert gsm8k_grade("\\boxed{seventy}", "72", truncated=False) == (
        False,
        "boxed_unparsable",
    )
    assert gsm8k_grade("no digits", "72", truncated=False) == (False, "no_number")
    assert gsm8k_grade("\\boxed{71.999}", "72", truncated=False) == (False, "boxed")


def test_bbh_options():
    assert bbh_option("(C)") == ("C", "one_option")
    assert bbh_option("\\text{(B)}") == ("B", "one_option")
    assert bbh_option("(A) or (B)") == (None, "several_options")
    assert bbh_option("none") == (None, "no_option")
    assert bbh_grade("</think> \\boxed{(C)}", "(C)", truncated=False) == (
        True,
        "final_answer",
    )
    assert bbh_grade("So the answer is (D).", "(C)", truncated=False) == (
        False,
        "final_answer",
    )
    assert bbh_grade("\\boxed{(A) (B)}", "(A)", truncated=False) == (
        False,
        "boxed:several_options",
    )
    assert bbh_grade("I think so", "(A)", truncated=False) == (False, "no_answer")
    with pytest.raises(ValueError):
        bbh_grade("x", "C or D", truncated=False)


@pytest.mark.parametrize(
    "trace",
    [
        # review of 6b409001, R4: both used to grade CORRECT against (C)
        r"</think> \boxed{(A)} or \boxed{(C)}",
        r"</think> \boxed{(C)}. Actually, the answer is (D).",
        r"</think> The answer is (C). Wait, the answer is (B).",
    ],
)
def test_bbh_conflicting_final_answers_are_incorrect(trace):
    assert bbh_grade(trace, "(C)", truncated=False) == (
        False,
        "conflicting_final_answers",
    )


def test_bbh_consistent_repeats_and_reasoning_boxes():
    t = r"<think>maybe \boxed{(A)}</think> \boxed{(C)} so the answer is (C)."
    assert bbh_grade(t, "(C)", truncated=False) == (True, "final_answer")


def test_gsm8k_conflicting_final_answers_are_incorrect():
    assert gsm8k_grade(
        r"</think> \boxed{6}, so the answer is 7", "6", truncated=False
    ) == (
        False,
        "conflicting_final_answers",
    )
    assert gsm8k_grade(r"\boxed{5} \boxed{6}", "6", truncated=False)[1] == (
        "conflicting_final_answers"
    )
    assert gsm8k_grade("the answer is 1,250 clips", "1250", truncated=False) == (
        True,
        "answer_is",
    )

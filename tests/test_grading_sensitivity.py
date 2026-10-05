"""Secondary format-aware scoring (S.4 investigation): each rule with a positive
case and an adversarial negative, the boundary statuses, and both transition
directions. The frozen primary grader is never altered by this module."""

from __future__ import annotations

import pytest

from kvcache import grading_sensitivity as gs

ASK_ALL = "Find all real solutions. Enter all the solutions, separated by commas."
ASK_ORDERED = "Enter the solutions in increasing order, separated by commas."


def _row(trace, gold, *, problem="Compute it.", hit_cap=False, primary=False):
    return gs.score_row(
        trace=trace, gold=gold, problem_text=problem, hit_cap=hit_cap, primary=primary
    )


@pytest.mark.parametrize(
    "trace, hit_cap, want",
    [
        ("<think>x</think> so \\boxed{3}", False, "closed_with_answer"),
        ("<think>x</think> I cannot finish", False, "closed_no_answer"),
        ("<think>so \\boxed{3} wait", True, "answer_inside_unclosed_thinking"),
        ("<think>still thinking", True, "truncated_no_answer"),
        ("<think>still thinking", False, "no_answer"),
    ],
)
def test_boundary_status(trace, hit_cap, want):
    assert gs.boundary_status(trace, hit_cap=hit_cap) == want


def test_text_wrapper_rescues_and_does_not_invent():
    r = _row("<think>..</think> The winner is \\boxed{Evelyn}.", "\\text{Evelyn}")
    assert (r["secondary_verdict"], r["secondary_rule"]) == (True, "TEXT_WRAPPER")
    assert r["transition"] == "INCORRECT_TO_CORRECT"
    r = _row("<think>..</think> \\boxed{Evelyn}", "\\text{Bob}")
    assert r["secondary_verdict"] is False


def test_unordered_only_when_the_problem_asks_for_an_unordered_list():
    trace = "<think>..</think> \\boxed{-2, 1}"
    r = _row(trace, "1,-2", problem=ASK_ALL)
    assert (r["secondary_verdict"], r["secondary_rule"]) == (True, "UNORDERED_LIST")
    assert _row(trace, "1,-2", problem=ASK_ORDERED)["secondary_verdict"] is False
    # v2: no positive unordered wording -> ambiguous, never a rescue
    r = _row(trace, "1,-2", problem="Compute x and y.")
    assert (r["secondary_verdict"], r["secondary_rule"]) == (None, "ORDER_UNCERTAIN")
    # a tuple keeps its order even when the problem says "all"
    r = _row("<think>..</think> \\boxed{(2, 1)}", "(1,2)", problem=ASK_ALL)
    assert r["secondary_verdict"] is False
    # multiset, not set: a repeated element must not match a different one
    r = _row("<think>..</think> \\boxed{1, 1}", "1,-2", problem=ASK_ALL)
    assert r["secondary_verdict"] is False


def test_terminal_box_block_joins_only_adjacent_final_boxes():
    trace = (
        "<think>..</think> The answers are "
        "$\\boxed{3}$, $\\boxed{5}$, and $\\boxed{7}$."
    )
    ans, rule = gs.secondary_extract(trace)
    assert (ans, rule) == ("3, 5, 7", "TERMINAL_BOX_BLOCK")
    r = _row(trace, "3, 5, 7", problem=ASK_ALL)
    assert r["secondary_verdict"] is True
    # boxes separated by prose are never collected: the last box is the answer
    trace = "<think>..</think> First $\\boxed{3}$ is a guess. Then we get \\boxed{7}."
    assert gs.secondary_extract(trace) == ("7", "LAST_BOX")
    # boxes inside the reasoning are never part of the final block
    trace = "<think> \\boxed{3} </think> \\boxed{7}"
    assert gs.secondary_extract(trace) == ("7", "LAST_BOX")


def test_base_subscript_is_ambiguous_not_rescued():
    r = _row("<think>..</think> \\boxed{52}", "52_8", primary=False)
    assert (r["secondary_verdict"], r["secondary_rule"]) == (None, "BASE_SUBSCRIPT")
    assert r["transition"] == "AMBIGUOUS"
    assert r["secondary_point"] is False  # keeps the primary score
    assert _row("<think>..</think> \\boxed{53}", "52_8")["secondary_verdict"] is False


def test_unclosed_thinking_is_ambiguous_in_both_directions():
    for primary in (True, False):
        r = _row("<think> so \\boxed{3}", "3", hit_cap=False, primary=primary)
        assert r["secondary_verdict"] is None
        assert r["secondary_rule"] == "UNCLOSED_THINKING"
        assert r["secondary_point"] is primary


def test_correct_to_incorrect_transition_is_kept():
    # primary credited a box inside the reasoning; no final answer after it
    r = _row("<think> \\boxed{3} </think> I am not sure.", "3", primary=True)
    assert r["primary_answer"] == "3"
    assert r["secondary_verdict"] is False
    assert r["transition"] == "CORRECT_TO_INCORRECT"


def test_plain_agreement_is_unchanged():
    r = _row("<think>..</think> \\boxed{\\frac{1}{2}}", "\\frac12", primary=True)
    assert (r["secondary_rule"], r["transition"]) == ("IS_EQUIV", "UNCHANGED")


def test_unordered_wording_is_not_read_as_ordered():
    ask = "Enter all the solutions, separated by commas, in any (unordered) form."
    r = _row("<think>..</think> \\boxed{-2, 1}", "1,-2", problem=ask)
    assert r["secondary_verdict"] is True


# ---------------------------------------------- v2 (review of fbf869f, R2)
@pytest.mark.parametrize(
    "problem",
    [
        "Find x and y, separated by commas.",
        "Give the coordinates x,y separated by commas.",
        "Give the numerator and denominator separated by commas.",
    ],
)
def test_v2_commas_alone_never_make_order_irrelevant(problem):
    verdict, rule = gs.secondary_equiv("1,2", "2,1", problem_text=problem)
    assert verdict is not True
    if "coordinates" in problem or "numerator" in problem:
        assert (verdict, rule) == (False, "ORDER_REQUIRED")
    else:
        assert (verdict, rule) == (None, "ORDER_UNCERTAIN")


def test_v2_positive_unordered_wording_still_rescues():
    verdict, rule = gs.secondary_equiv(
        "1,-2", "-2, 1", problem_text="Find all integers n. Enter all the solutions."
    )
    assert (verdict, rule) == (True, "UNORDERED_LIST")


def test_v2_later_correction_blocks_the_multibox_rescue():
    trace = (
        "<think>x</think> \\boxed{3}, \\boxed{5}, \\boxed{7}.\n\n"
        "Actually that list was a mistake. Final answer: 11."
    )
    assert gs.secondary_extract(trace) == (None, "MULTIBOX_NOT_TERMINAL")
    r = _row(trace, "3, 5, 7", problem=ASK_ALL, primary=False)
    assert r["secondary_verdict"] is None
    assert r["transition"] == "AMBIGUOUS" and r["secondary_point"] is False


def test_v2_terminal_block_with_only_delimiters_after_it_is_still_joined():
    trace = "<think>x</think> $$ \\boxed{3}, \\boxed{5}, \\boxed{7} $$\n"
    assert gs.secondary_extract(trace) == ("3, 5, 7", "TERMINAL_BOX_BLOCK")

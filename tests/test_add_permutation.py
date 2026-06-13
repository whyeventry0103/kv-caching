import pytest

from kvcache.metrics.add import (
    OTHER_ANSWER,
    add_tv,
    answer_distribution,
    coarsen_answer,
    entropy_miller_madow,
    excess_add_permutation,
    greedy_flip_rate,
    pass_at_k,
    truncation_rate,
)
from kvcache.stats import benjamini_hochberg


def test_add_and_per_problem_permutation_null():
    full = ["A"] * 4
    compressed = ["B"] * 4
    observed, p_value = excess_add_permutation(full, compressed)
    assert observed == 1.0
    assert 0.0 < p_value < 0.1


def test_add_support_entropy_passk_and_bh():
    assert add_tv(["A", "B"], ["A", "A"]) == 0.5
    assert entropy_miller_madow(["A", "B"]) > 0.69
    assert pass_at_k(10, 2, 1) == pytest.approx(0.2)
    assert greedy_flip_rate(["A", "B"], ["A", "C"]) == 0.5
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.04, 0.04])


def test_invalid_answers_share_one_coarsened_bucket():
    assert coarsen_answer(None) == OTHER_ANSWER
    assert coarsen_answer("", truncated=True) == OTHER_ANSWER
    distribution = answer_distribution(["A", "junk", None], valid_support={"A", "B"})
    assert distribution == {
        "A": pytest.approx(1 / 3),
        OTHER_ANSWER: pytest.approx(2 / 3),
    }


def test_truncated_rows_remain_in_denominator():
    rows = [
        {"truncated": False, "correct": True},
        {"truncated": True, "correct": False},
    ]
    assert truncation_rate(rows) == 0.5

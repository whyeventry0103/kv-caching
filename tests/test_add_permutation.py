import pytest

from kvcache.metrics.add import (
    add_tv,
    entropy_miller_madow,
    excess_add_permutation,
    greedy_flip_rate,
    pass_at_k,
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

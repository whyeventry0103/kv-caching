import pytest

from kvcache.metrics.tdp import (
    auc_tv,
    content_token_profile,
    fraction_above,
    onset_aligned,
    segment_aggregate,
    summarize_profile,
    tdp,
)
from kvcache.replay import replay_tv_profile, topk_tv_bounds, total_variation


def test_total_variation_and_profile():
    assert total_variation([0.75, 0.25], [0.25, 0.75]) == 0.5
    profile = replay_tv_profile([[0.5, 0.5], [0.9, 0.1]], [[0.5, 0.5], [0.6, 0.4]])
    assert profile == pytest.approx([0.0, 0.3])
    assert auc_tv(profile) == pytest.approx(0.15)
    assert fraction_above(profile, 0.1) == 0.5
    assert tdp(profile, 0.1) == 1
    assert set(summarize_profile(profile)["frac_above"]) == {
        "p05",
        "p10",
        "p15",
        "p20",
    }
    assert onset_aligned(profile, 1) == {-1: 0.0, 0: pytest.approx(0.3)}
    assert content_token_profile(["word", "2"], profile) == pytest.approx([0.3])
    assert segment_aggregate(profile, [("all", 0, 2)])["all"] == pytest.approx(0.15)


def test_topk_tail_bound_contains_estimate():
    result = topk_tv_bounds({1: 0.7}, {1: 0.6}, p_tail=0.3, q_tail=0.4)
    assert result.lower <= result.estimate <= result.upper
    assert result.upper <= 1.0


def test_topk_requires_union_support_probabilities():
    with pytest.raises(ValueError, match="union"):
        topk_tv_bounds({1: 0.7}, {2: 0.6}, p_tail=0.3, q_tail=0.4)

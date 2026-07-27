import pytest

from kvcache.metrics.tdp import (
    auc_tv,
    content_token_profile,
    fraction_above,
    onset_aligned,
    segment_aggregate,
    summarize_profile,
    tdp,
    total_tv,
)
from kvcache.replay import replay_tv_profile, topk_tv_bounds, total_variation


def test_total_variation_and_profile():
    assert total_variation([0.75, 0.25], [0.25, 0.75]) == 0.5
    profile = replay_tv_profile([[0.5, 0.5], [0.9, 0.1]], [[0.5, 0.5], [0.6, 0.4]])
    assert profile == pytest.approx([0.0, 0.3])
    assert auc_tv(profile) == pytest.approx(0.15)
    assert total_tv(profile) == pytest.approx(0.3)
    assert fraction_above(profile, 0.1) == 0.5
    assert tdp(profile, 0.1) == 1
    assert set(summarize_profile(profile)["frac_above"]) == {
        "p05",
        "p10",
        "p15",
        "p20",
    }
    assert summarize_profile(profile)["n_steps"] == 2
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


def test_windowed_tv_splits_on_onset_and_uses_two_estimators():
    from kvcache.metrics.tdp import windowed_tv

    # onset at index 2: steps 0,1 are "before"; the rest are "first50".
    profiles = [([0.0, 0.0, 0.2, 0.4], 2), ([0.0, 0.0, 0.6, 0.8], 2)]
    out = windowed_tv(profiles)
    assert out["n_profiles"] == 2
    assert out["windows"]["before"]["per_profile"] == 0.0
    # per-profile: mean(0.3, 0.7) == pooled mean here because both traces are
    # the same length; the estimators separate only for unequal lengths.
    assert out["windows"]["first50"]["per_profile"] == pytest.approx(0.5)
    assert out["windows"]["first50"]["n_steps"] == 4


def test_windowed_tv_per_profile_and_pooled_differ_on_unequal_lengths():
    from kvcache.metrics.tdp import windowed_tv

    profiles = [([0.0, 1.0], 1), ([0.0, 0.0, 0.0, 0.0], 1)]
    out = windowed_tv(profiles)["windows"]["first50"]
    assert out["per_profile"] == pytest.approx(0.5)  # mean(1.0, 0.0)
    assert out["pooled"] == pytest.approx(0.25)  # one 1.0 among four steps


def test_windowed_tv_skips_profiles_without_onset():
    from kvcache.metrics.tdp import windowed_tv

    assert windowed_tv([([0.1, 0.2], None), ([0.0, 0.5], 1)])["n_profiles"] == 1


def test_onset_window_boundaries():
    from kvcache.metrics.tdp import onset_window

    assert onset_window(-1) == "before"
    assert onset_window(0) == "first50"
    assert onset_window(49) == "first50"
    assert onset_window(50) == "later"


def test_windowed_tv_reports_median_onset():
    """The paper quotes the median onset (65); it must have a source."""
    from kvcache.metrics.tdp import windowed_tv

    profiles = [([0.0, 0.1], 1), ([0.0, 0.0, 0.2], 2), ([0.0, 0.3], 9)]
    assert windowed_tv(profiles)["median_onset"] == 2

    # profiles skipped for a missing onset must not enter the median either
    skipped = [([0.0, 0.1], None), ([0.0, 0.2], 4), ([0.0, 0.3], 6)]
    assert windowed_tv(skipped)["median_onset"] == 5


def test_windowed_tv_median_onset_is_none_without_profiles():
    from kvcache.metrics.tdp import windowed_tv

    assert windowed_tv([])["median_onset"] is None

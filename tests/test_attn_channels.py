"""Hand-computed checks for the Amendment Q.3 attention channels."""

from __future__ import annotations

import pytest

from kvcache.metrics.attn_channels import (
    DEFAULT_COLLATERAL_TAU,
    collateral_fraction,
    mass_loss,
    renormalised_restriction,
    shape_tv,
)

# full-model attention over 4 positions; retained set keeps 0 and 2
FULL = [0.5, 0.3, 0.15, 0.05]
KEEP = [0, 2]


def test_mass_loss_is_the_evicted_mass():
    # kept 0.5 + 0.15 = 0.65 -> lost 0.35
    assert mass_loss(FULL, KEEP) == pytest.approx(0.35)


def test_mass_loss_is_zero_when_nothing_is_evicted():
    assert mass_loss(FULL, [0, 1, 2, 3]) == pytest.approx(0.0)


def test_renormalised_restriction_sums_to_one():
    # 0.5/0.65, 0.15/0.65
    r = renormalised_restriction(FULL, KEEP)
    assert r == pytest.approx([0.5 / 0.65, 0.15 / 0.65])
    assert sum(r) == pytest.approx(1.0)


def test_shape_tv_is_zero_when_the_compressed_model_matches_the_restriction():
    """The undamaged-state case: all of the effect lives in M, none in Delta."""
    ref = renormalised_restriction(FULL, KEEP)
    assert shape_tv(ref, FULL, KEEP) == pytest.approx(0.0)


def test_shape_tv_hand_computed():
    # ref = [0.769230..., 0.230769...]; comp = [0.5, 0.5]
    # TV = 0.5 * (|0.5-0.76923| + |0.5-0.23077|) = 0.26923...
    assert shape_tv([0.5, 0.5], FULL, KEEP) == pytest.approx(0.5 - 0.15 / 0.65)


def test_shape_tv_rejects_misaligned_lengths():
    with pytest.raises(ValueError, match="sorted position order"):
        shape_tv([1.0], FULL, KEEP)


def test_restriction_is_none_when_no_mass_survives():
    """Undefined, not zero: a uniform substitute would read as no distortion."""
    assert renormalised_restriction([0.0, 1.0], [0]) is None
    assert shape_tv([1.0], [0.0, 1.0], [0]) is None


def test_collateral_counts_a_position_wanted_by_any_head_in_the_group():
    # position 1 evicted; head 1 placed 0.9 on it -> collateral 1.0
    heads = [[0.9, 0.05, 0.05], [0.05, 0.90, 0.05]]
    assert collateral_fraction(heads, [1]) == pytest.approx(1.0)


def test_collateral_is_zero_when_no_head_wanted_the_evicted_position():
    heads = [[0.9, 0.005, 0.095], [0.9, 0.005, 0.095]]
    assert collateral_fraction(heads, [1]) == pytest.approx(0.0)


def test_collateral_is_a_fraction_over_evicted_positions():
    # two evicted (1 and 2); only 1 is wanted -> 0.5
    heads = [[0.8, 0.19, 0.005], [0.8, 0.005, 0.005]]
    assert collateral_fraction(heads, [1, 2]) == pytest.approx(0.5)


def test_collateral_with_nothing_evicted_is_a_true_zero():
    assert collateral_fraction([[1.0]], []) == pytest.approx(0.0)


def test_collateral_tau_is_the_amendment_value():
    """Fixed in Q.3 before the cell ran so it cannot be tuned afterwards."""
    assert DEFAULT_COLLATERAL_TAU == 0.01


def test_gqa_shape_28_query_heads_over_4_kv_groups():
    """The primary model's real geometry: 7 query heads share one eviction."""
    n_heads, n_kv = 28, 4
    assert n_heads % n_kv == 0
    assert n_heads // n_kv == 7


# ---------------------------------------------------------------------------
# Amendment R.4 reference implementations.

import math  # noqa: E402

from kvcache.metrics.attn_channels import (  # noqa: E402
    align_to_positions,
    collateral_event,
    restricted_from_logits,
)


def test_alignment_reorders_score_ordered_rows_by_absolute_position():
    """kvpress keeps entries in score order; identity is by position."""
    vals = [0.7, 0.2, 0.1]  # cache order
    pos = [40, 3, 17]  # absolute positions in that cache order
    assert align_to_positions(vals, pos, [3, 17, 40]) == [0.2, 0.1, 0.7]


def test_alignment_refuses_missing_or_duplicated_positions():
    with pytest.raises(ValueError, match="absent"):
        align_to_positions([0.5, 0.5], [1, 2], [1, 3])
    with pytest.raises(ValueError, match="duplicate"):
        align_to_positions([0.5, 0.5], [1, 1], [1])


def test_restriction_from_logits_matches_probabilities_hand_computed():
    logits = [math.log(0.5), math.log(0.3), math.log(0.15), math.log(0.05)]
    m, cond = restricted_from_logits(logits, [0, 2])
    assert m == pytest.approx(0.35)
    assert cond == pytest.approx([0.5 / 0.65, 0.15 / 0.65])


def test_restriction_survives_underflow_that_would_zero_a_probability_path():
    """exp(-2000) underflows to 0.0; the log-space path keeps the shape."""
    logits = [0.0, -2000.0, -2001.0]
    m, cond = restricted_from_logits(logits, [1, 2])
    assert m == pytest.approx(1.0)
    assert sum(cond) == pytest.approx(1.0)
    assert cond[0] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))


def test_collateral_needs_both_conditions():
    # 3 heads, positions 0..4; budget cut = 2 by group-mean attention.
    heads = [
        [0.40, 0.30, 0.02, 0.26, 0.02],
        [0.40, 0.30, 0.02, 0.26, 0.02],
        [0.10, 0.10, 0.70, 0.08, 0.02],  # only this head wants position 2
    ]
    pos = [0, 1, 2, 3, 4]
    out = collateral_event(heads, pos, evicted=[2, 3, 4], budget=2)
    # group means: 0.30, 0.233, 0.247, 0.20, 0.02 -> top-2 = {0, 2}
    # evicted 2: wanted, but IN top-2 -> not collateral
    # evicted 3: wanted (0.26 > .01), outside top-2 -> collateral
    # evicted 4: wanted (0.02 > .01), outside top-2 -> collateral
    assert out["n_evicted_eligible"] == 3
    assert out["n_wanted_any_head"] == 3
    assert out["n_collateral"] == 2
    assert out["collateral"] == pytest.approx(2 / 3)
    assert out["wanted_any_head"] == pytest.approx(1.0)


def test_collateral_differs_from_the_any_head_statistic():
    """The old collateral_fraction is the any-head part only."""
    heads = [[0.5, 0.49, 0.01], [0.5, 0.49, 0.01]]
    out = collateral_event(heads, [0, 1, 2], evicted=[1], budget=2)
    assert out["wanted_any_head"] == 1.0
    assert out["collateral"] == 0.0  # position 1 is inside the group's top-2
    assert collateral_fraction(heads, [1]) == 1.0


def test_collateral_with_nothing_eligible_is_unavailable_not_zero():
    heads = [[0.5, 0.5]]
    out = collateral_event(heads, [0, 1], evicted=[0], budget=1, protected=[0])
    assert out["n_evicted_eligible"] == 0
    assert out["collateral"] is None and out["wanted_any_head"] is None


def test_collateral_ties_break_toward_the_lower_position():
    heads = [[0.25, 0.25, 0.25, 0.25]]
    out = collateral_event(heads, [0, 1, 2, 3], evicted=[1, 2, 3], budget=2)
    # top-2 by (-mean, position) = {0, 1}; evicted 2 and 3 are collateral
    assert out["n_collateral"] == 2

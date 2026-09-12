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

"""Hand-computed checks for the Amendment R.1 paired contrast."""

from __future__ import annotations

import pytest

from kvcache.metrics.paired import paired_flip_contrast, trace_identity


def test_hand_computed_flips_and_total():
    fc = {("p1", 11): True, ("p1", 23): True, ("p2", 11): False, ("p2", 23): True}
    comp = {("p1", 11): False, ("p1", 23): True, ("p2", 11): True, ("p2", 23): True}
    out = paired_flip_contrast(fc, comp, n_resamples=200)
    # conditions: lost, same, gained, same -> net 0, lost 1/4, gained 1/4
    assert out["total_change"]["mean"] == pytest.approx(0.0)
    assert out["harmful_flip_mass"]["mean"] == pytest.approx(0.25)
    assert out["beneficial_flip_mass"]["mean"] == pytest.approx(0.25)
    assert out["n_matched"] == 4


def test_net_flip_equals_beneficial_minus_harmful():
    fc = {("a", s): s % 2 == 0 for s in range(8)} | {("b", s): True for s in range(8)}
    comp = {("a", s): True for s in range(8)} | {("b", s): s < 3 for s in range(8)}
    out = paired_flip_contrast(fc, comp, n_resamples=200)
    assert out["total_change"]["mean"] == pytest.approx(
        out["beneficial_flip_mass"]["mean"] - out["harmful_flip_mass"]["mean"]
    )


def test_unmatched_conditions_are_counted_not_imputed():
    fc = {("p1", 11): True, ("p1", 23): True}
    comp = {("p1", 11): False, ("p9", 11): True}
    out = paired_flip_contrast(fc, comp, n_resamples=50)
    assert out["n_matched"] == 1
    assert out["n_only_fc"] == 1 and out["n_only_comp"] == 1
    assert out["total_change"]["mean"] == pytest.approx(-1.0)


def test_no_overlap_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="no matched"):
        paired_flip_contrast({("a", 1): True}, {("b", 1): True})


def test_resampling_unit_is_the_problem():
    fc = {(f"p{i}", s): True for i in range(5) for s in range(4)}
    comp = {(f"p{i}", s): i != 0 for i in range(5) for s in range(4)}
    out = paired_flip_contrast(fc, comp, n_resamples=200)
    assert out["total_change"]["n_clusters"] == 5
    assert out["total_change"]["n_observations"] == 20


def test_trace_identity_counts_and_locates_the_first_difference():
    a = {("p", 1): [1, 2, 3], ("p", 2): [1, 2, 3], ("q", 1): [5], ("r", 1): [7]}
    b = {("p", 1): [1, 2, 3], ("p", 2): [1, 9, 3], ("q", 1): [5, 6], ("s", 1): [0]}
    out = trace_identity(a, b)
    assert out["n_shared"] == 3
    assert out["n_identical"] == 1
    assert out["n_only_a"] == 1 and out["n_only_b"] == 1
    # a strict prefix differs at the end of the shorter trace
    assert out["first_difference"] == {"('p', 2)": 1, "('q', 1)": 1}

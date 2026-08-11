"""Hand-computed checks for the Convention-2 additivity test (Amendment P.1)."""

import pytest

from kvcache.additivity import additivity_test, interaction_on


def _tbl(pairs):
    return {(pid, seed): ok for pid, seed, ok in pairs}


def test_interaction_is_zero_when_the_channels_do_not_overlap():
    """Constructed so readout_error + residual_rev lands exactly on delta_tot.

    fc all correct; free all wrong; readout loses half; reverse loses the other
    half. Then readout_error = -0.5, residual_rev = -0.5, delta_tot = -1.0 and
    the interaction is exactly 0 -- the channels partition the damage.
    """
    keys = [("p1", 0), ("p2", 0), ("p3", 0), ("p4", 0)]
    fc = dict.fromkeys(keys, True)
    free = dict.fromkeys(keys, False)
    readout = {keys[0]: False, keys[1]: False, keys[2]: True, keys[3]: True}
    reverse = {keys[0]: True, keys[1]: True, keys[2]: False, keys[3]: False}
    out = interaction_on(keys, fc, free, readout, reverse)
    assert out["delta_tot"] == pytest.approx(-1.0)
    assert out["readout_error"] == pytest.approx(-0.5)
    assert out["residual_rev"] == pytest.approx(-0.5)
    assert out["interaction"] == pytest.approx(0.0)


def test_overlapping_damage_makes_the_terms_over_explain():
    """Both channels destroy the SAME problems -> sum overshoots, interaction > 0."""
    keys = [("p1", 0), ("p2", 0), ("p3", 0), ("p4", 0)]
    fc = dict.fromkeys(keys, True)
    free = dict.fromkeys(keys, False)
    readout = {keys[0]: False, keys[1]: False, keys[2]: True, keys[3]: True}
    reverse = dict(readout)  # identical failures
    out = interaction_on(keys, fc, free, readout, reverse)
    assert out["readout_error"] == pytest.approx(-0.5)
    assert out["residual_rev"] == pytest.approx(-0.5)
    # sum = -1.0 against delta_tot = -1.0 ... still 0 here; make free partial:
    free2 = {keys[0]: False, keys[1]: True, keys[2]: True, keys[3]: True}
    out2 = interaction_on(keys, fc, free2, readout, reverse)
    assert out2["delta_tot"] == pytest.approx(-0.25)
    assert out2["sum_of_terms"] == pytest.approx(-1.0)
    assert out2["interaction"] == pytest.approx(0.75)  # over-explained


def test_common_population_excludes_conditions_missing_from_any_protocol():
    fc = _tbl([("p1", 0, True), ("p2", 0, True), ("p3", 0, True)])
    free = _tbl([("p1", 0, False), ("p2", 0, False), ("p3", 0, True)])
    readout = _tbl([("p1", 0, True), ("p2", 0, False)])  # p3 undefined
    reverse = _tbl([("p1", 0, True), ("p2", 0, True)])  # p3 undefined
    out = additivity_test(fc, free, readout, reverse, n_resamples=200)
    assert out["n_conditions"] == 2
    assert out["n_problems"] == 2


def test_ci_brackets_the_point_estimate_and_records_provenance():
    keys = [(f"p{i}", 0) for i in range(20)]
    fc = dict.fromkeys(keys, True)
    free = {k: (i % 4 == 0) for i, k in enumerate(keys)}
    readout = {k: (i % 3 != 0) for i, k in enumerate(keys)}
    reverse = {k: (i % 2 == 0) for i, k in enumerate(keys)}
    out = additivity_test(fc, free, readout, reverse, n_resamples=500, seed=0)
    lo, hi = out["interaction_ci95"]
    assert lo <= out["interaction"] <= hi
    assert out["n_resamples"] == 500 and out["seed"] == 0
    assert out["convention"] == "2"


def test_refuses_when_no_condition_is_defined_everywhere():
    with pytest.raises(ValueError):
        additivity_test(
            _tbl([("p1", 0, True)]),
            _tbl([("p1", 0, True)]),
            _tbl([("p2", 0, True)]),
            _tbl([("p2", 0, True)]),
        )


def test_selection_is_bounded_both_ways_not_absorbed():
    """The dropped conditions must be scored both ways, per Amendment K layer 1."""
    keys = [(f"p{i}", 0) for i in range(10)]
    fc = dict.fromkeys(keys, True)
    free = {k: (i % 2 == 0) for i, k in enumerate(keys)}
    # Last three conditions are undefined for readout and reverse -> dropped.
    readout = {k: True for k in keys[:7]}
    reverse = {k: False for k in keys[:7]}
    out = additivity_test(fc, free, readout, reverse, n_resamples=200)
    assert out["n_conditions"] == 7
    assert out["n_dropped_by_selection"] == 3
    lo, hi = out["primary_bound_interaction"]
    assert lo <= hi
    assert "easier subset" in out["estimand_scope"]

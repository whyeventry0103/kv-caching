import pytest

from kvcache.decomposition import decompose


def test_residual_is_arithmetic_and_honestly_named():
    result = decompose(total_change=-0.30, readout_change=-0.10)
    assert result.unexplained_residual == pytest.approx(-0.20)
    assert not hasattr(result, "causal_trajectory_effect")
    assert result.residual_bracket is None


def test_reverse_replay_brackets_the_residual_and_sign_is_carried():
    result = decompose(
        total_change=-0.30,
        readout_change=-0.10,
        reverse_readout_change=-0.25,
        flip_mass_net=-0.20,
    )
    assert result.unexplained_residual == pytest.approx(-0.20)
    # Endpoints: forward = total - readout = -0.20; reverse = reverse_readout_change
    # itself (acc_rev - acc_fc), the residual estimated with readout measured on
    # the compressed model's own states.
    low, high = result.residual_bracket
    assert low == pytest.approx(-0.25)
    assert high == pytest.approx(-0.20)
    assert result.flip_mass_net == pytest.approx(-0.20)


def test_paired_policy_contrast_reports_wins_ties_losses_separately():
    """'non-negative on N' is not 'beats on N' -- the breakdown must survive."""
    from kvcache.stats import paired_policy_contrast

    a = {"p1": [True, True], "p2": [True, False], "p3": [False, False]}
    b = {"p1": [False, False], "p2": [True, False], "p3": [True, True]}
    out = paired_policy_contrast(a, b, n_resamples=200)
    assert out["n_problems"] == 3
    assert (out["wins"], out["ties"], out["losses"]) == (1, 1, 1)
    assert out["non_negative"] == 2
    assert out["mean_difference"] == pytest.approx(0.0)


def test_paired_policy_contrast_pairs_on_the_problem():
    from kvcache.stats import paired_policy_contrast

    a = {"p1": [True, True, True, True]}
    b = {"p1": [True, False, False, False]}
    out = paired_policy_contrast(a, b, n_resamples=100)
    assert out["mean_difference"] == pytest.approx(0.75)
    assert out["wins"] == 1


def test_paired_policy_contrast_requires_common_problems():
    import pytest as _pytest

    from kvcache.stats import paired_policy_contrast

    with _pytest.raises(ValueError):
        paired_policy_contrast({"a": [True]}, {"b": [True]})


def test_percentile_ci_mc_error_separates_a_safe_margin_from_a_fragile_one():
    """The point of the measurement: distinguish a real margin from a seed."""
    from kvcache.stats import percentile_ci_mc_error

    # Comfortably negative: every resample stays below zero.
    safe = percentile_ci_mc_error([-0.9, -0.8, -0.85, -0.95] * 25, n_seeds=20)
    assert safe["upper"]["max"] < 0
    assert safe["upper"]["crosses_zero"] == 0
    assert safe["upper"]["margin_sd"] is None or safe["upper"]["margin_sd"] > 3

    # Straddling zero: the upper endpoint is positive regardless of seed.
    fragile = percentile_ci_mc_error([-1.0, 1.0] * 50, n_seeds=20)
    assert fragile["upper"]["max"] > 0
    assert fragile["n_resamples"] == 2000
    assert fragile["n_units"] == 100


def test_percentile_ci_mc_error_rejects_an_empty_sample():
    import pytest as _pytest

    from kvcache.stats import percentile_ci_mc_error

    with _pytest.raises(ValueError):
        percentile_ci_mc_error([])


def test_paired_policy_contrast_records_bootstrap_provenance():
    """A percentile CI with no recorded seed is not reproducible.

    The published CI upper endpoint could not be reproduced exactly because the
    original run's seed was never recorded; pinning it here is the fix.
    """
    from kvcache.stats import paired_policy_contrast

    a = {"p1": [True, True], "p2": [True, False], "p3": [False, False]}
    b = {"p1": [False, False], "p2": [True, False], "p3": [True, True]}
    out = paired_policy_contrast(a, b, n_resamples=256, seed=7)
    assert out["seed"] == 7
    assert out["n_resamples"] == 256
    # same seed reproduces the interval exactly; a different seed need not
    again = paired_policy_contrast(a, b, n_resamples=256, seed=7)
    assert again["ci95"] == out["ci95"]


def test_paired_policy_contrast_histogram_is_a_sufficient_statistic():
    """The histogram must account for every problem, so a reader can
    recompute the bootstrap from the artifact alone."""
    from kvcache.stats import paired_policy_contrast

    a = {"p1": [True, True], "p2": [True, False], "p3": [False, False]}
    b = {"p1": [False, False], "p2": [True, False], "p3": [True, True]}
    out = paired_policy_contrast(a, b, n_resamples=200)
    histogram = out["difference_histogram"]
    assert sum(count for _, count in histogram) == out["n_problems"]
    assert histogram == sorted(histogram)
    assert dict(histogram)[-1.0] == 1  # p3: 0.0 - 1.0
    assert dict(histogram)[0.0] == 1  # p2: tie


def test_paired_residual_contrast_identical_presses_give_zero():
    from kvcache.stats import paired_residual_contrast

    r = {"a": {"p1": -0.3, "p2": -0.4}, "b": {"p1": -0.3, "p2": -0.4}}
    got = paired_residual_contrast(r, n_resamples=200, seed=0)
    assert got["mean_abs_pairwise_difference"] == pytest.approx(0.0)
    assert got["pairs"]["a_vs_b"]["mean_difference"] == pytest.approx(0.0)
    assert got["pairs"]["a_vs_b"]["excludes_zero"] is False


def test_paired_residual_contrast_constant_offset_is_detected():
    """Every problem shifted by the same amount: zero paired variance, so the
    CI collapses onto the offset and the difference is unambiguous -- exactly
    the case an unpaired comparison of wide marginal CIs would call a tie."""
    from kvcache.stats import paired_residual_contrast

    r = {
        "a": {f"p{i}": -0.30 for i in range(20)},
        "b": {f"p{i}": -0.35 for i in range(20)},
    }
    got = paired_residual_contrast(r, n_resamples=200, seed=0)
    pair = got["pairs"]["a_vs_b"]
    assert pair["mean_difference"] == pytest.approx(0.05)
    assert pair["ci95"][0] == pytest.approx(0.05)
    assert pair["ci95"][1] == pytest.approx(0.05)
    assert pair["excludes_zero"] is True


def test_paired_residual_contrast_pairs_on_common_problems_only():
    from kvcache.stats import paired_residual_contrast

    r = {"a": {"p1": -0.3, "p2": -0.4, "p3": -0.5}, "b": {"p1": -0.3, "p2": -0.4}}
    got = paired_residual_contrast(r, n_resamples=50, seed=0)
    assert got["n_problems"] == 2


def test_paired_residual_contrast_needs_two_presses():
    from kvcache.stats import paired_residual_contrast

    with pytest.raises(ValueError, match="at least two"):
        paired_residual_contrast({"a": {"p1": -0.3}}, n_resamples=10)


def test_paired_residual_contrast_mapd_is_not_max_minus_min():
    """With three presses at -0.30/-0.34/-0.38 the pairwise gaps are
    0.04/0.04/0.08, mean 0.0533 -- max-min alone would report 0.08 and be
    driven entirely by the two extreme cells."""
    from kvcache.stats import paired_residual_contrast

    r = {
        "a": {f"p{i}": -0.30 for i in range(10)},
        "b": {f"p{i}": -0.34 for i in range(10)},
        "c": {f"p{i}": -0.38 for i in range(10)},
    }
    got = paired_residual_contrast(r, n_resamples=50, seed=0)
    assert got["mean_abs_pairwise_difference"] == pytest.approx(0.053333, abs=1e-5)
    assert got["n_pairs_excluding_zero"] == 3


def test_resolution_spread_collapses_when_readout_cannot_separate():
    """The AIME case in miniature: free-running separates the presses, readout
    gives every press the same score. The difference must be the free-run
    spread exactly, and rank agreement must NOT be credited for the ties."""
    from kvcache.stats import resolution_spread

    pids = [f"p{i}" for i in range(20)]
    free = {
        "a": {p: 1.0 for p in pids},
        "b": {p: 0.5 for p in pids},
        "c": {p: 0.0 for p in pids},
    }
    readout = {name: {p: 0.5 for p in pids} for name in "abc"}
    got = resolution_spread(free, readout, n_resamples=100, seed=0)
    assert got["readout_spread"] == pytest.approx(0.0)
    assert got["free_run_spread"] == pytest.approx(0.6666667, abs=1e-6)
    assert got["spread_difference"] == pytest.approx(0.6666667, abs=1e-6)
    assert got["spread_ratio_descriptive"] == pytest.approx(0.0)
    assert got["rank_agreement"] == pytest.approx(0.0)


def test_resolution_spread_is_zero_when_both_protocols_agree():
    from kvcache.stats import resolution_spread

    pids = [f"p{i}" for i in range(20)]
    table = {
        "a": {p: 1.0 for p in pids},
        "b": {p: 0.5 for p in pids},
    }
    got = resolution_spread(table, table, n_resamples=100, seed=0)
    assert got["spread_difference"] == pytest.approx(0.0)
    assert got["spread_difference_excludes_zero"] is False
    assert got["rank_agreement"] == pytest.approx(1.0)
    assert got["spread_ratio_descriptive"] == pytest.approx(1.0)


def test_resolution_spread_uses_mean_abs_pairwise_not_max_min():
    from kvcache.stats import resolution_spread

    pids = [f"p{i}" for i in range(10)]
    free = {
        "a": {p: 1.0 for p in pids},
        "b": {p: 0.9 for p in pids},
        "c": {p: 0.0 for p in pids},
    }
    got = resolution_spread(free, free, n_resamples=50, seed=0)
    # gaps 0.1, 1.0, 0.9 -> mean 0.6667; max-min would say 1.0
    assert got["free_run_spread"] == pytest.approx(0.6666667, abs=1e-6)


def test_resolution_spread_flags_itself_post_hoc():
    """The label is part of the result, not something a caller may forget."""
    from kvcache.stats import resolution_spread

    pids = [f"p{i}" for i in range(5)]
    t = {"a": {p: 1.0 for p in pids}, "b": {p: 0.0 for p in pids}}
    got = resolution_spread(t, t, n_resamples=20, seed=0)
    assert "post-hoc" in got["analysis_status"]

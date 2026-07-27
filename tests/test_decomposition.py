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

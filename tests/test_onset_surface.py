"""Hand-computed checks for the Amendment Q.2 exposure surface."""

from __future__ import annotations

import pytest

from kvcache.metrics.onset_surface import (
    exposure_slope,
    matched_positions,
    window_divergence,
)


def test_window_divergence_is_the_mean_over_the_window():
    est = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert window_divergence(est, 2, 2) == pytest.approx(2.5)
    assert window_divergence(est, 0, 6) == pytest.approx(2.5)


def test_window_refuses_a_partial_window_rather_than_averaging_it():
    """A short window is a different estimator; mixing them biases the slope."""
    est = [0.0, 1.0, 2.0]
    assert window_divergence(est, 2, 2) is None
    assert window_divergence(est, 0, 4) is None


def test_window_rejects_degenerate_arguments():
    assert window_divergence([1.0, 2.0], -1, 2) is None
    assert window_divergence([1.0, 2.0], 0, 0) is None


def test_matched_positions_start_a_full_window_after_the_latest_onset():
    """The latest arm must be compressing for a whole window before it is read."""
    pos = matched_positions(
        [0, 512, 2048], width=256, trace_len=10000, step=1024, count=3
    )
    assert pos == [2304, 3328, 4352]


def test_matched_positions_stop_when_the_trace_runs_out():
    pos = matched_positions([0, 2048], width=256, trace_len=2700, step=1024, count=5)
    assert pos == [2304]
    assert (
        matched_positions([0, 2048], width=256, trace_len=2000, step=1024, count=5)
        == []
    )


def test_slope_is_positive_when_divergence_grows_with_exposure():
    # exposures 1000, 2000, 3000; divergence 0.1, 0.2, 0.3 -> slope 1e-4
    assert exposure_slope([1000, 2000, 3000], [0.1, 0.2, 0.3]) == pytest.approx(1e-4)


def test_slope_is_zero_when_divergence_is_flat_in_exposure():
    """The RATIO-ONLY shape: dose fixed, exposure varies, nothing happens."""
    assert exposure_slope([1000, 2000, 3000], [0.2, 0.2, 0.2]) == pytest.approx(0.0)


def test_slope_is_negative_when_divergence_falls_with_exposure():
    assert exposure_slope([1000, 2000], [0.3, 0.1]) == pytest.approx(-2e-4)


def test_slope_is_none_for_a_degenerate_design_not_zero():
    """Identical exposures are a broken design, not a measured null."""
    assert exposure_slope([1000, 1000, 1000], [0.1, 0.2, 0.3]) is None
    assert exposure_slope([1000], [0.1]) is None
    assert exposure_slope([], []) is None


def test_the_q2_geometry_holds_ratio_fixed_while_exposure_varies():
    """Why this design escapes r = B/t: at matched t, r is identical by construction."""
    B, t = 256, 4096
    for _t0 in (0, 512, 2048):
        assert B / t == pytest.approx(0.0625)  # dose identical across arms
    assert [t - t0 for t0 in (0, 512, 2048)] == [4096, 3584, 2048]  # exposure differs

"""Amendment Q.5 latency-sweep verdict.

Effects are ``none`` minus arm, so POSITIVE means divergence was reduced --
reinsertion helped. P.5 measured only the far regime and found no help; the
sweep asks whether help exists at short lag and decays.
"""

from __future__ import annotations

from kvcache.metrics.restoration import classify_latency_sweep


def _p(lag, eff, lo, hi):
    return (lag, eff, (lo, hi))


def test_decay_when_the_shortest_lag_helps_and_effects_fall_monotonically():
    pts = [
        _p(1, 0.030, 0.020, 0.040),
        _p(4, 0.015, 0.008, 0.022),
        _p(16, 0.004, -0.002, 0.010),
        _p(64, -0.001, -0.003, 0.001),
    ]
    out = classify_latency_sweep(pts)
    assert out["verdict"] == "DECAY"
    assert out["monotone_decreasing"] is True


def test_flat_null_when_no_lag_helps_including_the_shortest():
    """The P.5 result extended: composition matters more than contents."""
    pts = [
        _p(1, -0.0013, -0.0019, -0.0007),
        _p(4, -0.0014, -0.0020, -0.0008),
        _p(256, -0.0015, -0.0022, -0.0009),
    ]
    assert classify_latency_sweep(pts)["verdict"] == "FLAT-NULL"


def test_non_monotone_when_help_exists_but_the_trend_is_not_monotone():
    pts = [
        _p(1, 0.001, -0.002, 0.004),
        _p(4, 0.030, 0.020, 0.040),
        _p(16, 0.002, -0.001, 0.005),
    ]
    out = classify_latency_sweep(pts)
    assert out["verdict"] == "NON-MONOTONE"


def test_help_requires_the_ci_to_exclude_zero_not_just_a_positive_point():
    """A positive point estimate with a CI spanning zero is not help."""
    pts = [_p(1, 0.010, -0.005, 0.025), _p(4, 0.001, -0.004, 0.006)]
    assert classify_latency_sweep(pts)["verdict"] == "FLAT-NULL"


def test_points_are_ordered_by_lag_regardless_of_input_order():
    pts = [_p(64, -0.001, -0.003, 0.001), _p(1, 0.030, 0.020, 0.040)]
    out = classify_latency_sweep(pts)
    assert out["lags"] == [1, 64]
    assert out["verdict"] == "DECAY"


def test_a_single_point_cannot_support_a_monotone_claim():
    out = classify_latency_sweep([_p(1, 0.03, 0.02, 0.04)])
    assert out["verdict"] == "INSUFFICIENT"
    assert classify_latency_sweep([])["verdict"] == "INSUFFICIENT"

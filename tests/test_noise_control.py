from __future__ import annotations

import pytest

from kvcache.metrics.noise_control import calibration_error, match_gate, window_means


def _compounding(first: float, later: float, n_first: int = 50, n_later: int = 150):
    return [first] * n_first + [later] * n_later


def test_calibration_error_is_mean_absolute_gap():
    assert calibration_error([0.1, 0.2], [0.2, 0.2]) == pytest.approx(0.05)


def test_calibration_error_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        calibration_error([0.1], [0.1, 0.2])


def test_window_means_split_at_fifty():
    got = window_means(_compounding(0.002, 0.088))
    assert got["before"] == 0.0
    assert got["first50"] == pytest.approx(0.002)
    assert got["later"] == pytest.approx(0.088)


def test_match_gate_passes_on_an_exact_match():
    target = _compounding(0.0018, 0.0884)
    got = match_gate(target, list(target))
    assert got["passed"] is True
    assert got["reasons"] == []


def test_match_gate_fails_a_flat_profile_with_the_right_average():
    """The core reason the shape condition exists. A flat profile can carry the
    right overall mean while having none of the compounding under test, and
    compounding is the property the control is supposed to reproduce."""
    target = _compounding(0.0018, 0.0884)
    mean = sum(target) / len(target)
    flat = [mean] * len(target)
    got = match_gate(target, flat)
    assert got["passed"] is False
    assert any("growth shape" in r for r in got["reasons"])


def test_match_gate_fails_when_magnitude_is_off_but_shape_is_right():
    """The mirror case: doubling every step preserves the growth ratio exactly."""
    target = _compounding(0.0018, 0.0884)
    doubled = [2 * v for v in target]
    got = match_gate(target, doubled)
    assert got["passed"] is False
    assert any("window means" in r for r in got["reasons"])
    assert got["shape_relative_error"] == pytest.approx(0.0)


def test_match_gate_allows_small_deviation_within_tolerance():
    target = _compounding(0.0018, 0.0884)
    close = _compounding(0.0018 * 1.05, 0.0884 * 1.05)
    got = match_gate(target, close)
    assert got["passed"] is True


def test_match_gate_carries_the_do_not_report_contract():
    """The refusal must travel with the result, not live only in a docstring."""
    target = _compounding(0.0018, 0.0884)
    got = match_gate(target, [0.5] * len(target))
    assert got["passed"] is False
    assert "MUST NOT be reported" in str(got["contract"])


def test_match_gate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        match_gate([0.1, 0.2], [0.1])

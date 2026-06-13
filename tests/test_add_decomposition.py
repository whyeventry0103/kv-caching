import pytest

from kvcache.metrics.add import (
    add_decomposition,
    aggregate_flip_mass,
    flip_mass,
)


def test_termination_split_separates_length_from_reasoning():
    full = ["A", "A", "B", "B"]
    compressed = ["A", "A", "B", None]
    result = add_decomposition(full, compressed)
    assert result.termination_mass_full == pytest.approx(0.0)
    assert result.termination_mass_compressed == pytest.approx(0.25)
    assert result.termination_shift == pytest.approx(0.25)
    assert result.within_answered_tv == pytest.approx(1 / 6)
    # The bound add_tv >= 0.5 * |termination_shift| always holds.
    assert result.add_tv >= 0.5 * abs(result.termination_shift) - 1e-9


def test_pure_termination_contamination_reads_zero_reasoning_shift():
    full = ["A", "B"]
    compressed = ["A", "B", None, None]  # same answered shape, extra OTHER mass
    result = add_decomposition(full, compressed)
    assert result.within_answered_tv == pytest.approx(0.0)
    assert result.add_tv == pytest.approx(0.5)
    assert result.termination_shift == pytest.approx(0.5)


def test_within_answered_is_none_when_no_answered_mass():
    result = add_decomposition(["A", "B"], [None, None])
    assert result.within_answered_tv is None


def test_signed_flip_mass_and_aggregation():
    gained = flip_mass(["2", "2", "3", "3"], ["2", "2", "2", "3"], gold_raw="2")
    assert gained == pytest.approx(0.25)
    lost = flip_mass(["2", "2", "2", "3"], ["3", "3", "3", "3"], gold_raw="2")
    assert lost == pytest.approx(-0.75)
    totals = aggregate_flip_mass([0.25, -0.5, 0.1])
    assert totals == {
        "gained": pytest.approx(0.35),
        "lost": pytest.approx(0.5),
        "net": pytest.approx(-0.15),
    }

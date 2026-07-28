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


def test_add_is_invariant_to_answer_encounter_order():
    """The 1-ULP non-determinism, asserted away rather than documented.

    Dict iteration is insertion order, and insertion order follows whichever
    answer a run happened to see first -- which follows the order the row files
    were globbed. Summing in that order made per-problem within_answered_tv
    reproducible only to within a ULP, while the byte-diff regenerability check
    and the nightly parity check both assume bitwise determinism.
    """
    import random

    from kvcache.metrics.add import add_decomposition, add_tv

    answers = ["2", "3/4", "\\pi", "-1", "0.5", None, "7", "x+1"]
    rng = random.Random(20260729)
    full = [answers[i % len(answers)] for i in range(40)]
    comp = [answers[(i * 3) % len(answers)] for i in range(40)]

    baseline = add_decomposition(full, comp)
    baseline_tv = add_tv(full, comp)
    for _ in range(25):
        f = list(full)
        c = list(comp)
        rng.shuffle(f)
        rng.shuffle(c)
        got = add_decomposition(f, c)
        # Shuffling the samples changes nothing about the distributions, so
        # every field must reproduce EXACTLY -- not approximately.
        assert got.add_tv == baseline.add_tv
        assert got.within_answered_tv == baseline.within_answered_tv
        assert got.termination_shift == baseline.termination_shift
        assert add_tv(f, c) == baseline_tv

"""Finite-channel information reference (Meeting 5, M5-C): the section 10.3
reference cases, hand-computed noisy channels, nuisance marginalization,
category-permutation invariance, OTHER handling, and grid validation."""

from __future__ import annotations

import math

import pytest

from kvcache import task_information as ti
from kvcache.task_information import K, Readout


def _channel(fn, greedy=None):
    """A complete grid whose S-readout row is fn(s, n) -> K probabilities."""
    out = []
    for s in range(K):
        for n in range(K):
            ps = fn(s, n)
            lp = tuple(math.log(p) if p > 0 else -math.inf for p in ps)
            out.append(Readout(s, n, lp, greedy(s, n) if greedy else None))
    return out


def _noisy(correct, s):
    wrong = (1 - correct) / (K - 1)
    return [correct if z == s else wrong for z in range(K)]


@pytest.mark.parametrize(
    "kind, mi, correct",
    [
        ("identity", 2.0, 1.0),
        ("uniform", 0.0, 0.25),
        ("permutation", 2.0, 0.0),
        ("always_other", 0.0, 0.0),
    ],
)
def test_review_reference_table(kind, mi, correct):
    summ = ti.channel_summary(ti.reference_channel(kind), "s")
    assert summ["mi_bits"] == pytest.approx(mi, abs=1e-12)
    assert summ["expected_correct_prob"] == pytest.approx(correct, abs=1e-12)


def test_reference_extras():
    perm = ti.channel_summary(ti.reference_channel("permutation"), "s")
    assert perm["greedy_accuracy"] == 0.0 and perm["log_loss_bits"] == math.inf
    other = ti.channel_summary(ti.reference_channel("always_other"), "s")
    assert other["other_prob"] == 1.0
    ident = ti.channel_summary(ti.reference_channel("identity", query="n"), "n")
    assert ident["mi_bits"] == pytest.approx(2.0)
    assert ident["mi_other_variable_bits"] == pytest.approx(0.0, abs=1e-12)
    assert ident["log_loss_bits"] == 0.0 and ident["reader_score_bits"] == 2.0


def test_target_and_nuisance_are_scored_separately():
    copy = ti.channel_summary(ti.reference_channel("copy_other"), "s")
    assert copy["mi_bits"] == pytest.approx(0.0, abs=1e-12)  # reports N, not S
    assert copy["mi_other_variable_bits"] == pytest.approx(2.0)
    assert copy["expected_correct_prob"] == pytest.approx(0.25)


def test_symmetric_noise_hand_computed():
    # pbar uniform, so I = 2 + sum_z p log2 p = 2 + .7 log2 .7 + .3 log2 .1
    rows = _channel(lambda s, n: _noisy(0.7, s))
    want = 2 + 0.7 * math.log2(0.7) + 0.3 * math.log2(0.1)
    assert want == pytest.approx(0.643221, abs=1e-6)
    summ = ti.channel_summary(rows, "s")
    assert summ["mi_bits"] == pytest.approx(want, abs=1e-12)
    assert summ["log_loss_bits"] == pytest.approx(-math.log2(0.7))
    assert summ["expected_correct_prob"] == pytest.approx(0.7)


def test_other_mass_carries_no_information_and_is_not_renormalized():
    # correct 0.5, OTHER 0.5: pbar = (1/8,1/8,1/8,1/8,1/2); I = .5*log2(4) = 1 bit
    rows = _channel(lambda s, n: [0.5 if z == s else 0.0 for z in range(K)])
    summ = ti.channel_summary(rows, "s")
    assert summ["mi_bits"] == pytest.approx(1.0)
    assert summ["other_prob"] == pytest.approx(0.5)
    assert summ["log_loss_bits"] == pytest.approx(1.0)
    assert summ["reader_score_bits"] == pytest.approx(1.0)


def test_nuisance_is_marginalized_not_conditioned_on():
    # identity when n is even, uniform when odd: p(z|s) = .625 on s, .125 else
    def fn(s, n):
        return [1.0 if z == s else 0.0 for z in range(K)] if n % 2 == 0 else [0.25] * K

    want = 2 + 0.625 * math.log2(0.625) + 0.375 * math.log2(0.125)
    assert want == pytest.approx(0.451205, abs=1e-6)
    assert ti.mutual_information_bits(_channel(fn), "s") == pytest.approx(want)


def test_mi_is_invariant_to_relabelling_categories_but_accuracy_is_not():
    perm = [2, 0, 3, 1]
    base = _channel(lambda s, n: _noisy(0.6, s))
    moved = _channel(lambda s, n: [_noisy(0.6, s)[perm.index(z)] for z in range(K)])
    a, b = ti.channel_summary(base, "s"), ti.channel_summary(moved, "s")
    assert a["mi_bits"] == pytest.approx(b["mi_bits"], abs=1e-12)
    assert a["expected_correct_prob"] == pytest.approx(0.6)
    assert b["expected_correct_prob"] == pytest.approx(0.4 / 3)


def test_reader_score_can_be_negative():
    rows = _channel(lambda s, n: [0.1 if z == s else 0.3 for z in range(K)])
    summ = ti.channel_summary(rows, "s")
    assert summ["reader_score_bits"] == pytest.approx(2 - math.log2(10))
    assert summ["reader_score_bits"] < 0


def test_greedy_accuracy_uses_the_native_top1_flag():
    rows = _channel(lambda s, n: _noisy(0.7, s), greedy=lambda s, n: s if n else None)
    assert ti.channel_summary(rows, "s")["greedy_accuracy"] == pytest.approx(0.75)


def test_channel_row_other_mass_is_accurate_when_tiny():
    lp = [math.log(0.25 - 2.5e-13)] * K  # OTHER = 1e-12
    row, flag = ti.channel_row(lp)
    assert not flag
    assert math.exp(row[ti.OTHER]) == pytest.approx(1e-12, rel=1e-3)


def test_channel_row_rounding_excess_flagged_and_large_excess_refused():
    row, flag = ti.channel_row([math.log(0.25 + 1e-9)] * K)
    assert flag and row[ti.OTHER] == -math.inf
    with pytest.raises(ValueError, match="exceeds"):
        ti.channel_row([math.log(0.3)] * K)
    for bad in ([0.1, -1, -1, -1], [math.nan, -1, -1, -1], [-1, -1, -1]):
        with pytest.raises(ValueError):
            ti.channel_row(bad)


def test_grid_status_final_partial_invalid():
    full = ti.reference_channel("identity")
    assert ti.grid_status(full) == (ti.FINAL, [])
    status, why = ti.grid_status(full[:-1])
    assert status == ti.PARTIAL and "missing variants [(3, 3)]" in why[0]
    with pytest.raises(ValueError, match="PARTIAL"):
        ti.mutual_information_bits(full[:-1], "s")  # never re-weights the prior
    assert ti.grid_status(full + full[:1])[0] == ti.INVALID
    bad = [Readout(4, 0, full[0].logprobs, None)] + full[1:]
    assert ti.grid_status(bad)[0] == ti.INVALID
    neg = [Readout(0, 0, (0.5, -1.0, -1.0, -1.0), None)] + full[1:]
    assert ti.grid_status(neg)[0] == ti.INVALID
    g = [Readout(0, 0, full[0].logprobs, 7)] + full[1:]
    assert ti.grid_status(g)[0] == ti.INVALID


def test_reference_table_and_labels():
    rows = {r["channel"]: r for r in ti.reference_table()}
    assert rows["permutation"]["mi_bits"] == pytest.approx(2.0)
    assert "not total cache MI" == ti.NOT_LABEL
    assert "output-channel MI" in ti.ENDPOINT_LABEL

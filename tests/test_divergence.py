"""Hand-computed checks for the per-step divergence measures (Amendment P.4)."""

import math

import pytest

from kvcache.metrics.divergence import measures_from_probs


def test_identical_distributions_are_zero_everywhere():
    p = [0.5, 0.3, 0.2]
    out = measures_from_probs(p, p)
    assert out["tv"] == pytest.approx(0.0)
    assert out["jsd"] == pytest.approx(0.0)
    assert out["top1_agree"] is True
    assert out["rank_displacement"] == 0
    assert out["kl_fwd"] == pytest.approx(0.0)
    assert out["kl_rev"] == pytest.approx(0.0)


def test_disjoint_support_saturates_jsd_at_one_bit_and_diverges_kl():
    """The exact case JSD exists for: KL is infinite, JSD is 1 bit."""
    out = measures_from_probs([1.0, 0.0], [0.0, 1.0])
    assert out["tv"] == pytest.approx(1.0)
    assert out["jsd"] == pytest.approx(1.0)
    assert out["kl_fwd"] is None  # non-finite, reported as such not clamped
    assert out["kl_rev"] is None
    assert out["top1_agree"] is False


def test_tv_matches_the_hand_computation():
    out = measures_from_probs([0.7, 0.2, 0.1], [0.4, 0.4, 0.2])
    assert out["tv"] == pytest.approx(0.5 * (0.3 + 0.2 + 0.1))


def test_rank_displacement_counts_tokens_now_above_the_full_argmax():
    # full argmax is index 0; under test it sits behind two others.
    out = measures_from_probs([0.6, 0.3, 0.1], [0.1, 0.5, 0.4])
    assert out["rank_displacement"] == 2
    assert out["top1_agree"] is False


def test_rank_displacement_is_zero_while_top1_still_agrees():
    out = measures_from_probs([0.6, 0.3, 0.1], [0.4, 0.35, 0.25])
    assert out["top1_agree"] is True
    assert out["rank_displacement"] == 0


def test_jsd_is_symmetric_where_kl_is_not():
    p, q = [0.9, 0.1], [0.4, 0.6]
    a, b = measures_from_probs(p, q), measures_from_probs(q, p)
    assert a["jsd"] == pytest.approx(b["jsd"])
    assert a["kl_fwd"] != pytest.approx(a["kl_rev"])


def test_jsd_never_exceeds_one_bit():
    for q in ([0.5, 0.5], [0.99, 0.01], [0.01, 0.99]):
        assert 0.0 <= measures_from_probs([0.8, 0.2], q)["jsd"] <= 1.0


def test_mismatched_support_refuses():
    with pytest.raises(ValueError):
        measures_from_probs([0.5, 0.5], [1.0])


def test_torch_path_agrees_with_the_reference_implementation():
    """The live path must reproduce the formulas the CPU tests pin down."""
    torch = pytest.importorskip("torch")
    logits_full = torch.tensor([2.0, 1.0, 0.1, -0.5])
    logits_test = torch.tensor([1.2, 1.9, 0.3, -0.2])
    from kvcache.metrics.divergence import measures_from_logits

    live = measures_from_logits(logits_full, logits_test)
    p = torch.softmax(logits_full, dim=-1).tolist()
    q = torch.softmax(logits_test, dim=-1).tolist()
    ref = measures_from_probs(p, q)
    for key in ("tv", "jsd"):
        assert live[key] == pytest.approx(ref[key], abs=1e-9)
    assert live["top1_agree"] == ref["top1_agree"]
    assert live["rank_displacement"] == ref["rank_displacement"]
    assert math.isclose(live["kl_fwd"], ref["kl_fwd"], abs_tol=1e-9)

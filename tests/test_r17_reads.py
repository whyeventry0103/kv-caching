"""R.17 reads: hand-computed cases, including the edge cases each branch rule
has to get right before any real number passes through it."""

from __future__ import annotations

import pytest

from kvcache.metrics import r17_reads as r


def test_auroc_hand_cases_and_undefined_class():
    assert r.auroc([3, 2, 1], [True, False, False]) == pytest.approx(1.0)
    assert r.auroc([1, 2, 3], [True, False, False]) == pytest.approx(0.0)
    # one tie among 2 x 2 comparisons: (1 + 1 + 0.5 + 0) / 4
    assert r.auroc([2, 1, 1, 0], [True, True, False, False]) == pytest.approx(
        (1 + 1 + 1 + 0.5) / 4
    )
    assert r.auroc([1, 2], [False, False]) is None


def test_auroc_bootstrap_perfect_separator_is_significant():
    scores, labels = {}, {}
    for i in range(10):
        for s in range(4):
            y = s < 2
            scores[(f"p{i}", s)] = 1.0 if y else 0.0
            labels[(f"p{i}", s)] = y
    o = r.auroc_cluster_bootstrap(scores, labels, n_resamples=200)
    assert o["auroc"] == 1.0 and o["ci95"] == [1.0, 1.0]
    assert o["p_two_sided"] == 0.0 and o["n_positive"] == 20


def test_h2_labels():
    big = {"spread": 0.3, "median_half_width": 0.03}
    small = {"spread": 0.01, "median_half_width": 0.03}
    assert r.h2_label(m=big, d=small) == "MIRROR"
    assert r.h2_label(m=small, d=big) == "INVERTED"
    assert r.h2_label(m=big, d=big) == "NEITHER"


def test_spread_read():
    o = r.spread_read(
        {"a": {"mean": 0.5, "ci95": [0.4, 0.6]}, "b": {"mean": 0.2, "ci95": [0.1, 0.5]}}
    )
    assert o["spread"] == pytest.approx(0.3)
    assert o["median_half_width"] == pytest.approx((0.1 + 0.2) / 2)


def test_h3_holds_only_when_ci_entirely_negative():
    m_r = {(f"p{i}", s): 0.5 for i in range(6) for s in range(2)}
    lower = {k: 0.2 for k in m_r}
    assert r.h3_read(lower, m_r, n_resamples=100)["label"] == "HOLDS"
    same = dict(m_r)
    assert r.h3_read(same, m_r, n_resamples=100)["label"] == "FAILS"


def test_restriction_uses_reference_length_and_reports_attrition():
    keys = [(f"p{i}", s) for i in range(4) for s in range(2)]
    fc_len = {k: (5000 if k[0] in ("p0", "p1") else 1000) for k in keys}
    fc_correct = {k: 1.0 for k in keys}
    comp, exposed = {}, {}
    for t0 in ("0", "512", "2048"):
        comp[("random", "256", t0)] = {
            # short traces: damaged early, spared late; long traces: always wrong
            k: (1.0 if (t0 == "2048" and fc_len[k] < 2304) else 0.0)
            for k in keys
        }
        exposed[("random", "256", t0)] = {
            k: not (t0 == "2048" and fc_len[k] < 2304) for k in keys
        }
    o = r.restricted_onset_contrasts(comp, exposed, fc_correct, fc_len, n_resamples=100)
    assert o["retained_fraction"] == 0.5
    c = o["cells"]["random-256"]
    assert c["t0_0_vs_2048"]["unrestricted"]["mean"] == pytest.approx(0.5)
    assert c["t0_0_vs_2048"]["restricted"]["mean"] == pytest.approx(0.0)
    assert c["unexposed_at_t0_2048_within_retained"] == 0
    assert c["registered_reading"].startswith("onset effect NOT ESTABLISHED")


def test_per_layer_omega_excludes_layers_without_finite_values():
    rows = {
        ("p0", 1): {"sum_omega": [4.0, 0.0], "n_finite": [2, 0]},
        ("p1", 1): {"sum_omega": [6.0, 3.0], "n_finite": [2, 1]},
    }
    out = r.per_layer_omega(rows, n_resamples=50)
    assert out[0]["mean"] == pytest.approx(2.5) and out[0]["n_problems"] == 2
    assert out[1]["mean"] == pytest.approx(3.0) and out[1]["n_problems"] == 1


def test_eligibility_profile_splits_and_requires_reference_rows():
    elig = {("a", 1): True, ("a", 2): True, ("b", 1): False}
    fc = {
        ("a", 1): {"correct": True, "truncated": False},
        ("a", 2): {"correct": False, "truncated": True},
        ("b", 1): {"correct": True, "truncated": False},
    }
    toks = {("a", 1): 100, ("a", 2): 300, ("b", 1): 50}
    o = r.eligibility_profile(elig, fc, toks, n_resamples=50)
    assert o["eligible"]["fc_accuracy"]["mean"] == pytest.approx(0.5)
    assert o["eligible"]["median_trace_tokens"] == 200
    assert o["ineligible"]["n_problems"] == 1
    assert "realized_ratio" not in o["eligible"]
    pt = {("a", 1): 20, ("a", 2): 40, ("b", 1): 10}
    o2 = r.eligibility_profile(elig, fc, toks, prompt_tokens=pt, n_resamples=50)
    # 256 / (mean prompt 30 + mean trace 200)
    assert o2["eligible"]["realized_ratio"] == pytest.approx(256 / 230)
    with pytest.raises(ValueError):
        r.eligibility_profile({("c", 1): True}, fc, toks)


def test_summary_window_matches_full_window_and_refuses_partials():
    est = [0.1] * 256 + [0.3] * 256 + [0.5] * 100
    summary = {
        "window": 256,
        "n_steps": len(est),
        "window_mean_tv": [0.1, 0.3, 0.5],
    }
    assert r.summary_window_tv(summary, 256) == pytest.approx(r.window_tv(est, 256))
    assert r.summary_window_tv(summary, 512) is None  # partial last window
    with pytest.raises(ValueError):
        r.summary_window_tv(summary, 100)
    with pytest.raises(ValueError):
        r.window_tv(est, 512)


def test_collapse_labels():
    inc = {"mean": 0.001, "ci95": [-0.01, 0.01], "mismatch_bound": 0.0}
    assert r.collapse_label([inc] * 3, tv_range=0.1) == "COLLAPSE"
    pos = {"mean": 0.02, "ci95": [0.01, 0.03], "mismatch_bound": 0.001}
    assert r.collapse_label([pos] * 3, 0.1) == "NO COLLAPSE: absolute budget matters"
    neg = {"mean": -0.02, "ci95": [-0.03, -0.01], "mismatch_bound": 0.001}
    assert r.collapse_label([neg] * 3, 0.1) == "NO COLLAPSE: horizon matters"
    assert r.collapse_label([pos, neg, inc], 0.1) == "MIXED"
    # excludes zero but inside the residual r-mismatch: counts as including zero
    within = {"mean": 0.02, "ci95": [0.01, 0.03], "mismatch_bound": 0.05}
    assert r.collapse_label([within] * 3, 0.1) == "COLLAPSE"
    # includes zero but large relative to the between-r range: not a collapse
    wide = {"mean": 0.05, "ci95": [-0.01, 0.11], "mismatch_bound": 0.0}
    assert r.collapse_label([wide] * 3, 0.1) == "MIXED"

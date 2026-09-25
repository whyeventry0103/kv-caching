"""Amendment R.2 direct functional control, on a tiny random-init Qwen2.

Skipped without torch/transformers/kvpress (runs on the remote CPU with the
pinned stack). Pins: forks happen before every physical removal, the
deterministic B-vs-B control is exactly reproducible, masking reproduces
removal to floating-point precision with identical next selections, and the
check is sensitive -- with the mask disabled, arm B differs grossly.
"""

from __future__ import annotations

import pytest


def _tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("kvpress")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = transformers.Qwen2ForCausalLM._from_config(
        cfg, attn_implementation="eager"
    ).eval()
    prompt = torch.randint(3, 97, (12,)).tolist()
    trace = torch.randint(3, 97, (70,)).tolist()
    return model, prompt, trace


def _press(policy, interval, delay):
    from kvcache.fork_control import DeferredRemovalPress
    from kvcache.presses import build_press

    base = build_press(policy, budget=16, compression_interval=interval, seed=5)
    return DeferredRemovalPress(
        base.base_press, budget=16, interval=interval, delay=delay
    )


@pytest.mark.parametrize("policy", ["random", "streamingllm"])
@pytest.mark.parametrize("interval,delay", [(8, 6), (4, 6)])  # (4, 6) overlaps
def test_masking_reproduces_removal(policy, interval, delay):
    model, prompt, trace = _tiny()
    from kvcache.fork_control import run_fork_control

    out = run_fork_control(model, _press(policy, interval, delay), prompt, trace)
    assert out["n_forks"] > 3
    assert out["deterministic_control_max_abs_logit"] == 0.0
    assert out["max_abs_logit"] < 1e-4
    assert out["selection_agreement"] == 1.0
    assert all(s - delay >= 0 for s in out["removal_steps"])
    assert len(out["tv_profile"]) == len(trace)


def test_the_control_is_sensitive_when_masking_is_off(monkeypatch):
    model, prompt, trace = _tiny()
    from kvcache.fork_control import DeferredRemovalPress, run_fork_control

    real = DeferredRemovalPress._pre

    def unmasked(self, module, args, kwargs):
        res = real(self, module, args, kwargs)
        if res is not None and not self.compact:
            a, kw = res
            kw["attention_mask"] = kw["attention_mask"].clamp(min=0)
            return a, kw
        return res

    monkeypatch.setattr(DeferredRemovalPress, "_pre", unmasked)
    out = run_fork_control(model, _press("random", 8, 6), prompt, trace)
    assert out["max_abs_logit"] > 1e-3


def _reader():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import report_fork_control

    return report_fork_control


def _rows(agree, det=1e-6, diff=2e-6, n=3):
    prof = [0.0] * 8 + [0.1, 0.0, 0.0, 0.0] * 10
    return [
        {
            "pid": f"p{k}",
            "seed": 11,
            "press": "random",
            "interval": 4,
            "deterministic_control_max_abs_logit": det,
            "first_removal_step": 8,
            "tv_profile": prof,
            "forks": [
                {"max_abs_logit": diff, "max_tv": 0.0, "selection_agrees": agree}
            ],
        }
        for k in range(n)
    ]


def test_reader_tolerance_rules_and_the_legacy_label():
    rf = _reader()
    assert rf.classify(1e-6, 1e-5) == "PASS"
    assert rf.classify(1e-4, 1e-5) == "FAIL"
    assert rf.classify(0.0, 0.0) == "EXACT"
    assert rf.classify(1e-7, 0.0) == "TOLERANCE-DEGENERATE"
    exp = {(f"p{k}", 11, "random", 4) for k in range(3)}
    out = rf.read(
        _rows(True), n_resamples=50, seed=0, expected=exp, check_identity=False
    )
    c = out["cells"]["random-i4"]
    assert out["tolerance"] == pytest.approx(1e-5)
    assert c["logit_gate"] == "PASS" and c["overall"] == "PASS"
    assert out["overall"] == "PASS"
    assert c["legacy_removal_anchored_gate"]["label"].startswith("NOT VALID")


def test_selection_disagreement_fails_even_within_logit_tolerance():
    """Review of 4e38224: tolerance 1e-5, A-B 1e-7, selection disagrees."""
    rf = _reader()
    exp = {(f"p{k}", 11, "random", 4) for k in range(3)}
    out = rf.read(
        _rows(False, det=1e-6, diff=1e-7),
        n_resamples=50,
        seed=0,
        expected=exp,
        check_identity=False,
    )
    c = out["cells"]["random-i4"]
    assert c["logit_gate"] == "PASS"  # R.11's numerical rule, unchanged
    assert c["selection_gate"]["status"] == "FAIL"
    assert c["overall"] == "FAIL" and out["overall"] == "FAIL"


def test_no_comparisons_and_incomplete_coverage_are_never_pass():
    rf = _reader()
    exp = {(f"p{k}", 11, "random", 4) for k in range(3)}
    out = rf.read(
        _rows(None), n_resamples=50, seed=0, expected=exp, check_identity=False
    )
    c = out["cells"]["random-i4"]
    assert c["selection_gate"]["status"] == "NO-COMPARISONS"
    assert c["selection_gate"]["n_ineligible_trace_ended"] == 3
    assert c["overall"] == "INCONCLUSIVE"
    out = rf.read(
        _rows(True, n=2), n_resamples=50, seed=0, expected=exp, check_identity=False
    )
    assert out["coverage"]["status"] == "PARTIAL"
    assert out["overall"] == "INCONCLUSIVE"


def test_malformed_fork_evidence_is_invalid_before_classification():
    """Review of cfa30cd: an infinite deterministic control made tolerance inf
    and turned A-B = 1.0 into PASS."""
    rf = _reader()
    rows = _rows(True, det=float("inf"), diff=1.0)
    exp = {(f"p{k}", 11, "random", 4) for k in range(3)}
    out = rf.read(rows, n_resamples=50, seed=0, expected=exp, check_identity=False)
    assert out["status"] == "INVALID" and out["overall"] == "INVALID-EVIDENCE"
    assert out["cells"] == {}
    neg = _rows(True, det=1e-6, diff=-1.0)
    assert (
        rf.read(neg, n_resamples=50, seed=0, expected=exp, check_identity=False)[
            "status"
        ]
        == "INVALID"
    )
    noprov = _rows(True)
    out = rf.read(noprov, n_resamples=50, seed=0, expected=exp)  # identity checked
    assert out["status"] == "INVALID" and out["provenance_failures"]

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


def test_reader_tolerance_rules_and_the_legacy_label():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from report_fork_control import classify, read

    assert classify(1e-6, 1e-5) == "PASS"
    assert classify(1e-4, 1e-5) == "FAIL"
    assert classify(0.0, 0.0) == "EXACT"
    assert classify(1e-7, 0.0) == "TOLERANCE-DEGENERATE"
    prof = [0.0] * 8 + [0.1, 0.0, 0.0, 0.0] * 10
    rows = [
        {
            "press": "random",
            "interval": 4,
            "deterministic_control_max_abs_logit": 1e-6,
            "first_removal_step": 8,
            "tv_profile": prof,
            "forks": [{"max_abs_logit": 2e-6, "max_tv": 0.0, "selection_agrees": True}],
        }
        for _ in range(3)
    ]
    out = read(rows, n_resamples=50, seed=0)
    c = out["cells"]["random-i4"]
    assert out["tolerance"] == pytest.approx(1e-5) and c["verdict"] == "PASS"
    assert c["selection_agreement"] == 1.0
    assert c["legacy_removal_anchored_gate"]["label"].startswith("NOT VALID")

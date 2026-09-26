"""R.17.0.5 fork-precision instruments.

The pure statistics run everywhere. The model paths need torch/transformers/
kvpress and run on the GPU host's CPU with the pinned stack, on a tiny
random-init Qwen2: in fp32 the two fork arms agree to rounding, the diagnosis
reports where the gap lives, and a model compared with itself has a floor of
exactly zero.
"""

from __future__ import annotations

import pytest

from kvcache import fork_precision as fp


def test_window_max_p95_hand_case():
    blocks = [[0.0, 1.0, 0.0, 2.0, 5.0], [3.0, 0.0]]
    # windows of 2, no crossing, trailing partial dropped: maxima 1, 2, 3
    o = fp.window_max_p95(blocks, 2)
    assert o["n_windows"] == 3 and o["max"] == 3.0
    assert o["p95"] == sorted([1.0, 2.0, 3.0])[int(0.95 * 2)]
    assert fp.window_max_p95([[1.0]], 4) == {"p95": None, "n_windows": 0}


def test_block_steps_keep_only_blocks_that_fit():
    assert fp.block_steps(10, [0, 2, 8], 3) == [2, 3, 4]  # 0 is prefill, 8+3>10
    assert fp.block_steps(10, [1, 7], 3) == [1, 2, 3, 7, 8, 9]


def test_gate_read_needs_every_arm():
    floor = {16: 0.01, 64: 0.02}
    ok = fp.gate_read({("random", 16): 0.015, ("random", 64): 0.04}, floor)
    assert ok["verdict"] == "PASS"
    bad = fp.gate_read({("random", 16): 0.015, ("random", 64): 0.05}, floor)
    assert bad["verdict"] == "FAIL" and not bad["arms"]["random-i64"]["passed"]
    missing = fp.gate_read({("random", 256): 0.01}, floor)
    assert missing["verdict"] == "FAIL"


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


def _press(interval=8, delay=6):
    from kvcache.fork_control import DeferredRemovalPress
    from kvcache.presses import build_press

    base = build_press("random", budget=16, compression_interval=interval, seed=5)
    return DeferredRemovalPress(
        base.base_press, budget=16, interval=interval, delay=delay
    )


def test_first_fork_diagnosis_fp32_arms_agree_and_are_located():
    model, prompt, trace = _tiny()
    out = fp.first_fork_diagnosis(model, _press(), prompt, trace)
    assert out["fork_step"] is not None and out["n_compared"] >= 1
    s = out["steps"][0]
    assert {"tv", "max_abs_logit", "argmax_token_rank_in_b", "top10_overlap"} <= set(s)
    assert max(x["max_abs_logit"] for x in out["steps"]) < 1e-4
    assert all(x["top1_agree"] for x in out["steps"])
    # the fork step equals run_fork_control's first removal on the same inputs
    from kvcache.fork_control import run_fork_control

    ref = run_fork_control(model, _press(), prompt, trace)
    assert out["fork_step"] == ref["first_removal_step"]


def test_the_diagnosis_is_sensitive_when_masking_is_off(monkeypatch):
    model, prompt, trace = _tiny()
    from kvcache.fork_control import DeferredRemovalPress

    real = DeferredRemovalPress._pre

    def unmasked(self, module, args, kwargs):
        res = real(self, module, args, kwargs)
        if res is not None and not self.compact:
            a, kw = res
            kw["attention_mask"] = kw["attention_mask"].clamp(min=0)
            return a, kw
        return res

    monkeypatch.setattr(DeferredRemovalPress, "_pre", unmasked)
    out = fp.first_fork_diagnosis(model, _press(), prompt, trace)
    assert max(x["max_abs_logit"] for x in out["steps"]) > 1e-3


def test_floor_of_a_model_against_itself_is_zero_and_detects_a_perturbation():
    torch = pytest.importorskip("torch")
    model, prompt, trace = _tiny()
    ref = fp.collect_block_logits(model, prompt, trace, starts=[5, 40], length=10)
    assert len(ref) == 20
    same = fp.compare_block_logits(model, prompt, trace, ref)
    assert max(v["max_abs_logit"] for v in same.values()) == 0.0
    with torch.no_grad():
        model.lm_head.weight.add_(1e-2)
    moved = fp.compare_block_logits(model, prompt, trace, ref)
    assert max(v["max_abs_logit"] for v in moved.values()) > 0.0

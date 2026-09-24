"""Amendment R.3: the decode-step gate for free-running onset-staggered generation.

The CPU-only checks pin the counter convention and the exposure arithmetic.
The tiny-model checks (skipped without torch/transformers/kvpress) are the
"verified by test before science" gate R.3 registers: a gate of None must
reproduce legacy generation token for token, and a gated arm must be
identical to full-cache generation until its first effective prune.
"""

from __future__ import annotations

import pytest

from kvcache.generation import decode_step_gate, exposure_from_events


class _FakeHandle:
    def __init__(self, hooks, fn):
        self._hooks, self._fn = hooks, fn

    def remove(self):
        self._hooks.remove(self._fn)


class _FakeModel:
    """Calls its forward pre-hooks the way nn.Module.__call__ does."""

    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, fn):
        self.hooks.append(fn)
        return _FakeHandle(self.hooks, fn)

    def __call__(self):
        for fn in list(self.hooks):
            fn(self, ())


class _FakePress:
    enabled = False
    profile_step = None
    compression_start_step = None


def test_gate_numbers_forwards_like_the_replay_loop():
    """Prefill is step 0; the forward producing token j's logits is step j."""
    model, press, seen = _FakeModel(), _FakePress(), []
    with decode_step_gate(model, press):
        assert press.enabled is True
        for _ in range(4):
            model()
            seen.append(press.profile_step)
    assert seen == [0, 1, 2, 3]
    assert press.enabled is False
    assert model.hooks == []  # the hook never outlives the generation


def test_gate_restores_state_and_removes_the_hook_on_error():
    model, press = _FakeModel(), _FakePress()
    with pytest.raises(RuntimeError), decode_step_gate(model, press):
        model()
        raise RuntimeError("generation failed")
    assert press.enabled is False
    assert model.hooks == []


def test_gate_refuses_an_uninstrumented_press():
    """A plain DecodingPress has no counter; gating it would silently not gate."""

    class Plain:
        pass

    with (
        pytest.raises(TypeError, match="instrumented"),
        decode_step_gate(_FakeModel(), Plain()),
    ):
        pass


def test_exposure_counts_tokens_sampled_after_the_first_effective_prune():
    events = [
        {"layer": 0, "before": 100, "after": 100, "step": 64},  # under budget
        {"layer": 0, "before": 300, "after": 256, "step": 128},
        {"layer": 1, "before": 300, "after": 256, "step": 128},
    ]
    out = exposure_from_events(events, n_tokens=1000)
    assert out == {
        "eviction_onset_step": 129,
        "achieved_exposure": 1000 - 129,
        "exposed": True,
    }


def test_a_generation_that_ends_before_eviction_is_unexposed_not_dropped():
    events = [{"layer": 0, "before": 100, "after": 100, "step": 64}]
    assert exposure_from_events(events, n_tokens=90) == {
        "eviction_onset_step": None,
        "achieved_exposure": 0,
        "exposed": False,
    }
    # The last prune affected nothing that was sampled afterwards.
    last = [{"layer": 0, "before": 300, "after": 256, "step": 99}]
    assert exposure_from_events(last, n_tokens=100)["exposed"] is False


def test_exposure_refuses_events_from_an_ungated_press():
    with pytest.raises(ValueError, match="not gated"):
        exposure_from_events([{"before": 3, "after": 2, "step": None}], 10)


# ------------------------------------------------------------- tiny model
PROMPT_LEN, NEW, BUDGET, INTERVAL = 12, 48, 16, 4


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
        max_position_embeddings=256,
    )
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    ids = torch.randint(3, 97, (1, PROMPT_LEN))
    return torch, model, ids


def _generate(torch, model, ids, press, *, gate: bool, start=None):
    from contextlib import nullcontext

    if press is not None and start is not None:
        press.compression_start_step = start
    torch.manual_seed(1234)
    ctx = press(model) if press is not None else nullcontext()
    step = decode_step_gate(model, press) if gate else nullcontext()
    with torch.inference_mode(), ctx, step:
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            max_new_tokens=NEW,
            min_new_tokens=NEW,
            pad_token_id=0,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    return out.sequences[0, ids.shape[-1] :].tolist(), [s[0] for s in out.scores]


def _press(instrumented: bool):
    from kvcache.presses import build_press

    return build_press(
        "random",
        budget=BUDGET,
        compression_interval=INTERVAL,
        seed=7,
        instrumented=instrumented,
    )


def test_gate_none_reproduces_legacy_generation_token_for_token():
    torch, model, ids = _tiny()
    legacy, legacy_scores = _generate(torch, model, ids, _press(False), gate=False)
    gated = _press(True)
    via_gate, gate_scores = _generate(torch, model, ids, gated, gate=True, start=None)
    assert via_gate == legacy
    assert all(
        torch.equal(a, b) for a, b in zip(gate_scores, legacy_scores, strict=True)
    )
    # Non-vacuous: compression really ran and changed the trajectory.
    assert any(e["after"] < e["before"] for e in gated.compressions)
    full, _ = _generate(torch, model, ids, None, gate=False)
    assert full != legacy


def test_gated_arm_equals_full_cache_until_its_first_effective_prune():
    torch, model, ids = _tiny()
    start = 20
    press = _press(True)
    gated, g_scores = _generate(torch, model, ids, press, gate=True, start=start)
    full, f_scores = _generate(torch, model, ids, None, gate=False)
    effective = [e["step"] for e in press.compressions if e["after"] < e["before"]]
    # Counted forwards start at `start`; the interval fills INTERVAL forwards
    # later, as in the replay (t0=512 -> first prune at 575, onset 576).
    assert min(effective) == start + INTERVAL - 1
    info = exposure_from_events(press.compressions, len(gated))
    onset = info["eviction_onset_step"]
    assert onset == start + INTERVAL
    assert gated[:onset] == full[:onset]
    # Single-step resolution: sampled tokens can coincide under a slightly
    # moved distribution, the logits cannot. Every step before the onset is
    # bit-identical to the full cache and the onset step itself is not (an
    # off-by-one in the counter passed a token-level check; mutation-tested).
    assert all(torch.equal(g_scores[j], f_scores[j]) for j in range(onset))
    assert not torch.equal(g_scores[onset], f_scores[onset])
    assert gated != full  # non-vacuous: eviction did change the trajectory
    assert info["achieved_exposure"] == NEW - onset
    assert all(e["step"] >= start for e in press.compressions)

import pytest

from kvcache.presses import (
    build_press,
    build_replay_press,
    get_press,
    press_needs_eager_attention,
    press_seed,
)


def test_press_seed_is_deterministic_and_decorrelated():
    # Stable across calls (reproducible).
    assert press_seed("p1", 11) == press_seed("p1", 11)
    # Varies by problem and by seed (decorrelates random eviction).
    assert press_seed("p1", 11) != press_seed("p2", 11)
    assert press_seed("p1", 11) != press_seed("p1", 23)
    assert isinstance(press_seed("p1", 11), int)


def test_h2o_needs_eager_attention_others_do_not():
    assert press_needs_eager_attention("h2o") is True
    assert press_needs_eager_attention("streamingllm") is False
    assert press_needs_eager_attention("random") is False


def test_rkv_refuses_without_smoke_marker(monkeypatch, tmp_path):
    # R-KV is parity-verified but must not enter analysis until the real-generation
    # smoke marker exists (Amendment E). No marker -> refuse.
    monkeypatch.delenv("KVCACHE_RKV_SMOKE_MARKER", raising=False)
    monkeypatch.delenv("KVCACHE_BASE_RUNS_ROOT", raising=False)
    with pytest.raises(NotImplementedError, match="smoke"):
        get_press("rkv")


def test_rkv_promoted_by_smoke_marker(monkeypatch, tmp_path):
    marker = tmp_path / "rkv-gen-smoke-PASS.json"
    marker.write_text('{"passed": true}')
    monkeypatch.setenv("KVCACHE_RKV_SMOKE_MARKER", str(marker))
    assert get_press("rkv") == "rkv"


def test_rkv_promoted_by_base_runs_root_marker(monkeypatch, tmp_path):
    # launch.sh exports KVCACHE_BASE_RUNS_ROOT; the smoke writes the marker there.
    monkeypatch.delenv("KVCACHE_RKV_SMOKE_MARKER", raising=False)
    monkeypatch.setenv("KVCACHE_BASE_RUNS_ROOT", str(tmp_path))
    with pytest.raises(NotImplementedError):
        get_press("rkv")
    (tmp_path / "rkv-gen-smoke-PASS.json").write_text('{"passed": true}')
    assert get_press("rkv") == "rkv"


@pytest.mark.parametrize(
    "name",
    [
        "snapkv",
        "expected_attention",
        "rkv",
        "kivi2",
        "triattention",
    ],
)
def test_unimplemented_or_deferred_presses_refuse(name, monkeypatch):
    monkeypatch.delenv("KVCACHE_RKV_SMOKE_MARKER", raising=False)
    monkeypatch.delenv("KVCACHE_BASE_RUNS_ROOT", raising=False)
    with pytest.raises(NotImplementedError):
        get_press(name)


def test_unknown_press_is_rejected():
    with pytest.raises(KeyError):
        get_press("not-a-press")


class _FakeStreaming:
    pass


class _FakeRandom:
    def __init__(self, seed=None):
        self.seed = seed


class _FakeDecoding:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeObservedAttention:
    pass


class _FakeKVPress:
    StreamingLLMPress = _FakeStreaming
    RandomPress = _FakeRandom
    ObservedAttentionPress = _FakeObservedAttention
    DecodingPress = _FakeDecoding


@pytest.mark.parametrize("name", ["streamingllm", "random", "h2o"])
def test_build_press_constructs_decoding_press(name):
    press = build_press(
        name,
        budget=256,
        compression_interval=64,
        seed=11,
        kvpress_module=_FakeKVPress,
    )
    assert isinstance(press, _FakeDecoding)
    assert press.kwargs["target_size"] == 256
    assert press.kwargs["compression_interval"] == 64
    assert press.kwargs["hidden_states_buffer_size"] == 0
    if name == "random":
        assert press.kwargs["base_press"].seed == 11


def test_build_press_accepts_instrumented_decoding_class():
    class _InstrumentedDecoding(_FakeDecoding):
        pass

    press = build_press(
        "random",
        budget=256,
        seed=11,
        kvpress_module=_FakeKVPress,
        decoding_press_cls=_InstrumentedDecoding,
    )
    assert isinstance(press, _InstrumentedDecoding)


def test_fullcache_press_is_noop():
    assert build_press("fullcache", budget=256) is None


def test_replay_press_enabled_flag_scopes_forward_hook():
    class _ReplayDecoding:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.__post_init__()

        def __post_init__(self):
            self.parent_calls = 0

        def forward_hook(self, module, input, kwargs, output):
            self.parent_calls += 1
            return "compressed"

    class _ReplayKVPress(_FakeKVPress):
        DecodingPress = _ReplayDecoding

    press = build_replay_press(
        "streamingllm",
        budget=256,
        kvpress_module=_ReplayKVPress,
    )
    assert press.forward_hook(None, None, None, "full") == "full"
    press.enabled = True
    assert press.forward_hook(None, None, None, "full") == "compressed"
    assert press.parent_calls == 1


def test_replay_press_onset_is_step_after_prune():
    class _ReplayDecoding:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.__post_init__()

        def __post_init__(self):
            pass

        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            return keys, values

    class _ReplayKVPress(_FakeKVPress):
        DecodingPress = _ReplayDecoding

    class _Keys:
        shape = (1, 1, 320, 1)

    class _Module:
        layer_idx = 0

    press = build_replay_press(
        "streamingllm",
        budget=256,
        kvpress_module=_ReplayKVPress,
    )
    press.profile_step = 64
    press.compress(_Module(), None, _Keys(), _Keys(), None, {})
    assert press.onset_step == 65


def test_rkv_scorer_overrides_snapkv_dataclass_defaults():
    # SnapKVPress is a dataclass; a plain subclass's annotated class attributes are
    # shadowed by the generated __init__ (instance gets window_size=64/kernel_size=5).
    # The live smoke caught this (Jaccard 0.805). Lock the override behavior.
    from dataclasses import dataclass, field
    from types import SimpleNamespace

    from kvcache.presses.rkv import (
        RKV_KERNEL_SIZE,
        RKV_MIX_LAMBDA,
        RKV_WINDOW_SIZE,
        build_rkv_scorer,
    )

    @dataclass
    class FakeSnapKVPress:  # mirrors kvpress's dataclass defaults
        compression_ratio: float = 0.0
        window_size: int = 64
        kernel_size: int = 5
        extra: dict = field(default_factory=dict)

    fake_kvpress = SimpleNamespace(SnapKVPress=FakeSnapKVPress)
    scorer = build_rkv_scorer(fake_kvpress)
    assert scorer.window_size == RKV_WINDOW_SIZE == 8
    assert scorer.kernel_size == RKV_KERNEL_SIZE == 7
    assert scorer.mix_lambda == RKV_MIX_LAMBDA

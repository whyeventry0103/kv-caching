import pytest

from kvcache.presses import build_press, build_replay_press, get_press


@pytest.mark.parametrize(
    "name",
    [
        "snapkv",
        "expected_attention",
        "h2o",
        "rkv",
        "kivi2",
        "triattention",
    ],
)
def test_unimplemented_or_deferred_presses_refuse(name):
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


class _FakeKVPress:
    StreamingLLMPress = _FakeStreaming
    RandomPress = _FakeRandom
    DecodingPress = _FakeDecoding


@pytest.mark.parametrize("name", ["streamingllm", "random"])
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

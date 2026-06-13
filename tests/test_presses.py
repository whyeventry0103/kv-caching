import pytest

from kvcache.presses import build_press, get_press


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


def test_fullcache_press_is_noop():
    assert build_press("fullcache", budget=256) is None

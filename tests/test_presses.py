import pytest

from kvcache.presses import get_press


@pytest.mark.parametrize(
    "name",
    [
        "streamingllm",
        "snapkv",
        "random",
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

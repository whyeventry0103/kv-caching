import pytest

from kvcache.presses import get_press


def test_available_presses_are_explicit():
    assert get_press("streamingllm") == "streamingllm"
    assert get_press("expected_attention") == "expected_attention"


@pytest.mark.parametrize("name", ["h2o", "rkv", "kivi2", "triattention"])
def test_unimplemented_or_deferred_presses_refuse(name):
    with pytest.raises(NotImplementedError):
        get_press(name)

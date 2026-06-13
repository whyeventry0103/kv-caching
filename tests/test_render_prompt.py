import pytest

from kvcache.models import _revision, render_prompt


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"messages": messages, "kwargs": kwargs}


def test_render_prompt_has_only_one_user_turn():
    rendered = render_prompt(
        FakeTokenizer(),
        "Question?",
        {"system_prompt": None, "chat_template_kwargs": {"enable_thinking": True}},
    )
    assert rendered["messages"] == [{"role": "user", "content": "Question?"}]
    assert rendered["kwargs"]["add_generation_prompt"] is True
    assert rendered["kwargs"]["enable_thinking"] is True


def test_render_prompt_refuses_system_message():
    with pytest.raises(ValueError, match="system_prompt"):
        render_prompt(FakeTokenizer(), "Question?", {"system_prompt": "system"})


def test_model_revision_must_be_pinned():
    with pytest.raises(RuntimeError, match="revision"):
        _revision({"revision": "TODO(human)"})
    assert _revision({"revision": "abc123"}) == "abc123"

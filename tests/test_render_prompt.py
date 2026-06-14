import pytest

from kvcache.models import _revision, render_prompt, tokenize_rendered_prompt


class FakeTokenizer:
    def __init__(self):
        self.tokenize_calls = []

    def apply_chat_template(self, messages, **kwargs):
        return {"messages": messages, "kwargs": kwargs}

    def __call__(self, prompt, **kwargs):
        self.tokenize_calls.append((prompt, kwargs))
        return {"input_ids": [1, 2, 3]}


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


def test_rendered_prompt_tokenization_does_not_duplicate_bos():
    tokenizer = FakeTokenizer()
    encoded = tokenize_rendered_prompt(
        tokenizer,
        "<bos><user>Question?",
        return_tensors="pt",
    )
    assert encoded == {"input_ids": [1, 2, 3]}
    assert tokenizer.tokenize_calls == [
        (
            "<bos><user>Question?",
            {"add_special_tokens": False, "return_tensors": "pt"},
        )
    ]


def test_model_revision_must_be_pinned():
    with pytest.raises(RuntimeError, match="revision"):
        _revision({"revision": "TODO(human)"})
    assert _revision({"revision": "abc123"}) == "abc123"

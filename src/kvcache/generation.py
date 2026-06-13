"""Generation entry points; concrete heavy backends are remote-only."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def generate_full_cache(
    prompt: str, generator: Callable[..., Any], **kwargs: Any
) -> Any:
    return generator(prompt, press=None, **kwargs)


def generate_compressed(
    prompt: str, generator: Callable[..., Any], press: Any, **kwargs: Any
) -> Any:
    return generator(prompt, press=press, **kwargs)


def bank_backend(name: str) -> str:
    if name not in {"transformers", "vllm"}:
        raise ValueError(f"unsupported trace-bank backend: {name}")
    return name

"""Minimal press protocol used by generation and remote adapters."""

from __future__ import annotations

from typing import Protocol


class Press(Protocol):
    name: str

    def compression_ratio(self, original_tokens: int, budget: int) -> float:
        """Return retained/original KV ratio."""
        ...


def retained_ratio(original_tokens: int, budget: int) -> float:
    if original_tokens <= 0 or budget <= 0:
        raise ValueError("token counts and budgets must be positive")
    return min(1.0, budget / original_tokens)

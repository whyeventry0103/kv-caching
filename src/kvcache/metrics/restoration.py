"""Restoration intervention summaries."""

from __future__ import annotations


def restoration_effect(tv_before: float, tv_after_reinsertion: float) -> float:
    """Positive values mean reinserting evicted KV reduced spike TV."""
    return tv_before - tv_after_reinsertion

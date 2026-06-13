"""Press registry.

Expected-attention and TriAttention must be verified in the pinned KVPress
release or ported explicitly. They intentionally fail until that remote audit.
"""

from __future__ import annotations

SUPPORTED = {"streamingllm", "h2o", "snapkv", "rkv", "kivi2", "random"}
VERIFY_OR_PORT = {"expected_attention", "triattention"}


def get_press(name: str) -> str:
    normalized = name.lower()
    if normalized in VERIFY_OR_PORT:
        raise NotImplementedError(
            f"{normalized}: verify in pinned kvpress else port - TODO(human)"
        )
    if normalized not in SUPPORTED:
        raise KeyError(f"unknown press: {name}")
    return normalized

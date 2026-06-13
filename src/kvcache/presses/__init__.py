"""Press registry with explicit implementation and deferral states."""

from __future__ import annotations

SUPPORTED = {"streamingllm", "snapkv", "random", "expected_attention"}
PORT_REQUIRED = {"h2o", "rkv"}
DEFERRED = {"kivi2", "triattention"}


def get_press(name: str) -> str:
    normalized = name.lower()
    if normalized in PORT_REQUIRED:
        raise NotImplementedError(
            f"{normalized}: paper-faithful adapter/port required - TODO(human)"
        )
    if normalized in DEFERRED:
        raise NotImplementedError(
            f"{normalized}: deferred and not eligible for execution - TODO(human)"
        )
    if normalized not in SUPPORTED:
        raise KeyError(f"unknown press: {name}")
    return normalized

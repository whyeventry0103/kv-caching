"""Lazy press registry for verified decoding-time baselines."""

from __future__ import annotations

from typing import Any

EXECUTABLE = {"streamingllm", "random"}
PREFILL_ONLY = {"snapkv", "expected_attention"}
PORT_REQUIRED = {"h2o", "rkv"}
DEFERRED = {"kivi2", "triattention"}


def get_press(name: str) -> str:
    normalized = name.lower()
    if normalized in EXECUTABLE:
        return normalized
    if normalized in PREFILL_ONLY:
        raise NotImplementedError(
            f"{normalized}: decoding-time adapter required before execution"
        )
    if normalized in PORT_REQUIRED:
        raise NotImplementedError(
            f"{normalized}: paper-faithful adapter/port required - TODO(human)"
        )
    if normalized in DEFERRED:
        raise NotImplementedError(
            f"{normalized}: deferred and not eligible for execution - TODO(human)"
        )
    if normalized not in EXECUTABLE | PREFILL_ONLY | PORT_REQUIRED | DEFERRED:
        raise KeyError(f"unknown press: {name}")
    raise AssertionError("unreachable")


def build_press(
    name: str,
    *,
    budget: int,
    compression_interval: int = 64,
    seed: int | None = None,
    kvpress_module: Any | None = None,
) -> Any | None:
    """Build a verified decoding press; imports kvpress only when executed."""
    normalized = name.lower()
    if normalized == "fullcache":
        return None
    get_press(normalized)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if compression_interval <= 0:
        raise ValueError("compression_interval must be positive")
    if kvpress_module is None:
        import kvpress as kvpress_module

    if normalized == "streamingllm":
        base_press = kvpress_module.StreamingLLMPress()
    elif normalized == "random":
        base_press = kvpress_module.RandomPress(seed=seed)
    else:
        raise AssertionError("unreachable")
    return kvpress_module.DecodingPress(
        base_press=base_press,
        target_size=budget,
        compression_interval=compression_interval,
        hidden_states_buffer_size=0,
    )

"""Press registry with explicit implementation and deferral states.

Pinned kvpress 0.2.10 applies its native presses during prefill only. This project
studies cache growth during long decoding, so those classes are not executable
baselines until a decoding-time absolute-budget adapter is implemented.
"""

from __future__ import annotations

PREFILL_ONLY = {"streamingllm", "snapkv", "random", "expected_attention"}
PORT_REQUIRED = {"h2o", "rkv"}
DEFERRED = {"kivi2", "triattention"}


def get_press(name: str) -> str:
    normalized = name.lower()
    if normalized in PREFILL_ONLY:
        raise NotImplementedError(
            f"{normalized}: pinned kvpress 0.2.10 is prefill-only; "
            "decoding-time absolute-budget adapter required"
        )
    if normalized in PORT_REQUIRED:
        raise NotImplementedError(
            f"{normalized}: paper-faithful adapter/port required - TODO(human)"
        )
    if normalized in DEFERRED:
        raise NotImplementedError(
            f"{normalized}: deferred and not eligible for execution - TODO(human)"
        )
    if normalized not in PREFILL_ONLY | PORT_REQUIRED | DEFERRED:
        raise KeyError(f"unknown press: {name}")
    raise AssertionError("unreachable")

"""Lazy press registry for verified decoding-time baselines."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

EXECUTABLE = {"streamingllm", "random", "h2o"}
PREFILL_ONLY = {"snapkv", "expected_attention"}
PORT_REQUIRED = {"rkv"}
DEFERRED = {"kivi2", "triattention"}
# Attention-score presses read attention weights, which require eager attention.
EAGER_ATTENTION = {"h2o"}

# R-KV promotion gate (PREREGISTRATION Amendment E): the port is parity-verified
# (retained-set Jaccard 1.0 vs the authors' code, docs/design/RKV_PORT.md);
# the remaining
# gate is the REAL-GENERATION smoke. scripts/rkv_smoke.py writes this marker ONLY
# when every smoke check passes, which mechanically promotes rkv to executable --
# so the nightly queue can chain smoke -> rkv cells without a human in the loop.
RKV_SMOKE_MARKER_ENV = "KVCACHE_RKV_SMOKE_MARKER"
RKV_SMOKE_MARKER_NAME = "rkv-gen-smoke-PASS.json"


def rkv_smoke_marker_path() -> Path | None:
    explicit = os.environ.get(RKV_SMOKE_MARKER_ENV)
    if explicit:
        return Path(explicit)
    base = os.environ.get("KVCACHE_BASE_RUNS_ROOT")
    return Path(base) / RKV_SMOKE_MARKER_NAME if base else None


def rkv_certified() -> bool:
    marker = rkv_smoke_marker_path()
    return bool(marker and marker.exists())


def press_needs_eager_attention(name: str) -> bool:
    return name.lower() in EAGER_ATTENTION


def press_seed(pid: str, seed: int) -> int:
    """Stable per-(problem, seed) RNG seed for stochastic presses (RandomPress).

    Using one cell seed across all problems would correlate random eviction and
    risks a "lucky seed" artifact (a reviewer concern). This decorrelates eviction
    per problem while staying fully deterministic/reproducible.
    """
    digest = hashlib.sha256(f"{pid}|{seed}".encode()).hexdigest()
    return int(digest[:8], 16)


class NoOpReplayPress:
    """Explicitly toggled no-op used by the full-cache self-replay control."""

    enabled = False
    profile_step = None
    onset_step = None
    compression_interval = None
    compressions: list[dict[str, int]] = []

    @contextmanager
    def __call__(self, model):
        yield


def get_press(name: str) -> str:
    normalized = name.lower()
    if normalized in EXECUTABLE:
        return normalized
    if normalized in PREFILL_ONLY:
        raise NotImplementedError(
            f"{normalized}: decoding-time adapter required before execution"
        )
    if normalized in PORT_REQUIRED:
        if normalized == "rkv" and rkv_certified():
            return normalized  # parity Jaccard 1.0 + generation smoke marker present
        raise NotImplementedError(
            f"{normalized}: parity-verified (Jaccard 1.0) but the real-generation "
            "smoke has not passed. Run the queue job rkv-smoke (or scripts/"
            "rkv_smoke.py --write-pass-marker) so the PASS marker exists before any "
            "analysis row (PREREGISTRATION Amendment E)."
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
    decoding_press_cls: type | None = None,
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
    elif normalized == "h2o":
        # kvpress's ObservedAttentionPress is the in-library H2O (its own docstring
        # cites arXiv:2306.14048); requires the model loaded with eager attention.
        base_press = kvpress_module.ObservedAttentionPress()
    elif normalized == "random":

        class DeviceSafeRandomPress(kvpress_module.RandomPress):
            """Work around kvpress 0.5.3 creating a CPU generator for CUDA scores."""

            def score(
                self,
                module,
                hidden_states,
                keys,
                values,
                attentions,
                kwargs,
            ):
                import torch

                generator = None
                if self.seed is not None:
                    generator = torch.Generator(device=keys.device)
                    generator.manual_seed(self.seed)
                return torch.rand(
                    *keys.shape[:-1],
                    generator=generator,
                    device=keys.device,
                    dtype=keys.dtype,
                )

        base_press = DeviceSafeRandomPress(seed=seed)
    elif normalized == "rkv":
        from kvcache.presses.rkv import build_rkv_scorer

        base_press = build_rkv_scorer(kvpress_module)
    else:
        raise AssertionError("unreachable")
    press_cls = decoding_press_cls or kvpress_module.DecodingPress
    # R-KV scores importance over the last-8-query observation window, so DecodingPress
    # must buffer those hidden states; the other presses score from keys alone.
    if normalized == "rkv":
        from kvcache.presses.rkv import RKV_WINDOW_SIZE

        buffer_size = RKV_WINDOW_SIZE
    else:
        buffer_size = 0
    return press_cls(
        base_press=base_press,
        target_size=budget,
        compression_interval=compression_interval,
        hidden_states_buffer_size=buffer_size,
    )


def build_replay_press(
    name: str,
    *,
    budget: int,
    compression_interval: int = 64,
    seed: int | None = None,
    kvpress_module: Any | None = None,
) -> Any:
    """Build an instrumented decoding press toggled explicitly by replay."""
    if name.lower() == "fullcache":
        return NoOpReplayPress()
    if kvpress_module is None:
        import kvpress as kvpress_module

    class EnabledDecodingPress(kvpress_module.DecodingPress):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.enabled = False
            self.profile_step: int | None = None
            self.onset_step: int | None = None
            self.compressions: list[dict[str, int]] = []

        def forward_hook(self, module, input, kwargs, output):
            if not self.enabled:
                return output
            return super().forward_hook(module, input, kwargs, output)

        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            before = int(keys.shape[2])
            compressed_keys, compressed_values = super().compress(
                module, hidden_states, keys, values, attentions, kwargs
            )
            after = int(compressed_keys.shape[2])
            self.compressions.append(
                {"layer": int(module.layer_idx), "before": before, "after": after}
            )
            if self.onset_step is None:
                # DecodingPress prunes in a forward hook after the current token's
                # attention output is computed. The pruned cache first affects the
                # distribution produced by the following replay forward.
                self.onset_step = (
                    None if self.profile_step is None else self.profile_step + 1
                )
            return compressed_keys, compressed_values

    return build_press(
        name,
        budget=budget,
        compression_interval=compression_interval,
        seed=seed,
        kvpress_module=kvpress_module,
        decoding_press_cls=EnabledDecodingPress,
    )


def build_rkv_press(
    *,
    budget: int,
    compression_interval: int = 64,
    kvpress_module: Any | None = None,
    decoding_press_cls: type | None = None,
) -> Any:
    """Build the parity-gated R-KV decoding press (for the parity harness only).

    R-KV is refused by get_press until the parity gate passes; this constructor exists
    so scripts/rkv_parity.py can exercise the port against the authors' code.
    """
    from kvcache.presses.rkv import RKV_WINDOW_SIZE, build_rkv_scorer

    if kvpress_module is None:
        import kvpress as kvpress_module
    press_cls = decoding_press_cls or kvpress_module.DecodingPress
    return press_cls(
        base_press=build_rkv_scorer(kvpress_module),
        target_size=budget,
        compression_interval=compression_interval,
        hidden_states_buffer_size=RKV_WINDOW_SIZE,
    )

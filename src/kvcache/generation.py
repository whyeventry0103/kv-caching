"""Generation entry points; concrete heavy backends are remote-only."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from kvcache.canonicalize import canonicalize, extract_answer
from kvcache.datasets import Problem
from kvcache.determinism import set_determinism
from kvcache.grading import is_equiv
from kvcache.metrics.add import OTHER_ANSWER
from kvcache.models import render_prompt, tokenize_rendered_prompt

_ANSWER_MARKER = re.compile(r"\\boxed\{|(?:final\s+)?answer\s*(?:is|:)", re.I)


def has_answer_marker(trace: str) -> bool:
    """Whether the trace contains a recognized final-answer marker."""
    return bool(_ANSWER_MARKER.search(trace))


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    trace: str
    answer_raw: str
    answer_canon: str
    correct: bool
    n_tokens: int
    truncated: bool
    wallclock_s: float
    hit_token_cap: bool
    has_answer_marker: bool
    finish_reason: str
    prompt_token_ids: tuple[int, ...] = ()
    trace_token_ids: tuple[int, ...] = ()


def _move_inputs(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def finalize_generation(
    *,
    prompt: str,
    trace: str,
    n_tokens: int,
    max_new_tokens: int,
    gold_answer: str,
    wallclock_s: float,
    prompt_token_ids: tuple[int, ...] = (),
    trace_token_ids: tuple[int, ...] = (),
) -> GenerationResult:
    """Convert decoded text into a scored result without requiring Torch.

    Correctness is decided by :func:`kvcache.grading.is_equiv` on the *raw* gold
    and *raw* predicted answer strings. ``answer_canon`` is retained only as the
    ADD bucketing key, never as the correctness oracle. ``gold_answer`` must be the
    raw dataset answer (not pre-canonicalized) for equivalence to be sound.
    """
    has_marker = has_answer_marker(trace)
    hit_token_cap = n_tokens >= max_new_tokens
    # truncated is the length-failure case only: capped before emitting an answer.
    # A trace that emits a marker exactly at the cap is hit_token_cap but NOT
    # truncated, and is scored on its answer (see intermediate_algebra/1994 s89).
    truncated = hit_token_cap and not has_marker
    finish_reason = "marker" if has_marker else ("length" if hit_token_cap else "eos")
    answer_raw = extract_answer(trace) if has_marker else ""
    answer_canon = canonicalize(answer_raw) if has_marker else OTHER_ANSWER
    correct = (not truncated) and has_marker and is_equiv(gold_answer, answer_raw)
    return GenerationResult(
        prompt=prompt,
        trace=trace,
        answer_raw=answer_raw,
        answer_canon=answer_canon,
        correct=correct,
        n_tokens=n_tokens,
        truncated=truncated,
        wallclock_s=wallclock_s,
        hit_token_cap=hit_token_cap,
        has_answer_marker=has_marker,
        finish_reason=finish_reason,
        prompt_token_ids=prompt_token_ids,
        trace_token_ids=trace_token_ids,
    )


def generate_full_cache(
    model: Any,
    tokenizer: Any,
    problem: Problem,
    model_cfg: dict[str, Any],
    sampling_cfg: dict[str, Any],
    *,
    seed: int,
    max_new_tokens: int,
) -> GenerationResult:
    """Generate one full-cache trace and retain truncation failures."""
    import torch

    prompt = render_prompt(tokenizer, problem.question, model_cfg)
    inputs = _move_inputs(
        tokenize_rendered_prompt(tokenizer, prompt, return_tensors="pt"),
        model.device,
    )
    input_ids = inputs["input_ids"]
    set_determinism(seed)
    started = time.monotonic()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(sampling_cfg["temperature"]),
            top_p=float(sampling_cfg["top_p"]),
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
    wallclock_s = time.monotonic() - started
    generated = output_ids[0, input_ids.shape[-1] :]
    n_tokens = int(generated.shape[-1])
    trace = tokenizer.decode(generated, skip_special_tokens=True)
    return finalize_generation(
        prompt=prompt,
        trace=trace,
        n_tokens=n_tokens,
        max_new_tokens=max_new_tokens,
        gold_answer=problem.gold_answer,
        wallclock_s=wallclock_s,
        prompt_token_ids=tuple(int(token) for token in input_ids[0].tolist()),
        trace_token_ids=tuple(int(token) for token in generated.tolist()),
    )


STEP_GATE_VERSION = "forward-pre-hook-v1"


@contextmanager
def decode_step_gate(model: Any, press: Any) -> Iterator[None]:
    """Drive an instrumented press's step counter from free-running generation.

    Amendment R.3. The onset gate (``compression_active``) compares
    ``press.profile_step`` with ``compression_start_step``; in replay the loop
    writes that counter, and in ``model.generate`` nothing does. A forward
    pre-hook on the top-level model sets it before every forward, with the
    replay loop's convention: the prefill is step 0 and the forward that
    consumes generated token ``j - 1`` (and produces token ``j``'s logits) is
    step ``j``. The press is explicitly enabled for the duration, because the
    instrumented subclass is a no-op unless enabled.

    The hook only counts; it never touches the sampling RNG, and the Random
    policy's generator is its own, so policy randomness and sampling
    randomness stay separate.
    """
    for attr in ("enabled", "profile_step", "compression_start_step"):
        if not hasattr(press, attr):
            raise TypeError(
                f"step gate needs an instrumented press (missing {attr!r}); build "
                "it with build_press(..., instrumented=True)"
            )
    counter = {"step": 0}

    def pre_hook(_module: Any, _args: Any) -> None:
        press.profile_step = counter["step"]
        counter["step"] += 1

    handle = model.register_forward_pre_hook(pre_hook)
    previous = press.enabled
    press.enabled = True
    try:
        yield
    finally:
        handle.remove()
        press.enabled = previous


def exposure_from_events(
    events: Sequence[dict[str, Any]], n_tokens: int
) -> dict[str, Any]:
    """Achieved exposure of one generation, from the press's prune records.

    An event is effective when it removed entries (``after < before``); an
    event on a cache still under budget is a no-op and is not exposure. The
    first effective prune at forward ``k`` first affects token ``k + 1``, so
    the tokens sampled from a pruned cache number ``n_tokens - (k + 1)``. A
    generation that ends first is an UNEXPOSED assigned observation (exposure
    0), which stays in every denominator (R.3).
    """
    steps = [
        int(e["step"])
        for e in events
        if e.get("step") is not None and int(e["after"]) < int(e["before"])
    ]
    if any(e.get("step") is None for e in events):
        raise ValueError("prune event without a step: the press was not gated")
    if not steps:
        return {"eviction_onset_step": None, "achieved_exposure": 0, "exposed": False}
    onset = min(steps) + 1
    exposure = max(0, int(n_tokens) - onset)
    return {
        "eviction_onset_step": onset,
        "achieved_exposure": exposure,
        "exposed": exposure > 0,
    }


def generate_with_press(
    model: Any,
    tokenizer: Any,
    problem: Problem,
    model_cfg: dict[str, Any],
    sampling_cfg: dict[str, Any],
    *,
    press: Any,
    seed: int,
    max_new_tokens: int,
    step_gate: bool = False,
) -> tuple[GenerationResult, int]:
    """Generate once under a verified press context and report prompt length.

    ``step_gate`` drives an instrumented press's decode-step counter
    (:func:`decode_step_gate`); the default path is the legacy one, untouched.
    """
    import torch

    prompt = render_prompt(tokenizer, problem.question, model_cfg)
    inputs = _move_inputs(
        tokenize_rendered_prompt(tokenizer, prompt, return_tensors="pt"),
        model.device,
    )
    input_ids = inputs["input_ids"]
    set_determinism(seed)
    started = time.monotonic()
    gate = decode_step_gate(model, press) if step_gate else _no_gate()
    with torch.inference_mode(), press(model), gate:
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(sampling_cfg["temperature"]),
            top_p=float(sampling_cfg["top_p"]),
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
    wallclock_s = time.monotonic() - started
    generated = output_ids[0, input_ids.shape[-1] :]
    n_tokens = int(generated.shape[-1])
    trace = tokenizer.decode(generated, skip_special_tokens=True)
    return (
        finalize_generation(
            prompt=prompt,
            trace=trace,
            n_tokens=n_tokens,
            max_new_tokens=max_new_tokens,
            gold_answer=problem.gold_answer,
            wallclock_s=wallclock_s,
            prompt_token_ids=tuple(int(token) for token in input_ids[0].tolist()),
            trace_token_ids=tuple(int(token) for token in generated.tolist()),
        ),
        int(input_ids.shape[-1]),
    )


@contextmanager
def _no_gate() -> Iterator[None]:
    yield


def generate_compressed(
    model: Any,
    tokenizer: Any,
    problem: Problem,
    model_cfg: dict[str, Any],
    sampling_cfg: dict[str, Any],
    *,
    press: Any,
    seed: int,
    max_new_tokens: int,
    step_gate: bool = False,
) -> GenerationResult:
    """Generate one compressed trace using the verified DecodingPress path."""
    result, _ = generate_with_press(
        model,
        tokenizer,
        problem,
        model_cfg,
        sampling_cfg,
        press=press,
        seed=seed,
        max_new_tokens=max_new_tokens,
        step_gate=step_gate,
    )
    return result


def bank_backend(name: str) -> str:
    if name not in {"transformers", "vllm"}:
        raise ValueError(f"unsupported trace-bank backend: {name}")
    return name

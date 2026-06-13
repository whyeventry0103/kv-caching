"""Generation entry points; concrete heavy backends are remote-only."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kvcache.canonicalize import canonicalize, extract_answer
from kvcache.datasets import Problem
from kvcache.determinism import set_determinism
from kvcache.grading import is_equiv
from kvcache.metrics.add import OTHER_ANSWER
from kvcache.models import render_prompt

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
    inputs = _move_inputs(tokenizer(prompt, return_tensors="pt"), model.device)
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
    )


def generate_compressed(
    prompt: str, generator: Callable[..., Any], press: Any, **kwargs: Any
) -> Any:
    return generator(prompt, press=press, **kwargs)


def bank_backend(name: str) -> str:
    if name not in {"transformers", "vllm"}:
        raise ValueError(f"unsupported trace-bank backend: {name}")
    return name

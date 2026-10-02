"""Exact slot-dynamics simulation of the eviction presses (Meeting #4, Q1).

Scientific rationale. ``kvpress.RandomPress`` reseeds its generator at every
``score`` call, so for a given cache size the score of each cache SLOT is a
fixed number, and ``ScorerPress`` gathers the top-k in score order. Eviction is
therefore a fixed pattern over slots, not a fresh uniform draw. Because the
scores depend only on the tensor shape, dtype, device and seed, the retention
process can be simulated exactly without a model: the same ``torch.rand`` call
on the same generator reproduces the press's own scores. StreamingLLM keeps
``n_sink`` slots plus the most recent ones, which on CUDA at production sizes
is position order (PREREGISTRATION S addendum A, fact 1).

What this is for: a description of WHICH positions each press retains, so the
accuracy ordering can be discussed against it. It is not a measurement of
accuracy and licenses no causal claim about it.
"""

from __future__ import annotations

import statistics
from typing import Any


def simulate_slots(
    policy: str,
    *,
    prompt_len: int,
    budget: int,
    interval: int,
    steps: int,
    seed: int,
    n_kv: int,
    device: str = "cpu",
    dtype_name: str = "bfloat16",
    n_sink: int = 4,
) -> dict[str, Any]:
    """Absolute positions held by every KV head after ``steps`` decode steps.

    Random: ``torch.rand(1, n_kv, k_len)`` from a generator reseeded with
    ``seed`` at every prune, top-``budget`` by score (the press's own code
    path). StreamingLLM: ``n_sink`` first slots plus the most recent, in
    position order. Returns the per-head position lists and the logical length.
    """
    import torch

    dtype = getattr(torch, dtype_name)
    heads = [list(range(prompt_len)) for _ in range(n_kv)]
    logical = prompt_len
    gen = torch.Generator(device=device)
    n_prunes = 0
    for t in range(1, steps + 1):
        for h in heads:
            h.append(logical)
        logical += 1
        if t % interval == 0 and len(heads[0]) > budget:
            k_len = len(heads[0])
            if policy == "random":
                gen.manual_seed(seed)
                r = torch.rand(
                    1, n_kv, k_len, generator=gen, device=device, dtype=dtype
                )
                idx = r.topk(budget, dim=-1).indices[0].tolist()
                heads = [[h[i] for i in idx[j]] for j, h in enumerate(heads)]
            elif policy == "streamingllm":
                heads = [h[:n_sink] + h[-(budget - n_sink) :] for h in heads]
            else:
                raise ValueError(f"no simulation for {policy!r}")
            n_prunes += 1
    return {"heads": heads, "logical": logical, "n_prunes": n_prunes}


def retention_summary(
    sim: dict[str, Any], *, prompt_len: int, budget: int
) -> dict[str, Any]:
    heads, logical = sim["heads"], sim["logical"]
    allpos = [p for h in heads for p in h]
    ages = sorted(logical - 1 - p for p in allpos)
    return {
        "prompt_kept_per_head": [sum(1 for p in h if p < prompt_len) for h in heads],
        "last_64_kept_per_head": [
            sum(1 for p in h if p >= logical - 64) for h in heads
        ],
        "older_than_1024_per_head": [
            sum(1 for p in h if p < logical - 1024) for h in heads
        ],
        "distinct_positions_union": len(set(allpos)),
        "union_over_budget": len(set(allpos)) / budget,
        "median_age": statistics.median(ages),
        "p90_age": ages[int(0.9 * (len(ages) - 1))],
        "n_prunes": sim["n_prunes"],
    }


def cohort_survival(
    policy: str,
    *,
    enter_step: int,
    cohort: int,
    horizons: list[int],
    **kw: Any,
) -> dict[str, list[float]]:
    """Fraction of a ``cohort``-token block entering at ``enter_step`` still
    cached after each horizon, per head."""
    prompt_len = int(kw["prompt_len"])
    lo, hi = prompt_len + enter_step, prompt_len + enter_step + cohort
    out: dict[str, list[float]] = {}
    for horizon in horizons:
        sim = simulate_slots(policy, steps=enter_step + cohort + horizon, **kw)
        out[str(horizon)] = [
            round(sum(1 for p in h if lo <= p < hi) / cohort, 4) for h in sim["heads"]
        ]
    return out

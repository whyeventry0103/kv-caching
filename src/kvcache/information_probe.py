"""Information-probe execution core (Meeting 5, M5-C; review of 6b409001, R3;
prepared 2026-10-14 -- NOT approved, no GPU run).

Scientific rationale. The finite-channel MI (``kvcache.task_information``) is
only meaningful if each readout depends on the target solely through the
declared cache at a fixed checkpoint, read by a declared reader that cannot
change that cache. This module builds exactly that, backend-agnostic so the
whole path is tested on CPU with a stub model:

1. the protected instruction is prefilled, then the fact-bearing body is fed
   token by token through the SAME decode path that evicts trace tokens, with
   the policy's eviction enabled;
2. at the checkpoint the cache is CLONED twice per query; each clone receives
   only the query tokens, with eviction DISABLED, and the native log-softmax
   over the whole vocabulary gives the four symbol log-probabilities, the
   native top-1 token and its log-probability;
3. controls, written into every row: the two clones agree exactly
   (``clone_agreement``: no shared mutable state between readouts); the
   readout evicted nothing (``no_readout_eviction``: prune count unchanged);
   and two independent full-cache arms of the same context agree exactly
   (``full_full_identity``: determinism of the instrument itself);
4. the retained fraction of each fact span (absolute positions, from the
   press's position maps where it keeps them) is logged per row.

The backend supplies ``new_cache``, ``forward(ids, cache) -> logits``,
``clone(cache)``, ``log_softmax(logits) -> list[float]``; the press handle
supplies ``enabled(flag)`` (a context manager), ``n_prunes()`` and
``retained()``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

from kvcache.information_tasks import Variant
from kvcache.provenance import GENERATION_SOURCES
from kvcache.task_information import ROW_SCHEMA


class Backend(Protocol):
    def new_cache(self) -> Any: ...
    def forward(self, token_ids: Sequence[int], cache: Any) -> Any: ...
    def clone(self, cache: Any) -> Any: ...
    def log_softmax(self, logits: Any) -> list[float]: ...


class PressHandle(Protocol):
    def attach(self) -> AbstractContextManager[Any]: ...
    def enabled(self, flag: bool) -> AbstractContextManager[Any]: ...
    def n_prunes(self) -> int: ...
    def retained(self) -> dict[int, list[list[int]]] | None: ...


@dataclass(frozen=True)
class NoPress:
    """The full-cache arm: nothing is ever evicted."""

    def attach(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def enabled(self, flag: bool) -> AbstractContextManager[Any]:
        return nullcontext()

    def n_prunes(self) -> int:
        return 0

    def retained(self) -> dict[int, list[list[int]]] | None:
        return None


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _readout(
    backend: Backend, cache: Any, query_ids: Sequence[int], symbol_ids: Sequence[int]
) -> dict[str, Any]:
    lp = backend.log_softmax(backend.forward(list(query_ids), cache))
    top = max(range(len(lp)), key=lambda i: (lp[i], -i))
    sym = [float(lp[t]) for t in symbol_ids]
    return {
        "symbol_logprobs": sym,
        "greedy": symbol_ids.index(top) if top in symbol_ids else None,
        "top1_logprob": float(lp[top]),
    }


def retained_fraction(
    retained: Mapping[int, list[list[int]]] | None, span: tuple[int, int]
) -> float | None:
    """Mean over layers and heads of the fraction of span positions kept."""
    if not retained:
        return None
    want = set(range(*span))
    fracs = [
        len(want & set(head)) / len(want)
        for heads in retained.values()
        for head in heads
    ]
    return sum(fracs) / len(fracs) if fracs else None


def run_variant(
    backend: Backend,
    press: PressHandle,
    variant: Variant,
    *,
    symbol_ids: Sequence[int],
    queries: Sequence[str] = ("s", "n"),
) -> dict[str, dict[str, Any]]:
    """Execute one (S, N) variant under one policy; readouts per query."""
    with press.attach():
        return _run_attached(backend, press, variant, symbol_ids, queries)


def _run_attached(
    backend: Backend,
    press: PressHandle,
    variant: Variant,
    symbol_ids: Sequence[int],
    queries: Sequence[str],
) -> dict[str, dict[str, Any]]:
    cache = backend.new_cache()
    with press.enabled(True):
        backend.forward(list(variant.prompt_ids), cache)
        for tok in variant.body_ids:
            backend.forward([tok], cache)
    prunes_at_checkpoint = press.n_prunes()
    retained = press.retained()
    base = len(variant.prompt_ids)
    spans = {
        k: retained_fraction(retained, (base + a, base + b))
        for k, (a, b) in variant.fact_spans.items()
    }
    out = {}
    with press.enabled(False):
        for q in queries:
            a = _readout(
                backend, backend.clone(cache), variant.query_ids[q], symbol_ids
            )
            b = _readout(
                backend, backend.clone(cache), variant.query_ids[q], symbol_ids
            )
            out[q] = {
                **a,
                "clone_agreement": a == b,
                "no_readout_eviction": press.n_prunes() == prunes_at_checkpoint,
                "n_prunes": prunes_at_checkpoint,
                "retained_fraction": spans,
                "e2e_max_abs_diff": None,
            }
        # End-to-end control (calibration criterion C4): the LAST query is also
        # fed straight into the original cache -- no clone -- and must agree
        # with its cloned readout. The original is discarded afterwards.
        last = queries[-1]
        e2e = _readout(backend, cache, variant.query_ids[last], symbol_ids)
        out[last]["e2e_max_abs_diff"] = max(
            0.0 if x == y else abs(x - y)  # equal (incl. -inf) is no difference
            for x, y in zip(
                e2e["symbol_logprobs"], out[last]["symbol_logprobs"], strict=True
            )
        )
        out[last]["no_readout_eviction"] = (
            out[last]["no_readout_eviction"]
            and press.n_prunes() == prunes_at_checkpoint
        )
    return out


def run_context(
    backend: Backend,
    variants: Sequence[Variant],
    *,
    policies: Mapping[str, Callable[[int], PressHandle]],
    eviction_seeds: Sequence[int],
    symbol_ids: Sequence[int],
    provenance: Mapping[str, Any],
    queries: Sequence[str] = ("s", "n"),
) -> list[dict[str, Any]]:
    """Every row of one context block: all variants x policies x seeds x queries.

    ``policies`` maps a name to a press factory taking the eviction seed; the
    name ``full`` must map to a no-eviction arm. The full/full identity control
    runs the full arm twice on the block's first variant.
    """
    if "full" not in policies:
        raise ValueError("the design needs a full-cache arm named 'full'")
    first = variants[0]
    f1 = run_variant(backend, policies["full"](0), first, symbol_ids=symbol_ids)
    f2 = run_variant(backend, policies["full"](0), first, symbol_ids=symbol_ids)
    full_full = all(
        f1[q]["symbol_logprobs"] == f2[q]["symbol_logprobs"]
        and f1[q]["top1_logprob"] == f2[q]["top1_logprob"]
        for q in queries
    )
    rows = []
    for name, factory in policies.items():
        for seed in eviction_seeds:
            for v in variants:
                res = run_variant(
                    backend, factory(seed), v, symbol_ids=symbol_ids, queries=queries
                )
                for q in queries:
                    r = res[q]
                    rows.append(
                        {
                            "schema": ROW_SCHEMA,
                            "policy": name,
                            "query": q,
                            "context_id": v.block_id,
                            "eviction_seed": int(seed),
                            "s": v.s,
                            "n": v.n,
                            "symbol_logprobs": r["symbol_logprobs"],
                            "greedy": r["greedy"],
                            "top1_logprob": r["top1_logprob"],
                            "provenance": dict(provenance),
                            "controls": {
                                "full_full_identity": full_full,
                                "clone_agreement": r["clone_agreement"],
                                "no_readout_eviction": r["no_readout_eviction"],
                            },
                            "n_prunes": r["n_prunes"],
                            "e2e_max_abs_diff": r["e2e_max_abs_diff"],
                            "retained_fraction": r["retained_fraction"],
                            "fact_spans": {k: list(s) for k, s in v.fact_spans.items()},
                        }
                    )
    return rows


def resume_problems(
    existing: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> list[str]:
    """Refuse a resume into rows produced under another identity."""
    return [
        f"row {i}: provenance differs"
        for i, r in enumerate(existing)
        if r.get("provenance") != dict(provenance)
    ]


INFO_KIND = "info_probe"
INFO_SOURCES = (
    "src/kvcache/information_probe.py",
    "src/kvcache/information_tasks.py",
    "src/kvcache/task_information.py",
    "src/kvcache/presses/prompt_protect.py",
    "src/kvcache/presses/protected_selectors.py",
    "src/kvcache/presses/rkv_prompt_protect.py",
    "src/kvcache/presses/rkv.py",
    "src/kvcache/presses/__init__.py",
    "src/kvcache/replay.py",
    "src/kvcache/models.py",
    "scripts/run_information_probe.py",
)
if GENERATION_SOURCES.setdefault(INFO_KIND, INFO_SOURCES) != INFO_SOURCES:
    raise RuntimeError(f"provenance family {INFO_KIND!r} registered twice")


class TransformersBackend:
    """A Hugging Face causal LM as a ``Backend``: absolute positions are kept
    per cache (so pruning never shifts RoPE positions), and a clone is a deep
    copy carrying its own position counter."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._pos: dict[int, int] = {}

    def new_cache(self) -> Any:
        from transformers import DynamicCache

        cache = DynamicCache()
        self._pos[id(cache)] = 0
        return cache

    def clone(self, cache: Any) -> Any:
        import copy

        new = copy.deepcopy(cache)
        self._pos[id(new)] = self._pos[id(cache)]
        return new

    def forward(self, token_ids: Sequence[int], cache: Any) -> Any:
        import torch

        start = self._pos[id(cache)]
        dev = self.model.device
        with torch.inference_mode():
            out = self.model(
                input_ids=torch.tensor([list(token_ids)], device=dev),
                past_key_values=cache,
                cache_position=torch.arange(start, start + len(token_ids), device=dev),
                use_cache=True,
            )
        self._pos[id(cache)] = start + len(token_ids)
        return out.logits[0, -1]

    def log_softmax(self, logits: Any) -> list[float]:
        import torch

        return torch.log_softmax(logits.float(), dim=-1).tolist()


class KVPressHandle:
    """A protected decoding press (instrumented, so ``enabled`` is honoured) as
    a ``PressHandle``; retained positions come from its position maps."""

    def __init__(self, press: Any, model: Any) -> None:
        self.press, self.model = press, model

    def attach(self) -> AbstractContextManager[Any]:
        return self.press(self.model)

    def enabled(self, flag: bool) -> AbstractContextManager[Any]:
        from kvcache.replay import press_enabled

        return press_enabled(self.press, flag)

    def n_prunes(self) -> int:
        return int(self.press.protect.n_prunes)

    def retained(self) -> dict[int, list[list[int]]] | None:
        maps = self.press.protect.maps
        return {int(k): v.tolist() for k, v in maps.items()} or None

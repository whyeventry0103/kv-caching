"""Protected M / Delta / omega capture (Meeting 5, M5A-CAPTURE-PP; prepared
2026-10-20 on CPU -- NOT approved, no GPU run).

Scientific rationale. The Random-vs-StreamingLLM accuracy gap survives prompt
protection (M5-A strict read: +0.054, descriptive), but every M, Delta and
omega the project has measured is UNPROTECTED (capture-b256). Whether the
metrics still rank the policies against their protected accuracy is therefore
unmeasured, and protected accuracy may never be combined with unprotected
captures. This module measures the same three quantities, with the same
replay, on the protected presses.

What changes, and only this. The validated capture (``kvcache.capture``)
recovers each prune's selection with ``EventTap``: it re-applies ``topk`` to
the base scorer's scores and refuses unless the result reproduces the
compressed cache bitwise. A protected press forces every prompt position in
(and the StreamingLLM / H2O-style / R-KV variants add their own ordering or
recency rules), so its selection is NOT the base scores' top-k and EventTap
would correctly refuse. Every protected family, however, records the
absolute positions it kept (``press.protect.maps[layer]``, in the exact order
it gathered the cache). ``ProtectedEventTap`` reads that record and verifies
it the same way EventTap verifies its re-derivation: the pre-prune keys,
gathered at the record's local indices, must equal the retained keys
bitwise, and no prompt position may be evicted. Everything downstream -- the
full branch, the position tracker, M / M_event / Delta, collateral, the
latent split and omega, the layer-0 key check, the pre-eviction bitwise
equality -- is ``kvcache.capture`` unchanged.

Why a separate module, and why ``capture.replay_condition`` is reused by a
scoped swap rather than edited. ``capture.py``'s bytes are part of six
generation identities with complete stored runs (capture, capture_v2, basis,
exact_lag, restore_v2, restore_v3); editing it would orphan them. The
protected replay therefore calls the validated ``replay_condition`` with
``capture.EventTap`` replaced by ``ProtectedEventTap`` for the duration of
the call only (restored on any exit), so the protected and unprotected
captures run one replay loop. This module has its own identity family,
``capture_pp``.

torch is imported inside functions so the CPU-only gate never needs it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from kvcache.provenance import GENERATION_SOURCES

CAPTURE_PP_KIND = "capture_pp"
PROTECTED_POLICIES = ("random_pp", "streamingllm_pp", "h2o_style_pp_v1", "rkv_pp_v1")
CAPTURE_PP_SOURCES = (
    "src/kvcache/capture.py",
    "src/kvcache/capture_protected.py",
    "src/kvcache/presses/__init__.py",
    "src/kvcache/presses/prompt_protect.py",
    "src/kvcache/presses/protected_selectors.py",
    "src/kvcache/presses/rkv_prompt_protect.py",
    "src/kvcache/presses/rkv.py",
    "src/kvcache/replay.py",
    "src/kvcache/models.py",
    "scripts/run_capture_pp.py",
)
if (
    GENERATION_SOURCES.setdefault(CAPTURE_PP_KIND, CAPTURE_PP_SOURCES)
    != CAPTURE_PP_SOURCES
):
    raise RuntimeError(f"provenance family {CAPTURE_PP_KIND!r} registered twice")


def build_protected_capture_press(
    name: str,
    *,
    budget: int,
    compression_interval: int,
    seed: int,
    forced_prompt_len: int | None = None,
    kvpress_module: Any | None = None,
    decoding_press_cls: type | None = None,
    min_trace_slots: int | None = None,
) -> Any:
    """An INSTRUMENTED protected press of one registered family.

    ``forced_prompt_len`` is the identity control only (0 for Random and R-KV:
    the press then reproduces the unprotected one). ``min_trace_slots`` is for
    tiny-model tests only; production uses each family's registered default.
    """
    from kvcache.presses.prompt_protect import build_prompt_protected_press
    from kvcache.presses.protected_selectors import build_protected_selector_press
    from kvcache.presses.rkv_prompt_protect import build_rkv_prompt_protected_press

    builders = {
        "random_pp": build_prompt_protected_press,
        "streamingllm_pp": build_prompt_protected_press,
        "h2o_style_pp_v1": build_protected_selector_press,
        "rkv_pp_v1": build_rkv_prompt_protected_press,
    }
    if name not in builders:
        raise KeyError(
            f"not a protected capture policy: {name!r} ({PROTECTED_POLICIES})"
        )
    return builders[name](
        name,
        budget=int(budget),
        compression_interval=int(compression_interval),
        seed=int(seed),
        kvpress_module=kvpress_module,
        decoding_press_cls=decoding_press_cls,
        instrumented=True,
        forced_prompt_len=forced_prompt_len,
        **(
            {} if min_trace_slots is None else {"min_trace_slots": int(min_trace_slots)}
        ),
    )


def local_indices(candidates: Any, kept: Any) -> Any:
    """Per KV head, the index of every kept absolute position among the
    candidates, in the kept order. Raises unless every kept position is a
    candidate (positions are unique within a head)."""
    import torch

    if candidates.shape[0] != kept.shape[0]:
        raise RuntimeError("candidates and kept positions disagree on KV heads")
    srt, perm = torch.sort(candidates, dim=-1)
    loc = torch.searchsorted(srt, kept)
    if bool((loc >= srt.shape[-1]).any()):
        raise RuntimeError("a kept position is not among the candidates")
    if not torch.equal(srt.gather(1, loc), kept):
        raise RuntimeError("a kept position is not among the candidates")
    return perm.gather(1, loc)


class ProtectedEventTap:
    """``EventTap``'s interface for a protected press: the selection comes from
    the press's own position record, verified bitwise against the cache."""

    def __init__(self, press: Any, tracker: Any, *, stash_kv: bool = False) -> None:
        if stash_kv:
            raise RuntimeError("the protected capture does not stash evicted K/V")
        state = getattr(press, "protect", None)
        if state is None:
            raise RuntimeError("ProtectedEventTap needs a protected press")
        self.press = press
        self.state = state
        self.tracker = tracker
        self.events: list[Any] = []
        self.listeners: list[Any] = []
        self._prunes_before = 0
        press.event_tap = self

    def begin(self) -> None:
        self._prunes_before = int(self.state.n_prunes)

    def end(
        self,
        module: Any,
        keys_before: Any,
        keys_after: Any,
        step: Any,
        values_before: Any = None,
    ) -> None:
        import torch

        from kvcache.capture import PruneEvent

        k_len, n_kept = int(keys_before.shape[2]), int(keys_after.shape[2])
        pruned = int(self.state.n_prunes) - self._prunes_before
        if n_kept == k_len:
            if pruned:
                raise RuntimeError(
                    "protected tap: a prune was recorded but nothing left"
                )
            return
        if pruned != 1:
            raise RuntimeError(
                f"protected tap: the cache shrank but the press recorded "
                f"{pruned} prunes"
            )
        if step is None:
            raise RuntimeError("protected tap: press has no profile_step")
        layer = int(module.layer_idx)
        dev = keys_before.device
        kept = self.state.maps[layer].to(dev)
        if tuple(kept.shape) != (int(keys_after.shape[1]), n_kept):
            raise RuntimeError(
                f"protected tap: position record {tuple(kept.shape)} does not match "
                f"the retained cache at layer {layer}"
            )
        cand = self.tracker.current(layer, k_len, device=dev)
        idx = local_indices(cand, kept)
        gathered = keys_before.gather(
            2, idx.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, keys_before.shape[-1])
        )
        if not torch.equal(gathered, keys_after):
            raise RuntimeError(
                f"protected tap: the press's position record does not reproduce the "
                f"retained cache at layer {layer}, step {step}"
            )
        cand, kept2 = self.tracker.prune(layer, k_len, idx)
        if not torch.equal(kept2, kept):
            raise RuntimeError("protected tap: tracker and press disagree on positions")
        prompt_len = int(self.state.prompt_len or 0)
        if bool(((kept < prompt_len).sum(-1) != prompt_len).any()):
            raise RuntimeError(
                f"protected tap: a prompt position was evicted (layer {layer})"
            )
        mask = torch.ones_like(cand, dtype=torch.bool)
        mask.scatter_(1, idx, False)
        evicted = cand[mask].view(cand.shape[0], k_len - n_kept)
        ev = PruneEvent(int(step), layer, cand, evicted)
        self.events.append(ev)
        for fn in self.listeners:
            fn(ev)


@contextmanager
def _protected_tap() -> Iterator[None]:
    """Swap ``capture.EventTap`` for ``ProtectedEventTap`` for one call only."""
    from kvcache import capture

    original = capture.EventTap
    capture.EventTap = ProtectedEventTap
    try:
        yield
    finally:
        capture.EventTap = original


def replay_condition_protected(
    model: Any,
    press: Any,
    full: Any,
    prompt_ids: Sequence[int],
    trace_ids: Sequence[int],
    **kwargs: Any,
) -> dict[str, Any]:
    """``kvcache.capture.replay_condition`` for a protected press: the same
    replay loop, instruments and live checks, with the selection read from the
    press's own verified record. Adds the press's protection report."""
    from kvcache import capture

    if getattr(press, "protect", None) is None:
        raise RuntimeError("replay_condition_protected needs a protected press")
    with _protected_tap():
        out = capture.replay_condition(
            model, press, full, prompt_ids, trace_ids, **kwargs
        )
    out["protect"] = press.protect.report()
    return out

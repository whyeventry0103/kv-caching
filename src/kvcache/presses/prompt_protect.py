"""Prompt-protected decoding presses (PREREGISTRATION Amendment S.1).

Scientific rationale. None of the presses in ``kvcache.presses`` protects the
prompt: ``RandomPress`` scores every cached position, and ``StreamingLLMPress``
keeps four sink tokens plus the recent window, so the question itself is
evicted once the trace outgrows the budget. Concurrent work (arXiv:2609.03430)
attributes most of the gap between eviction selectors to whether the prompt is
kept. Until a prompt-protected arm runs through the decomposition, the
trajectory residual is open to a prompt-loss reading. These presses are that
arm, and they differ from the unprotected ones in exactly one respect.

* ``random_pp``: the unprotected Random press's own scores, with the score of
  every entry whose ABSOLUTE position lies in the prompt raised to the dtype
  maximum before the top-k.
* ``streamingllm_pp``: ``StreamingLLMPress`` with ``n_sink`` set to the prompt
  length (the four sink tokens are the first four prompt tokens).

Why positions are tracked rather than assumed. ``ScorerPress.compress`` gathers
``scores.topk(n).indices``, which are in score order, so after the first prune
a cache slot's index says nothing about its absolute position. The press keeps
a per-layer, per-KV-head position map, protects by position, and checks at
every prune that every prompt position survived (and, for StreamingLLM+prompt,
that the retained trace entries are the most recent). A violation raises: a
row that exists passed the check at every prune.

Nothing in ``kvcache.presses.__init__`` is modified. Its bytes are part of
every generation family's identity, which is why this is a separate module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PP_SUFFIX = "_pp"
PP_BASES = ("random", "streamingllm")
MIN_TRACE_SLOTS = 64


def pp_base_name(name: str) -> str:
    """``random_pp`` -> ``random``; anything else is refused."""
    normalized = name.lower()
    if not normalized.endswith(PP_SUFFIX):
        raise KeyError(f"not a prompt-protected press name: {name!r}")
    base = normalized[: -len(PP_SUFFIX)]
    if base not in PP_BASES:
        raise KeyError(
            f"no prompt-protected variant of {base!r}; registered: {PP_BASES}"
        )
    return base


def effective_budget(
    budget: int, prompt_len: int, min_trace_slots: int = MIN_TRACE_SLOTS
) -> tuple[int, bool]:
    """(cache target, budget_relaxed) under the S.1 rule.

    The target is the nominal budget including the prompt. Only when the prompt
    leaves fewer than ``min_trace_slots`` trace slots is it raised, to
    ``prompt_len + min_trace_slots``, and the condition flagged.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if prompt_len < 0:
        raise ValueError("prompt_len cannot be negative")
    need = int(prompt_len) + int(min_trace_slots)
    return max(int(budget), need), need > int(budget)


def require_pp_config(config: dict[str, Any], *, replay: bool = False) -> None:
    """Refuse a config that is not an S.1 prompt-protected cell."""
    if config.get("prompt_protected") is not True:
        raise SystemExit("REFUSED: config must set prompt_protected: true")
    for name in config["presses"]:
        try:
            pp_base_name(str(name))
        except KeyError as exc:
            raise SystemExit(f"REFUSED: {exc.args[0]}") from None
    for key in ("compression_start_step", "matched_ratio", "restoration"):
        if key in config:
            raise SystemExit(
                f"REFUSED: {key} is not registered for prompt-protected cells"
            )
    if replay and config.get("mode") != "readout":
        raise SystemExit("REFUSED: prompt-protected replay is registered for readout")


@dataclass
class ProtectState:
    """What the press knows and has checked; read by the smoke gate."""

    base: str
    budget: int
    min_trace_slots: int = MIN_TRACE_SLOTS
    prompt_len: int | None = None
    target: int | None = None
    budget_relaxed: bool | None = None
    prompt_len_forced: bool = False
    n_prunes: int = 0
    maps: dict[int, Any] = field(default_factory=dict)
    logical: dict[int, int] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "budget": self.budget,
            "prompt_len": self.prompt_len,
            "prompt_len_forced": self.prompt_len_forced,
            "target": self.target,
            "budget_relaxed": self.budget_relaxed,
            "n_prunes": self.n_prunes,
            "n_layers_pruned": len(self.maps),
        }


def _set_prompt(press: Any, state: ProtectState, prompt_len: int) -> None:
    state.prompt_len = int(prompt_len)
    state.target, state.budget_relaxed = effective_budget(
        state.budget, state.prompt_len, state.min_trace_slots
    )
    press.target_size = state.target
    if state.base == "streamingllm":
        press.base_press.n_sink = state.prompt_len


def current_positions(state: ProtectState, layer: int, n_kv: int, k_len: int) -> Any:
    """Absolute position of every cache slot of ``layer``, shape [n_kv, k_len].

    Before the first prune the cache is in absolute order. Afterwards the slots
    are the previous prune's survivors (in gather order) followed by the
    entries appended since, whose positions continue the logical count.
    """
    import torch

    if layer not in state.maps:
        state.logical[layer] = k_len
        return torch.arange(k_len).unsqueeze(0).expand(n_kv, k_len).clone()
    prev = state.maps[layer]
    n_new = k_len - int(prev.shape[1])
    if n_new < 0:
        raise RuntimeError(
            f"layer {layer}: cache shrank outside the press "
            f"({int(prev.shape[1])} -> {k_len})"
        )
    start = state.logical[layer]
    new = torch.arange(start, start + n_new).unsqueeze(0).expand(n_kv, n_new)
    state.logical[layer] = start + n_new
    return torch.cat([prev, new], dim=1)


def check_retained(state: ProtectState, before: Any, after: Any, layer: int) -> None:
    """Raise unless the prune kept what S.1 says it keeps, in every KV head."""
    import torch

    prompt_len = int(state.prompt_len or 0)
    for head in range(int(after.shape[0])):
        kept = after[head]
        if int((kept < prompt_len).sum()) != prompt_len:
            raise RuntimeError(
                f"layer {layer} head {head}: a prompt position was evicted "
                f"(prune {state.n_prunes})"
            )
        if int(torch.unique(kept).numel()) != int(kept.numel()):
            raise RuntimeError(f"layer {layer} head {head}: duplicated position")
        if state.base == "streamingllm":
            trace = before[head][before[head] >= prompt_len]
            n_trace = int(kept.numel()) - prompt_len
            want = torch.sort(trace).values[int(trace.numel()) - n_trace :]
            got = torch.sort(kept[kept >= prompt_len]).values
            if not torch.equal(want, got):
                raise RuntimeError(
                    f"layer {layer} head {head}: retained trace entries are not "
                    "the most recent"
                )


def build_prompt_protected_press(
    name: str,
    *,
    budget: int,
    compression_interval: int = 64,
    seed: int | None = None,
    kvpress_module: Any | None = None,
    decoding_press_cls: type | None = None,
    instrumented: bool = False,
    forced_prompt_len: int | None = None,
    min_trace_slots: int = MIN_TRACE_SLOTS,
) -> Any:
    """A decoding press that never evicts the prompt (Amendment S.1).

    The signature is ``build_press``'s, so the unmodified runners can call it.
    ``forced_prompt_len`` exists for the smoke gate's identity control only: it
    fixes the protected length (0 for Random, 4 for StreamingLLM reproduce the
    unprotected press) instead of reading it from the prefill forward.
    """
    from kvcache.presses import build_press

    base = pp_base_name(name)
    press = build_press(
        base,
        budget=budget,
        compression_interval=compression_interval,
        seed=seed,
        kvpress_module=kvpress_module,
        decoding_press_cls=decoding_press_cls,
        instrumented=instrumented,
    )
    scorer = press.base_press
    if "compress" in vars(scorer):
        raise RuntimeError("the base scorer already carries a patched compress")
    state = ProtectState(base=base, budget=int(budget), min_trace_slots=min_trace_slots)
    press.protect = state
    if forced_prompt_len is not None:
        state.prompt_len_forced = True
        _set_prompt(press, state, int(forced_prompt_len))

    inner_hook = press.forward_hook

    def forward_hook(module: Any, input: Any, kwargs: dict, output: Any) -> Any:
        q_len = int(kwargs["hidden_states"].shape[1])
        last = int(kwargs["cache_position"][-1])
        prefill = last + 1 == q_len
        if state.prompt_len is None:
            if not prefill:
                raise RuntimeError(
                    "the first forward is not a pure prefill, so the prompt "
                    "length cannot be read (Amendment S.1)"
                )
            _set_prompt(press, state, q_len)
        elif prefill and q_len > 1 and not state.prompt_len_forced:
            if state.n_prunes or q_len != state.prompt_len:
                raise RuntimeError(
                    "a second prompt reached a used press; build one press per "
                    "sequence"
                )
        return inner_hook(module, input, kwargs, output)

    press.forward_hook = forward_hook
    inner_score = scorer.score

    def compress(
        module: Any,
        hidden_states: Any,
        keys: Any,
        values: Any,
        attentions: Any,
        kwargs: dict,
    ) -> tuple[Any, Any]:
        import torch

        if scorer.compression_ratio == 0:
            return keys, values
        if state.prompt_len is None:
            raise RuntimeError("prune before the prompt length was read")
        layer = int(module.layer_idx)
        if int(keys.shape[0]) != 1:
            raise RuntimeError("prompt protection is registered for batch size 1")
        n_kv, k_len = int(keys.shape[1]), int(keys.shape[2])
        before = current_positions(state, layer, n_kv, k_len)
        scores = inner_score(module, hidden_states, keys, values, attentions, kwargs)
        n_kept = int(k_len * (1 - scorer.compression_ratio))
        if n_kept < state.prompt_len:
            raise RuntimeError(
                f"target {n_kept} is smaller than the prompt ({state.prompt_len})"
            )
        protected = (before < state.prompt_len).to(scores.device)
        scores = scores.clone()
        scores[0][protected] = torch.finfo(scores.dtype).max
        indices = scores.topk(n_kept, dim=-1).indices
        if state.base == "streamingllm":
            # StreamingLLMPress scores by SLOT, so it needs slots in position
            # order. Tied top-k returns them that way on CUDA (observed, not
            # guaranteed); sorting makes it hold. The retained set is unchanged.
            indices = torch.sort(indices, dim=-1).values
        after = before.gather(1, indices[0].to(before.device))
        check_retained(state, before, after, layer)
        state.maps[layer] = after
        state.n_prunes += 1
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values

    scorer.compress = compress
    return press


# ------------------------------------------------------------------ smoke gate
def select_smoke_traces(
    candidates: list[dict[str, Any]], min_trace_tokens: int
) -> list[dict[str, Any]]:
    """The S.1 smoke traces: longest, median and shortest prompt.

    Among candidates with at least ``min_trace_tokens`` trace tokens, ordered
    by (prompt length, pid). The rule is fixed so the smoke cannot be steered
    toward traces on which the presses happen to behave.
    """
    pool = sorted(
        (c for c in candidates if int(c["trace_len"]) >= int(min_trace_tokens)),
        key=lambda c: (int(c["prompt_len"]), str(c["pid"])),
    )
    if len(pool) < 3:
        raise SystemExit("REFUSED: fewer than three traces are long enough to smoke")
    return [pool[-1], pool[len(pool) // 2], pool[0]]


def effective_prunes(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prune events that removed something (the press also logs no-op calls)."""
    return [e for e in events if int(e["before"]) > int(e["after"])]


def smoke_failures(records: list[dict[str, Any]]) -> list[str]:
    """Every registered S.1 smoke check that did not hold (empty = PASS)."""
    fails: list[str] = []
    if not records:
        return ["no smoke record"]
    for r in records:
        tag = f"{r['kind']} {r['press']} {r['pid']}"
        if "raised" in r:
            fails.append(f"{tag}: raised: {r['raised']}")
            continue
        protect = r["protect"]
        if protect["prompt_len"] != r["prompt_len"]:
            fails.append(
                f"{tag}: press read prompt length {protect['prompt_len']}, "
                f"true length {r['prompt_len']}"
            )
        if r["kind"] == "generation":
            if protect["n_prunes"] < 1:
                fails.append(f"{tag}: no prune in {r['n_tokens']} generated tokens")
            continue
        if protect["target"] != r["expected_target"]:
            fails.append(f"{tag}: target {protect['target']} != {r['expected_target']}")
        if r["first_effective_prune"] is None:
            fails.append(f"{tag}: no effective prune")
            continue
        if protect["n_layers_pruned"] != r["n_layers"]:
            fails.append(
                f"{tag}: pruned {protect['n_layers_pruned']} of {r['n_layers']} layers"
            )
        if r["pre_eviction_max_tv"] != 0.0:
            fails.append(f"{tag}: pre-eviction TV {r['pre_eviction_max_tv']} != 0")
        if r["post_prune_sizes"] != [r["expected_target"]]:
            fails.append(
                f"{tag}: post-prune sizes {r['post_prune_sizes']} != "
                f"[{r['expected_target']}]"
            )
        if r["identity_max_abs_diff"] != 0.0:
            fails.append(
                f"{tag}: identity control differs by {r['identity_max_abs_diff']}"
            )
        if not r["protection_max_abs_diff"] > 0.0:
            fails.append(f"{tag}: protection changed nothing")
    return fails

"""Replay-time capture for Meeting #3 Tasks 2 and 3 (Amendments R.4, R.6, R.9).

One teacher-forced replay feeds both instruments, because their grids are the
same (policy x budget on the Q.2 subset) and replay is the expensive part:

* **Q.3 attention channels** (``attn-profile-v2``). At every effective prune
  event: GQA collateral from the full-reference attention at the pruning
  forward; and at the first forward after it, per query head, ``M`` (full mass
  outside the compressed history), ``M_event`` (full mass on the positions
  evicted at this event) and ``Delta`` (TV between the compressed branch's own
  attention and the full attention renormalised onto the same absolute
  positions), then the unweighted mean over each KV group's query heads.
* **Q.6 latent split** (``latent-proj-v1``). At every decode forward, the
  displacement of each decoder block's output, standardised coordinate-wise by
  the frozen full-only basis, split into its component in and orthogonal to the
  basis span.

**Why the full branch runs once per trace.** The full-cache branch is identical
for every (policy, budget) condition on a trace, so it is replayed once
(``FullRecord``: per-step block outputs, queries at the steps an event can
touch, and the final cache, whose keys are final because a full cache is never
pruned) and each condition then replays only its compressed branch. Both runs
are the same stepwise computation as the paired replay, so the project's exact
controls still hold and are enforced live: before the first effective prune the
compressed block outputs must equal the full ones bitwise, and at every event
the compressed layer-0 keys at their mapped positions must equal the full
layer-0 keys bitwise (layer-0 keys depend only on token and position).

**Identity is by absolute position.** kvpress keeps entries in score order and
each KV head keeps its own set, so ``PositionTracker`` carries the absolute
position of every compressed-cache entry per (layer, KV head), updated from the
indices the policy actually selected. ``EventTap`` recovers those indices by
re-applying ``topk`` to the scores the policy returned and REFUSES unless the
result reproduces the compressed cache bitwise (gather equality).

torch is imported inside functions so the CPU-only gate never needs it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

ATTN_SCHEMA = "attn-profile-v2"
LATENT_SCHEMA = "latent-proj-v1"
N_SAMPLED_POSITIONS = 256
LAYER0_DELTA_TOL = 1e-5  # fp32 LSE renormalisation vs softmax on the same logits


def sampled_decode_positions(
    n_trace_tokens: int, k: int = N_SAMPLED_POSITIONS
) -> list[int]:
    """Up to ``k`` uniformly spaced decode forwards, endpoints included (R.6).

    Forward ``j`` (1 <= j <= T - 1) consumes generated token ``j - 1`` and
    predicts token ``j``; forward 0 is the prefill, whose prediction position is
    the prompt's last token and is excluded. Fixed from the trace length alone,
    so fitting and projection read the same positions.
    """
    last = int(n_trace_tokens) - 1
    if last < 1:
        return []
    n = min(int(k), last)
    if n == 1:
        return [1]
    return sorted({round(1 + i * (last - 1) / (n - 1)) for i in range(n)})


def armed_steps(step: int, interval: int) -> bool:
    """Whether the full branch must keep its queries at forward ``step``.

    With no onset gate, DecodingPress prunes at every ``interval``-th decode
    forward (step % interval == 0), and the natural measurement reads the forward
    after (step % interval == 1). Every effective prune is checked against this
    at capture time, so a mismatch is a loud error, never a silently missing
    event.
    """
    return step >= 1 and step % int(interval) in (0, 1)


# --------------------------------------------------------------------- positions
class PositionTracker:
    """Absolute positions of the compressed cache's entries, per layer and KV head."""

    def __init__(self, n_kv_heads: int) -> None:
        self.n_kv = int(n_kv_heads)
        self.maps: dict[int, Any] = {}
        self.next_abs: dict[int, int] = {}

    def current(self, layer: int, cache_len: int, device: Any = None) -> Any:
        """Positions of all ``cache_len`` entries, in cache order."""
        import torch

        m = self.maps.get(layer)
        if m is None:
            return torch.arange(cache_len, device=device).expand(self.n_kv, -1)
        n_new = int(cache_len) - int(m.shape[-1])
        if n_new < 0:
            raise RuntimeError(
                f"layer {layer}: cache has {cache_len} entries but {m.shape[-1]} "
                "are tracked; the cache shrank outside a recorded prune"
            )
        start = self.next_abs[layer]
        ext = torch.arange(start, start + n_new, device=m.device).expand(self.n_kv, -1)
        return torch.cat([m, ext], dim=-1)

    def prune(self, layer: int, cache_len: int, indices: Any) -> tuple[Any, Any]:
        """Apply one prune; returns (candidate positions, retained positions)."""
        cand = self.current(layer, cache_len, device=indices.device)
        m = self.maps.get(layer)
        self.next_abs[layer] = (
            int(cache_len)
            if m is None
            else self.next_abs[layer] + int(cache_len) - int(m.shape[-1])
        )
        kept = cand.gather(1, indices)
        self.maps[layer] = kept
        return cand, kept


# --------------------------------------------------------------------- event tap
@dataclass
class PruneEvent:
    step: int
    layer: int
    candidates: Any  # LongTensor [n_kv, k_len]  absolute positions before the prune
    evicted: Any  # LongTensor [n_kv, n_evicted] absolute positions removed


class EventTap:
    """Recover the policy's actual selection at every effective prune.

    Installed on an instrumented press (``press.event_tap``). It wraps the base
    scorer's ``score`` to keep the scores it returned, then after the prune
    re-applies ``topk`` and verifies the gathered keys equal the compressed
    cache bitwise. A policy whose selection cannot be reproduced this way is an
    error here rather than a silently wrong position map downstream.
    """

    def __init__(self, press: Any, tracker: PositionTracker) -> None:
        self.press = press
        self.tracker = tracker
        self.events: list[PruneEvent] = []
        self.listeners: list[Any] = []
        self._scores = None
        base = press.base_press
        original = base.score

        def score(*args: Any, **kwargs: Any) -> Any:
            s = original(*args, **kwargs)
            self._scores = s
            return s

        base.score = score
        press.event_tap = self

    def begin(self) -> None:
        # compress() returns early WITHOUT scoring when the cache is under
        # budget, so a stale tensor from the previous layer must never survive.
        self._scores = None

    def end(self, module: Any, keys_before: Any, keys_after: Any, step: Any) -> None:
        import torch

        k_len, n_kept = int(keys_before.shape[2]), int(keys_after.shape[2])
        if n_kept == k_len:
            return  # under budget: a prune event that removed nothing
        if self._scores is None:
            raise RuntimeError("event tap: cache shrank but no scores were captured")
        if step is None:
            raise RuntimeError("event tap: press has no profile_step")
        idx = self._scores.topk(n_kept, dim=-1).indices  # [b, n_kv, n_kept]
        gathered = keys_before.gather(
            2, idx.unsqueeze(-1).expand(-1, -1, -1, keys_before.shape[-1])
        )
        if not torch.equal(gathered, keys_after):
            raise RuntimeError(
                f"event tap: topk re-application does not reproduce the retained "
                f"cache at layer {module.layer_idx}, step {step}"
            )
        cand, _kept = self.tracker.prune(int(module.layer_idx), k_len, idx[0])
        mask = torch.ones_like(cand, dtype=torch.bool)
        mask.scatter_(1, idx[0], False)
        evicted = cand[mask].view(cand.shape[0], k_len - n_kept)
        ev = PruneEvent(int(step), int(module.layer_idx), cand, evicted)
        self.events.append(ev)
        for fn in self.listeners:
            fn(ev)


# --------------------------------------------------------------------- attention
def query_logits(module: Any, hidden_states: Any, cos_sin: Any, keys: Any) -> Any:
    """fp32 attention logits of the current single query over ``keys``.

    ``[n_heads, L]``. The query is recomputed from the attention module's own
    input exactly as Qwen2 computes it (q_proj, then rotary at the current
    position), then scored against every cached key of its KV group.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    n_heads = int(module.config.num_attention_heads)
    d = int(module.head_dim)
    q = module.q_proj(hidden_states[:, -1:]).view(1, 1, n_heads, d).transpose(1, 2)
    cos, sin = cos_sin
    q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
    return logits_from_query(q[0, :, 0].float(), keys, float(module.scaling))


def logits_from_query(q: Any, keys: Any, scaling: float) -> Any:
    """``[n_heads, L]`` logits for queries ``q [n_heads, d]`` and GQA ``keys``."""
    kf = keys[0].float()  # [n_kv, L, d]
    n_heads, n_kv = q.shape[0], kf.shape[0]
    rep = n_heads // n_kv
    qg = q.view(n_kv, rep, -1)  # head h -> group h // rep, as repeat_kv lays it out
    return (qg @ kf.transpose(1, 2)).reshape(n_heads, -1) * scaling


def channel_stats(
    full_logits: Any, comp_logits: Any, comp_positions: Any, evicted: Any
) -> dict[str, Any]:
    """Per-group ``M``, ``M_event``, ``Delta`` at one forward (R.4 convention).

    ``full_logits [H, Lf]`` index absolute positions directly (the full cache is
    never pruned). ``comp_logits [H, Lc]`` are in compressed-cache order and
    ``comp_positions [G, Lc]`` maps them to absolute positions. ``evicted
    [G, E]`` are the positions removed at this event. Per query head, then the
    unweighted mean over each group's heads. Normalisation is by log-sum-exp;
    a non-finite value is counted, never replaced.
    """
    import torch

    n_heads = full_logits.shape[0]
    n_groups = comp_positions.shape[0]
    rep = n_heads // n_groups
    grp = torch.arange(n_heads, device=full_logits.device) // rep
    pos = comp_positions[grp]  # [H, Lc]
    lse_all = torch.logsumexp(full_logits, dim=-1)  # [H]
    sub = full_logits.gather(1, pos)  # [H, Lc]
    lse_s = torch.logsumexp(sub, dim=-1)
    m_head = -torch.expm1(lse_s - lse_all)
    a_full_s = torch.softmax(sub, dim=-1)
    a_comp = torch.softmax(comp_logits, dim=-1)
    delta_head = 0.5 * (a_comp - a_full_s).abs().sum(-1)
    ev = evicted[grp]
    m_event_head = torch.exp(full_logits.gather(1, ev) - lse_all[:, None]).sum(-1)
    finite = (
        torch.isfinite(m_head)
        & torch.isfinite(delta_head)
        & torch.isfinite(m_event_head)
    )
    return {
        "M": m_head.view(n_groups, rep).mean(-1),
        "M_event": m_event_head.view(n_groups, rep).mean(-1),
        "Delta": delta_head.view(n_groups, rep).mean(-1),
        "Delta_head": delta_head,
        "n_nonfinite": int((~finite).sum()),
    }


def collateral_stats(
    full_logits: Any, candidates: Any, evicted: Any, *, budget: int, tau: float
) -> dict[str, Any]:
    """GQA collateral per group at one prune (R.4), from full-reference attention.

    ``full_logits [H, L]`` at the pruning forward; ``candidates [G, Lc]`` are the
    group's pre-prune positions, ``evicted [G, E]`` the ones removed. A position
    is collateral iff some head of the group gives it > ``tau`` of its full
    attention AND the group-mean attention ranks it outside the top ``budget``
    candidates (ties toward the lower position). Matches the pure-Python
    ``kvcache.metrics.attn_channels.collateral_event`` (parity-tested).
    """
    import torch

    n_heads = full_logits.shape[0]
    n_groups = candidates.shape[0]
    rep = n_heads // n_groups
    probs = torch.softmax(full_logits.double(), dim=-1)
    n_wanted, n_coll, n_elig = [], [], []
    for g in range(n_groups):
        p = probs[g * rep : (g + 1) * rep][:, candidates[g]]  # [rep, Lc]
        mean = p.mean(0)
        cand = candidates[g]
        # rank by (-mean, position): stable sort on position, then on -mean
        order = torch.argsort(cand, stable=True)
        order = order[torch.argsort(-mean[order], stable=True)]
        top = torch.zeros_like(cand, dtype=torch.bool)
        top[order[: int(budget)]] = True
        srt, perm = torch.sort(cand)
        loc = torch.searchsorted(srt, evicted[g])
        if bool((loc >= srt.numel()).any()) or not torch.equal(
            srt[loc.clamp(max=srt.numel() - 1)], evicted[g]
        ):
            raise RuntimeError("evicted positions not among the candidates")
        ev_idx = perm[loc]
        wanted = (p[:, ev_idx] > tau).any(0)
        n_elig.append(int(ev_idx.numel()))
        n_wanted.append(int(wanted.sum()))
        n_coll.append(int((wanted & ~top[ev_idx]).sum()))
    return {"n_eligible": n_elig, "n_wanted": n_wanted, "n_collateral": n_coll}


# --------------------------------------------------------------------- latent
def split_norms(delta: Any, scale: Any, basis: Any) -> tuple[Any, Any]:
    """``(||d_par||, ||d_perp||)`` per layer for standardised displacements.

    ``delta [n_layers, D]`` raw block-output differences; ``scale [n_layers, D]``
    coordinate-wise scales (the mean cancels in a difference); ``basis
    [n_layers, D, k]`` orthonormal columns.
    """
    import torch

    z = delta.float() / scale
    coords = torch.einsum("ld,ldk->lk", z, basis)
    par = torch.einsum("lk,ldk->ld", coords, basis)
    return coords.norm(dim=-1), (z - par).norm(dim=-1)


@dataclass
class LatentAccumulator:
    """Online per-layer omega sums for one (trace, condition) -- R.6 storage rule."""

    n_layers: int
    sampled: list[int]
    sum_omega: list[float] = field(default_factory=list)
    n_finite: list[int] = field(default_factory=list)
    n_undefined: list[int] = field(default_factory=list)
    n_unbounded: list[int] = field(default_factory=list)
    sum_par: list[float] = field(default_factory=list)
    sum_perp: list[float] = field(default_factory=list)
    sampled_par: list[list[float]] = field(default_factory=list)
    sampled_perp: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = self.n_layers
        self.sum_omega, self.sum_par, self.sum_perp = [0.0] * n, [0.0] * n, [0.0] * n
        self.n_finite, self.n_undefined, self.n_unbounded = [0] * n, [0] * n, [0] * n
        self._sampled = set(self.sampled)

    def add(self, step: int, par: Sequence[float], perp: Sequence[float]) -> None:
        from kvcache.metrics.latent_split import (
            OMEGA_FINITE,
            OMEGA_UNBOUNDED,
            omega_with_reason,
        )

        for layer in range(self.n_layers):
            p, q = float(par[layer]), float(perp[layer])
            self.sum_par[layer] += p
            self.sum_perp[layer] += q
            value, reason = omega_with_reason(p, q)
            if reason == OMEGA_FINITE:
                self.sum_omega[layer] += float(value)
                self.n_finite[layer] += 1
            elif reason == OMEGA_UNBOUNDED:
                self.n_unbounded[layer] += 1
            else:
                self.n_undefined[layer] += 1
        if step in self._sampled:
            self.sampled_par.append([float(x) for x in par])
            self.sampled_perp.append([float(x) for x in perp])


# --------------------------------------------------------------------- summaries
def omega_trace_value(acc: dict[str, Sequence[float]]) -> float | None:
    """One trace's omega: finite omega averaged over positions within layer,
    then equal layer weights (R.6). Layers with no finite omega are excluded;
    ``None`` when no layer has one."""
    vals = [
        s / n for s, n in zip(acc["sum_omega"], acc["n_finite"], strict=True) if n > 0
    ]
    return (sum(vals) / len(vals)) if vals else None


def pooled_fraction(num: Sequence[float], den: Sequence[float]) -> float | None:
    """Pooled ratio sum(num)/sum(den); ``None`` when nothing was eligible."""
    d = float(sum(den))
    return (float(sum(num)) / d) if d > 0 else None


def finite_or_none(x: float) -> float | None:
    return x if (x is not None and math.isfinite(x)) else None


# --------------------------------------------------------------------- drivers
def _block_output(output: Any) -> Any:
    """A decoder block's output hidden state (tensor or first tuple element)."""
    return output[0] if isinstance(output, tuple) else output


def _layers(model: Any) -> Any:
    return model.model.layers


@dataclass
class FullRecord:
    """The full-cache branch of one trace, replayed once and shared by conditions."""

    prompt_len: int
    n_forwards: int
    hidden: Any  # [n_forwards, n_layers, D] native dtype: block outputs per forward
    queries: dict[int, list[Any]]  # armed step -> per layer rotated q [H, d] fp32
    keys: list[Any]  # per layer final full keys [1, n_kv, P + n_forwards - 1, d]
    scaling: list[float]


def record_full_branch(
    model: Any, prompt_ids: Sequence[int], trace_ids: Sequence[int], *, interval: int
) -> FullRecord:
    """Stepwise full-cache replay with the replay loop's forward convention."""
    import torch

    from kvcache.replay import TransformersReplayBackend

    layers = _layers(model)
    n_layers = len(layers)
    n_fwd = len(trace_ids)  # forwards 0..T-1: the last token is never consumed
    d_model = int(model.config.hidden_size)
    with torch.inference_mode():
        hidden = torch.empty(
            (n_fwd, n_layers, d_model), dtype=model.dtype, device=model.device
        )
    queries: dict[int, list[Any]] = {}
    state = {"step": 0}
    handles = []

    def block_hook(i: int):
        def fn(_m: Any, _a: Any, out: Any) -> None:
            hidden[state["step"], i] = _block_output(out)[0, -1]

        return fn

    def attn_hook(i: int):
        def fn(module: Any, _a: Any, kwargs: dict, _out: Any) -> None:
            step = state["step"]
            if not armed_steps(step, interval):
                return
            from transformers.models.qwen2.modeling_qwen2 import (
                apply_rotary_pos_emb,
            )

            h = kwargs["hidden_states"]
            n_heads = int(module.config.num_attention_heads)
            q = module.q_proj(h[:, -1:]).view(1, 1, n_heads, -1).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            q, _ = apply_rotary_pos_emb(q, q, cos[:, -1:], sin[:, -1:])
            queries.setdefault(step, [None] * n_layers)[i] = q[0, :, 0].float()

        return fn

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(block_hook(i)))
        handles.append(
            layer.self_attn.register_forward_hook(attn_hook(i), with_kwargs=True)
        )
    backend = TransformersReplayBackend(model)
    cache = backend.cache_factory()
    try:
        with torch.inference_mode():
            backend.forward(list(prompt_ids), cache)
            for j in range(1, n_fwd):
                state["step"] = j
                backend.forward([int(trace_ids[j - 1])], cache)
    finally:
        for h in handles:
            h.remove()
    keys = [cache.layers[i].keys for i in range(n_layers)]
    return FullRecord(
        prompt_len=len(prompt_ids),
        n_forwards=n_fwd,
        hidden=hidden,
        queries=queries,
        keys=keys,
        scaling=[float(layer.self_attn.scaling) for layer in layers],
    )


def replay_condition(
    model: Any,
    press: Any,
    full: FullRecord,
    prompt_ids: Sequence[int],
    trace_ids: Sequence[int],
    *,
    budget: int | None,
    interval: int,
    tau: float,
    basis: dict[str, Any] | None = None,
    trigger_step: int = 130,
) -> dict[str, Any]:
    """Replay one condition's compressed branch against ``full``; return summaries.

    ``press=None`` is the full/full instrumentation-parity control: the second
    branch is another full cache, and at every read step Delta and M must be
    exactly 0 and every block output bitwise equal.
    """
    import torch

    from kvcache.replay import TransformersReplayBackend

    layers = _layers(model)
    n_layers = len(layers)
    n_kv = int(model.config.num_key_value_heads)
    tracker = PositionTracker(n_kv)
    tap = EventTap(press, tracker) if press is not None else None
    state = {"step": 0, "pending": {}}
    with torch.inference_mode():
        comp_hidden = torch.empty_like(full.hidden[0])
    ev_rows: list[dict[str, Any]] = []  # one per (event, layer)
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    checks = {"layer0_delta_max": 0.0, "n_nonfinite": 0, "fullfull_max": 0.0}
    P = full.prompt_len

    def full_logits(step: int, layer: int) -> Any:
        q = full.queries.get(step)
        if q is None or q[layer] is None:
            raise RuntimeError(f"full queries were not armed at step {step}")
        k = full.keys[layer][:, :, : P + step]
        return logits_from_query(q[layer], k, full.scaling[layer])

    def on_event(ev: PruneEvent) -> None:
        if ev.step % interval != 0:
            raise RuntimeError(f"effective prune at step {ev.step} is off the grid")
        col = collateral_stats(
            full_logits(ev.step, ev.layer),
            ev.candidates,
            ev.evicted,
            budget=int(budget),
            tau=tau,
        )
        row = {"step": ev.step, "layer": ev.layer, **col}
        by_key[(ev.step, ev.layer)] = row
        ev_rows.append(row)
        state["pending"][ev.layer] = ev

    if tap is not None:
        tap.listeners.append(on_event)

    def block_hook(i: int):
        def fn(_m: Any, _a: Any, out: Any) -> None:
            comp_hidden[i] = _block_output(out)[0, -1]

        return fn

    def attn_hook(i: int):
        def fn(module: Any, _a: Any, kwargs: dict, _out: Any) -> None:
            step = state["step"]
            cache = kwargs["past_key_values"]
            if press is None:
                if step % interval != 1 or step < 1:
                    return
                keys = cache.layers[i].keys
                comp = query_logits(
                    module, kwargs["hidden_states"], kwargs["position_embeddings"], keys
                )
                ref = full_logits(step, i)
                checks["fullfull_max"] = max(
                    checks["fullfull_max"], float((comp - ref).abs().max())
                )
                return
            ev = state["pending"].get(i)
            if ev is None or ev.step != step - 1:
                return
            keys = cache.layers[i].keys
            positions = tracker.current(i, int(keys.shape[2]), device=keys.device)
            if i == 0:
                ref_k = full.keys[0][0].gather(
                    1, positions.unsqueeze(-1).expand(-1, -1, keys.shape[-1])
                )
                if not torch.equal(ref_k, keys[0]):
                    raise RuntimeError(
                        f"layer-0 keys at mapped positions differ from the full "
                        f"cache at step {step}: the position map is wrong"
                    )
            comp = query_logits(
                module, kwargs["hidden_states"], kwargs["position_embeddings"], keys
            )
            st = channel_stats(full_logits(step, i), comp, positions, ev.evicted)
            checks["n_nonfinite"] += st["n_nonfinite"]
            if i == 0:
                checks["layer0_delta_max"] = max(
                    checks["layer0_delta_max"], float(st["Delta_head"].max())
                )
            row = by_key[(ev.step, i)]
            row["M"] = st["M"].tolist()
            row["M_event"] = st["M_event"].tolist()
            row["Delta"] = st["Delta"].tolist()
            del state["pending"][i]

        return fn

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(block_hook(i)))
        handles.append(
            layer.self_attn.register_forward_hook(attn_hook(i), with_kwargs=True)
        )
    sampled = sampled_decode_positions(len(trace_ids))
    acc = LatentAccumulator(n_layers, sampled) if basis is not None else None
    backend = TransformersReplayBackend(model)
    cache = backend.cache_factory()
    press_ctx = press(model) if press is not None else _null()
    try:
        with torch.inference_mode(), press_ctx:
            if press is not None:
                press.enabled = True
            for j in range(full.n_forwards):
                state["step"] = j
                if press is not None:
                    press.profile_step = j
                n_events_before = len(tap.events) if tap is not None else 0
                if j == 0:
                    backend.forward(list(prompt_ids), cache)
                else:
                    backend.forward([int(trace_ids[j - 1])], cache)
                if n_events_before == 0 and not torch.equal(
                    comp_hidden, full.hidden[j]
                ):
                    raise RuntimeError(
                        f"compressed block outputs differ from the full branch at "
                        f"forward {j}, before any eviction: instrument defect"
                    )
                if acc is not None and j >= 1:
                    delta = comp_hidden.float() - full.hidden[j].float()
                    par, perp = split_norms(delta, basis["scale"], basis["basis"])
                    acc.add(j, par.tolist(), perp.tolist())
    finally:
        for h in handles:
            h.remove()
        if press is not None:
            press.enabled = False
    return _summarise(ev_rows, checks, acc, n_layers, n_kv, trigger_step)


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _summarise(
    ev_rows: list[dict[str, Any]],
    checks: dict[str, Any],
    acc: LatentAccumulator | None,
    n_layers: int,
    n_kv: int,
    trigger_step: int,
) -> dict[str, Any]:
    """Compact per-(trace, condition) output (R.9 storage rule)."""
    measured = [r for r in ev_rows if "M" in r]
    steps = sorted({r["step"] for r in ev_rows})
    lg = {f: [[0.0] * n_kv for _ in range(n_layers)] for f in ("M", "M_event", "Delta")}
    n_meas = [[0] * n_kv for _ in range(n_layers)]
    cnt = {
        f: [[0] * n_kv for _ in range(n_layers)]
        for f in ("n_eligible", "n_wanted", "n_collateral")
    }
    for r in ev_rows:
        for f in cnt:
            for g in range(n_kv):
                cnt[f][r["layer"]][g] += int(r[f][g])
    for r in measured:
        for g in range(n_kv):
            n_meas[r["layer"]][g] += 1
            for f in lg:
                lg[f][r["layer"]][g] += float(r[f][g])
    per_event = []
    for s in steps:
        rs = [r for r in measured if r["step"] == s]
        allr = [r for r in ev_rows if r["step"] == s]
        vals = {
            f: (sum(sum(r[f]) for r in rs) / (len(rs) * n_kv)) if rs else None
            for f in ("M", "M_event", "Delta")
        }
        per_event.append(
            {
                "step": s,
                **vals,
                **{f: sum(sum(r[f]) for r in allr) for f in cnt},
            }
        )
    trig = next((s for s in steps if s >= trigger_step), None)
    slices = {}
    for label, s in (
        ("first", steps[0] if steps else None),
        ("second", steps[1] if len(steps) > 1 else None),
        ("trigger", trig),
    ):
        if s is None:
            continue
        rs = sorted((r for r in ev_rows if r["step"] == s), key=lambda r: r["layer"])
        slices[label] = {
            "step": s,
            **{
                f: [r.get(f) for r in rs]
                for f in ("M", "M_event", "Delta", *cnt.keys())
            },
        }
    n_elig = sum(sum(x) for x in cnt["n_eligible"])
    out: dict[str, Any] = {
        "attn_schema": ATTN_SCHEMA,
        "n_events": len(steps),
        "event_steps": steps,
        "n_measured": sum(sum(x) for x in n_meas),
        "sum_by_layer_group": lg,
        "n_measured_by_layer_group": n_meas,
        "counts_by_layer_group": cnt,
        "per_event": per_event,
        "slices": slices,
        "collateral_pooled": pooled_fraction(
            [sum(x) for x in cnt["n_collateral"]], [sum(x) for x in cnt["n_eligible"]]
        ),
        "wanted_any_head_pooled": pooled_fraction(
            [sum(x) for x in cnt["n_wanted"]], [sum(x) for x in cnt["n_eligible"]]
        ),
        "n_evicted_eligible": n_elig,
        "M_mean": _mean_of(lg["M"], n_meas),
        "M_event_mean": _mean_of(lg["M_event"], n_meas),
        "Delta_mean": _mean_of(lg["Delta"], n_meas),
        "checks": checks,
        "first_prune_delta_max": "NOT APPLICABLE (R.4: natural measurement)",
    }
    if acc is not None:
        lat = {
            "latent_schema": LATENT_SCHEMA,
            "sum_omega": acc.sum_omega,
            "n_finite": acc.n_finite,
            "n_undefined": acc.n_undefined,
            "n_unbounded": acc.n_unbounded,
            "sum_par": acc.sum_par,
            "sum_perp": acc.sum_perp,
            "sampled_positions": acc.sampled,
        }
        lat["omega_trace"] = omega_trace_value(lat)
        out["latent"] = lat
        out["latent_sampled"] = {"par": acc.sampled_par, "perp": acc.sampled_perp}
    return out


def _mean_of(sums: list[list[float]], counts: list[list[int]]) -> float | None:
    n = sum(sum(c) for c in counts)
    return (sum(sum(s) for s in sums) / n) if n else None


# --------------------------------------------------------------------- basis
def validate_basis_arrays(
    mean: Any, scale: Any, basis: Any, *, tol: float = 1e-6
) -> dict[str, Any]:
    """The FrozenBasis checks, vectorised for [n_layers, D, k] arrays (R.6).

    Finite everywhere, positive scale, orthonormal columns to ``tol`` (checked
    in float64 on the stored values), full rank k. Raises on any failure.
    """
    import numpy as np

    for name, arr in (("mean", mean), ("scale", scale), ("basis", basis)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"basis artifact: non-finite values in {name}")
    if np.any(np.asarray(scale) <= 0):
        raise ValueError("basis artifact: non-positive scale")
    b = np.asarray(basis, dtype=np.float64)
    gram = np.einsum("ldk,ldj->lkj", b, b)
    err = np.abs(gram - np.eye(b.shape[-1])[None]).max(axis=(1, 2))
    if float(err.max()) > tol:
        raise ValueError(
            f"basis artifact: not orthonormal (max |U^T U - I| = {err.max():.3e} "
            f"> {tol:g}, layer {int(err.argmax())})"
        )
    return {"max_orthonormality_error": float(err.max()), "k": int(b.shape[-1])}


class MomentAccumulator:
    """Weighted float64 moments per layer on CPU for the full-only fit (R.6)."""

    def __init__(self, n_layers: int, dim: int) -> None:
        import numpy as np

        self.w = 0.0
        self.s1 = np.zeros((n_layers, dim))
        self.s2 = np.zeros((n_layers, dim))
        self.sxx = np.zeros((n_layers, dim, dim))

    def add(self, x: Any, weight: float) -> None:
        """``x [n_layers, n, D]`` samples, each carrying ``weight``."""
        import numpy as np

        x = np.asarray(x, dtype=np.float64)
        self.w += weight * x.shape[1]
        self.s1 += weight * x.sum(axis=1)
        self.s2 += weight * (x * x).sum(axis=1)
        for i in range(x.shape[0]):
            self.sxx[i] += weight * (x[i].T @ x[i])

    def solve(self, k: int) -> dict[str, Any]:
        """Coordinate-wise standardisation, then exact per-layer eigh."""
        import numpy as np

        mean = self.s1 / self.w
        var = self.s2 / self.w - mean**2
        if np.any(var <= 0):
            raise ValueError("fit: a coordinate has zero variance; cannot standardise")
        scale = np.sqrt(var)
        n_layers, dim = mean.shape
        basis = np.zeros((n_layers, dim, k))
        eig = np.zeros((n_layers, k))
        explained = np.zeros(n_layers)
        for i in range(n_layers):
            cov = self.sxx[i] / self.w - np.outer(mean[i], mean[i])
            cov = cov / np.outer(scale[i], scale[i])
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]
            usable = int((vals > 1e-12 * max(vals[0], 1e-300)).sum())
            if usable < k:
                raise ValueError(
                    f"fit: layer {i} has {usable} usable directions (< k={k})"
                )
            basis[i], eig[i] = vecs[:, :k], vals[:k]
            explained[i] = float(vals[:k].sum() / vals[vals > 0].sum())
        return {
            "mean": mean,
            "scale": scale,
            "basis": basis,
            "eigenvalues": eig,
            "explained_fraction": explained,
        }

"""Amendment R.17.2.1: capture-v2 = capture-v1 plus per-head M/Delta and nested k.

capture-v1 (``kvcache.capture``) averages M, M_event and Delta over each KV
group's seven query heads before storing them, and keeps the latent split only
as norms at k = 64. Two registered reads need more: the within-group dispersion
of M (R.17.1.6) and the k-sensitivity of omega (R.17.1.7). This module adds
exactly those, reduced in the same pass, and nothing else:

* per (layer, query head) sums over events of M, M_event and Delta, plus the
  per-layer count of measured events;
* per (layer, k) sums of ||d_par_k||, ||d_perp_k|| and finite omega_k (with
  the R.6 singular-case counts) for nested k, from a k = 256 refit basis whose
  leading 64 directions must reproduce the frozen basis (acceptance checked by
  the runner). The registered omega stays the v1 quantity on the frozen basis.

``replay_condition_v2`` is ``capture.replay_condition`` copied verbatim with the
additions spliced in, because v1 accumulates inside closures that cannot be
wrapped without guessing which layer a call belongs to. ``capture.py`` itself
is NOT modified: its bytes are the identity of the finished b256 rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kvcache.capture import (
    EventTap,
    FullRecord,
    LatentAccumulator,
    PositionTracker,
    PruneEvent,
    _block_output,
    _layers,
    _null,
    _summarise,
    channel_stats,
    collateral_stats,
    logits_from_query,
    query_logits,
    sampled_decode_positions,
    split_norms,
)

CAPTURE_V2_SCHEMA = "capture-v2"
NESTED_KS = (8, 16, 32, 64, 128, 256)


def head_stats(full_logits: Any, comp_positions: Any, evicted: Any) -> dict[str, Any]:
    """Per query head M and M_event -- the values ``channel_stats`` averages.

    Same operations in the same order as ``channel_stats``: per event, the
    group mean of these equals its group value exactly. Summed over events the
    two orders agree to float32 rounding (tested at 1e-6 relative).
    """
    import torch

    n_heads = full_logits.shape[0]
    n_groups = comp_positions.shape[0]
    rep = n_heads // n_groups
    grp = torch.arange(n_heads, device=full_logits.device) // rep
    pos = comp_positions[grp]
    lse_all = torch.logsumexp(full_logits, dim=-1)
    sub = full_logits.gather(1, pos)
    lse_s = torch.logsumexp(sub, dim=-1)
    m_head = -torch.expm1(lse_s - lse_all)
    ev = evicted[grp]
    m_event_head = torch.exp(full_logits.gather(1, ev) - lse_all[:, None]).sum(-1)
    return {"M_head": m_head, "M_event_head": m_event_head}


def summarise_heads(ev_rows: list[dict[str, Any]], n_layers: int) -> dict[str, Any]:
    """Per (layer, head) sums over measured events, columnar."""
    measured = [r for r in ev_rows if "M_head" in r]
    n_heads = len(measured[0]["M_head"]) if measured else 0
    sums = {
        f: [[0.0] * n_heads for _ in range(n_layers)]
        for f in ("M_head", "M_event_head", "Delta_head")
    }
    n_meas = [0] * n_layers
    for r in measured:
        n_meas[r["layer"]] += 1
        for f in sums:
            for h in range(n_heads):
                sums[f][r["layer"]][h] += float(r[f][h])
    return {
        "head_schema": "per-layer-per-query-head-sums-v1",
        "sum_by_layer_head": sums,
        "n_measured_by_layer": n_meas,
    }


def split_norms_nested(
    delta: Any, scale: Any, basis: Any, ks: Sequence[int]
) -> tuple[Any, Any]:
    """``(||d_par_k||, ||d_perp_k||)`` as [n_layers, len(ks)] for nested k.

    Computed directly, like ``capture.split_norms`` (perp = ||z - P_k z||, not
    a subtraction of squared norms), from the leading k columns of ``basis``.
    """
    import torch

    z = delta.float() / scale
    coords = torch.einsum("ld,ldk->lk", z, basis)
    par, perp = [], []
    for k in ks:
        c = coords[:, :k]
        proj = torch.einsum("lk,ldk->ld", c, basis[:, :, :k])
        par.append(c.norm(dim=-1))
        perp.append((z - proj).norm(dim=-1))
    return torch.stack(par, dim=-1), torch.stack(perp, dim=-1)


@dataclass
class NestedAccumulator:
    """Per (layer, k) sums with the R.6 omega rule, for every step."""

    n_layers: int
    ks: list[int]
    sum_par: list[list[float]] = field(default_factory=list)
    sum_perp: list[list[float]] = field(default_factory=list)
    sum_omega: list[list[float]] = field(default_factory=list)
    n_finite: list[list[int]] = field(default_factory=list)
    n_undefined: list[list[int]] = field(default_factory=list)
    n_unbounded: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        z = [[0.0] * len(self.ks) for _ in range(self.n_layers)]
        self.sum_par, self.sum_perp, self.sum_omega = (
            [r[:] for r in z],
            [r[:] for r in z],
            [r[:] for r in z],
        )
        n = [[0] * len(self.ks) for _ in range(self.n_layers)]
        self.n_finite, self.n_undefined, self.n_unbounded = (
            [r[:] for r in n],
            [r[:] for r in n],
            [r[:] for r in n],
        )

    def add(self, par: Sequence[Sequence[float]], perp: Sequence[Sequence[float]]):
        from kvcache.metrics.latent_split import (
            OMEGA_FINITE,
            OMEGA_UNBOUNDED,
            omega_with_reason,
        )

        for layer in range(self.n_layers):
            for i in range(len(self.ks)):
                p, q = float(par[layer][i]), float(perp[layer][i])
                self.sum_par[layer][i] += p
                self.sum_perp[layer][i] += q
                value, reason = omega_with_reason(p, q)
                if reason == OMEGA_FINITE:
                    self.sum_omega[layer][i] += float(value)
                    self.n_finite[layer][i] += 1
                elif reason == OMEGA_UNBOUNDED:
                    self.n_unbounded[layer][i] += 1
                else:
                    self.n_undefined[layer][i] += 1

    def summary(self) -> dict[str, Any]:
        return {
            "ks": list(self.ks),
            "sum_par": self.sum_par,
            "sum_perp": self.sum_perp,
            "sum_omega": self.sum_omega,
            "n_finite": self.n_finite,
            "n_undefined": self.n_undefined,
            "n_unbounded": self.n_unbounded,
        }


def basis_acceptance(
    frozen: dict[str, Any], nested: dict[str, Any], *, k: int = 64
) -> dict[str, Any]:
    """R.17.2.1 acceptance of the k = 256 refit against the frozen k = 64 fit.

    mean and scale must agree within 1e-6 relative, and every principal angle
    between the refit's leading k columns and the frozen basis must be <= 1e-3
    rad, in every layer. numpy arrays in, a verdict out.
    """
    import numpy as np

    def rel(a: Any, b: Any) -> float:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))

    mean_rel = rel(nested["mean"], frozen["mean"])
    scale_rel = rel(nested["scale"], frozen["scale"])
    fb = np.asarray(frozen["basis"], dtype=np.float64)
    nb = np.asarray(nested["basis"], dtype=np.float64)[:, :, :k]
    worst = 0.0
    for layer in range(fb.shape[0]):
        s = np.linalg.svd(fb[layer].T @ nb[layer], compute_uv=False)
        ang = float(np.arccos(np.clip(s.min(), -1.0, 1.0)))
        worst = max(worst, ang)
    passed = mean_rel <= 1e-6 and scale_rel <= 1e-6 and worst <= 1e-3
    return {
        "passed": bool(passed),
        "mean_max_rel_diff": mean_rel,
        "scale_max_rel_diff": scale_rel,
        "max_principal_angle_rad": worst,
        "k_compared": k,
        "n_layers": int(fb.shape[0]),
        "nested_k_max": int(np.asarray(nested["basis"]).shape[-1]),
    }


def replay_condition_v2(
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
    nested_basis: dict[str, Any] | None = None,
    nested_ks: Sequence[int] = NESTED_KS,
) -> dict[str, Any]:
    """``capture.replay_condition`` plus the R.17.2.1 per-head and nested-k sums.

    Every v1 field is computed by the v1 code, copied verbatim; the additions
    only read values v1 already computed (tested: v1 fields are identical).

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
            fl_i = full_logits(step, i)
            st = channel_stats(fl_i, comp, positions, ev.evicted)
            hs = head_stats(fl_i, positions, ev.evicted)
            checks["n_nonfinite"] += st["n_nonfinite"]
            if i == 0:
                checks["layer0_delta_max"] = max(
                    checks["layer0_delta_max"], float(st["Delta_head"].max())
                )
            row = by_key[(ev.step, i)]
            row["M"] = st["M"].tolist()
            row["M_event"] = st["M_event"].tolist()
            row["Delta"] = st["Delta"].tolist()
            row["M_head"] = hs["M_head"].tolist()
            row["M_event_head"] = hs["M_event_head"].tolist()
            row["Delta_head"] = st["Delta_head"].tolist()
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
    nacc = (
        NestedAccumulator(n_layers, list(nested_ks))
        if nested_basis is not None and basis is not None
        else None
    )
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
                    if nacc is not None:
                        npar, nperp = split_norms_nested(
                            delta,
                            nested_basis["scale"],
                            nested_basis["basis"],
                            nacc.ks,
                        )
                        nacc.add(npar.tolist(), nperp.tolist())
    finally:
        for h in handles:
            h.remove()
        if press is not None:
            press.enabled = False
    out = _summarise(ev_rows, checks, acc, n_layers, n_kv, trigger_step)
    out.update(summarise_heads(ev_rows, n_layers))
    if nacc is not None:
        out["latent_nested"] = nacc.summary()
    return out

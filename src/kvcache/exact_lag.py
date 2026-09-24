"""Amendment R.5: identity-repaired restoration with an exact-lag protocol.

Replaces the unexecuted Q.5 sweep, whose press stashed the tail of a
score-ordered tensor (not the evicted complement), shared its stash across
layers, and could not realize most requested lags.

**Identity repair.** ``kvcache.capture.EventTap`` (``stash_kv=True``) keeps, per
layer, the true evicted complement of K and V at every effective prune, taken
from the pre-compression tensors at the indices the policy actually selected
(topk re-applied and verified by gather equality), in eviction order.
``verify_stash`` then checks K AND V bitwise against the full cache: every
layer-0 entry (layer-0 K/V depend only on token and position), and in every
layer every entry computed before the first effective prune. Reinsertion is
gated on keys AND values, and each arm's gate report is recorded in the row.

**Exact lag.** The cohort is the first effective prune at or after decode step
``trigger_step`` (it is the same forward in every layer; checked). One paired
replay (full + compressed, no intervention = the ``none`` arm) runs to
``e + max(lags) + window``, snapshotting the compressed state immediately before
forward ``e + lag`` for each lag. Each intervention arm restores that snapshot,
applies the P.5 surgery (``plan_reinsertion`` / ``apply_reinsertion``, gated
bitwise by ``retained_set_report``), and replays ``window`` forwards. So
reinsertion lands at exactly ``e + lag`` whether or not that is a pruning step,
and all arms of a lag share the pre-intervention state. The first prediction
that sees reinsertion is forward ``e + lag``; the read is the mean per-step TV
over ``[e + lag, e + lag + window)``, and the effect is ``none`` minus arm.

Absolute read position moves with the lag, so this is a recovery-after-delay
experiment and does not isolate staleness from position (R.5).
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from typing import Any

LAGS = (1, 4, 16, 64, 256)
INTERVENTION_ARMS = ("matched", "random", "relaxed")


def _forward(model: Any, cache: Any, pos: int, token_ids: Sequence[int]) -> Any:
    import torch

    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=model.device)
    cp = torch.arange(pos, pos + len(token_ids), dtype=torch.long, device=model.device)
    out = model(input_ids=ids, past_key_values=cache, cache_position=cp, use_cache=True)
    return out.logits[0, -1].float()


def _tv(full: Any, comp: Any) -> float:
    import torch

    p = torch.softmax(full, dim=-1)
    q = torch.softmax(comp, dim=-1)
    return min(1.0, float(0.5 * (p - q).abs().sum()))


def _press_state(press: Any) -> dict[str, Any]:
    return {
        "layer_step_counts": copy.deepcopy(press.layer_step_counts),
        "hidden_states_buffer": copy.deepcopy(press.hidden_states_buffer),
    }


def _restore_press(press: Any, state: dict[str, Any]) -> None:
    press.layer_step_counts = copy.deepcopy(state["layer_step_counts"])
    press.hidden_states_buffer = copy.deepcopy(state["hidden_states_buffer"])


def reinsert(
    cache: Any,
    stash: dict[int, list[tuple[int, Any, Any, Any]]],
    *,
    arm: str,
    n_reinsert: int,
    rng_key: str,
) -> dict[str, Any]:
    """Apply one P.5 arm to every (layer, KV head); return the gate report."""
    import torch

    from kvcache.metrics.restoration import (
        apply_reinsertion,
        plan_reinsertion,
        retained_set_report,
    )

    passed, n_layers = True, 0
    for layer, entries in sorted(stash.items()):
        ek = torch.cat([e[2] for e in entries], dim=1)  # [n_kv, S, d]
        ev = torch.cat([e[3] for e in entries], dim=1)
        keys, values = cache.layers[layer].keys, cache.layers[layer].values
        new_k, new_v = [], []
        for h in range(keys.shape[1]):
            plan = plan_reinsertion(
                evicted_order=list(range(ek.shape[1])),
                retained=list(range(keys.shape[2])),
                n_reinsert=n_reinsert,
                arm=arm,
                rng=random.Random(f"{rng_key}|{layer}|{h}|reinsert"),
                evict_rng=random.Random(f"{rng_key}|{layer}|{h}|evict"),
            )
            k_h, v_h = apply_reinsertion(
                keys[:, h : h + 1],
                values[:, h : h + 1],
                evicted_keys=ek[h][None, None],
                evicted_values=ev[h][None, None],
                reinsert_local=plan["reinsert"],
                evict_local=plan["evict"],
            )
            idx = torch.tensor(plan["reinsert"], device=ek.device, dtype=torch.long)
            gate = retained_set_report(
                k_h, ek[h].index_select(0, idx)[None, None], n_reinserted=n_reinsert
            )
            # values too: a surgery that moved the right keys but corrupted
            # their values would otherwise pass (review of 4e38224)
            vgate = retained_set_report(
                v_h, ev[h].index_select(0, idx)[None, None], n_reinserted=n_reinsert
            )
            passed = passed and bool(gate["passed"]) and bool(vgate["passed"])
            new_k.append(k_h)
            new_v.append(v_h)
        cache.layers[layer].keys = torch.cat(new_k, dim=1)
        cache.layers[layer].values = torch.cat(new_v, dim=1)
        n_layers += 1
    return {
        "passed": passed,
        "n_layers": n_layers,
        "n_heads_checked": n_layers * int(keys.shape[1]),
        "checked": "reinserted keys AND values bitwise equal the stash, per head",
    }


def verify_stash(
    stash: dict[int, list[tuple[int, Any, Any, Any]]],
    full_cache: Any,
    clean_before: int,
) -> dict[str, int]:
    """Check the stashed K AND V against the full cache, in every layer.

    Layer 0's keys and values depend only on token and position, so every
    layer-0 stash entry must equal the full cache bitwise. In EVERY layer, an
    entry whose absolute position precedes ``clean_before`` (prompt + the first
    effective prune: computed before any eviction could alter a hidden state)
    must also equal the full cache bitwise, K and V. A corrupted value at a deep
    layer is therefore caught, not only a wrong layer-0 key.
    """
    import torch

    n_checked = 0
    for layer, entries in stash.items():
        fk = full_cache.layers[layer].keys[0]
        fv = full_cache.layers[layer].values[0]
        d = fk.shape[-1]
        for _s, pos, k_ev, v_ev in entries:
            sel = (
                pos < clean_before if layer else torch.ones_like(pos, dtype=torch.bool)
            )
            if not bool(sel.any()):
                continue
            idx = pos.unsqueeze(-1).expand(-1, -1, d)
            gk, gv = fk.gather(1, idx), fv.gather(1, idx)
            m = sel.unsqueeze(-1).expand_as(gk)
            if not torch.equal(gk[m], k_ev[m]):
                raise RuntimeError(
                    f"stashed keys differ from full keys (layer {layer})"
                )
            if not torch.equal(gv[m], v_ev[m]):
                raise RuntimeError(
                    f"stashed values differ from full values (layer {layer})"
                )
            n_checked += int(sel.sum())
    return {"n_entries_checked": n_checked, "clean_before_position": clean_before}


def run_exact_lag(
    model: Any,
    press: Any,
    prompt_ids: Sequence[int],
    trace_ids: Sequence[int],
    *,
    n_reinsert: int,
    rng_key: str,
    trigger_step: int = 130,
    window: int = 200,
    lags: Sequence[int] = LAGS,
) -> dict[str, Any]:
    """One (trace, policy): cohort, the none arm, and every arm at every lag."""
    import torch
    from transformers import DynamicCache

    from kvcache.capture import EventTap, PositionTracker

    n_kv = int(model.config.num_key_value_heads)
    tap = EventTap(press, PositionTracker(n_kv), stash_kv=True)
    n_fwd = len(trace_ids)
    cohort: dict[str, Any] = {"step": None}

    def on_event(ev: Any) -> None:
        if cohort["step"] is None and ev.step >= trigger_step:
            cohort["step"] = ev.step

    tap.listeners.append(on_event)
    full_cache, comp_cache = DynamicCache(), DynamicCache()
    none_tv: dict[int, float] = {}
    full_logits: dict[int, Any] = {}
    snaps: dict[int, Any] = {}
    horizon = n_fwd - 1
    first_effective = None
    identity_checks: dict[str, int] | None = None
    with torch.inference_mode(), press(model):
        for j in range(n_fwd):
            ids = list(prompt_ids) if j == 0 else [int(trace_ids[j - 1])]
            pos = 0 if j == 0 else len(prompt_ids) + j - 1
            e = cohort["step"]
            if e is not None and j - e in set(lags):
                snaps[j - e] = (
                    copy.deepcopy(comp_cache),
                    _press_state(press),
                    pos,
                )
            press.profile_step = j
            press.enabled = False
            fl = _forward(model, full_cache, pos, ids)
            press.enabled = True
            n_before = len(tap.events)
            cl = _forward(model, comp_cache, pos, ids)
            press.enabled = False
            tv = _tv(fl, cl)
            if first_effective is None and len(tap.events) > n_before:
                first_effective = j
            if first_effective is None and tv > 1e-6:
                raise RuntimeError(f"diverged before the first eviction at step {j}")
            none_tv[j] = tv
            if e is None and cohort["step"] is not None:
                e = cohort["step"]
                if j != e:
                    raise RuntimeError("cohort event not at the current forward")
                steps = {s for layer in tap.stash.values() for s, *_ in layer[-1:]}
                if len(tap.stash) != len(model.model.layers) or steps != {e}:
                    raise RuntimeError("cohort event is not present in every layer")
                identity_checks = verify_stash(
                    tap.stash, full_cache, len(prompt_ids) + first_effective
                )
                press.event_tap = None  # the stash is frozen at the cohort
                horizon = min(n_fwd - 1, e + max(lags) + window - 1)
            if e is not None and j > e:
                full_logits[j] = fl
            if cohort["step"] is not None and j >= horizon:
                break
        e = cohort["step"]
        out: dict[str, Any] = {
            "stash_identity_checks": identity_checks if e is not None else None,
            "cohort_step": e,
            "first_effective_prune": first_effective,
            "n_trace_tokens": n_fwd,
            "n_reinsert": n_reinsert,
            "lags": {},
        }
        if e is None:
            out["status"] = "INELIGIBLE: no eviction at or after the trigger"
            return out
        stash = tap.stash
        out["stash_size"] = min(sum(x[1].shape[1] for x in v) for v in stash.values())
        if out["stash_size"] < n_reinsert:
            out["status"] = "INELIGIBLE: stash smaller than the reinsertion count"
            return out
        out["status"] = "READ"
        for lag in lags:
            start = e + lag
            if start + window - 1 > n_fwd - 1 or lag not in snaps:
                out["lags"][str(lag)] = {"eligible": False}
                continue
            res: dict[str, Any] = {
                "eligible": True,
                "requested_lag": lag,
                "achieved_lag": lag,
                "applied_step": start,
                "none": sum(none_tv[j] for j in range(start, start + window)) / window,
            }
            for arm in INTERVENTION_ARMS:
                cache0, pstate, _pos = snaps[lag]
                cache = copy.deepcopy(cache0)
                _restore_press(press, pstate)
                gate = reinsert(
                    cache, stash, arm=arm, n_reinsert=n_reinsert, rng_key=rng_key
                )
                if not gate["passed"]:
                    raise RuntimeError(f"retained-set gate failed: {arm} lag {lag}")
                res.setdefault("gates", {})[arm] = gate
                tvs = []
                for j in range(start, start + window):
                    press.profile_step = j
                    press.enabled = True
                    cl = _forward(
                        model, cache, len(prompt_ids) + j - 1, [int(trace_ids[j - 1])]
                    )
                    press.enabled = False
                    tvs.append(_tv(full_logits[j], cl))
                res[arm] = sum(tvs) / window
            out["lags"][str(lag)] = res
    return out

"""Amendment R.17.2.3: restoration, redesigned so every long-enough trace is eligible.

The R.10 exact-lag probe (``kvcache.exact_lag``) triggers at the first prune at
or after step 130, where the stash reaches the 64-entry reinsertion count only
for long prompts: 8 of 25 problems were eligible, and they were the long-prompt,
low-accuracy ones. Triggering at the first prune at or after step 1024 makes the
stash at least 1024 + prompt - budget >= 768 entries at budget 256 for every
trace that reaches the trigger.

Arms, all from the same paired replay and the same verified stash:

* ``none``: no intervention.
* the exact cohort at lags {1, 4, 16, 64, 256}: matched / random / relaxed,
  exactly as R.10 (``exact_lag.reinsert``);
* **full self-restoration**: immediately before forward ``e + 1`` EVERY stashed
  evicted entry is returned and eviction is switched off, so no position is
  missing (checked: the cache's absolute positions are exactly 0..P+e-1 once,
  per KV head) and M = 0 from then on. The returned K/V are the compressed
  branch's own, identity-checked against the full cache where they must equal
  it (R.15.4), so what divergence remains is state drift.

M and Delta (R.4 channels, via ``kvcache.capture.channel_stats``) are measured
in the none and full-restoration arms at the first forward after the trigger
prune (``e + 1``, the first prediction that sees an intervention) and every 16th
forward of the 200-forward window. The full branch's attention at those forwards
is arm-independent and recorded once, during the main replay.

Nothing in ``exact_lag`` or ``capture`` is modified: both are live generation
families. The main loop below mirrors ``exact_lag.run_exact_lag`` with three
changes: the event tap stays attached after the cohort with stashing switched
off (so positions stay tracked), the tracker is snapshotted with the cache, and
the attention channels are measured.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from kvcache.exact_lag import (
    INTERVENTION_ARMS,
    LAGS,
    _forward,
    _press_state,
    _restore_press,
    _tv,
    reinsert,
    verify_stash,
)

TRIGGER_STEP = 1024
MEASURE_EVERY = 16


def measure_steps(e: int, window: int, every: int = MEASURE_EVERY) -> list[int]:
    """Forwards at which M and Delta are read: e + 1, then every ``every``-th."""
    return list(range(e + 1, e + 1 + window, every))


def full_restore(
    cache: Any,
    stash: dict[int, list[tuple[int, Any, Any, Any]]],
    tracker: Any,
    total_positions: int,
) -> dict[str, Any]:
    """Return every stashed entry to every layer; check nothing is missing.

    Appends the stash (eviction order) after the live entries and extends the
    tracker's position map to match. Gate: the reinserted slice equals the
    stash bitwise (K and V), and per KV head the cache now holds each absolute
    position 0..total_positions-1 exactly once.
    """
    import torch

    passed = True
    n_layers = 0
    for layer, entries in sorted(stash.items()):
        pos = torch.cat([e[1] for e in entries], dim=1)  # [n_kv, S]
        ek = torch.cat([e[2] for e in entries], dim=1)  # [n_kv, S, d]
        ev = torch.cat([e[3] for e in entries], dim=1)
        cl = cache.layers[layer]
        n_live = int(cl.keys.shape[2])
        cl.keys = torch.cat([cl.keys, ek[None].to(cl.keys.dtype)], dim=2)
        cl.values = torch.cat([cl.values, ev[None].to(cl.values.dtype)], dim=2)
        live = tracker.current(layer, n_live, device=pos.device)
        tracker.maps[layer] = torch.cat([live, pos], dim=1)
        same = torch.equal(cl.keys[0, :, n_live:], ek) and torch.equal(
            cl.values[0, :, n_live:], ev
        )
        full = torch.arange(total_positions, device=pos.device)
        complete = all(
            torch.equal(torch.sort(tracker.maps[layer][h]).values, full)
            for h in range(pos.shape[0])
        )
        passed = passed and bool(same) and bool(complete)
        n_layers += 1
    return {
        "passed": passed,
        "n_layers": n_layers,
        "checked": "returned K and V bitwise equal the stash; every absolute "
        "position present exactly once per KV head",
    }


def run_restore_v2(
    model: Any,
    press: Any,
    prompt_ids: Sequence[int],
    trace_ids: Sequence[int],
    *,
    n_reinsert: int,
    rng_key: str,
    trigger_step: int = TRIGGER_STEP,
    window: int = 200,
    lags: Sequence[int] = LAGS,
    every: int = MEASURE_EVERY,
) -> dict[str, Any]:
    """One (trace, policy): cohort arms at every lag, full restoration, channels."""
    import torch
    from transformers import DynamicCache

    from kvcache.capture import EventTap, PositionTracker, channel_stats, query_logits

    n_kv = int(model.config.num_key_value_heads)
    layers = model.model.layers
    tracker = PositionTracker(n_kv)
    tap = EventTap(press, tracker, stash_kv=True)
    n_fwd = len(trace_ids)
    cohort: dict[str, Any] = {"step": None}

    def on_event(ev: Any) -> None:
        if cohort["step"] is None and ev.step >= trigger_step:
            cohort["step"] = ev.step

    tap.listeners.append(on_event)

    # --- attention channels: the full branch recorded once, arms measured
    state: dict[str, Any] = {"branch": None, "step": -1, "measure": set()}
    full_attn: dict[int, dict[int, Any]] = {}  # step -> layer -> [H, P+s] logits
    channels: dict[str, dict[int, list[dict[str, float]]]] = {}
    arm_tracker: dict[str, Any] = {}

    def attn_hook(i: int):
        def fn(module: Any, _a: Any, kwargs: dict, _out: Any) -> None:
            branch, step = state["branch"], state["step"]
            if branch is None:
                return
            keys = kwargs["past_key_values"].layers[i].keys
            e = cohort["step"]
            if branch == "full":
                if e is not None and step in state["measure"]:
                    full_attn.setdefault(step, {})[i] = query_logits(
                        module,
                        kwargs["hidden_states"],
                        kwargs["position_embeddings"],
                        keys,
                    )
                return
            if step not in state["measure"] or step not in full_attn:
                return
            tr = arm_tracker[branch]
            positions = tr.current(i, int(keys.shape[2]), device=keys.device)
            comp = query_logits(
                module, kwargs["hidden_states"], kwargs["position_embeddings"], keys
            )
            empty = positions[:, :0]
            st = channel_stats(full_attn[step][i], comp, positions, empty)
            channels.setdefault(branch, {}).setdefault(step, []).append(
                {
                    "layer": i,
                    "M": float(st["M"].mean()),
                    "Delta": float(st["Delta"].mean()),
                    "n_nonfinite": int(st["n_nonfinite"]),
                }
            )

        return fn

    handles = [
        layer.self_attn.register_forward_hook(attn_hook(i), with_kwargs=True)
        for i, layer in enumerate(layers)
    ]
    full_cache, comp_cache = DynamicCache(), DynamicCache()
    none_tv: dict[int, float] = {}
    full_logits: dict[int, Any] = {}
    snaps: dict[int, Any] = {}
    horizon = n_fwd - 1
    first_effective = None
    identity_checks: dict[str, int] | None = None
    arm_tracker["none"] = tracker
    try:
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
                        copy.deepcopy(tracker),
                    )
                press.profile_step = j
                state["step"] = j
                press.enabled = False
                state["branch"] = "full"
                fl = _forward(model, full_cache, pos, ids)
                press.enabled = True
                state["branch"] = "none"
                n_before = len(tap.events)
                cl = _forward(model, comp_cache, pos, ids)
                press.enabled = False
                state["branch"] = None
                tv = _tv(fl, cl)
                if first_effective is None and len(tap.events) > n_before:
                    first_effective = j
                if first_effective is None and tv > 1e-6:
                    raise RuntimeError(
                        f"diverged before the first eviction at step {j}"
                    )
                none_tv[j] = tv
                if e is None and cohort["step"] is not None:
                    e = cohort["step"]
                    if j != e:
                        raise RuntimeError("cohort event not at the current forward")
                    steps = {s for lay in tap.stash.values() for s, *_ in lay[-1:]}
                    if len(tap.stash) != len(layers) or steps != {e}:
                        raise RuntimeError("cohort event is not present in every layer")
                    identity_checks = verify_stash(
                        tap.stash, full_cache, len(prompt_ids) + first_effective
                    )
                    # freeze the stash, keep tracking positions (the none arm's
                    # channels need the live position map through the window)
                    tap.stash_kv = False
                    state["measure"] = set(measure_steps(e, window, every))
                    horizon = min(n_fwd - 1, e + max(lags) + window - 1)
                if e is not None and j > e:
                    full_logits[j] = fl
                if cohort["step"] is not None and j >= horizon:
                    break
            e = cohort["step"]
            press.event_tap = None  # as exact_lag: arm replays are not tracked
            out: dict[str, Any] = {
                "stash_identity_checks": identity_checks if e is not None else None,
                "cohort_step": e,
                "first_effective_prune": first_effective,
                "n_trace_tokens": n_fwd,
                "n_reinsert": n_reinsert,
                "trigger_step": trigger_step,
                "lags": {},
            }
            if e is None:
                out["status"] = "INELIGIBLE: no eviction at or after the trigger"
                return out
            stash = {k: v for k, v in tap.stash.items()}
            out["stash_size"] = min(
                sum(x[1].shape[1] for x in v) for v in stash.values()
            )
            if out["stash_size"] < n_reinsert:
                out["status"] = "INELIGIBLE: stash smaller than the reinsertion count"
                return out
            out["status"] = "READ"
            # --- cohort arms, exactly as R.10
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
                    "none": sum(none_tv[j] for j in range(start, start + window))
                    / window,
                }
                for arm in INTERVENTION_ARMS:
                    cache0, pstate, _pos, _tr = snaps[lag]
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
                        c = _forward(
                            model,
                            cache,
                            len(prompt_ids) + j - 1,
                            [int(trace_ids[j - 1])],
                        )
                        press.enabled = False
                        tvs.append(_tv(full_logits[j], c))
                    res[arm] = sum(tvs) / window
                out["lags"][str(lag)] = res
            # --- full self-restoration from the lag-1 snapshot (before e + 1)
            start = e + 1
            fs: dict[str, Any] = {"eligible": False}
            if 1 in snaps and start + window - 1 <= n_fwd - 1:
                cache0, pstate, _pos, tr0 = snaps[1]
                cache = copy.deepcopy(cache0)
                tr = copy.deepcopy(tr0)
                _restore_press(press, pstate)
                gate = full_restore(cache, stash, tr, len(prompt_ids) + e)
                if not gate["passed"]:
                    raise RuntimeError("full-restoration gate failed")
                arm_tracker["full_restore"] = tr
                len_at_restore = int(cache.layers[0].keys.shape[2])
                tvs = []
                for j in range(start, start + window):
                    press.profile_step = j
                    press.enabled = False  # eviction off: M stays 0
                    state["step"] = j
                    state["branch"] = "full_restore"
                    c = _forward(
                        model, cache, len(prompt_ids) + j - 1, [int(trace_ids[j - 1])]
                    )
                    state["branch"] = None
                    tvs.append(_tv(full_logits[j], c))
                fs = {
                    "eligible": True,
                    "applied_step": start,
                    "gate": gate,
                    "window_mean_tv": sum(tvs) / window,
                    "none_window_mean_tv": sum(
                        none_tv[j] for j in range(start, start + window)
                    )
                    / window,
                    "cache_len_at_restore": len_at_restore,
                    "cache_len_after": int(cache.layers[0].keys.shape[2]),
                }
            out["full_restore"] = fs
            out["channels"] = {
                arm: {
                    str(s): {
                        "M": sum(x["M"] for x in rows) / len(rows),
                        "Delta": sum(x["Delta"] for x in rows) / len(rows),
                        "n_layers": len(rows),
                        "n_nonfinite": sum(x["n_nonfinite"] for x in rows),
                        "per_layer": {
                            "M": [
                                x["M"] for x in sorted(rows, key=lambda r: r["layer"])
                            ],
                            "Delta": [
                                x["Delta"]
                                for x in sorted(rows, key=lambda r: r["layer"])
                            ],
                        },
                    }
                    for s, rows in sorted(by_step.items())
                }
                for arm, by_step in channels.items()
            }
            out["measure_steps"] = measure_steps(e, window, every)
    finally:
        for h in handles:
            h.remove()
    return out

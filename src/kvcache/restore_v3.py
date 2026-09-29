"""Amendment S.2: restoration by origin. Is the restorable divergence the prompt?

restore-v2 (R.17.2.3) found that returning EVERY evicted entry removes about
two thirds of the divergence, while returning the 64 most recently evicted
entries does nothing measurable. Concurrent work (arXiv:2609.03430) holds that
the prompt is the fragile part of the cache. This instrument asks which evicted
entries carry the restorable part, by returning them by ORIGIN.

Four arms, all applied to the same lag-1 snapshot immediately before forward
``e + 1`` (``e`` = the first effective prune at or after ``trigger_step``), and
all with eviction switched OFF for the read window:

* ``frozen``: nothing is returned;
* ``prompt``: every stashed entry whose absolute position is in the prompt;
* ``trace``: as many stashed entries per KV head as ``prompt`` returned, drawn
  uniformly without replacement from the evicted non-prompt positions;
* ``full``: every stashed entry (restore-v2's arm, ``restore_v2.full_restore``).

``frozen`` is the baseline for every ratio because full restoration changes two
things at once: it returns entries and it stops eviction.

Rectangular cache. Random retains different positions in each KV head, so the
number of evicted prompt entries differs by head, while a layer's cache is one
tensor. ``n`` is the largest such count over heads and layers; every head
returns all of its evicted prompt entries and is padded to ``n`` with its own
most recently evicted trace entries. The pad is recorded.

Nothing in ``restore_v2``, ``exact_lag`` or ``capture`` is modified: all are
live generation families. The replay loop mirrors ``run_restore_v2``'s, without
the cohort arms and the attention channels.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from typing import Any

from kvcache.exact_lag import (
    _forward,
    _press_state,
    _restore_press,
    _tv,
    verify_stash,
)
from kvcache.restore_v2 import TRIGGER_STEP, full_restore

ARMS = ("frozen", "prompt", "trace", "full")
PARTIAL_ARMS = ("prompt", "trace")


def prompt_return_count(positions: dict[int, list[list[int]]], prompt_len: int) -> int:
    """``n``: the largest number of evicted prompt entries in any (layer, head)."""
    return max(
        (
            sum(1 for p in head if p < prompt_len)
            for heads in positions.values()
            for head in heads
        ),
        default=0,
    )


def plan_origin_return(
    head_positions: Sequence[int],
    *,
    prompt_len: int,
    n: int,
    arm: str,
    rng: random.Random,
) -> tuple[list[int], int]:
    """Stash indices one KV head returns under ``arm``, and its pad count.

    ``head_positions`` are the absolute positions of this head's stashed
    entries, in eviction order. ``prompt`` returns every prompt entry, padded
    to ``n`` with the most recently evicted trace entries; ``trace`` returns
    ``n`` trace entries drawn uniformly without replacement.
    """
    if arm not in PARTIAL_ARMS:
        raise ValueError(f"no return plan for arm {arm!r}")
    prompt_idx = [i for i, p in enumerate(head_positions) if p < prompt_len]
    trace_idx = [i for i, p in enumerate(head_positions) if p >= prompt_len]
    if len(set(head_positions)) != len(head_positions):
        raise ValueError("a position was stashed twice in one head")
    if arm == "prompt":
        pad = n - len(prompt_idx)
        if pad < 0:
            raise ValueError("n is smaller than this head's evicted prompt count")
        if pad > len(trace_idx):
            raise ValueError("too few evicted trace entries to pad with")
        return prompt_idx + (trace_idx[len(trace_idx) - pad :] if pad else []), pad
    if n > len(trace_idx):
        raise ValueError("too few evicted trace entries to draw from")
    return sorted(rng.sample(trace_idx, n)), 0


def _stash_positions(stash: dict[int, list[tuple[int, Any, Any, Any]]]) -> dict:
    import torch

    return {
        layer: torch.cat([e[1] for e in entries], dim=1).tolist()
        for layer, entries in stash.items()
    }


def origin_restore(
    cache: Any,
    stash: dict[int, list[tuple[int, Any, Any, Any]]],
    tracker: Any,
    *,
    arm: str,
    prompt_len: int,
    n: int,
    rng_key: str,
) -> dict[str, Any]:
    """Return one arm's entries to every layer; gate what came back.

    Gate: the returned K and V equal the stash bitwise; no position is held
    twice in any head; and after ``prompt`` every KV head of every layer holds
    every prompt position.
    """
    import torch

    passed, n_layers, pad_max = True, 0, 0
    for layer, entries in sorted(stash.items()):
        pos = torch.cat([e[1] for e in entries], dim=1)  # [n_kv, S]
        ek = torch.cat([e[2] for e in entries], dim=1)  # [n_kv, S, d]
        ev = torch.cat([e[3] for e in entries], dim=1)
        plans = []
        for h, head in enumerate(pos.tolist()):
            idx, pad = plan_origin_return(
                head,
                prompt_len=prompt_len,
                n=n,
                arm=arm,
                # keyed by head, not layer: a policy that evicts the same
                # positions in every layer returns the same ones in every layer
                rng=random.Random(f"{rng_key}|{h}|{arm}"),
            )
            plans.append(idx)
            pad_max = max(pad_max, pad)
        sel = torch.tensor(plans, dtype=torch.long, device=pos.device)  # [n_kv, n]
        d = int(ek.shape[-1])
        rk = ek.gather(1, sel.unsqueeze(-1).expand(-1, -1, d))
        rv = ev.gather(1, sel.unsqueeze(-1).expand(-1, -1, d))
        rpos = pos.gather(1, sel)
        cl = cache.layers[layer]
        n_live = int(cl.keys.shape[2])
        cl.keys = torch.cat([cl.keys, rk[None].to(cl.keys.dtype)], dim=2)
        cl.values = torch.cat([cl.values, rv[None].to(cl.values.dtype)], dim=2)
        live = tracker.current(layer, n_live, device=pos.device)
        tracker.maps[layer] = torch.cat([live, rpos], dim=1)
        same = torch.equal(cl.keys[0, :, n_live:], rk) and torch.equal(
            cl.values[0, :, n_live:], rv
        )
        want = torch.arange(prompt_len, device=pos.device)
        ok = True
        for h in range(int(pos.shape[0])):
            held = tracker.maps[layer][h]
            ok = ok and int(torch.unique(held).numel()) == int(held.numel())
            if arm == "prompt":
                ok = ok and torch.equal(
                    torch.sort(held[held < prompt_len]).values, want
                )
        passed = passed and bool(same) and bool(ok)
        n_layers += 1
    return {
        "passed": passed,
        "n_layers": n_layers,
        "n_returned_per_head": n,
        "pad_max": pad_max,
        "checked": "returned K and V bitwise equal the stash; no position held "
        "twice; after the prompt arm every head holds every prompt position",
    }


def run_restore_v3(
    model: Any,
    press: Any,
    prompt_ids: Sequence[int],
    trace_ids: Sequence[int],
    *,
    rng_key: str,
    trigger_step: int = TRIGGER_STEP,
    window: int = 200,
) -> dict[str, Any]:
    """One (trace, policy): the no-intervention read and the four origin arms."""
    import torch
    from transformers import DynamicCache

    from kvcache.capture import EventTap, PositionTracker

    n_kv = int(model.config.num_key_value_heads)
    layers = model.model.layers
    tracker = PositionTracker(n_kv)
    tap = EventTap(press, tracker, stash_kv=True)
    n_fwd = len(trace_ids)
    prompt_len = len(prompt_ids)
    cohort: dict[str, Any] = {"step": None}

    def on_event(ev: Any) -> None:
        if cohort["step"] is None and ev.step >= trigger_step:
            cohort["step"] = ev.step

    tap.listeners.append(on_event)
    full_cache, comp_cache = DynamicCache(), DynamicCache()
    none_tv: dict[int, float] = {}
    full_logits: dict[int, Any] = {}
    snap: Any = None
    horizon = n_fwd - 1
    first_effective = None
    identity_checks: dict[str, int] | None = None
    with torch.inference_mode(), press(model):
        for j in range(n_fwd):
            ids = list(prompt_ids) if j == 0 else [int(trace_ids[j - 1])]
            pos = 0 if j == 0 else prompt_len + j - 1
            e = cohort["step"]
            if e is not None and j - e == 1:
                snap = (
                    copy.deepcopy(comp_cache),
                    _press_state(press),
                    copy.deepcopy(tracker),
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
                steps = {s for lay in tap.stash.values() for s, *_ in lay[-1:]}
                if len(tap.stash) != len(layers) or steps != {e}:
                    raise RuntimeError("cohort event is not present in every layer")
                identity_checks = verify_stash(
                    tap.stash, full_cache, prompt_len + first_effective
                )
                tap.stash_kv = False
                horizon = min(n_fwd - 1, e + window)
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
            "prompt_len": prompt_len,
            "trigger_step": trigger_step,
            "window": window,
            "arms": {},
        }
        if e is None:
            out["status"] = "INELIGIBLE: no eviction at or after the trigger"
            return out
        start = e + 1
        if snap is None or start + window - 1 > n_fwd - 1:
            out["status"] = "INELIGIBLE: the read window does not fit the trace"
            return out
        stash = {k: v for k, v in tap.stash.items()}
        out["stash_size"] = min(sum(x[1].shape[1] for x in v) for v in stash.values())
        n = prompt_return_count(_stash_positions(stash), prompt_len)
        out["n_prompt_return"] = n
        if n == 0:
            out["status"] = "INELIGIBLE: no prompt entry had been evicted"
            return out
        out["status"] = "READ"
        out["none_window_mean_tv"] = (
            sum(none_tv[j] for j in range(start, start + window)) / window
        )
        cache0, pstate, tracker0 = snap
        for arm in ARMS:
            cache = copy.deepcopy(cache0)
            tr = copy.deepcopy(tracker0)
            _restore_press(press, pstate)
            if arm == "full":
                gate = full_restore(cache, stash, tr, prompt_len + e)
            elif arm == "frozen":
                gate = {"passed": True, "n_layers": 0, "checked": "nothing returned"}
            else:
                gate = origin_restore(
                    cache,
                    stash,
                    tr,
                    arm=arm,
                    prompt_len=prompt_len,
                    n=n,
                    rng_key=rng_key,
                )
            if not gate["passed"]:
                raise RuntimeError(f"restoration gate failed: {arm}")
            len_at_restore = int(cache.layers[0].keys.shape[2])
            tvs = []
            for j in range(start, start + window):
                press.profile_step = j
                press.enabled = False  # eviction off in every arm
                c = _forward(model, cache, prompt_len + j - 1, [int(trace_ids[j - 1])])
                tvs.append(_tv(full_logits[j], c))
            out["arms"][arm] = {
                "window_mean_tv": sum(tvs) / window,
                "gate": gate,
                "cache_len_at_restore": len_at_restore,
                "cache_len_after": int(cache.layers[0].keys.shape[2]),
            }
    return out

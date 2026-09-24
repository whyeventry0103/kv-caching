"""Amendment R.2: the direct functional control for Q.1 (mask now, remove later).

The registered removal-anchored gate cannot identify physical-removal effects
(``phase_amplitude`` is rotation-invariant), so R.2 adds a control that asks the
question directly: does PHYSICALLY removing an evicted key change anything
beyond making it unattendable?

``DeferredRemovalPress`` separates the two acts. Every ``interval`` decode
forwards it selects over the LOGICALLY ACTIVE entries only (masked entries never
count toward the budget or re-enter scoring; scoring sees the gathered active
set, so the Random policy's fixed-seed draws are identical whatever is still
physically present), makes the non-selected entries unattendable at once (an
additive -inf mask per KV head, injected into each layer's attention; exp(-inf)
is exactly 0 in the fp32 softmax), and physically removes them ``delay``
forwards later. Overlapping pending removals are tracked per entry.

``run_fork_control`` replays a trace paired with the full cache (arm A = the
press as described, physically removing on schedule; the TV profile against the
full cache feeds the legacy removal-anchored gate, recorded for the record).
Immediately before every physical removal the state is forked: arm B keeps the
due entries masked instead and is replayed on the identical next tokens,
positions and policy state until the next selection event. Read per fork: max
|logit difference| and next-token TV between the arms at every compared step,
and agreement of the next selection decision (absolute positions). A
deterministic control replays arm B twice at the first forks of each trace; its
maximum fixes the tolerance (x10) in the reader.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

DELAY = 32


class DeferredRemovalPress:
    """Select on the active set, mask immediately, remove ``delay`` steps later."""

    def __init__(
        self,
        base_press: Any,
        *,
        budget: int,
        interval: int,
        delay: int = DELAY,
    ) -> None:
        self.base = base_press
        self.budget, self.interval, self.delay = int(budget), int(interval), int(delay)
        self.enabled = False
        self.compact = True  # arm A; arm B sets False (keep due entries masked)
        self.step = 0
        self.counts: dict[int, int] = {}
        # per layer: due [n_kv, L] (-1 = active, else step of physical removal)
        # and positions [n_kv, L] absolute token positions, in cache order
        self.due: dict[int, Any] = {}
        self.pos: dict[int, Any] = {}
        self.next_pos: dict[int, int] = {}
        self.decisions: list[tuple[int, int, Any]] = []  # (step, layer, masked pos)
        self._handles: list[Any] = []

    # -- state ---------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "counts": self.counts,
                "due": self.due,
                "pos": self.pos,
                "next_pos": self.next_pos,
            }
        )

    def restore(self, st: dict[str, Any]) -> None:
        st = copy.deepcopy(st)
        self.counts, self.due = st["counts"], st["due"]
        self.pos, self.next_pos = st["pos"], st["next_pos"]

    def _sync(self, layer: int, cache_len: int, device: Any) -> None:
        """Extend the bookkeeping to ``cache_len`` entries (new tokens active)."""
        import torch

        if layer not in self.due:
            n_kv = self._n_kv
            self.due[layer] = torch.full((n_kv, 0), -1, dtype=torch.long, device=device)
            self.pos[layer] = torch.zeros((n_kv, 0), dtype=torch.long, device=device)
            self.next_pos[layer] = 0
        n_new = cache_len - self.due[layer].shape[1]
        if n_new < 0:
            raise RuntimeError("cache shorter than the bookkeeping")
        if n_new:
            n_kv = self.due[layer].shape[0]
            start = self.next_pos[layer]
            new_pos = torch.arange(start, start + n_new, device=device)
            self.pos[layer] = torch.cat(
                [self.pos[layer], new_pos.expand(n_kv, -1)], dim=1
            )
            self.due[layer] = torch.cat(
                [
                    self.due[layer],
                    torch.full((n_kv, n_new), -1, dtype=torch.long, device=device),
                ],
                dim=1,
            )
            self.next_pos[layer] = start + n_new

    def due_now(self, step: int) -> bool:
        return any(
            bool(((d >= 0) & (d <= step)).any()) for d in self.due.values() if d.numel()
        )

    # -- hooks ---------------------------------------------------------------
    def __call__(self, model: Any) -> DeferredRemovalPress:
        self._n_kv = int(model.config.num_key_value_heads)
        self._rep = int(model.config.num_attention_heads) // self._n_kv
        for layer in model.model.layers:
            attn = layer.self_attn
            self._handles.append(
                attn.register_forward_pre_hook(self._pre, with_kwargs=True)
            )
            self._handles.append(
                attn.register_forward_hook(self._post, with_kwargs=True)
            )
        return self

    def __enter__(self) -> DeferredRemovalPress:
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _pre(self, module: Any, args: Any, kwargs: dict) -> Any:
        import torch

        if not self.enabled or kwargs["hidden_states"].shape[1] != 1:
            return None
        layer = int(module.layer_idx)
        cache = kwargs["past_key_values"]
        if layer >= len(cache.layers):
            return None
        cl = cache.layers[layer]
        self._sync(layer, int(cl.keys.shape[2]), cl.keys.device)
        due = self.due[layer]
        if self.compact:
            gone = (due >= 0) & (due <= self.step)
            if bool(gone.any()):
                keep = ~gone
                n_keep = int(keep[0].sum())
                if not bool((keep.sum(1) == n_keep).all()):
                    raise RuntimeError("unequal removal counts across KV heads")
                idx = torch.nonzero(keep)[:, 1].view(due.shape[0], n_keep)
                d = cl.keys.shape[-1]
                g = idx[None, :, :, None].expand(1, -1, -1, d)
                cl.keys = cl.keys.gather(2, g)
                cl.values = cl.values.gather(2, g)
                self.due[layer] = due.gather(1, idx)
                self.pos[layer] = self.pos[layer].gather(1, idx)
                due = self.due[layer]
        masked = due >= 0  # [n_kv, L_past]
        n_heads = masked.shape[0] * self._rep
        dtype = kwargs["hidden_states"].dtype
        mask = torch.zeros(
            (1, n_heads, 1, masked.shape[1] + 1), dtype=dtype, device=masked.device
        )
        mask[0, :, 0, :-1] = torch.where(
            masked.repeat_interleave(self._rep, dim=0),
            torch.tensor(float("-inf"), dtype=dtype, device=masked.device),
            torch.tensor(0.0, dtype=dtype, device=masked.device),
        )
        kwargs["attention_mask"] = mask
        return args, kwargs

    def _post(self, module: Any, args: Any, kwargs: dict, output: Any) -> Any:
        import torch

        if not self.enabled or kwargs["hidden_states"].shape[1] != 1:
            return output
        layer = int(module.layer_idx)
        self.counts[layer] = self.counts.get(layer, 0) + 1
        if self.counts[layer] < self.interval:
            return output
        self.counts[layer] = 0
        cl = kwargs["past_key_values"].layers[layer]
        self._sync(layer, int(cl.keys.shape[2]), cl.keys.device)
        due = self.due[layer]
        active = due < 0
        n_act = int(active[0].sum())
        if not bool((active.sum(1) == n_act).all()):
            raise RuntimeError("unequal active counts across KV heads")
        if n_act <= self.budget:
            return output
        aidx = torch.nonzero(active)[:, 1].view(due.shape[0], n_act)
        d = cl.keys.shape[-1]
        g = aidx[None, :, :, None].expand(1, -1, -1, d)
        k_act, v_act = cl.keys.gather(2, g), cl.values.gather(2, g)
        scores = self.base.score(
            module, kwargs["hidden_states"], k_act, v_act, None, kwargs
        )
        kept = scores.topk(self.budget, dim=-1).indices[0]  # [n_kv, budget]
        drop = torch.ones_like(aidx, dtype=torch.bool)
        drop.scatter_(1, kept, False)
        cols = aidx[drop].view(due.shape[0], n_act - self.budget)
        self.due[layer] = due.scatter(1, cols, self.step + self.delay)
        self.decisions.append(
            (self.step, layer, self.pos[layer].gather(1, cols).sort(dim=1).values)
        )
        return output


def _forward(model: Any, cache: Any, pos: int, ids: Sequence[int]) -> Any:
    import torch

    t = torch.tensor([list(ids)], dtype=torch.long, device=model.device)
    cp = torch.arange(pos, pos + len(ids), dtype=torch.long, device=model.device)
    return model(input_ids=t, past_key_values=cache, cache_position=cp).logits[0, -1]


def _tv(a: Any, b: Any) -> float:
    import torch

    return min(
        1.0,
        float(
            0.5
            * (torch.softmax(a.float(), -1) - torch.softmax(b.float(), -1)).abs().sum()
        ),
    )


def run_fork_control(
    model: Any,
    press: DeferredRemovalPress,
    prompt_ids: Sequence[int],
    trace_ids: Sequence[int],
    *,
    n_deterministic: int = 3,
) -> dict[str, Any]:
    """One trace: arm A with forks to arm B before every physical removal."""
    import torch
    from transformers import DynamicCache

    n_fwd = len(trace_ids)
    full, main = DynamicCache(), DynamicCache()
    tv_profile: list[float] = []
    removal_steps: list[int] = []
    forks: list[dict[str, Any]] = []
    det_max = 0.0
    pending: dict[int, list[tuple[int, Any]]] = {}  # step -> [(fork id, B logits)]
    b_decisions: dict[int, Any] = {}
    with torch.inference_mode(), press(model):
        for j in range(n_fwd):
            ids = list(prompt_ids) if j == 0 else [int(trace_ids[j - 1])]
            pos = 0 if j == 0 else len(prompt_ids) + j - 1
            press.step = j
            if j > 0 and press.due_now(j):
                removal_steps.append(j)
                fid = len(forks)
                snap_cache, snap_state = copy.deepcopy(main), press.state()
                n_dec = len(press.decisions)
                runs = 2 if fid < n_deterministic else 1
                b_runs = []
                for _ in range(runs):
                    cache_b = copy.deepcopy(snap_cache)
                    press.restore(snap_state)
                    del press.decisions[n_dec:]
                    press.compact, press.enabled = False, True
                    logits_b = []
                    for k in range(j, n_fwd):
                        press.step = k
                        kid = list(prompt_ids) if k == 0 else [int(trace_ids[k - 1])]
                        logits_b.append(
                            (k, _forward(model, cache_b, len(prompt_ids) + k - 1, kid))
                        )
                        if len(press.decisions) > n_dec:
                            break
                    b_runs.append((logits_b, list(press.decisions[n_dec:])))
                if runs == 2:
                    det_max = max(
                        det_max,
                        max(
                            float((x[1] - y[1]).abs().max())
                            for x, y in zip(b_runs[0][0], b_runs[1][0], strict=True)
                        ),
                    )
                press.restore(snap_state)
                del press.decisions[n_dec:]
                press.compact, press.step = True, j
                for k, lg in b_runs[0][0]:
                    pending.setdefault(k, []).append((fid, lg))
                b_decisions[fid] = b_runs[0][1]
                forks.append(
                    {
                        "step": j,
                        "n_steps": len(b_runs[0][0]),
                        "max_abs_logit": 0.0,
                        "first_tv": None,
                        "max_tv": 0.0,
                        "a_decisions_start": n_dec,
                    }
                )
            press.enabled = False
            fl = _forward(model, full, pos, ids)
            press.enabled = True
            al = _forward(model, main, pos, ids)
            press.enabled = False
            tv_profile.append(_tv(fl, al))
            for fid, lg in pending.pop(j, []):
                f = forks[fid]
                diff = float((al.float() - lg.float()).abs().max())
                tv = _tv(al, lg)
                f["max_abs_logit"] = max(f["max_abs_logit"], diff)
                f["max_tv"] = max(f["max_tv"], tv)
                if f["first_tv"] is None:
                    f["first_tv"] = tv
                    f["first_abs_logit"] = diff
    agree = []
    for fid, f in enumerate(forks):
        bd = b_decisions[fid]
        ad = press.decisions[f["a_decisions_start"] : f["a_decisions_start"] + len(bd)]
        if not bd or len(ad) < len(bd):
            f["selection_agrees"] = None
            continue
        same = all(
            a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])
            for a, b in zip(ad, bd, strict=True)
        )
        f["selection_agrees"] = bool(same)
        agree.append(same)
    for f in forks:
        f.pop("a_decisions_start", None)
    return {
        "n_forward": n_fwd,
        "n_forks": len(forks),
        "removal_steps": removal_steps,
        "first_removal_step": removal_steps[0] if removal_steps else None,
        "deterministic_control_max_abs_logit": det_max,
        "max_abs_logit": max((f["max_abs_logit"] for f in forks), default=None),
        "max_tv": max((f["max_tv"] for f in forks), default=None),
        "selection_agreement": (sum(agree) / len(agree)) if agree else None,
        "n_selection_compared": len(agree),
        "forks": forks,
        "tv_profile": tv_profile,
    }

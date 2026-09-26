"""Amendment R.17.0.5: is the fork control's arm A / arm B gap numerics?

The direct fork control (``kvcache.fork_control``, R.2/R.11) compares arm A
(evicted entries physically compacted out of the cache) with arm B (the same
entries kept in place under an additive -inf mask). The two are the same
computation in exact arithmetic -- exp(-inf) is exactly 0 in the softmax -- but
not in bf16: a shorter key axis changes the kernel's reduction order. R.11's
tolerance is 10x a replay control that is exactly zero, so it can never pass;
this module supplies the two measurements that replace it.

* ``first_fork_diagnosis``: at one trace's first physical removal, replay both
  arms for the compared span and record per step the TV, the max |logit|
  difference, WHERE that difference lives (the arm-B vocabulary rank of the
  token carrying it) and the top-10 overlap. Run once in bf16 and once with the
  model in fp32: a gap that collapses in fp32 is kernel numerics.
* ``collect_block_logits`` / ``compare_block_logits``: a teacher-forced
  full-cache pass in one dtype stores the logits of fixed step blocks (bf16
  logits are stored losslessly as bf16); a second pass in the other dtype
  compares against them. That is the precision floor the fork gap is judged
  against (PASS iff fork p95 <= 2 x floor p95, fixed in R.17.0.5).

Nothing here edits ``fork_control``: its generation family is live.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any


def _step_stats(a: Any, b: Any) -> dict[str, Any]:
    import torch

    from kvcache.fork_control import _tv

    a, b = a.float(), b.float()
    d = (a - b).abs()
    m = int(d.argmax())
    top_a = set(torch.topk(a, 10).indices.tolist())
    top_b = set(torch.topk(b, 10).indices.tolist())
    return {
        "tv": _tv(a, b),
        "max_abs_logit": float(d.max()),
        "argmax_token_rank_in_b": int((b > b[m]).sum()),
        "b_logit_at_argmax": float(b[m]),
        "b_top1_logit": float(b.max()),
        "top10_overlap": len(top_a & top_b),
        "top1_agree": int(a.argmax()) == int(b.argmax()),
    }


def first_fork_diagnosis(
    model: Any,
    press: Any,
    prompt_ids: Sequence[int],
    trace_ids: Sequence[int],
) -> dict[str, Any]:
    """Both arms from the snapshot at the first physical removal, step by step.

    Mirrors ``run_fork_control`` exactly up to that removal: the same forward
    positions, the same press state, the same stopping rule (the next selection
    event). Arm A compacts, arm B masks; nothing else differs.
    """
    import torch
    from transformers import DynamicCache

    from kvcache.fork_control import _forward

    n_fwd = len(trace_ids)
    main = DynamicCache()
    with torch.inference_mode(), press(model):
        for j in range(n_fwd):
            ids = list(prompt_ids) if j == 0 else [int(trace_ids[j - 1])]
            pos = 0 if j == 0 else len(prompt_ids) + j - 1
            press.step = j
            if j > 0 and press.due_now(j):
                snap_cache, snap_state = copy.deepcopy(main), press.state()
                n_dec = len(press.decisions)
                arms: dict[str, list[Any]] = {}
                for name, compact in (("A_remove", True), ("B_mask", False)):
                    cache = copy.deepcopy(snap_cache)
                    press.restore(snap_state)
                    del press.decisions[n_dec:]
                    press.compact, press.enabled = compact, True
                    logits = []
                    for k in range(j, n_fwd):
                        press.step = k
                        kid = [int(trace_ids[k - 1])]
                        out = _forward(model, cache, len(prompt_ids) + k - 1, kid)
                        logits.append(out.detach().to("cpu"))
                        if len(press.decisions) > n_dec:
                            break
                    arms[name] = logits
                press.enabled = False
                steps = [
                    {"step": j + i, **_step_stats(a, b)}
                    for i, (a, b) in enumerate(
                        zip(arms["A_remove"], arms["B_mask"], strict=False)
                    )
                ]
                return {
                    "fork_step": j,
                    "n_steps_a": len(arms["A_remove"]),
                    "n_steps_b": len(arms["B_mask"]),
                    "n_compared": len(steps),
                    "logit_dtype": str(arms["B_mask"][0].dtype),
                    "steps": steps,
                }
            press.enabled = True
            _forward(model, main, pos, ids)
            press.enabled = False
    return {"fork_step": None, "n_compared": 0, "steps": []}


def block_steps(n_fwd: int, starts: Sequence[int], length: int) -> list[int]:
    """Forward indices covered by the blocks that fit the trace completely."""
    out: list[int] = []
    for s in starts:
        if s >= 1 and s + length <= n_fwd:
            out.extend(range(s, s + length))
    return out


def _teacher_forced(
    model: Any, prompt_ids: Sequence[int], trace_ids: Sequence[int], wanted: set[int]
) -> dict[int, Any]:
    import torch
    from transformers import DynamicCache

    from kvcache.fork_control import _forward

    cache = DynamicCache()
    out: dict[int, Any] = {}
    last = max(wanted) if wanted else -1
    with torch.inference_mode():
        for j in range(last + 1):
            ids = list(prompt_ids) if j == 0 else [int(trace_ids[j - 1])]
            pos = 0 if j == 0 else len(prompt_ids) + j - 1
            lg = _forward(model, cache, pos, ids)
            if j in wanted:
                out[j] = lg.detach().to("cpu")
    return out


def collect_block_logits(
    model: Any,
    prompt_ids: Sequence[int],
    trace_ids: Sequence[int],
    *,
    starts: Sequence[int],
    length: int,
) -> dict[int, Any]:
    """Teacher-forced full-cache logits at the block steps (stored as produced)."""
    steps = set(block_steps(len(trace_ids), starts, length))
    return _teacher_forced(model, prompt_ids, trace_ids, steps)


def compare_block_logits(
    model: Any,
    prompt_ids: Sequence[int],
    trace_ids: Sequence[int],
    reference: dict[int, Any],
) -> dict[int, dict[str, Any]]:
    """Per block step: TV and max |logit| between this model and ``reference``."""
    cur = _teacher_forced(model, prompt_ids, trace_ids, set(reference))
    return {j: _step_stats(cur[j], reference[j]) for j in sorted(reference)}


def window_max_p95(
    series_by_block: Sequence[Sequence[float]], window: int
) -> dict[str, Any]:
    """95th percentile over non-overlapping windows of the window-maximum.

    ``series_by_block`` holds contiguous per-step values; windows never cross a
    block boundary and a trailing partial window is dropped (R.17.0.5).
    """
    import statistics

    maxima = [
        max(block[i : i + window])
        for block in series_by_block
        for i in range(0, len(block) - window + 1, window)
    ]
    if not maxima:
        return {"p95": None, "n_windows": 0}
    ms = sorted(maxima)
    return {
        "p95": ms[int(0.95 * (len(ms) - 1))],
        "n_windows": len(ms),
        "median": statistics.median(ms),
        "max": ms[-1],
    }


def fork_p95(values: Sequence[float]) -> float | None:
    """95th percentile of per-fork ``max_tv`` (same index rule as above)."""
    vs = sorted(float(v) for v in values)
    return vs[int(0.95 * (len(vs) - 1))] if vs else None


def gate_read(
    fork: dict[tuple[str, int], float | None],
    floor: dict[int, float | None],
    *,
    multiple: float = 2.0,
) -> dict[str, Any]:
    """PASS iff every (policy, interval) fork p95 <= multiple x floor p95(I)."""
    arms = {}
    ok = True
    for (policy, interval), f in sorted(fork.items()):
        fl = floor.get(interval)
        passed = f is not None and fl is not None and f <= multiple * fl
        ok = ok and passed
        arms[f"{policy}-i{interval}"] = {
            "fork_p95_max_tv": f,
            "floor_p95_window_max_tv": fl,
            "ratio": (f / fl) if (f is not None and fl) else None,
            "passed": passed,
        }
    return {"multiple": multiple, "arms": arms, "verdict": "PASS" if ok else "FAIL"}

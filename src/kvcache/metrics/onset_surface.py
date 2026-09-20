"""Exposure-vs-dose surface for Amendment Q.2 (onset-staggered replay).

The identity ``r = B/t`` couples the realised compression ratio to context
length, so no slice of ``(B, t)`` isolates how long compression has been
running: hold the budget and the ratio falls as the trace grows; hold the ratio
and the absolute cache grows instead. Amendment M.3 ran the iso-ratio slice and
recorded that qualifier explicitly.

Compression onset breaks the identity because it is not in it. With compression
inactive until decode step ``t0``, two arms read at the SAME absolute position
``t`` and the same budget ``B`` have the same realised ratio ``r = B/t`` while
differing in exposure ``h = t - t0``. The slope of divergence against exposure
at matched ``r`` is therefore the quantity ``r = B/t`` otherwise forbids, and it
is what Amendment Q.2 pre-registered a read on.

Two things this module refuses to do, both because they would quietly answer a
different question:

* It never mixes positions. ``r`` is matched only at equal ``t``, so a slope is
  computed within one ``(budget, position)`` cell and never pooled across them.
* It never reads a position before an arm's onset. Exposure is undefined there,
  and including those steps would compare a compressed arm against an
  uncompressed one and call the difference exposure.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

__all__ = ["window_divergence", "exposure_slope", "matched_positions"]


def window_divergence(
    estimate: Sequence[float], position: int, width: int
) -> float | None:
    """Mean per-step divergence over ``[position, position + width)``.

    A single step is too noisy to regress on, and a window mean keeps the
    quantity on the same per-step scale as everything else in the project.
    Returns ``None`` when the trace is too short to fill the window, rather than
    averaging a partial one: a short window is a different estimator, and
    silently mixing the two across arms would put trace length into the slope.
    """
    lo = int(position)
    hi = lo + int(width)
    if lo < 0 or int(width) <= 0 or hi > len(estimate):
        return None
    seg = [float(v) for v in estimate[lo:hi]]
    if not seg:
        return None
    return statistics.fmean(seg)


def matched_positions(
    onsets: Sequence[int], width: int, trace_len: int, *, step: int, count: int
) -> list[int]:
    """Read positions where EVERY arm is post-onset and has a full window.

    The first admissible position is ``max(onsets) + width``: the latest arm
    must have been compressing for at least a full window before it is read, or
    its window straddles its own onset and the comparison is between a
    compressing arm and a partly uncompressed one.
    """
    if not onsets or int(width) <= 0:
        return []
    first = max(int(o) for o in onsets) + int(width)
    out: list[int] = []
    p = first
    while len(out) < int(count) and p + int(width) <= int(trace_len):
        out.append(p)
        p += int(step)
    return out


def exposure_slope(
    exposures: Sequence[float], divergences: Sequence[float]
) -> float | None:
    """Ordinary least-squares slope of divergence on exposure, per cell.

    ``None`` when fewer than two arms survive or every arm has the same
    exposure, which is a degenerate design rather than a zero slope and must not
    be reported as one.
    """
    xs = [float(x) for x in exposures]
    ys = [float(y) for y in divergences]
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.fmean(xs)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return None
    my = statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom

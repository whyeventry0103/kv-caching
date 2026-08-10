"""Shape statistics for onset-aligned per-step divergence profiles (Amendment P.2).

Per-step TV is memoryless: it cannot distinguish "the model absorbed the shock
and recovered" from "damage is compounding", because both can show the same
instantaneous value. Three views that can:

* **running integral** -- cumulative divergence at fixed offsets from onset. A
  plateau in the *rate* of accumulation is recovery; a straight or steepening
  climb is compounding.
* **phase profile** -- mean divergence by position within the prune cycle. If
  eviction is the shock, divergence should peak just after a prune and decay
  before the next one. Flat phase means divergence arises between prunes and
  eviction events are not the proximate trigger.
* **per-trace shape** -- monotone drift vs spike-and-recover, so the per-policy
  mixture can be reported rather than a single averaged curve that hides it.

All three are DESCRIPTIVE. These are TV-scale quantities and per Definition 6
cannot license an accuracy-scale claim by adjacency; the only sanctioned bridge
is the cross-cell prediction in Amendment P.3.

Thresholds are fixed here, before the run, and echoed into the artifact.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

# Offsets from onset at which the running integral is reported.
CHECKPOINTS: tuple[int, ...] = (50, 100, 200, 400, 800, 1600)

# A trace counts as DRIFT if its last quarter averages at least this multiple of
# its first quarter (post-onset). 1.5 is deliberately generous: we are asking
# whether divergence grew materially, not whether it grew at all.
DRIFT_RATIO = 1.5

# NO per-trace recovery threshold. A first version classified a trace as
# "recovering" when its per-trace phase profile fell by >=10% from peak to
# trough. That statistic is noise: with `interval` buckets of noisy per-step
# values, peak-minus-trough is dominated by sampling variation. A sham control
# (phase labels shuffled within the trace, destroying all real phase structure)
# reproduced it almost exactly -- observed median 0.752 against sham 0.737 on
# 200 Random traces. Per-trace recovery is therefore NOT measurable at this
# noise level and is not reported. Eviction alignment is tested only on the
# AGGREGATE phase profile, where averaging over hundreds of traces and dozens
# of cycles each makes the amplitude estimable, and always against its sham.
SHAM_SEED = 0


def running_integral(estimate: Sequence[float], onset: int) -> dict[str, float | None]:
    """Cumulative divergence at each checkpoint offset past onset."""
    out: dict[str, float | None] = {}
    total = 0.0
    reached: dict[int, float] = {}
    for offset, index in enumerate(range(onset, len(estimate))):
        total += float(estimate[index])
        if offset + 1 in CHECKPOINTS:
            reached[offset + 1] = total
    for checkpoint in CHECKPOINTS:
        out[f"cum_at_{checkpoint}"] = reached.get(checkpoint)
    out["cum_total"] = total
    out["n_steps_after_onset"] = float(max(0, len(estimate) - onset))
    return out


def phase_profile(
    estimate: Sequence[float], onset: int, interval: int
) -> list[float | None]:
    """Mean divergence by position within the prune cycle, 0 = the prune step."""
    buckets: list[list[float]] = [[] for _ in range(interval)]
    for index in range(onset, len(estimate)):
        buckets[(index - onset) % interval].append(float(estimate[index]))
    return [statistics.fmean(b) if b else None for b in buckets]


def classify_shape(
    estimate: Sequence[float], onset: int, interval: int
) -> dict[str, Any]:
    """Trace-level growth only.

    The recovery axis was removed: see SHAM_SEED above. Growth is measured over
    quarter-of-trace means (hundreds to thousands of steps each), which is
    robust where a 64-bucket per-trace phase profile is not.
    """
    tail = [float(v) for v in estimate[onset:]]
    if len(tail) < 4 * interval:
        return {"shape": "too-short", "n_steps": len(tail)}
    quarter = len(tail) // 4
    first, last = statistics.fmean(tail[:quarter]), statistics.fmean(tail[-quarter:])
    ratio = (last / first) if first > 0 else None
    drifts = ratio is not None and ratio >= DRIFT_RATIO
    return {
        "first_quarter_mean": first,
        "last_quarter_mean": last,
        "drift_ratio": ratio,
        "drifts": drifts,
        "shape": "monotone-drift" if drifts else "flat",
        "n_steps": len(tail),
    }


def phase_amplitude(
    phase: Sequence[float | None],
) -> float | None:
    """Peak-to-trough of an aggregate phase profile, as a fraction of its peak."""
    values = [v for v in phase if v is not None]
    if not values:
        return None
    peak = max(values)
    return (peak - min(values)) / peak if peak > 0 else None


def summarise_profiles(
    profiles: Iterable[tuple[Sequence[float], int | None, int]],
) -> dict[str, Any]:
    """Aggregate shape statistics over a set of (estimate, onset, interval)."""
    shapes: dict[str, int] = {}
    integrals: dict[str, list[float]] = {f"cum_at_{c}": [] for c in CHECKPOINTS}
    auc: list[float] = []
    phase_sum: list[list[float]] = []
    drift_ratios: list[float] = []
    sham_sum: list[list[float]] = []
    n = 0
    for estimate, onset, interval in profiles:
        if not estimate or onset is None:
            continue
        n += 1
        auc.append(sum(float(v) for v in estimate) / len(estimate))
        integ = running_integral(estimate, int(onset))
        for key in integrals:
            if integ.get(key) is not None:
                integrals[key].append(float(integ[key]))
        shape = classify_shape(estimate, int(onset), int(interval))
        shapes[shape["shape"]] = shapes.get(shape["shape"], 0) + 1
        if shape.get("drift_ratio") is not None:
            drift_ratios.append(float(shape["drift_ratio"]))
        phase = phase_profile(estimate, int(onset), int(interval))
        if not phase_sum:
            phase_sum = [[] for _ in phase]
            sham_sum = [[] for _ in phase]
        for i, value in enumerate(phase):
            if value is not None:
                phase_sum[i].append(value)
        # Sham: the same steps assigned to shuffled phases. Any amplitude the
        # sham reproduces is sampling variation, not eviction alignment.
        rng = random.Random(SHAM_SEED + n)
        tail = [float(v) for v in estimate[int(onset) :]]
        labels = [i % int(interval) for i in range(len(tail))]
        rng.shuffle(labels)
        buckets: list[list[float]] = [[] for _ in range(int(interval))]
        for value, label in zip(tail, labels, strict=True):
            buckets[label].append(value)
        for i, bucket in enumerate(buckets):
            if bucket:
                sham_sum[i].append(statistics.fmean(bucket))
    phase_mean = [statistics.fmean(v) if v else None for v in phase_sum]
    sham_mean = [statistics.fmean(v) if v else None for v in sham_sum]
    return {
        "n_profiles": n,
        "auc_tv_mean": statistics.fmean(auc) if auc else None,
        "running_integral_mean": {
            key: (statistics.fmean(v) if v else None) for key, v in integrals.items()
        },
        "running_integral_n": {key: len(v) for key, v in integrals.items()},
        "shape_counts": shapes,
        "shape_fractions": {k: v / n for k, v in shapes.items()} if n else {},
        "median_drift_ratio": statistics.median(drift_ratios) if drift_ratios else None,
        "phase_profile_mean": phase_mean,
        "phase_profile_sham_mean": sham_mean,
        "phase_amplitude": phase_amplitude(phase_mean),
        "phase_amplitude_sham": phase_amplitude(sham_mean),
        "phase_peak_index": (
            phase_mean.index(max(v for v in phase_mean if v is not None))
            if any(v is not None for v in phase_mean)
            else None
        ),
        "thresholds": {
            "drift_ratio": DRIFT_RATIO,
            "sham_seed": SHAM_SEED,
            "checkpoints": list(CHECKPOINTS),
        },
        "scale_discipline": (
            "TV-scale, descriptive only. Per Definition 6 this cannot license an "
            "accuracy-scale claim by adjacency; see Amendment P.3 for the only "
            "sanctioned bridge."
        ),
    }

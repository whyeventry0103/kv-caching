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
from collections.abc import Iterable, Mapping, Sequence
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


def phase_amplitude_excess(
    profiles: Sequence[tuple[Sequence[float], int | None, int]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate phase amplitude minus its sham, with a trace-clustered CI.

    The trace is the resampling unit. Observed and sham amplitudes are
    recomputed together on each resample, so the interval is on the EXCESS;
    resampling them independently would ignore their correlation and inflate it.

    Per-trace phase SUMS and COUNTS are precomputed once, so a resample is a
    vector add over traces rather than a re-bucketing of every step. The naive
    version re-bucketed ~2.6M steps per resample and did not finish.

    Only the excess licenses a claim. A raw amplitude is not evidence of
    eviction alignment: shuffled phase labels produce one too.
    """
    obs_sum: list[list[float]] = []
    obs_cnt: list[list[int]] = []
    sham_sum: list[list[float]] = []
    sham_cnt: list[list[int]] = []
    interval = 0
    for rank, (estimate, onset, iv) in enumerate(profiles):
        if not estimate or onset is None:
            continue
        iv = int(iv)
        interval = interval or iv
        if iv != interval:
            raise ValueError("profiles mix compression intervals")
        tail = [float(v) for v in estimate[int(onset) :]]
        o_s, o_c = [0.0] * iv, [0] * iv
        for offset, value in enumerate(tail):
            o_s[offset % iv] += value
            o_c[offset % iv] += 1
        rng = random.Random(SHAM_SEED + rank)
        labels = [k % iv for k in range(len(tail))]
        rng.shuffle(labels)
        s_s, s_c = [0.0] * iv, [0] * iv
        for value, label in zip(tail, labels, strict=True):
            s_s[label] += value
            s_c[label] += 1
        obs_sum.append(o_s)
        obs_cnt.append(o_c)
        sham_sum.append(s_s)
        sham_cnt.append(s_c)
    if not obs_sum:
        raise ValueError("no profiles with an onset")

    def amplitudes(index: Sequence[int]) -> tuple[float | None, float | None]:
        o_s, o_c = [0.0] * interval, [0] * interval
        s_s, s_c = [0.0] * interval, [0] * interval
        for i in index:
            for ph in range(interval):
                o_s[ph] += obs_sum[i][ph]
                o_c[ph] += obs_cnt[i][ph]
                s_s[ph] += sham_sum[i][ph]
                s_c[ph] += sham_cnt[i][ph]
        o = [(o_s[ph] / o_c[ph]) if o_c[ph] else None for ph in range(interval)]
        sh = [(s_s[ph] / s_c[ph]) if s_c[ph] else None for ph in range(interval)]
        return phase_amplitude(o), phase_amplitude(sh)

    whole = list(range(len(obs_sum)))
    obs_amp, sham_amp = amplitudes(whole)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_resamples):
        o, sh = amplitudes([rng.randrange(len(whole)) for _ in whole])
        if o is not None and sh is not None:
            draws.append(o - sh)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return {
        "n_profiles": len(obs_sum),
        "phase_amplitude": obs_amp,
        "phase_amplitude_sham": sham_amp,
        # Amendment R.2 (phase-excess-v2): `is not None`, not truthiness. The
        # v1 expression turned a legitimate 0.0 amplitude (a flat profile) into
        # a missing excess, while the bootstrap draws above already treated 0.0
        # correctly -- so the point estimate and its own interval disagreed on
        # what counted as data. Every committed Q.1/P.2 artifact had non-zero
        # amplitudes, so none of their excesses change.
        "excess": (
            (obs_amp - sham_amp)
            if (obs_amp is not None and sham_amp is not None)
            else None
        ),
        "analysis_version": "phase-excess-v2",
        "excess_ci95": [lo, hi],
        "excess_excludes_zero": (lo > 0.0) or (hi < 0.0),
        "n_resamples": n_resamples,
        "seed": seed,
    }


# --------------------------------------------------------------------------
# Floor-exit windowing (Amendments M.3 and P.2-v2).
#
# A ratio-matched press with min_budget=32 at ratio 0.038 keeps hitting the
# floor until the logical context reaches ~842 tokens. Below that the cell is a
# FIXED-32 budget whose realized ratio worsens as context grows -- the very
# confound the ratio matching removes -- and that period sits at the START of
# every trace. Any statistic computed over the whole trace is therefore
# contaminated by the harshest compression in the run.
#
# This lives in the package, not in a script, because two separate readers
# (the M.3 drift comparison and the P.2-v2 phase re-analysis) must use the SAME
# window or their verdicts are not comparable.
# --------------------------------------------------------------------------

MIN_RESTRICTED_WINDOW = 200  # a statistic over fewer steps than this is not reported


def floor_exit_step(
    trace: Sequence[Mapping[str, Any]], compression_interval: int
) -> int | None:
    """First step at which a ratio-matched press stopped sitting on ``min_budget``.

    ``trace`` is the ``trace`` list of a ``*.ratio_trace.json`` sidecar. The
    first compression sees prompt + one interval with nothing yet evicted, so
    ``prompt = first logical_context - interval`` and any event's step is
    ``logical_context - prompt``.

    Returns ``None`` when the press never left the floor, which is an exclusion,
    not a zero.
    """
    if not trace:
        return None
    prompt = int(trace[0]["logical_context"]) - int(compression_interval)
    for event in trace:
        if not event.get("at_floor"):
            return max(0, int(event["logical_context"]) - prompt)
    return None


def restrict_to_floor_exit(
    estimate: Sequence[float],
    onset: int | None,
    trace: Sequence[Mapping[str, Any]],
    compression_interval: int,
    *,
    min_window: int = MIN_RESTRICTED_WINDOW,
) -> tuple[list[float], int, str] | None:
    """Trim a profile to its above-floor window, re-basing the onset.

    Returns ``(trimmed_estimate, new_onset, "")`` or ``None`` with a reason when
    the profile is excluded.

    **The re-based onset must PRESERVE PHASE ALIGNMENT, not merely be
    non-negative.** ``phase_profile`` buckets on ``(index - onset) % interval``
    with bucket 0 defined as the prune step, so the re-based onset has to keep
    that congruence in the trimmed frame. Setting it to ``max(0, onset - start)``
    -- i.e. 0 whenever the trim starts after onset, which is always -- silently
    rotates every phase index by ``(start - onset) % interval``. On the
    replay-ratio-matched-m3-v2 profiles that offset is 63 on all 96 traces, so
    the whole phase profile came out rotated by exactly one bucket and the peak
    read 1 where the prune step is 0.

    The correct re-basing is the congruence itself: ``(onset - start) % interval``.
    """
    start = floor_exit_step(trace, compression_interval)
    if start is None:
        return None
    trimmed = list(estimate)[start:]
    if len(trimmed) < min_window:
        return None
    interval = int(compression_interval)
    new_onset = ((int(onset) if onset is not None else 0) - start) % interval
    return trimmed, new_onset, ""


def drift_normalized_curve(
    estimate: Sequence[float], n_bins: int = 40
) -> list[float] | None:
    """Bin a retained window into ``n_bins`` and divide by its first-quarter mean.

    The M.3 verdict is a statement about **drift** -- last-quarter mean over
    first-quarter mean, within one arm -- and deliberately not about level. The
    two arms sit at very different levels: over the retained window the
    ratio-matched press has been compressing far harder than the fixed-budget
    press (its floor is ``min_budget`` 32 against a budget of 256), so plotting
    the two raw profiles on shared axes invites a level reading that no
    measurement in this project supports.

    Normalising each trace by its own first-quarter mean before averaging makes
    the plotted quantity the same one the verdict reports: a curve that ends
    above 1.0 rose across its window, one that ends below 1.0 fell. It also
    keeps the per-trace weighting the drift statistic uses -- each trace
    contributes its own shape once, rather than long or high-divergence traces
    dominating a pooled mean.

    Returns ``None`` when the first quarter is degenerate (empty, or mean <= 0),
    which is the same guard the drift ratio itself applies.
    """
    series = list(estimate)
    q = len(series) // 4
    if q < 1 or n_bins < 1:
        return None
    base = statistics.fmean(series[:q])
    if base <= 0:
        return None
    n = len(series)
    out: list[float] = []
    for b in range(n_bins):
        lo = int(b * n / n_bins)
        hi = max(int((b + 1) * n / n_bins), lo + 1)
        out.append(statistics.fmean(series[lo:hi]) / base)
    return out

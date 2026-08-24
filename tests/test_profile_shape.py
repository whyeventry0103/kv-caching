"""Hand-constructed checks for the profile shape statistics (Amendment P.2)."""

import sys as _sys
from pathlib import Path as _Path

import pytest

from kvcache.metrics.profile_shape import (
    classify_shape,
    phase_profile,
    running_integral,
    summarise_profiles,
)


def test_running_integral_accumulates_only_after_onset():
    est = [0.0] * 10 + [1.0] * 100  # onset at 10
    out = running_integral(est, 10)
    assert out["cum_at_50"] == pytest.approx(50.0)
    assert out["cum_at_100"] == pytest.approx(100.0)
    assert out["cum_at_200"] is None  # trace ends first
    assert out["cum_total"] == pytest.approx(100.0)


def test_phase_profile_localises_a_spike_at_the_prune_step():
    interval = 4
    est = [0.0] * 8 + [1.0, 0.0, 0.0, 0.0] * 10  # spike every 4 steps from onset
    phase = phase_profile(est, 8, interval)
    assert phase[0] == pytest.approx(1.0)
    assert all(p == pytest.approx(0.0) for p in phase[1:])


def test_flat_profile_does_not_drift():
    interval = 8
    est = [0.0] * 8 + [0.05] * (8 * 8)
    out = classify_shape(est, 8, interval)
    assert out["shape"] == "flat"
    assert not out["drifts"]


def test_a_rising_profile_is_classified_as_drift():
    interval = 8
    tail = [0.01 + 0.001 * i for i in range(8 * 20)]
    out = classify_shape([0.0] * 8 + tail, 8, interval)
    assert out["drifts"] is True
    assert out["drift_ratio"] > 1.5


def test_aggregate_phase_amplitude_separates_signal_from_its_sham():
    """The per-trace recovery axis was removed because a sham reproduced it.

    Eviction alignment is only claimable where the AGGREGATE phase profile has
    an amplitude the shuffled-label sham does not reproduce.
    """
    interval = 4
    cycle = [1.0, 0.5, 0.2, 0.1]
    out = summarise_profiles([([0.0] * 4 + cycle * 40, 4, interval)] * 8)
    assert out["phase_peak_index"] == 0
    assert out["phase_amplitude"] == pytest.approx(0.9)
    # Shuffling destroys the structure, so the sham amplitude must be far lower.
    assert out["phase_amplitude_sham"] < 0.5


def test_recovery_axis_is_gone_so_it_cannot_be_reported_by_accident():
    out = classify_shape([0.0] * 4 + [1.0, 0.5, 0.2, 0.1] * 20, 4, 4)
    assert "recovers" not in out
    assert "within_cycle_decay_fraction" not in out


def test_summarise_reports_counts_fractions_and_its_own_thresholds():
    interval = 4
    flat = [0.0] * 4 + [0.05] * (4 * 10)
    spiky = [0.0] * 4 + [1.0, 0.5, 0.2, 0.1] * 10
    out = summarise_profiles([(flat, 4, interval), (spiky, 4, interval)])
    assert out["n_profiles"] == 2
    assert sum(out["shape_counts"].values()) == 2
    assert out["thresholds"]["drift_ratio"] == 1.5
    assert "sham_seed" in out["thresholds"]
    assert "Definition 6" in out["scale_discipline"]


def test_profiles_without_an_onset_are_skipped_not_counted():
    out = summarise_profiles([([0.1] * 40, None, 4), ([0.1] * 40, 4, 4)])
    assert out["n_profiles"] == 1


def test_phase_amplitude_excess_is_positive_only_for_real_phase_structure():
    """A genuinely eviction-locked profile must beat its own sham."""
    from kvcache.metrics.profile_shape import phase_amplitude_excess

    interval = 4
    locked = ([0.0] * 4 + [1.0, 0.5, 0.2, 0.1] * 40, 4, interval)
    out = phase_amplitude_excess([locked] * 12, n_resamples=200, seed=0)
    assert out["excess"] > 0
    assert out["excess_excludes_zero"] is True


def test_phase_amplitude_excess_does_not_fire_on_structureless_noise():
    """The control that retired the per-trace statistic, as a standing test."""
    import random as _random

    from kvcache.metrics.profile_shape import phase_amplitude_excess

    rng = _random.Random(7)
    interval = 8
    noisy = [
        ([0.0] * 8 + [rng.random() for _ in range(8 * 40)], 8, interval)
        for _ in range(12)
    ]
    out = phase_amplitude_excess(noisy, n_resamples=200, seed=0)
    # No eviction-locked structure exists, so the excess must not exclude zero.
    assert out["excess_excludes_zero"] is False


def test_floor_exit_step_and_restriction_are_hand_checkable():
    """Amendment P.2-v2 windowing. Shared with the M.3 drift comparison."""
    from kvcache.metrics.profile_shape import floor_exit_step, restrict_to_floor_exit

    # First event sees prompt + one interval, nothing evicted -> prompt = 164-64 = 100.
    trace = [
        {"logical_context": 164, "at_floor": True},
        {"logical_context": 228, "at_floor": True},
        {"logical_context": 900, "at_floor": False},
        {"logical_context": 964, "at_floor": False},
    ]
    assert floor_exit_step(trace, 64) == 800  # 900 - 100

    # Never leaves the floor -> excluded, not zero.
    assert floor_exit_step([{"logical_context": 164, "at_floor": True}], 64) is None
    assert floor_exit_step([], 64) is None

    # Trim re-bases the onset: the whole retained window is post-onset.
    trimmed, onset, _ = restrict_to_floor_exit([0.1] * 1200, 65, trace, 64)
    assert len(trimmed) == 400
    assert onset == 0

    # Shorter than the 200-step floor -> excluded.
    assert restrict_to_floor_exit([0.1] * 900, 65, trace, 64) is None


def test_the_restriction_actually_changes_the_statistic(tmp_path):
    """A restricted run must not silently equal the unrestricted one.

    Builds one cell whose early (floor-bound) segment carries a strong
    prune-aligned spike and whose late segment is flat, so trimming the floor
    period MUST lower the phase amplitude. If the flag were a no-op this fails.
    """
    import importlib.util
    import json as _json

    root = _Path(__file__).resolve().parents[1]
    cell = tmp_path / "profiles" / "math500" / "model" / "random-256-s11"
    cell.mkdir(parents=True)
    interval, n = 64, 2000
    # Spiky before step 800, flat after.
    series = [
        (0.9 if (i % interval == 0) else 0.05) if i < 800 else 0.05 for i in range(n)
    ]
    (cell / "aa.json").write_text(
        _json.dumps(
            {
                "estimate": series,
                "onset_step": 64,
                "compression_interval": interval,
                "schema_version": "tv-profile-v2",
            }
        ),
        encoding="utf-8",
    )
    (cell / "aa.ratio_trace.json").write_text(
        _json.dumps(
            {
                "matched_ratio": 0.038,
                "min_budget": 32,
                "trace": [
                    {"logical_context": 64 + interval, "at_floor": True},
                    {"logical_context": 800 + 64, "at_floor": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location(
        "rps", root / "scripts" / "report_profile_shape.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def run(restrict):
        out = tmp_path / f"o{int(restrict)}.json"
        argv = [
            "report_profile_shape.py",
            "--profiles-dir",
            str(tmp_path / "profiles"),
            "--label",
            "t",
            "--out",
            str(out),
        ]
        if restrict:
            argv.append("--restrict-to-floor-exit")
        _sys.argv = argv
        assert mod.main() == 0
        return _json.loads(out.read_text(encoding="utf-8"))

    unres, res = run(False), run(True)
    assert unres["window"].startswith("WHOLE TRACE")
    assert res["window"].startswith("RESTRICTED")
    assert res["excluded"] == {
        "never_left_floor": 0,
        "window_too_short": 0,
        "no_ratio_trace": 0,
    }
    a_un = unres["per_policy"]["random"]["phase_amplitude"]
    a_re = res["per_policy"]["random"]["phase_amplitude"]
    assert a_re < a_un, f"restriction did not reduce amplitude: {a_re} vs {a_un}"

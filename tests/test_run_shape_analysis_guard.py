"""The floor-exit guard in run_shape_analysis (Amendments P.2-v2 and Q.1).

The original guard refused `restrict_to_floor_exit: false` outright, on the
grounds that "the unrestricted numbers already exist and are not the
pre-registered test". That is true of Amendment P.2-v2's ratio-matched cell and
is NOT a general truth: the Q.1 sawtooth arms are fixed-budget, have no
min_budget floor and no ratio_trace sidecar, so floor-exit restriction is
undefined for them and the whole-trace profile is the pre-registered test.

So the guard is now conditional on something detectable rather than declared --
only a ratio-matched press writes ratio_trace sidecars -- with an explicit
rationale required in every unrestricted config, so the window a verdict is read
on can never be chosen by omission.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_shape_analysis.py"


def _run(cfg: Path, runs_root: Path) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "REMOTE_RUNS_ROOT": str(runs_root)}
    return subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(cfg), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


def _cfg(tmp_path: Path, **over) -> Path:
    base = {
        "name": "shape_test",
        "source_run": "some-run",
        "budget": "256",
        "restrict_to_floor_exit": False,
        "unrestricted_rationale": "fixed-budget arm, no floor to exit",
        "run_artifact": "shape-test/out.json",
        "out_artifact": "results/diag/amendment-q/out.json",
    }
    base.update(over)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    return p


def _profiles(runs_root: Path, run: str, *, ratio_matched: bool) -> None:
    d = runs_root / run / "profiles" / "math500" / "m" / "random-256-s11"
    d.mkdir(parents=True)
    (d / "aaaa.json").write_text(
        '{"estimate":[0.0,0.1],"onset_step":0}', encoding="utf-8"
    )
    if ratio_matched:
        (d / "aaaa.ratio_trace.json").write_text('{"trace":[]}', encoding="utf-8")


def test_unrestricted_without_a_rationale_is_refused(tmp_path):
    runs = tmp_path / "runs"
    _profiles(runs, "some-run", ratio_matched=False)
    cfg = _cfg(tmp_path, unrestricted_rationale=None)
    r = _run(cfg, runs)
    assert r.returncode != 0
    assert "unrestricted_rationale" in (r.stdout + r.stderr)


def test_unrestricted_on_a_ratio_matched_run_is_refused_even_with_a_rationale(tmp_path):
    """The P.2-v2 case: the floor-bound early trace would dominate."""
    runs = tmp_path / "runs"
    _profiles(runs, "some-run", ratio_matched=True)
    cfg = _cfg(tmp_path)
    r = _run(cfg, runs)
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "ratio-matched" in out and "must be true" in out


def test_unrestricted_on_a_fixed_budget_run_with_a_rationale_is_allowed(tmp_path):
    runs = tmp_path / "runs"
    _profiles(runs, "some-run", ratio_matched=False)
    cfg = _cfg(tmp_path)
    r = _run(cfg, runs)
    assert r.returncode == 0, r.stdout + r.stderr


def test_restricted_still_works_on_a_ratio_matched_run(tmp_path):
    runs = tmp_path / "runs"
    _profiles(runs, "some-run", ratio_matched=True)
    cfg = _cfg(tmp_path, restrict_to_floor_exit=True, unrestricted_rationale=None)
    r = _run(cfg, runs)
    assert r.returncode == 0, r.stdout + r.stderr


def test_each_sawtooth_arm_has_its_own_config_because_intervals_cannot_mix(tmp_path):
    """phase_amplitude_excess raises on mixed intervals; one arm per config."""
    cfgs = sorted((ROOT / "configs" / "experiments").glob("shape_sawtooth_i*_q1.yaml"))
    assert len(cfgs) == 3, [c.name for c in cfgs]
    sources = set()
    for c in cfgs:
        d = yaml.safe_load(c.read_text(encoding="utf-8"))
        assert d["restrict_to_floor_exit"] is False
        assert d.get("unrestricted_rationale")
        sources.add(d["source_run"])
    assert sources == {
        "replay-sawtooth-i16",
        "replay-sawtooth-i64",
        "replay-sawtooth-i256",
    }


# --------------------------------------------------------------------------
# launch.sh re-exports REMOTE_RUNS_ROOT="$BASE_RUNS_ROOT/$RUN_NAME" so a
# generator writes inside its own run directory, and separately exports
# KVCACHE_BASE_RUNS_ROOT for the unscoped tree. An analysis job reads ANOTHER
# run's profiles, so it must use the base. The earlier tests set only
# REMOTE_RUNS_ROOT and therefore could not see this: they simulated a bare shell,
# not the launcher. These simulate the launcher.


def _run_env(cfg: Path, env_extra: dict) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(cfg), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


def test_source_profiles_resolve_against_the_base_root_under_launch_sh(tmp_path):
    """The real bug: three cells were marked stuck by a doubled run name."""
    base = tmp_path / "runs"
    _profiles(base, "replay-sawtooth-i16", ratio_matched=False)
    cfg = _cfg(tmp_path, source_run="replay-sawtooth-i16")
    r = _run_env(
        cfg,
        {
            # exactly what launch.sh exports
            "KVCACHE_BASE_RUNS_ROOT": str(base),
            "REMOTE_RUNS_ROOT": str(base / "shape-sawtooth-i16"),
        },
    )
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "replay-sawtooth-i16/profiles" in out
    assert (
        "shape-sawtooth-i16/replay-sawtooth-i16" not in out
    ), "run name appeared twice: the scoped root was used for the source"


def test_scoped_root_alone_still_works_when_there_is_no_base(tmp_path):
    """Run by hand with only REMOTE_RUNS_ROOT set: fall back, do not crash."""
    base = tmp_path / "runs"
    _profiles(base, "replay-sawtooth-i16", ratio_matched=False)
    cfg = _cfg(tmp_path, source_run="replay-sawtooth-i16")
    import os

    env = {k: v for k, v in os.environ.items() if k != "KVCACHE_BASE_RUNS_ROOT"}
    env["REMOTE_RUNS_ROOT"] = str(base)
    r = subprocess.run(
        [sys.executable, str(RUNNER), "--config", str(cfg), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_artifact_also_lands_under_the_base_so_the_marker_check_finds_it(tmp_path):
    """nightly_autorun resolves job.marker against the BASE runs root."""
    base = tmp_path / "runs"
    _profiles(base, "replay-sawtooth-i16", ratio_matched=False)
    cfg = _cfg(
        tmp_path,
        source_run="replay-sawtooth-i16",
        run_artifact="shape-sawtooth-i16/shape_sawtooth_i16.json",
    )
    r = _run_env(
        cfg,
        {
            "KVCACHE_BASE_RUNS_ROOT": str(base),
            "REMOTE_RUNS_ROOT": str(base / "shape-sawtooth-i16"),
        },
    )
    assert r.returncode == 0, r.stdout + r.stderr
    expected = str(base / "shape-sawtooth-i16" / "shape_sawtooth_i16.json")
    assert expected in (r.stdout + r.stderr), (
        f"artifact must land at {expected} or the orchestrator's marker check "
        "will never see it and the cell is stuck forever"
    )

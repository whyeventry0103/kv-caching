"""The nightly email must say when the queue cannot advance.

For days the report listed restoration-lag1 as "next" and flagged the night
CORE DONE, while the orchestrator's storage guard refused that job on every
tick. A blocked queue read exactly like runs silently failing. The report now
evaluates the same guard the orchestrator applies and flags BLOCKED.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location(
        "nightly_report_under_test", ROOT / "scripts" / "nightly_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nightly_report_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _setup(monkeypatch, tmp_path, used_gb):
    rep = _report()
    na = rep.na
    runs = tmp_path / "runs"
    runs.mkdir()
    job = na.Job(
        name="restoration-lag1",
        command="replay",
        config="c.yaml",
        run_name="restoration-lag1",
        total=800,
        seeds=[11, 23, 47, 89],
        min_free_gb=28,
        progress_kind="grid",
        requires=None,
        est_output_gb=0.45,
    )
    monkeypatch.setitem(na.BASE_ENV, "REMOTE_RUNS_ROOT", str(runs))
    monkeypatch.setattr(na, "load_queue", lambda _q: (True, [job]))
    monkeypatch.setattr(na, "load_state", lambda: {})
    monkeypatch.setattr(na, "tmux_session_names", lambda: [])
    monkeypatch.setattr(na, "storage_used_gb", lambda *a, **k: used_gb)
    monkeypatch.setattr(rep, "PREV_SNAPSHOT", tmp_path / "prev.json")
    monkeypatch.setattr(rep, "CORE_JOBS", set())
    return rep


def test_a_refused_next_job_flags_the_night_blocked(monkeypatch, tmp_path):
    # 71.0 of the 72 GiB allocation (R.15): 1.0 GiB headroom < 0.45 + 1 margin
    rep = _setup(monkeypatch, tmp_path, used_gb=71.0)
    subject, body = rep.generate_report(write_snapshot=False)
    assert "BLOCKED (storage: restoration-lag1)" in subject
    assert "WILL NOT LAUNCH" in body
    assert "REFUSING TO LAUNCH restoration-lag1" in body


def test_a_job_that_fits_is_not_flagged(monkeypatch, tmp_path):
    rep = _setup(monkeypatch, tmp_path, used_gb=40.0)
    subject, body = rep.generate_report(write_snapshot=False)
    assert "BLOCKED" not in subject
    assert "WILL NOT LAUNCH" not in body
    assert "next: restoration-lag1" in body


@pytest.mark.parametrize("used", [71.0, 40.0])
def test_the_report_never_raises_whatever_the_storage_state(
    monkeypatch, tmp_path, used
):
    """generate_report's contract: failures go into the body, never out."""
    rep = _setup(monkeypatch, tmp_path, used_gb=used)
    rep.generate_report(write_snapshot=False)

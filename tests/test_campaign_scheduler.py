"""The scheduler enforces cumulative campaign caps through its real main()
(Meeting 5 review, G0.2). Side effects (tmux, preflight, GPUs, disk) are
stubbed; queue, ledger, launch bounds and refusals are the production code."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "na_campaign", ROOT / "scripts" / "nightly_autorun.py"
)
M = importlib.util.module_from_spec(SPEC)
sys.modules["na_campaign"] = M
SPEC.loader.exec_module(M)

from kvcache import campaign_budget as cb  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    queue = tmp_path / "queue.yaml"
    queue.write_text(
        "version: 1\nenabled: true\njobs:\n"
        "  - name: pilot\n    command: grid_protsel\n"
        "    config: configs/experiments/grid_math500_protsel_pilot.yaml\n"
        "    run_name: pilot\n    total: 16\n    seeds: [11]\n"
        "    min_free_gb: 28\n    progress_kind: grid\n    requires: null\n"
        "    campaign: P3\n"
    )
    camps = tmp_path / "campaigns.yaml"
    camps.write_text("campaigns:\n  P3:\n    cap_gpu_hours: 6\n")
    monkeypatch.setattr(M, "CAMPAIGNS_PATH", camps)
    monkeypatch.setattr(M, "LEDGER_PATH", tmp_path / "ledger.json")
    monkeypatch.setattr(M, "LAUNCH_RECORDS", tmp_path / "launches")
    monkeypatch.setattr(M, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setitem(M.BASE_ENV, "REMOTE_RUNS_ROOT", str(runs))
    monkeypatch.setattr(M, "run_preflight", lambda job: (True, ""))
    monkeypatch.setattr(M, "storage_used_gb", lambda *a, **k: 0.0)
    monkeypatch.setattr(M, "tmux_session_names", lambda: [])
    monkeypatch.setattr(M, "write_runner", lambda spec: None)
    launched: list = []
    monkeypatch.setattr(M, "launch_session", lambda spec: launched.append(spec))
    return {"queue": queue, "launched": launched, "tmp": tmp_path}


def _main(e, monkeypatch, now="2026-10-15T01:45:00"):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nightly_autorun",
            "--queue",
            str(e["queue"]),
            "--now",
            now,
            "--fake-gpus",
            "0:47000,1:47000",
            "--force",
        ],
    )
    return M.main()


def test_launch_is_bounded_by_the_campaign_and_reserved(env, monkeypatch, capsys):
    assert _main(env, monkeypatch) == 0
    (spec,) = env["launched"]  # one seed -> one shard
    assert spec.env["KVCACHE_LAUNCH_ID"].startswith("P3-")
    # window is ~9 h, but the P3 cap (6 h) bounds the process to 5.95 h
    assert float(spec.env["KVCACHE_MAX_HOURS"]) == pytest.approx(5.95)
    led = json.loads((env["tmp"] / "ledger.json").read_text())
    (rec,) = led["launches"].values()
    assert rec["status"] == cb.RESERVED and rec["reserved_hours"] == pytest.approx(6.0)


def test_an_exhausted_campaign_is_refused_with_its_deficit(env, monkeypatch, capsys):
    caps = {"P3": 6.0}
    from datetime import datetime, timedelta

    t = datetime(2026, 10, 14, 2, 0)
    cb.record_historical(
        env["tmp"] / "ledger.json",
        caps,
        launch_id="h",
        campaign="P3",
        job="pilot",
        gpu=0,
        started=t,
        ended=t + timedelta(hours=5.9),
        evidence="test",
    )
    assert _main(env, monkeypatch) == 0
    assert env["launched"] == []
    out = capsys.readouterr().out
    assert "CAP pilot" in out and "deficit" in out


def test_a_failed_tmux_launch_releases_its_reservation(env, monkeypatch, capsys):
    def boom(spec):
        raise subprocess.CalledProcessError(1, "tmux")

    monkeypatch.setattr(M, "launch_session", boom)
    assert _main(env, monkeypatch) == 0
    led = json.loads((env["tmp"] / "ledger.json").read_text())
    (rec,) = led["launches"].values()
    assert rec["status"] == cb.SETTLED and rec["settled_hours"] == 0.0
    assert "launch FAILED" in capsys.readouterr().out


def test_finished_shards_are_settled_on_the_next_tick(env, monkeypatch, capsys):
    assert _main(env, monkeypatch) == 0
    (spec,) = env["launched"]
    lid = spec.env["KVCACHE_LAUNCH_ID"]
    rec_dir = env["tmp"] / "launches"
    rec_dir.mkdir()
    (rec_dir / f"{lid}.json").write_text(
        json.dumps({"start_epoch": 1000.0, "end_epoch": 1000.0 + 1800, "rc": 0})
    )
    env["launched"].clear()
    _main(env, monkeypatch, now="2026-10-15T02:30:00")
    assert "settled" in capsys.readouterr().out
    s = cb.summary(env["tmp"] / "ledger.json", {"P3": 6.0})["P3"]
    # 0.5 h settled for the first shard + the second launch's new reservation
    assert s["charged"] == pytest.approx(0.5 + 5.5)


def test_queue_refuses_an_unknown_campaign(tmp_path, monkeypatch):
    q = tmp_path / "q.yaml"
    q.write_text(
        "version: 1\nenabled: true\njobs:\n  - name: x\n    command: grid\n"
        "    config: c.yaml\n    run_name: x\n    total: 1\n    seeds: [11]\n"
        "    min_free_gb: 1\n    progress_kind: grid\n    requires: null\n"
        "    campaign: P9\n"
    )
    with pytest.raises(SystemExit, match="unknown campaign"):
        M.load_queue(q)

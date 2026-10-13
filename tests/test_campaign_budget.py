"""Cumulative campaign GPU-hour caps (Meeting 5 review, G0.2): concurrency,
restart, failed launches, two-GPU charging, the combined P3 cap, unreconciled
crashes, and refusal with the deficit named."""

from __future__ import annotations

import json
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kvcache import campaign_budget as cb

CAPS = {"P1": 20.0, "P2": 2.0, "P3": 6.0}
NOW = datetime(2026, 10, 15, 1, 45)


def _res(path, campaign="P3", job="j", n=1, window=8.98, now=NOW):
    shards = [cb.Shard(f"{job}-g{i}", i) for i in range(n)]
    return cb.reserve(
        path,
        CAPS,
        campaign=campaign,
        job=job,
        shards=shards,
        window_hours=window,
        now=now,
    )


def test_reservation_is_bounded_by_the_campaign_not_the_window(tmp_path):
    led = tmp_path / "ledger.json"
    out, msg = _res(led, "P3", n=1, window=8.98)
    ((lid, bound),) = out.items()
    # P3 cap 6 h: one shard may reserve all 6 h, process bound = 6 - grace
    assert bound == pytest.approx(5.95)
    assert bound <= 6 - cb.GRACE_HOURS
    assert cb.summary(led, CAPS)["P3"]["remaining"] == pytest.approx(0.0)


def test_two_gpus_are_charged_twice_and_split_fairly(tmp_path):
    led = tmp_path / "ledger.json"
    out, _ = _res(led, "P2", n=2, window=8.98)
    assert len(out) == 2
    assert all(b == pytest.approx(0.95) for b in out.values())  # 2 h / 2 - grace
    assert cb.summary(led, CAPS)["P2"]["charged"] == pytest.approx(2.0)


def test_settle_releases_unused_time_and_restart_keeps_spending(tmp_path):
    led = tmp_path / "ledger.json"
    out, _ = _res(led, "P3", n=1)
    ((lid, _),) = out.items()
    cb.settle(led, CAPS, lid, started=NOW, ended=NOW + timedelta(hours=1.5), rc=0)
    # a "restart": a fresh read from disk sees the settled spending
    s = cb.summary(led, CAPS)["P3"]
    assert s["charged"] == pytest.approx(1.5) and s["remaining"] == pytest.approx(4.5)
    out2, _ = _res(led, "P3", n=1, now=NOW + timedelta(days=1))
    assert list(out2.values())[0] == pytest.approx(4.45)  # resume does not reset


def test_combined_p3_cap_across_three_jobs(tmp_path):
    """grid + readout + reverse share ONE 6 h allowance."""
    led = tmp_path / "ledger.json"
    for job, hours in (("grid", 2.0), ("readout", 2.0), ("reverse", 1.5)):
        out, _ = _res(led, "P3", job=job, n=1, window=hours)
        ((lid, _),) = out.items()
        cb.settle(led, CAPS, lid, started=NOW, ended=NOW + timedelta(hours=hours), rc=0)
    assert cb.summary(led, CAPS)["P3"]["charged"] == pytest.approx(5.5)
    # 0.5 h left funds exactly one shard bounded at 0.45 h ...
    out, _ = _res(led, "P3", job="grid", n=1)
    assert list(out.values()) == [pytest.approx(0.45)]
    # ... and then nothing: the next request is refused with the deficit
    out, why = _res(led, "P3", job="readout", n=1)
    assert out is None and "deficit" in why


def test_refusal_names_the_deficit_when_exhausted(tmp_path):
    led = tmp_path / "ledger.json"
    out, _ = _res(led, "P2", n=1)
    ((lid, _),) = out.items()
    cb.settle(led, CAPS, lid, started=NOW, ended=NOW + timedelta(hours=1.9), rc=0)
    out, why = _res(led, "P2", n=1)
    assert out is None
    assert "P2" in why and "deficit" in why and "remaining allowance 0.100" in why


def test_failed_launch_and_unreconciled_crash(tmp_path):
    led = tmp_path / "ledger.json"
    ends = tmp_path / "launches"
    ends.mkdir()
    out, _ = _res(led, "P1", n=2, window=8.0)
    a, b = sorted(out)
    cb.release_unlaunched(led, CAPS, a)  # tmux refused: zero charge
    # b's session is gone and left no end record: stays charged in full
    notes = cb.reconcile(led, CAPS, live_sessions=[], end_records_dir=ends)
    assert any("UNRECONCILED" in n for n in notes)
    d = json.loads(led.read_text())
    assert d["launches"][a]["settled_hours"] == 0.0
    assert d["launches"][b]["status"] == cb.UNRECONCILED
    assert cb.summary(led, CAPS)["P1"]["charged"] == pytest.approx(8.05)


def test_reconcile_settles_from_the_launchers_end_record(tmp_path):
    led = tmp_path / "ledger.json"
    ends = tmp_path / "launches"
    ends.mkdir()
    out, _ = _res(led, "P1", n=1, window=8.0)
    ((lid, _),) = out.items()
    t0 = NOW.timestamp()
    (ends / f"{lid}.json").write_text(json.dumps({"start_epoch": t0}))
    # still running -> left open even though a start record exists
    assert cb.reconcile(led, CAPS, live_sessions=["j-g0"], end_records_dir=ends) == []
    (ends / f"{lid}.json").write_text(
        json.dumps({"start_epoch": t0, "end_epoch": t0 + 3600 * 2.25, "rc": 124})
    )
    notes = cb.reconcile(led, CAPS, live_sessions=[], end_records_dir=ends)
    assert notes and "2.250" in notes[0]
    assert cb.summary(led, CAPS)["P1"]["charged"] == pytest.approx(2.25)


def _worker(args):
    path, i = args
    out, _ = cb.reserve(
        Path(path),
        CAPS,
        campaign="P3",
        job=f"j{i}",
        shards=[cb.Shard(f"j{i}-g0", 0)],
        window_hours=1.0,
        now=NOW,
    )
    return out is not None


def test_concurrent_reservations_never_exceed_the_cap(tmp_path):
    led = tmp_path / "ledger.json"
    # 12 concurrent requests of up to 1.05 h each against a 6 h cap
    with mp.get_context("fork").Pool(6) as pool:
        granted = pool.map(_worker, [(str(led), i) for i in range(12)])
    assert sum(granted) >= 1
    assert cb.summary(led, CAPS)["P3"]["charged"] <= 6.0 + 1e-9


def test_historical_charge_is_idempotent(tmp_path):
    led = tmp_path / "ledger.json"
    for _ in range(2):
        cb.record_historical(
            led,
            CAPS,
            launch_id="P2-hist-20261013",
            campaign="P2",
            job="protsel-smoke",
            gpu=0,
            started=NOW,
            ended=NOW + timedelta(hours=0.2584),
            evidence="bundle",
        )
    assert cb.summary(led, CAPS)["P2"]["charged"] == pytest.approx(0.2584)

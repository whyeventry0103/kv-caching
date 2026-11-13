"""CPU acceptance for the GSM8K timing scheduler integration (approved by the
owner 2026-11-13: campaign M5B-GSM8K-TIMING, cap 3 GPU-h)."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nightly_autorun as scheduler  # noqa: E402
from run_bank_manifest import resolve_design  # noqa: E402

from kvcache.cli import load_config  # noqa: E402
from kvcache.io_schema import result_path  # noqa: E402
from kvcache.token_ids import token_ids_path  # noqa: E402


def _job():
    _, jobs = scheduler.load_queue(scheduler.QUEUE_DEFAULT)
    return next(j for j in jobs if j.name == "bank-gsm8k-timing-m5")


def test_pilot_is_released_only_under_its_capped_campaign(tmp_path):
    job = _job()
    cfg = load_config(ROOT / job.config)
    assert cfg["held"] is False and cfg["scale_authorized"] is True
    assert job.blocked is None and job.campaign == "M5B-GSM8K-TIMING"
    assert scheduler.load_campaign_caps()["M5B-GSM8K-TIMING"] == 3.0
    eligible, _ = scheduler.launch_candidates(
        tmp_path, [job], skip=set(), live={}, state={}
    )
    assert [j.name for j, _ in eligible] == ["bank-gsm8k-timing-m5"]


def test_held_copy_of_the_pilot_is_still_blocked(tmp_path):
    job = dataclasses.replace(_job(), blocked="held", campaign=None)
    eligible, notes = scheduler.launch_candidates(
        tmp_path, [job], skip=set(), live={}, state={}
    )
    assert eligible == []
    assert any("BLOCKED bank-gsm8k-timing-m5" in s for s in notes)


def test_campaign_reservation_never_exceeds_the_3h_cap(tmp_path):
    from kvcache import campaign_budget as cb

    caps = scheduler.load_campaign_caps()
    ledger = tmp_path / "ledger.json"
    granted, why = cb.reserve(
        ledger,
        caps,
        campaign="M5B-GSM8K-TIMING",
        job="bank-gsm8k-timing-m5",
        shards=[cb.Shard("s11", 0), cb.Shard("s23", 1)],
        window_hours=8.0,
        now=datetime(2026, 11, 14, 1, 45),
    )
    assert granted is not None, why
    assert all(lid.startswith("M5B-GSM8K-TIMING-") for lid in granted)
    assert sum(granted.values()) <= 3.0 + 1e-9
    assert cb.summary(ledger, caps)["M5B-GSM8K-TIMING"]["remaining"] <= 1e-9


def test_scheduler_counts_only_unique_expected_pairs(tmp_path):
    job = _job()
    cfg = load_config(ROOT / job.config)
    design = resolve_design(cfg)
    args = (
        tmp_path / job.run_name,
        job.bench,
        cfg["models"][0],
        "fullcache",
        "full",
        11,
    )
    result, side = result_path(*args), token_ids_path(*args)
    row = {
        "pid": design["pids"][0],
        "bench": args[1],
        "model": args[2],
        "press": args[3],
        "budget": args[4],
        "decoding": {"seed": 11},
    }
    sidecar = {k: v for k, v in row.items() if k != "decoding"}
    sidecar["seed"] = 11
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert scheduler.count_progress(tmp_path, job) == 0
    side.parent.mkdir(parents=True)
    side.write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
    assert scheduler.count_per_seed(tmp_path, job) == {11: 1, 23: 0}
    result.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    assert scheduler.count_progress(tmp_path, job) == 0


def test_direct_alias_refuses_without_cap():
    job = _job()
    env = dict(os.environ, KVCACHE_REMOTE="1")
    env.pop("KVCACHE_LAUNCH_ID", None)
    p = subprocess.run(
        ["bash", "scripts/launch.sh", "--i-understand", job.command, job.config],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 2
    assert "requires its capped scheduler reservation" in p.stderr

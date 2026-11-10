"""CPU acceptance for the held GSM8K scheduler integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def test_pilot_is_held_and_no_cap_is_active(tmp_path):
    job = _job()
    cfg = load_config(ROOT / job.config)
    assert cfg["held"] is True and cfg["scale_authorized"] is False
    assert job.blocked and job.campaign is None
    assert "M5B-GSM8K-TIMING" not in scheduler.load_campaign_caps()
    eligible, notes = scheduler.launch_candidates(
        tmp_path, [job], skip=set(), live={}, state={}
    )
    assert eligible == []
    assert any("BLOCKED bank-gsm8k-timing-m5" in s for s in notes)


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

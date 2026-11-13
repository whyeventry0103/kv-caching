"""M5-B replication scheduler integration (prepared 2026-11-13; GSM8K released
2026-11-14): GSM8K cells run only under their stage campaigns, BBH cells stay
blocked with no campaign, grid progress counts only complete pairs of the
design's own keys on the new benchmarks, and the launch aliases refuse without
a capped M5-B reservation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nightly_autorun as scheduler  # noqa: E402

from kvcache.cli import load_config  # noqa: E402
from kvcache.io_schema import result_path  # noqa: E402
from kvcache.manifest_grid import resolve_grid_design  # noqa: E402
from kvcache.token_ids import token_ids_path  # noqa: E402


def _jobs():
    _, jobs = scheduler.load_queue(scheduler.QUEUE_DEFAULT)
    return {j.name: j for j in jobs if j.name.endswith("-m5b")}


# Released 2026-11-14 by the owner: GSM8K under its two stage campaigns with a
# frozen design; BBH stays held, blocked and uncapped until its own read.
STAGE_CAMPAIGNS = {"M5B-GSM8K-S1": 140, "M5B-GSM8K-S2": 132}


def test_gsm8k_cells_are_released_under_their_stage_caps_and_bbh_stays_held():
    jobs = _jobs()
    assert len(jobs) == 22
    caps = scheduler.load_campaign_caps()
    for name, cap in STAGE_CAMPAIGNS.items():
        assert caps[name] == cap
    assert not [c for c in caps if c.startswith("M5B-BBH-S")]
    for job in jobs.values():
        cfg = load_config(ROOT / job.config)
        if "-gsm8k-" in job.name:
            s2 = "-base256-" in job.name or "-h2o256-" in job.name
            stage = "S2" if s2 else "S1"
            assert not job.blocked and job.campaign == f"M5B-GSM8K-{stage}", job.name
            assert cfg["held"] is False and cfg["scale_authorized"] is True
            want = "final"
        else:
            assert job.blocked and job.campaign is None, job.name
            assert cfg["held"] is True and cfg["scale_authorized"] is False
            want = "pending"
        if job.command != "bank_manifest_m5b":
            assert cfg["design_status"] == want
        if job.command == "readout_manifest":
            assert job.requires == cfg["reference_run_name"]


@pytest.mark.parametrize("bench_job", ["grid-gsm8k-pp256-m5b", "grid-bbh-pp256-m5b"])
def test_manifest_grid_progress_counts_only_complete_design_pairs(tmp_path, bench_job):
    job = _jobs()[bench_job]
    design = resolve_grid_design(load_config(ROOT / job.config), kind="grid")
    run_dir = tmp_path / job.run_name

    def write(pid, press, seed, *, side=True):
        args = (run_dir, design["benchmark"], design["model"], press, "256", seed)
        row = {"pid": pid, "press": press, "budget": "256", "decoding": {"seed": seed}}
        for path, rec in (
            (result_path(*args), row),
            (token_ids_path(*args), {**row, "seed": seed}) if side else (None, None),
        ):
            if path is None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    pids = design["pids"]
    write(pids[0], "random_pp", 11)  # complete pair: counts
    write(pids[1], "random_pp", 11, side=False)  # orphan row: does not
    write("not-in-split", "random_pp", 23)  # foreign key: does not
    assert scheduler.count_progress(tmp_path, job) == 1
    assert scheduler.count_per_seed(tmp_path, job) == {11: 1}


def _launch(tmp_path, command, config, env):
    """Run a COPY of launch.sh under tmp_path, so a launch record written for a
    fake KVCACHE_LAUNCH_ID never lands in the checkout's logs/."""
    import shutil

    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copy(ROOT / "scripts/launch.sh", tmp_path / "scripts/launch.sh")
    (tmp_path / config).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / config).write_text((ROOT / config).read_text())
    return subprocess.run(
        ["bash", "scripts/launch.sh", "--i-understand", command, config],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "command, config, launch_id",
    [
        ("grid_manifest", "configs/experiments/grid_gsm8k_pp256_m5b.yaml", None),
        (
            "grid_manifest",
            "configs/experiments/grid_gsm8k_pp256_m5b.yaml",
            "M5B-GSM8K-TIMING-x",
        ),
        (
            "grid_manifest",
            "configs/experiments/grid_gsm8k_template_m5b.yaml",
            "M5B-GSM8K-S1-x",
        ),
        (
            "readout_manifest",
            "configs/experiments/readout_math500_pp256_s1.yaml",
            "M5B-GSM8K-S1-x",
        ),
        (
            "bank_manifest_m5b",
            "configs/experiments/bank_gsm8k_timing_m5.yaml",
            "M5B-GSM8K-S1-x",
        ),
    ],
)
def test_replication_aliases_refuse_without_a_capped_m5b_reservation(
    tmp_path, command, config, launch_id
):
    env = dict(os.environ, KVCACHE_REMOTE="1", KVCACHE_MAX_HOURS="1")
    env.pop("KVCACHE_LAUNCH_ID", None)
    if launch_id:
        env["KVCACHE_LAUNCH_ID"] = launch_id
    p = _launch(tmp_path, command, config, env)
    assert p.returncode == 2
    assert "requires its capped scheduler reservation" in p.stderr

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from kvcache.io_schema import ResultRow
from kvcache.replay import ReplayProfile, build_replay_row, write_replay_profile

ROOT = Path(__file__).resolve().parents[1]


def _reference_row():
    return asdict(
        ResultRow(
            bench="math500",
            pid="p1",
            model="model",
            press="fullcache",
            budget="full",
            compression_ratio=1.0,
            decoding={"temp": 0.6, "top_p": 0.95, "max_new": 16, "seed": 11},
            trace=r"\boxed{2}",
            answer_raw="2",
            answer_canon="2",
            correct=True,
            n_tokens=1,
            truncated=False,
            wallclock_s=1.0,
            runmeta={
                "commit": "abc",
                "dirty": False,
                "config_hash": "config",
                "host": "host",
                "gpu": "gpu",
                "cuda_visible": "0",
            },
            versions={"torch": "x", "transformers": "y", "kvpress": "z"},
            ts="2026-06-14T00:00:00Z",
            has_answer_marker=True,
            finish_reason="marker",
        )
    )


def test_replay_audit_accepts_scalar_profile(tmp_path):
    profile = ReplayProfile(
        estimate=(0.0,),
        lower=(0.0,),
        upper=(0.0,),
        reference_token_ids=(1,),
        onset_step=None,
        compression_interval=None,
        compression_events=0,
        max_before_prune=None,
        max_after_prune=None,
    )
    path = write_replay_profile(
        tmp_path,
        reference_key=("math500", "p1", "model", "fullcache", "full", 11),
        press_name="fullcache",
        budget="full",
        pid_hash="pid",
        profile=profile,
    )
    row = build_replay_row(
        _reference_row(),
        press_name="fullcache",
        budget="full",
        compression_ratio=1.0,
        replay_of="reference",
        tv_profile_path=str(path.relative_to(tmp_path)),
        profile=profile,
    )
    result_path = tmp_path / "math500/model/fullcache-full-s11.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(row) + "\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_replay.py"),
            "--run-root",
            str(tmp_path),
            "--expected-rows",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

"""launch.sh end records carry the RUNNER's exit code (2026-10-15).

The run block is piped into tee, so the exit trap used to see tee's status:
every launch -- a timer stop (124) or a crash -- was recorded as rc 0 in the
campaign ledger. Hours were unaffected (start/end times); the label was wrong.
This runs the real launch.sh with a fake runner that exits with a chosen code.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rc", [0, 3, 124])
def test_end_record_carries_the_runner_exit_code(tmp_path, rc):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    fake = venv / "bin" / "python"
    fake.write_text(f"#!/usr/bin/env bash\nexit {rc}\n")
    fake.chmod(0o755)
    tag = uuid.uuid4().hex[:8]
    lid, run = f"TEST-{tag}", f"test-launch-rc-{tag}"
    env = {
        **os.environ,
        "KVCACHE_REMOTE": "1",
        "KVCACHE_PERSONAL_FREE_GB": "20",
        "KVCACHE_MIN_FILESYSTEM_FREE_GB": "0",
        "REMOTE_RUNS_ROOT": str(tmp_path / "runs"),
        "REMOTE_CACHE_ROOT": str(tmp_path / "cache"),
        "REMOTE_DATA_ROOT": str(tmp_path / "data"),
        "ENVIRONMENT_NAME": str(venv),
        "KVCACHE_RUN_NAME": run,
        "KVCACHE_LAUNCH_ID": lid,
    }
    (tmp_path / "runs").mkdir()
    rec = ROOT / "logs" / "launches" / f"{lid}.json"
    log = ROOT / "logs" / f"{run}.log"
    try:
        p = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/launch.sh"),
                "--i-understand",
                "grid",
                "configs/experiments/grid_math500_dissociation.yaml",
                # as the scheduler launches it; also avoids bash 3.2's
                # "unbound variable" on an empty array under set -u (the
                # server's bash 5 is unaffected)
                "--resume",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr  # launch.sh's own status is unchanged
        d = json.loads(rec.read_text())
        assert d["launch_id"] == lid and d["rc"] == rc
        assert d["end_epoch"] >= d["start_epoch"]
        assert not rec.with_suffix(".rc").exists()  # the side file is consumed
    finally:
        for f in (rec, rec.with_suffix(".rc"), log):
            f.unlink(missing_ok=True)

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("build_bank.py", "smoke_fullcache.yaml"),
    ("build_bank.py", "smoke_fullcache_4096.yaml"),
    ("build_bank.py", "pilot_fullcache.yaml"),
    ("build_bank.py", "pilot_fullcache_v2.yaml"),
    ("build_bank.py", "pilot_fullcache_tokenized.yaml"),
    ("probe_decoding.py", "probe_decoding.yaml"),
    ("probe_decoding.py", "probe_decoding_random.yaml"),
    ("run_grid.py", "pilot_compression.yaml"),
    ("build_bank.py", "bank_fullcache.yaml"),
    ("run_grid.py", "grid_compression.yaml"),
    ("run_replay.py", "replay_decomposition.yaml"),
    ("run_replay.py", "pilot_replay_self_smoke.yaml"),
    ("run_replay.py", "pilot_replay_self.yaml"),
    ("run_replay.py", "pilot_replay_compression.yaml"),
    ("run_replay.py", "pilot_readout.yaml"),
    ("run_metrics.py", "add_permutation.yaml"),
    ("run_mechanism.py", "mechanism_ladder.yaml"),
    ("make_figures.py", "negative_control.yaml"),
]


@pytest.mark.parametrize(("script", "config"), CASES)
def test_every_experiment_script_dry_runs(script, config):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--config",
            str(ROOT / "configs" / "experiments" / config),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout


def test_costly_execution_refuses_without_confirmation():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_grid.py"),
            "--config",
            str(ROOT / "configs" / "experiments" / "grid_compression.yaml"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "REFUSED" in result.stderr


def test_heavy_modules_import_without_torch_or_cuda():
    import kvcache.datasets
    import kvcache.generation
    import kvcache.models

    assert callable(kvcache.datasets.data_root)
    assert callable(kvcache.generation.generate_full_cache)
    assert callable(kvcache.models.load_model)


def test_every_queued_runner_accepts_the_orchestrator_argv_contract():
    """nightly_autorun appends --resume --seeds to EVERY job it launches.

    A runner that rejects them dies at argparse with rc=2 before doing any
    work, and the orchestrator retries until it marks the job stuck. That is
    exactly what happened to shape-p2v2-restricted on 2026-08-25: three failed
    launches, an empty run directory, and no artifact. Dry-running a runner by
    hand does not catch it, because the hand invocation omits the flags the
    orchestrator adds -- so this test asserts the contract directly, for every
    command the queue can dispatch.

    Checked via --help, which exercises the parser definition without running
    anything: these runners are heavy and some have no --dry-run at all.
    """
    import subprocess
    import sys as _sys

    import yaml

    root = Path(__file__).resolve().parents[1]
    queue = yaml.safe_load((root / "configs" / "nightly_queue.yaml").read_text())
    jobs = queue["jobs"] if isinstance(queue, dict) else queue
    launch = (root / "scripts" / "launch.sh").read_text()

    for command in sorted({j["command"] for j in jobs}):
        match = re.search(
            rf'^\s*{re.escape(command)}\)\s*PY_SCRIPT="([^"]+)"', launch, re.M
        )
        assert match, f"launch.sh has no dispatch for queued command {command!r}"
        script = root / match.group(1)
        assert script.exists(), f"{script} missing for command {command!r}"

        helped = subprocess.run(
            [_sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        assert helped.returncode == 0, f"{script.name} --help failed"
        for flag in ("--resume", "--seeds"):
            assert flag in helped.stdout, (
                f"{script.name} does not accept {flag}, which the nightly "
                f"orchestrator appends to every launch -- it would die rc=2"
            )

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("build_bank.py", "bank_fullcache.yaml"),
    ("run_grid.py", "grid_compression.yaml"),
    ("run_replay.py", "replay_decomposition.yaml"),
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

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_replay", ROOT / "scripts" / "run_replay.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_readout_path_integer_budget(tmp_path):
    path = MODULE._readout_path(tmp_path, "streamingllm", 256, 11)
    assert path == tmp_path / "readout" / "streamingllm-256-s11.jsonl"


def test_readout_path_full_budget(tmp_path):
    # Regression: the FC-readout resampling control (readout_math500_fc_control)
    # uses budgets [full]; the readout branch cast int(budget) and crashed with
    # ValueError on 'full' across three launches (STUCK, 2026-07-11). "full" must
    # pass through, matching the bank file-naming convention (fullcache-full-sN).
    path = MODULE._readout_path(tmp_path, "fullcache", "full", 11)
    assert path == tmp_path / "readout" / "fullcache-full-s11.jsonl"

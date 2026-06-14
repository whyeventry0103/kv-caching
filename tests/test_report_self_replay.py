import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_self_replay", ROOT / "scripts" / "report_self_replay.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_self_replay_report_summarizes_exact_identity(tmp_path):
    run_root = tmp_path / "run"
    row_path = run_root / "math500" / "model" / "fullcache-full-s11.jsonl"
    profile_path = run_root / "profiles" / "profile.json"
    row_path.parent.mkdir(parents=True)
    profile_path.parent.mkdir(parents=True)
    row_path.write_text(
        json.dumps(
            {
                "pid": "p1",
                "decoding": {"seed": 11},
                "tv_profile_path": "profiles/profile.json",
            }
        )
        + "\n"
    )
    profile_path.write_text(
        json.dumps(
            {
                "estimate": [0.0, 0.0, 0.0],
                "onset_step": None,
                "compression_events": 0,
            }
        )
    )

    summary, profiles = MODULE.build_summary(run_root)

    assert summary["rows"] == 1
    assert summary["steps"] == 3
    assert summary["global_max_tv"] == 0.0
    assert summary["no_tdp_crossing_fraction"] == 1.0
    assert profiles[0]["threshold_crossings"]["p05"] == 0

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_m4", ROOT / "scripts" / "report_m4.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_m4_report_uses_accuracy_scale_residual(tmp_path):
    full = tmp_path / "full"
    comp = tmp_path / "comp"
    replay = tmp_path / "replay"
    readout = tmp_path / "readout"
    full_rows = [
        {"pid": f"p{i}", "decoding": {"seed": i}, "correct": True, "truncated": False}
        for i in range(20)
    ]
    comp_rows = [
        {
            "pid": f"p{i}",
            "decoding": {"seed": i},
            "press": "streamingllm",
            "budget": 256,
            "correct": i < 4,
            "truncated": False,
            "n_tokens": 10,
        }
        for i in range(20)
    ]
    replay_rows = [
        {
            "press": "streamingllm",
            "budget": 256,
            "auc_tv": 0.1,
            "tv_profile_path": f"profiles/{i}.json",
            "frac_above": {"p05": 0.2, "p10": 0.1},
            "onset_step": 65,
        }
        for i in range(20)
    ]
    readout_rows = [
        {
            "reference_key": ["math500", f"p{i}", "model", "fullcache", "full", i],
            "condition": {"press": "streamingllm", "budget": 256},
            "readout_correct": i < 9,
            "generated_token_ids": [1, 2],
        }
        for i in range(20)
    ]
    _write(full / "math500" / "rows.jsonl", full_rows)
    _write(comp / "math500" / "rows.jsonl", comp_rows)
    _write(replay / "math500" / "rows.jsonl", replay_rows)
    for i in range(20):
        profile_path = replay / "profiles" / f"{i}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps({"estimate": [0.4, 0.6]}))
    _write(readout / "readout" / "rows.jsonl", readout_rows)

    summary, cells = MODULE.build_report(
        fullcache_root=full,
        compression_root=comp,
        replay_root=replay,
        readout_root=readout,
    )

    assert summary["full_cache"]["accuracy"] == 1.0
    assert cells[0]["compressed_accuracy"] == 0.2
    assert cells[0]["readout_accuracy"] == 0.45
    assert cells[0]["total_change"] == -0.8
    assert cells[0]["readout_change"] == -0.55
    assert cells[0]["unexplained_residual"] == -0.25
    assert cells[0]["signed_flip_mass"]["lost"] == 0.8

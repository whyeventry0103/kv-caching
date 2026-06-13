import json
import subprocess
import sys
from pathlib import Path

from kvcache.bank import append_result, build_result_row
from kvcache.generation import finalize_generation

ROOT = Path(__file__).resolve().parents[1]


def test_audit_bank_accepts_consistent_rows(tmp_path):
    generation = finalize_generation(
        prompt="prompt",
        trace=r"\boxed{2}",
        n_tokens=4,
        max_new_tokens=16,
        gold_answer="2",
        wallclock_s=0.1,
    )
    runmeta = {
        "commit": "abc",
        "dirty": False,
        "config_hash": "config",
        "host": "host",
        "gpu": "gpu",
        "cuda_visible": "0",
        "timestamp": "2026-06-13T00:00:00+00:00",
        "source_tree_hash": "source",
    }
    row = build_result_row(
        bench="math500",
        pid="p1",
        model_name="model",
        seed=11,
        max_new_tokens=16,
        sampling_cfg={"temperature": 0.6, "top_p": 0.95},
        generation=generation,
        runmeta=runmeta,
    )
    append_result(tmp_path, row)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_bank.py"),
            "--run-root",
            str(tmp_path),
            "--expected-rows",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["benchmarks"]["math500"]["accuracy"] == 1.0

from dataclasses import asdict

import pytest

from kvcache.io_schema import ResultRow, result_path, validate_row


def valid_row():
    row = ResultRow(
        bench="math500",
        pid="1",
        model="model",
        press="h2o",
        budget=256,
        compression_ratio=0.25,
        decoding={"temp": 0.6, "top_p": 0.95, "max_new": 128, "seed": 11},
        trace="runs/trace.txt",
        answer_raw="1",
        answer_canon="1",
        correct=True,
        n_tokens=10,
        truncated=False,
        wallclock_s=1.0,
        runmeta={
            "commit": "abc",
            "dirty": False,
            "config_hash": "def",
            "host": "host",
            "gpu": "gpu",
            "cuda_visible": "0",
        },
        versions={"torch": "x", "transformers": "y", "kvpress": "z"},
        ts="2026-01-01T00:00:00Z",
    )
    return asdict(row)


def test_schema_accepts_complete_row_and_path():
    validate_row(valid_row())
    assert str(result_path("runs", "math500", "model", "h2o", 256, 11)) == (
        "runs/math500/model/h2o-256-s11.jsonl"
    )


def test_schema_rejects_missing_versions_and_truncation():
    row = valid_row()
    row["versions"].pop("kvpress")
    with pytest.raises(ValueError, match="kvpress"):
        validate_row(row)
    row = valid_row()
    row["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        validate_row(row)

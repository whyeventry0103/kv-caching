from dataclasses import asdict

import pytest

from kvcache.io_schema import V2_FIELDS, ResultRow, result_path, validate_row


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


def test_schema_rejects_missing_versions():
    row = valid_row()
    row["versions"].pop("kvpress")
    with pytest.raises(ValueError, match="kvpress"):
        validate_row(row)


def test_schema_retains_truncation_as_incorrect_failure():
    row = valid_row()
    row["truncated"] = True
    row["hit_token_cap"] = True
    row["has_answer_marker"] = False
    row["finish_reason"] = "length"
    row["correct"] = False
    validate_row(row)
    row["correct"] = True
    with pytest.raises(ValueError, match="correct=false"):
        validate_row(row)


def test_schema_rejects_inconsistent_truncation_markers():
    row = valid_row()
    row["truncated"] = True  # but hit_token_cap stays False -> inconsistent
    row["correct"] = False
    with pytest.raises(ValueError, match="hit_token_cap"):
        validate_row(row)


def test_schema_v1_rows_without_version_skip_v2_fields():
    row = valid_row()
    for key in V2_FIELDS:
        row.pop(key)
    validate_row(row)  # retired 4096 pilot bank stays auditable

import json

import pytest

from kvcache.bank import append_result, build_result_row
from kvcache.generation import finalize_generation
from kvcache.io_schema import validate_row


def test_fullcache_row_is_schema_valid_and_writable(tmp_path):
    generation = finalize_generation(
        prompt="prompt",
        trace=r"\boxed{2}",
        n_tokens=4,
        max_new_tokens=16,
        gold_answer="2",
        wallclock_s=0.1,
    )
    row = build_result_row(
        bench="math500",
        pid="p1",
        model_name="r1_distill_qwen_7b",
        seed=11,
        max_new_tokens=16,
        sampling_cfg={"temperature": 0.6, "top_p": 0.95},
        generation=generation,
        runmeta={
            "commit": "abc",
            "dirty": False,
            "config_hash": "hash",
            "host": "host",
            "gpu": "gpu",
            "cuda_visible": "0",
            "timestamp": "2026-06-13T00:00:00+00:00",
        },
    )
    validate_row(row)
    assert row["compression_ratio"] == 1.0
    path = append_result(tmp_path, row)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["pid"] == "p1"

    with pytest.raises(FileExistsError, match="result already exists"):
        append_result(tmp_path, row)


def test_compressed_row_uses_requested_press_and_budget():
    generation = finalize_generation(
        prompt="prompt",
        trace=r"\boxed{2}",
        n_tokens=300,
        max_new_tokens=4096,
        gold_answer="2",
        wallclock_s=0.1,
    )
    row = build_result_row(
        bench="math500",
        pid="p1",
        model_name="r1_distill_qwen_7b",
        seed=11,
        max_new_tokens=4096,
        sampling_cfg={"temperature": 0.6, "top_p": 0.95},
        generation=generation,
        runmeta={
            "commit": "abc",
            "dirty": False,
            "config_hash": "hash",
            "host": "host",
            "gpu": "gpu",
            "cuda_visible": "0",
            "timestamp": "2026-06-13T00:00:00+00:00",
        },
        press="streamingllm",
        budget=256,
        compression_ratio=0.5,
    )
    validate_row(row)
    assert row["press"] == "streamingllm"
    assert row["budget"] == 256
    assert row["compression_ratio"] == 0.5

import json

import pytest

from kvcache.token_ids import (
    append_token_ids,
    build_token_ids_record,
    validate_token_ids_record,
)


class FakeTokenizer:
    def decode(self, token_ids, *, skip_special_tokens):
        values = {
            (1, 2): "prompt",
            (3, 4): "trace",
            (3,): "tr",
        }
        return values.get(tuple(token_ids), "")


def result_row():
    return {
        "bench": "math500",
        "pid": "p1",
        "model": "model",
        "press": "fullcache",
        "budget": "full",
        "decoding": {"seed": 11},
        "n_tokens": 2,
        "runmeta": {"source_tree_hash": "source", "config_hash": "config"},
    }


def test_token_sidecar_validates_and_refuses_duplicate(tmp_path):
    tokenizer = FakeTokenizer()
    record = build_token_ids_record(
        result_row=result_row(),
        prompt_token_ids=[1, 2],
        trace_token_ids=[3, 4],
        prompt_text="prompt",
        trace_text="trace",
        tokenizer=tokenizer,
        tokenizer_revision="revision",
    )
    path = append_token_ids(tmp_path, record)
    assert json.loads(path.read_text())["trace_token_ids"] == [3, 4]
    with pytest.raises(FileExistsError, match="already exist"):
        append_token_ids(tmp_path, record)


def test_token_sidecar_rejects_decode_mismatch():
    tokenizer = FakeTokenizer()
    record = build_token_ids_record(
        result_row=result_row(),
        prompt_token_ids=[1, 2],
        trace_token_ids=[3, 4],
        prompt_text="prompt",
        trace_text="trace",
        tokenizer=tokenizer,
        tokenizer_revision="revision",
    )
    record["trace_token_ids"] = [3, 3]
    with pytest.raises(ValueError, match="decode"):
        validate_token_ids_record(
            record,
            result_row=result_row(),
            prompt_text="prompt",
            trace_text="trace",
            tokenizer=tokenizer,
        )

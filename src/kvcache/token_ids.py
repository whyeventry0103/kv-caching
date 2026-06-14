"""Exact generation token-ID sidecars for replayable reference banks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TOKEN_IDS_SCHEMA_VERSION = "token-ids-v1"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TokenIdsRecord:
    schema_version: str
    bench: str
    pid: str
    model: str
    press: str
    budget: int | str
    seed: int
    prompt_token_ids: list[int]
    trace_token_ids: list[int]
    prompt_text_sha256: str
    trace_text_sha256: str
    tokenizer_revision: str
    source_tree_hash: str
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def token_ids_path(
    root: str | Path,
    bench: str,
    model: str,
    press: str,
    budget: int | str,
    seed: int,
) -> Path:
    return Path(root) / "token_ids" / bench / model / f"{press}-{budget}-s{seed}.jsonl"


def build_token_ids_record(
    *,
    result_row: dict[str, Any],
    prompt_token_ids: list[int] | tuple[int, ...],
    trace_token_ids: list[int] | tuple[int, ...],
    prompt_text: str,
    trace_text: str,
    tokenizer: Any,
    tokenizer_revision: str,
) -> dict[str, Any]:
    record = TokenIdsRecord(
        schema_version=TOKEN_IDS_SCHEMA_VERSION,
        bench=result_row["bench"],
        pid=result_row["pid"],
        model=result_row["model"],
        press=result_row["press"],
        budget=result_row["budget"],
        seed=int(result_row["decoding"]["seed"]),
        prompt_token_ids=[int(token) for token in prompt_token_ids],
        trace_token_ids=[int(token) for token in trace_token_ids],
        prompt_text_sha256=text_sha256(prompt_text),
        trace_text_sha256=text_sha256(trace_text),
        tokenizer_revision=tokenizer_revision,
        source_tree_hash=str(result_row["runmeta"].get("source_tree_hash", "")),
        config_hash=str(result_row["runmeta"]["config_hash"]),
    ).to_dict()
    validate_token_ids_record(
        record,
        result_row=result_row,
        prompt_text=prompt_text,
        trace_text=trace_text,
        tokenizer=tokenizer,
    )
    return record


def validate_token_ids_record(
    record: dict[str, Any],
    *,
    result_row: dict[str, Any],
    prompt_text: str,
    trace_text: str,
    tokenizer: Any,
) -> None:
    required = set(TokenIdsRecord.__dataclass_fields__)
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"missing token-ID fields: {', '.join(missing)}")
    if record["schema_version"] != TOKEN_IDS_SCHEMA_VERSION:
        raise ValueError(f"unsupported token-ID schema: {record['schema_version']}")
    identity = (
        "bench",
        "pid",
        "model",
        "press",
        "budget",
    )
    for key in identity:
        if record[key] != result_row[key]:
            raise ValueError(f"token-ID identity mismatch: {key}")
    if int(record["seed"]) != int(result_row["decoding"]["seed"]):
        raise ValueError("token-ID identity mismatch: seed")
    if len(record["trace_token_ids"]) != int(result_row["n_tokens"]):
        raise ValueError("trace token count does not match result n_tokens")
    if not record["tokenizer_revision"]:
        raise ValueError("tokenizer_revision is required")
    if not record["source_tree_hash"]:
        raise ValueError("source_tree_hash is required")
    if not record["config_hash"]:
        raise ValueError("config_hash is required")
    if record["source_tree_hash"] != result_row["runmeta"].get("source_tree_hash", ""):
        raise ValueError("token-ID source tree does not match result row")
    if record["config_hash"] != result_row["runmeta"]["config_hash"]:
        raise ValueError("token-ID config hash does not match result row")
    if record["prompt_text_sha256"] != text_sha256(prompt_text):
        raise ValueError("prompt text hash mismatch")
    if record["trace_text_sha256"] != text_sha256(trace_text):
        raise ValueError("trace text hash mismatch")
    decoded_prompt = tokenizer.decode(
        record["prompt_token_ids"], skip_special_tokens=False
    )
    if decoded_prompt != prompt_text:
        raise ValueError("prompt token IDs do not decode to the stored prompt")
    decoded_trace = tokenizer.decode(
        record["trace_token_ids"], skip_special_tokens=True
    )
    if decoded_trace != trace_text:
        raise ValueError("trace token IDs do not decode to the stored trace")


def append_token_ids(root: str | Path, record: dict[str, Any]) -> Path:
    path = token_ids_path(
        root,
        record["bench"],
        record["model"],
        record["press"],
        record["budget"],
        int(record["seed"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            existing = json.loads(line)
            if existing["pid"] == record["pid"]:
                raise FileExistsError(
                    f"token IDs already exist for {record['bench']}/{record['pid']} "
                    f"seed {record['seed']}: {path}"
                )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path

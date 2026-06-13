"""Frozen JSONL result schema and strict validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

BASE_FIELDS = {
    "bench",
    "pid",
    "model",
    "press",
    "budget",
    "compression_ratio",
    "decoding",
    "trace",
    "answer_raw",
    "answer_canon",
    "correct",
    "n_tokens",
    "truncated",
    "wallclock_s",
    "runmeta",
    "versions",
    "ts",
}
# v2 (2026-06-13): separate the length signal (hit_token_cap) from the answer
# signal (has_answer_marker) so truncation, extraction failure, and natural stop
# are never conflated. Rows without `schema_version` are treated as v1 and do not
# require these (the retired 4096 pilot bank), so re-auditing v1 still works.
V2_FIELDS = {
    "schema_version",
    "hit_token_cap",
    "has_answer_marker",
    "finish_reason",
}
_FINISH_REASONS = {"marker", "length", "eos"}
REPLAY_FIELDS = {
    "replay_of",
    "tv_profile_path",
    "auc_tv",
    "frac_above",
    "tdp",
    "onset_step",
}


@dataclass(frozen=True)
class ResultRow:
    bench: str
    pid: str
    model: str
    press: str
    budget: int | str
    compression_ratio: float
    decoding: dict[str, Any]
    trace: str
    answer_raw: str
    answer_canon: str
    correct: bool
    n_tokens: int
    truncated: bool
    wallclock_s: float
    runmeta: dict[str, Any]
    versions: dict[str, str]
    ts: str
    schema_version: str = "v2"
    hit_token_cap: bool = False
    has_answer_marker: bool = False
    finish_reason: str = "eos"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def validate_row(row: dict[str, Any], *, replay: bool = False) -> None:
    """Reject structurally invalid rows while retaining truncation failures."""
    version = row.get("schema_version", "v1")
    required = BASE_FIELDS | (REPLAY_FIELDS if replay else set())
    if version != "v1":
        required |= V2_FIELDS
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"missing schema fields: {', '.join(missing)}")
    for key in ("temp", "top_p", "max_new", "seed"):
        if key not in row["decoding"]:
            raise ValueError(f"missing decoding field: {key}")
    for key in ("commit", "dirty", "config_hash", "host", "gpu", "cuda_visible"):
        if key not in row["runmeta"]:
            raise ValueError(f"missing runmeta field: {key}")
    for key in ("torch", "transformers", "kvpress"):
        if not row["versions"].get(key):
            raise ValueError(f"missing required version: {key}")
    if row["truncated"] and row["correct"]:
        raise ValueError("truncated rows must be scored correct=false")
    if version != "v1":
        if row["finish_reason"] not in _FINISH_REASONS:
            raise ValueError(f"invalid finish_reason: {row['finish_reason']}")
        if row["truncated"] != (row["hit_token_cap"] and not row["has_answer_marker"]):
            raise ValueError("truncated must equal hit_token_cap and not marker")
        expected = (
            "marker"
            if row["has_answer_marker"]
            else ("length" if row["hit_token_cap"] else "eos")
        )
        if row["finish_reason"] != expected:
            raise ValueError(
                f"finish_reason {row['finish_reason']} inconsistent with markers"
            )
    if replay:
        for key in ("p05", "p10", "p15", "p20"):
            if key not in row["frac_above"]:
                raise ValueError(f"missing frac_above field: {key}")


def result_path(
    root: str | Path,
    bench: str,
    model: str,
    press: str,
    budget: int | str,
    seed: int,
) -> Path:
    return Path(root) / bench / model / f"{press}-{budget}-s{seed}.jsonl"

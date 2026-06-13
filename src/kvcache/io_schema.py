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

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def validate_row(row: dict[str, Any], *, replay: bool = False) -> None:
    """Reject incomplete, unversioned, or truncated result rows."""
    required = BASE_FIELDS | (REPLAY_FIELDS if replay else set())
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
    if row["truncated"]:
        raise ValueError("truncated rows are invalid for primary result analysis")
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

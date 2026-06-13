"""Schema-valid full-cache bank writing for the guarded pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvcache.canonicalize import CANONICALIZER_VERSION
from kvcache.determinism import log_versions
from kvcache.generation import GenerationResult
from kvcache.grading import GRADER_VERSION
from kvcache.io_schema import ResultRow, result_path, validate_row


def build_result_row(
    *,
    bench: str,
    pid: str,
    model_name: str,
    seed: int,
    max_new_tokens: int,
    sampling_cfg: dict[str, Any],
    generation: GenerationResult,
    runmeta: dict[str, Any],
) -> dict[str, Any]:
    # Stamp the grader and canonicalizer versions so rows are re-gradeable: the
    # audit showed scoring depends on the grader (3/20 flips on 90 vs 90 degrees).
    versions = {
        **log_versions(),
        "grader": GRADER_VERSION,
        "canonicalizer": CANONICALIZER_VERSION,
    }
    row = ResultRow(
        bench=bench,
        pid=pid,
        model=model_name,
        press="fullcache",
        budget="full",
        compression_ratio=1.0,
        decoding={
            "temp": float(sampling_cfg["temperature"]),
            "top_p": float(sampling_cfg["top_p"]),
            "max_new": max_new_tokens,
            "seed": seed,
        },
        trace=generation.trace,
        answer_raw=generation.answer_raw,
        answer_canon=generation.answer_canon,
        correct=generation.correct,
        n_tokens=generation.n_tokens,
        truncated=generation.truncated,
        wallclock_s=generation.wallclock_s,
        hit_token_cap=generation.hit_token_cap,
        has_answer_marker=generation.has_answer_marker,
        finish_reason=generation.finish_reason,
        runmeta={
            "commit": runmeta["commit"],
            "dirty": runmeta["dirty"],
            "config_hash": runmeta["config_hash"],
            "host": runmeta["host"],
            "gpu": runmeta["gpu"],
            "cuda_visible": runmeta["cuda_visible"],
            "sync_method": runmeta.get("sync_method", "git"),
            "source_tree_hash": runmeta.get("source_tree_hash", ""),
            "status_hash": runmeta.get("status_hash", ""),
            "synced_at": runmeta.get("synced_at", ""),
        },
        versions=versions,
        ts=str(runmeta["timestamp"]),
    )
    result = json.loads(row.to_json())
    validate_row(result)
    return result


def append_result(
    root: str | Path,
    row: dict[str, Any],
) -> Path:
    path = result_path(
        root,
        row["bench"],
        row["model"],
        row["press"],
        row["budget"],
        row["decoding"]["seed"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                existing = json.loads(line)
                if existing["pid"] == row["pid"]:
                    raise FileExistsError(
                        f"result already exists for {row['bench']}/{row['pid']} "
                        f"seed {row['decoding']['seed']}: {path}"
                    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path

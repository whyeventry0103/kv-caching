"""Remote dataset access with explicit download permission."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kvcache.canonicalize import canonicalize
from kvcache.rsp import make_rsp_prompt


@dataclass(frozen=True)
class Problem:
    pid: str
    question: str
    gold_answer: str
    meta: dict[str, Any]


def data_root() -> Path:
    value = os.environ.get("REMOTE_DATA_ROOT")
    if not value:
        raise RuntimeError("REMOTE_DATA_ROOT is required")
    return Path(value)


def load_dataset_guarded(
    handle: str,
    *,
    split: str,
    revision: str,
    allow_download: bool = False,
    config: str | None = None,
) -> Any:
    """Load a dataset only on remote storage and never download implicitly."""
    root = data_root()
    if not allow_download and not (root / "datasets").exists():
        raise RuntimeError(
            "dataset unavailable locally; downloads require allow_download"
        )
    if allow_download and os.environ.get("KVCACHE_REMOTE") != "1":
        raise RuntimeError("dataset downloads are remote-only")
    if not revision or revision == "TODO(human)":
        raise RuntimeError("dataset revision must be pinned before loading")
    from datasets import DownloadConfig, load_dataset

    return load_dataset(
        handle,
        config,
        split=split,
        revision=revision,
        cache_dir=str(root / "datasets"),
        download_config=DownloadConfig(local_files_only=not allow_download),
    )


def _select_math500_rows(dataset: Any, subset: int | dict[str, Any] | None) -> Any:
    if subset is None:
        return dataset
    if isinstance(subset, int):
        return dataset.select(range(min(subset, len(dataset))))
    count = int(subset["count"])
    if subset.get("stratify_by") != "level":
        raise ValueError("MATH-500 pilot supports only stratify_by: level")
    selected_indices = []
    seen_levels = set()
    for index, row in enumerate(dataset):
        level = int(row["level"])
        if level not in seen_levels:
            selected_indices.append(index)
            seen_levels.add(level)
        if len(selected_indices) == count:
            break
    if len(selected_indices) != count:
        raise ValueError(f"could not select {count} distinct MATH-500 levels")
    return dataset.select(selected_indices)


def _math500_problems(
    dataset: Any, subset: int | dict[str, Any] | None
) -> list[Problem]:
    rows = _select_math500_rows(dataset, subset)
    # Keep gold RAW: correctness is decided by grading.is_equiv on raw strings, so
    # the gold must never pass through the brittle canonicalizer first. The canonical
    # form is retained in meta only as an ADD bucketing/diagnostic aid.
    return [
        Problem(
            pid=str(row["unique_id"]),
            question=str(row["problem"]),
            gold_answer=str(row["answer"]),
            meta={
                "level": int(row["level"]),
                "subject": str(row["subject"]),
                "gold_canon": canonicalize(str(row["answer"])),
            },
        )
        for row in rows
    ]


def _aime_problems(
    dataset: Any, subset: int | dict[str, Any] | None, *, name: str
) -> list[Problem]:
    """AIME rows (HuggingFaceH4/aime_2024 schema: id/problem/solution/answer/url/year).

    Gold stays RAW (AIME answers are integers 0-999 as strings); is_equiv grades raw
    vs raw exactly as for MATH-500. Rows are ordered by their integer id so subsetting
    is deterministic across runs.
    """
    rows = sorted(dataset, key=lambda row: int(row["id"]))
    if isinstance(subset, dict):
        subset = int(subset["count"])
    if isinstance(subset, int):
        rows = rows[: min(subset, len(rows))]
    return [
        Problem(
            pid=f"{name}-{int(row['id'])}",
            question=str(row["problem"]),
            gold_answer=str(row["answer"]),
            meta={
                "year": str(row.get("year", "")),
                "url": str(row.get("url", "")),
                "gold_canon": canonicalize(str(row["answer"])),
            },
        )
        for row in rows
    ]


def _rsp_problems(subset: int | None) -> list[Problem]:
    count = subset or 3
    problems = []
    for index in range(count):
        values = [index + 2, index + 5, index + 7]
        references = ["item 2", "END", "item 1"]
        problems.append(
            Problem(
                pid=f"rsp-{index:04d}",
                question=make_rsp_prompt(values, references),
                gold_answer=str(sum(values)),
                meta={"design": "interleaved-reference", "span": 3},
            )
        )
    return problems


def load_benchmark(
    name: str,
    benchmark_cfg: dict[str, Any],
    *,
    subset: int | dict[str, Any] | None = None,
    allow_download: bool = False,
) -> list[Problem]:
    """Load the implemented benchmarks: MATH-500, AIME-24, and synthetic RSP."""
    if name == "rsp":
        if isinstance(subset, dict):
            subset = int(subset["count"])
        return _rsp_problems(subset)
    if name not in {"math500", "aime24"}:
        raise NotImplementedError(f"benchmark not implemented for pilot: {name}")
    dataset = load_dataset_guarded(
        str(benchmark_cfg["handle"]),
        split=str(benchmark_cfg["split"]),
        revision=str(benchmark_cfg.get("revision", "")),
        config=benchmark_cfg.get("config"),
        allow_download=allow_download,
    )
    if name == "aime24":
        return _aime_problems(dataset, subset, name=name)
    return _math500_problems(dataset, subset)

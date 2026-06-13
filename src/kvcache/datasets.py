"""Remote dataset access with explicit download permission."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def data_root() -> Path:
    value = os.environ.get("REMOTE_DATA_ROOT")
    if not value:
        raise RuntimeError("REMOTE_DATA_ROOT is required")
    return Path(value)


def load_dataset_guarded(
    handle: str,
    *,
    split: str,
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
    from datasets import load_dataset

    return load_dataset(
        handle,
        config,
        split=split,
        cache_dir=str(root / "datasets"),
        download_mode="reuse_dataset_if_exists" if not allow_download else None,
    )

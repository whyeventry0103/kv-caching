"""Capture the run metadata required by remoteserver.md section 21."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from kvcache.determinism import log_versions


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sync_state(path: Path = Path(".sync-state")) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(
        line.split("=", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunMeta:
    run_name: str
    timestamp: str
    timezone: str
    commit: str
    dirty: bool
    config_path: str
    config_hash: str
    command: list[str]
    environment_name: str
    python: str
    versions: dict[str, str]
    host: str
    gpu: str
    gpu_count: str
    cuda_visible: str
    seed: int
    dataset_version: str
    input_checkpoint: str
    output_artifact: str
    notes: str
    sync_method: str
    source_tree_hash: str
    status_hash: str
    synced_at: str


def capture_runmeta(
    run_name: str,
    config_path: str | Path,
    seed: int,
    *,
    dataset_version: str = "unknown",
    input_checkpoint: str = "unknown",
    output_artifact: str = "unknown",
    notes: str = "",
) -> dict[str, object]:
    """Capture stable metadata without initializing CUDA."""
    config = Path(config_path)
    now = datetime.now(tz=ZoneInfo("UTC"))
    sync_state = _sync_state()
    empty_status_hash = hashlib.sha256(b"").hexdigest()
    commit = sync_state.get("source_commit", _git("rev-parse", "HEAD"))
    status_hash = sync_state.get("status_hash", "")
    meta = RunMeta(
        run_name=run_name,
        timestamp=now.isoformat(),
        timezone="UTC",
        commit=commit,
        dirty=(
            status_hash != empty_status_hash
            if sync_state
            else bool(_git("status", "--short").strip())
        ),
        config_path=str(config),
        config_hash=file_sha256(config),
        command=sys.argv.copy(),
        environment_name=os.environ.get(
            "VIRTUAL_ENV", os.environ.get("CONDA_DEFAULT_ENV", "unknown")
        ),
        python=platform.python_version(),
        versions=log_versions(),
        host=socket.gethostname(),
        gpu=os.environ.get("GPU_MODEL", "unknown"),
        gpu_count=os.environ.get("GPU_COUNT", "unknown"),
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        seed=seed,
        dataset_version=dataset_version,
        input_checkpoint=input_checkpoint,
        output_artifact=output_artifact,
        notes=notes,
        sync_method=sync_state.get("sync_method", "git"),
        source_tree_hash=sync_state.get("source_tree_hash", ""),
        status_hash=status_hash,
        synced_at=sync_state.get("synced_at", ""),
    )
    return asdict(meta)

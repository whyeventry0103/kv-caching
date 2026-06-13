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
    meta = RunMeta(
        run_name=run_name,
        timestamp=now.isoformat(),
        timezone="UTC",
        commit=_git("rev-parse", "HEAD"),
        dirty=bool(_git("status", "--short").strip()),
        config_path=str(config),
        config_hash=file_sha256(config),
        command=sys.argv.copy(),
        environment_name=os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
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
    )
    return asdict(meta)

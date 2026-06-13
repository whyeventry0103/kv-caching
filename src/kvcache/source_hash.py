"""Deterministic content hash for Policy B source synchronization."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

IGNORED_PARTS = {
    ".claude",
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "kvcache-storage",
    "logs",
    "results",
    "runs",
}
IGNORED_SUFFIXES = {".npy", ".pyc", ".safetensors"}
IGNORED_FILES = {".sync-state"}


def source_tree_hash(root: str | Path) -> str:
    root_path = Path(root).resolve()
    digest = hashlib.sha256()
    files = []
    for path in root_path.rglob("*"):
        relative = path.relative_to(root_path)
        if not path.is_file():
            continue
        if relative.name in IGNORED_FILES:
            continue
        if any(part.startswith(".venv-") for part in relative.parts):
            continue
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.suffix in IGNORED_SUFFIXES:
            continue
        files.append((relative, path))
    for relative, path in sorted(files):
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    print(source_tree_hash(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

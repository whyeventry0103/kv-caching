"""Deterministic content hash for Policy B source synchronization."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

# Matched at ANY depth: these are build/VCS artifacts that legitimately nest.
IGNORED_ANY_DEPTH = {
    ".claude",
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
# Matched ONLY at the repository root. These are the big generated trees, and
# the distinction is load-bearing: matching them at any depth silently excluded
# `docs/results/` -- PAPER_FINDINGS.md (the results record and #3 in the
# authority order), RESULTS_PLAYBOOK.md, RESULTS.md -- from the source-tree
# hash. The parity gate is what certifies local and remote hold the same
# source, so it was blind to divergence in exactly the document whose whole job
# is to be the shared record. `sync_source.sh` already draws this distinction
# correctly (`--exclude='/results/'` anchored, `--exclude='__pycache__/'` not),
# and this set now mirrors it so the hash covers exactly what sync transfers.
IGNORED_ROOT_ONLY = {
    "data",
    "kvcache-storage",
    "logs",
    "results",
    "runs",
}
# Retained for callers that imported the old name; it is the union and must not
# be used for path filtering.
IGNORED_PARTS = IGNORED_ANY_DEPTH | IGNORED_ROOT_ONLY
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
        if any(part in IGNORED_ANY_DEPTH for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] in IGNORED_ROOT_ONLY:
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

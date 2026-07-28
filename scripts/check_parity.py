#!/usr/bin/env python3
"""Report local/remote parity across commit, source tree, and result artifacts.

`make sync` has always verified the *hashed source tree*, which is necessary and
not sufficient. Two things it deliberately excludes turned out to matter: the
remote's git clone sat pinned at an old commit with none of our history, and it
had no ``results/`` at all -- so the artifacts every catalogued number resolves
to existed in exactly one place, and a run started there could not be checked
against the state that produced it.

This reports three layers separately, because they fail for different reasons:

* **commit** -- are both sides on the same history?
* **source tree** -- the existing hash guard, restated here so one command
  answers the whole question.
* **results** -- per-file checksums, with local-only and remote-only paths
  listed individually.

**Remote-only result files are EXPECTED, not drift.** A run in progress writes
artifacts the local side has never seen; that is the point of the GPU box. So
remote-only paths are reported under their own heading and do not by themselves
make parity fail. What fails parity is a *content mismatch* on a file both sides
have, or a local artifact missing remotely -- the two states in which a number
in the paper cannot be reproduced from what the remote is computing with.

    python scripts/check_parity.py            # human-readable
    python scripts/check_parity.py --strict   # non-zero exit on a mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.source_hash import source_tree_hash

REMOTE = "kvcache-gpu"
REMOTE_ROOT = "~/kv-caching"
# Only artifact trees are compared. results/runs is raw generations (19M+) and is
# deliberately remote-only; comparing it would report drift on every new row.
RESULT_SUBDIRS = ("diag",)


def _ssh(command: str, timeout: int = 180) -> str:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=20", REMOTE, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise SystemExit(f"remote command failed: {command}\n{result.stderr[-400:]}")
    return result.stdout


def _local_manifest(subdir: str) -> dict[str, str]:
    root = PROJECT_ROOT / "results" / subdir
    out: dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(PROJECT_ROOT / "results").as_posix()
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _remote_manifest(subdir: str) -> dict[str, str]:
    # find | sha256sum, one line per file, paths relative to results/.
    cmd = (
        f"cd {REMOTE_ROOT}/results 2>/dev/null && "
        f"find {subdir} -type f -exec sha256sum {{}} + 2>/dev/null | sort"
    )
    out: dict[str, str] = {}
    for line in _ssh(cmd).splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        out[rel.strip()] = digest.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    problems: list[str] = []

    local_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    ).stdout.strip()
    remote_commit = _ssh(f"cd {REMOTE_ROOT} && git rev-parse HEAD").strip()
    print("== commit")
    print(f"  local  {local_commit}")
    print(f"  remote {remote_commit}")
    if local_commit != remote_commit:
        problems.append("commit mismatch: remote is not on the local commit")
        print("  MISMATCH")
    else:
        print("  match")

    local_tree = source_tree_hash(PROJECT_ROOT)
    remote_tree = _ssh(
        f"cd {REMOTE_ROOT} && .venv/bin/python -c "
        "\"import sys; sys.path.insert(0, 'src'); "
        "from kvcache.source_hash import source_tree_hash; "
        "print(source_tree_hash('.'))\""
    ).strip()
    print("\n== source tree hash")
    print(f"  local  {local_tree}")
    print(f"  remote {remote_tree}")
    if local_tree != remote_tree:
        problems.append("source tree hash mismatch")
        print("  MISMATCH")
    else:
        print("  match")

    print("\n== result artifacts (per-file sha256)")
    total_local = total_remote = 0
    for subdir in RESULT_SUBDIRS:
        local = _local_manifest(subdir)
        remote = _remote_manifest(subdir)
        total_local += len(local)
        total_remote += len(remote)
        both = set(local) & set(remote)
        differing = sorted(p for p in both if local[p] != remote[p])
        local_only = sorted(set(local) - set(remote))
        remote_only = sorted(set(remote) - set(local))
        print(
            f"  results/{subdir}: local {len(local)}, remote {len(remote)}, "
            f"common {len(both)}, differing {len(differing)}, "
            f"local-only {len(local_only)}, remote-only {len(remote_only)}"
        )
        for path in differing:
            print(f"    CONTENT DIFFERS: {path}")
            problems.append(f"content differs: {path}")
        for path in local_only:
            print(f"    LOCAL-ONLY (not on remote): {path}")
            problems.append(f"local-only artifact: {path}")
        for path in remote_only:
            # Expected while a run is producing output; listed, never fatal.
            print(f"    remote-only (new output? not fatal): {path}")

    print(f"\n  totals: local {total_local}, remote {total_remote}")
    print("\n== verdict")
    if problems:
        print(f"  PARITY FAILED ({len(problems)})")
        for problem in problems:
            print(f"    {problem}")
    else:
        print("  PARITY OK (commit, source tree, and result artifacts agree)")
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())

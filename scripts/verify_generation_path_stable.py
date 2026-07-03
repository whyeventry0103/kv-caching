#!/usr/bin/env python3
"""Committed provenance check: the generation-determining source files are
byte-identical across every git commit recorded in a result bank's runmeta.

Reviewer ask (2026-06-19): the "generation path is byte-identical across mixed
provenance syncs" claim must be a re-runnable, committed artifact, not a manual diff.
It re-derives the distinct runmeta commits from the bank and asserts an empty `git diff`
over the generation-critical file set across them.

Scope (stated honestly): this covers the COMMITTED-history component of provenance. If
several rows share one commit but were synced from different working trees, their
`source_tree_hash` differs while the commit does not; that dirty-tree component is
separately bounded by `source_tree_hash` and the hard package-version check in
`audit_bank.py`. A single distinct commit therefore passes trivially and the tool says
so. If a recorded commit is not present in the current git history (e.g. the repo was
re-initialized after the bank was built), the tool reports it instead of silently
passing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

GENERATION_FILES = [
    "src/kvcache/generation.py",
    "src/kvcache/models.py",
    "src/kvcache/canonicalize.py",
    "src/kvcache/datasets.py",
    "src/kvcache/grading.py",
    "src/kvcache/io_schema.py",
    "src/kvcache/bank.py",
    "src/kvcache/token_ids.py",
    "src/kvcache/replay.py",
    "src/kvcache/presses",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--files", nargs="*", default=GENERATION_FILES)
    args = parser.parse_args()

    commits: set[str] = set()
    for path in sorted(args.run_root.rglob("*.jsonl")):
        if "token_ids" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                commits.add(json.loads(line)["runmeta"]["commit"])
    known = sorted(c for c in commits if c and c != "unknown")
    if not known:
        raise SystemExit("no usable runmeta commits found in bank")
    print(f"distinct runmeta commits ({len(known)}): {known}")
    if len(commits) != len(known):
        skipped = len(commits) - len(known)
        print(f"WARNING: {skipped} rows had commit 'unknown' (skipped)")
    if len(known) == 1:
        print("PASS (single commit): committed generation path identical by definition")
        return 0

    base = known[0]
    failures = []
    for other in known[1:]:
        diff = subprocess.run(
            ["git", "diff", "--stat", base, other, "--", *args.files],
            capture_output=True,
            text=True,
        )
        tag = f"{base[:10]}..{other[:10]}"
        if diff.returncode != 0:
            failures.append(f"{tag}: git error {diff.stderr.strip()}")
        elif diff.stdout.strip():
            failures.append(f"{tag}: generation path differs:\n{diff.stdout}")
    if failures:
        print("FAIL — generation path NOT verifiable/stable across bank commits:")
        for failure in failures:
            print(" ", failure)
        return 1
    print(f"PASS — generation path byte-identical across all {len(known)} bank commits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

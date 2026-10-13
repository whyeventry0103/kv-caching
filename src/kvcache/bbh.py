"""BIG-Bench Hard ``logical_deduction_five_objects`` loader (Meeting 5, M5-B;
prepared 2026-10-14; NOT pinned, NOT downloaded).

Scientific rationale. The review recommends one non-math task with finite
answer labels as the third task family. The data are a single JSON file in the
official repository, not a Hugging Face dataset, so the guard here is a git
blob hash: the file a person places must be byte-identical to the pinned blob.
The revision is a human decision (the repo's TODO(human) rule); the reviewer
inspected commit ``9ee07bd...`` and parsed 250 examples with five answer
labels, which is recorded as a PROPOSAL only. Until ``PINNED_COMMIT`` is set
by a person, ``load_bbh`` refuses.

Prompt (frozen with the rest of M5-B): the example's ``input`` verbatim as the
single user turn -- the same raw-question convention as MATH-500 and GSM8K.
Scoring: ``kvcache.task_graders.bbh_grade`` (exactly one valid option).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from kvcache.datasets import Problem

REPO = "suzgunmirac/BIG-Bench-Hard"
FILE = "bbh/logical_deduction_five_objects.json"
PROPOSED_COMMIT = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
PROPOSED_BLOB = "936cd6d80db71aeacaea857c34f9af87e1c34964"
PINNED_COMMIT = "TODO(human)"  # a person replaces this with the approved commit
PINNED_BLOB = "TODO(human)"
EXPECTED_N = 250
NAME = "bbh_lde5"
_TARGET = re.compile(r"\([A-E]\)")
_SHA1 = re.compile(r"[0-9a-f]{40}")


def git_blob_sha1(data: bytes) -> str:
    """The git object id of a file's bytes (what the repository tree records)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def parse_bbh(data: bytes, *, expected_blob: str) -> list[Problem]:
    """Problems from the file bytes, refusing any mismatch with the pin."""
    got = git_blob_sha1(data)
    if got != expected_blob:
        raise ValueError(f"blob {got} is not the pinned {expected_blob}")
    doc: Any = json.loads(data.decode("utf-8"))
    examples = doc.get("examples") if isinstance(doc, dict) else None
    if not isinstance(examples, list):
        raise ValueError("no 'examples' list")
    out = []
    for i, ex in enumerate(examples):
        question, target = str(ex.get("input", "")), str(ex.get("target", ""))
        if not question.strip() or not _TARGET.fullmatch(target):
            raise ValueError(f"example {i}: empty input or target {target!r}")
        out.append(
            Problem(
                pid=f"{NAME}/{i}",
                question=question,
                gold_answer=target,
                meta={"source": f"{REPO}@{PINNED_COMMIT}:{FILE}"},
            )
        )
    return out


def load_bbh(path: Path) -> list[Problem]:
    """Load the placed file; refuses until a person has pinned the revision."""
    if not (_SHA1.fullmatch(PINNED_COMMIT) and _SHA1.fullmatch(PINNED_BLOB)):
        raise RuntimeError(
            "BBH revision is not pinned (TODO(human)); the proposed commit "
            f"{PROPOSED_COMMIT} needs a person's decision"
        )
    problems = parse_bbh(Path(path).read_bytes(), expected_blob=PINNED_BLOB)
    if len(problems) != EXPECTED_N:
        raise ValueError(f"{len(problems)} examples, expected {EXPECTED_N}")
    return problems

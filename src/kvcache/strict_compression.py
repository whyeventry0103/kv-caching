"""Nominal-budget compression of the eight M5-A arms on the strict-fit cohort
(2026-11-10).

Scientific rationale. The framing rules require every absolute budget to be
reported with its realized compression ratio. The S.1 artifact
(``results/diag/amendment-s/realized_compression.json``) covers only Random
and StreamingLLM, on the 100-problem S.1 grids. The four-policy M5-A
comparison is read on a different population: the frozen 97-problem
strict-fit cohort (``m5a_read_strict.json``), and it adds H2O-style and R-KV.
Its ratios are therefore computed here, on exactly that cohort's 388
(problem, seed) keys per arm, and published as a SEPARATE artifact; nothing in
the S.1 artifact is changed or reused as a number.

The per-row quantity is the one ``kvcache.compression_summary`` defines: the
nominal-budget ratio min(1, B / (prompt + generated tokens)), not live
occupancy, K/V bytes or an eviction count. Each arm goes through
``compression_summary.cell_ratios``, which refuses a duplicate, missing or
extra key and any stored ratio that does not equal the recomputed one.

Evidence, not trust: the cohort keys come from the strict read's own
conditions table, accepted only when the read is FINAL and the table matches
its recorded hash; prompt lengths come from the S.1 B=256 conditions table,
accepted only when it matches the S.1 read's recorded hash. By construction
the strict cohort's prompts leave at least 64 trace slots at B=256, so no
protected arm may relax its budget; a relaxed row is an error, not a footnote.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BUDGET = 256
N_PROBLEMS = 97
N_SEEDS = 4


@dataclass(frozen=True)
class Arm:
    label: str  # the strict read's arm name, e.g. "random+P"
    policy: str
    protected: bool
    run: str  # grid run directory holding the rows
    press: str  # the press name stored in those rows


ARMS: tuple[Arm, ...] = (
    Arm("random-P", "random", False, "grid-diss256-tok", "random"),
    Arm("random+P", "random", True, "grid-pp256", "random_pp"),
    Arm("streamingllm-P", "streamingllm", False, "grid-diss256-tok", "streamingllm"),
    Arm("streamingllm+P", "streamingllm", True, "grid-pp256", "streamingllm_pp"),
    Arm("h2o-P", "h2o", False, "grid-diss256-h2o", "h2o"),
    Arm("h2o+P", "h2o", True, "grid-h2opp256-m5", "h2o_style_pp_v1"),
    Arm("rkv-P", "rkv", False, "grid-diss256-rkv", "rkv"),
    Arm("rkv+P", "rkv", True, "grid-rkvpp256-m5", "rkv_pp_v1"),
)


def strict_keys(read: Mapping[str, Any], tsv_bytes: bytes) -> set[tuple[str, int]]:
    """The strict cohort's (pid, seed) keys, from a FINAL read's own table."""
    if read.get("status") != "FINAL":
        raise ValueError(f"strict read is {read.get('status')!r}, not FINAL")
    if hashlib.sha256(tsv_bytes).hexdigest() != read.get(
        "sufficient_statistics_sha256"
    ):
        raise ValueError("strict conditions table differs from the read's hash")
    got = [
        (r["pid"], int(r["seed"]))
        for r in csv.DictReader(io.StringIO(tsv_bytes.decode()), delimiter="\t")
    ]
    keys = set(got)
    if len(got) != len(keys):
        raise ValueError("strict conditions table repeats a (pid, seed) key")
    pids = {p for p, _ in keys}
    if len(pids) != N_PROBLEMS or len(keys) != N_PROBLEMS * N_SEEDS:
        raise ValueError(
            f"strict cohort has {len(pids)} problems / {len(keys)} conditions, "
            f"expected {N_PROBLEMS} / {N_PROBLEMS * N_SEEDS}"
        )
    return keys


def verified_prompt_lengths(
    s1_read: Mapping[str, Any], s1_tsv_bytes: bytes
) -> dict[str, int]:
    """pid -> prompt length from the S.1 B=256 table, if it matches its read."""
    from kvcache.compression_summary import prompt_lengths

    if s1_read.get("status") != "FINAL":
        raise ValueError(f"S.1 read is {s1_read.get('status')!r}, not FINAL")
    if hashlib.sha256(s1_tsv_bytes).hexdigest() != s1_read.get(
        "sufficient_statistics_sha256"
    ):
        raise ValueError("S.1 conditions table differs from the S.1 read's hash")
    return prompt_lengths(s1_tsv_bytes)


def arm_ratios(
    rows: Sequence[Mapping[str, Any]],
    arm: Arm,
    keys: set[tuple[str, int]],
    prompt_len: Mapping[str, int],
) -> dict[str, Any]:
    """One arm's nominal ratios on exactly the strict keys (cross-checked)."""
    from kvcache.compression_summary import cell_ratios

    missing = {p for p, _ in keys} - set(prompt_len)
    if missing:
        raise ValueError(f"no prompt length for {len(missing)} strict problems")
    sel = [
        r
        for r in rows
        if r.get("press") == arm.press
        and (str(r["pid"]), int(r["decoding"]["seed"])) in keys
    ]
    out = cell_ratios(
        sel,
        press=arm.press,
        budget=BUDGET,
        keys=keys,
        prompt_len=prompt_len,
        protected=arm.protected,
    )
    if out["n_budget_relaxed"]:
        raise ValueError(
            f"{arm.label}: {out['n_budget_relaxed']} relaxed rows on the "
            "strict cohort, which excludes relaxed problems by construction"
        )
    return {
        "policy": arm.policy,
        "protected": arm.protected,
        "run": arm.run,
        "press": arm.press,
        **out,
    }

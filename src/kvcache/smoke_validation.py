"""Runtime validation of real-model smoke PASS markers (Meeting 5 review of
6b409001, R2; 2026-10-14).

Scientific rationale. A smoke marker is the only thing standing between a new
press family and hundreds of production rows. The first runtime gates checked
the marker's own PASS declaration, its identity and which record names were
present -- so a marker holding NO measurements, or one carrying a raised
control failure, verified. A self-declared PASS must never substitute for its
measurements. This module re-derives every acceptance decision from the
records themselves, for every smoke family:

* the marker parses to a mapping that says ``passed: true``, ``status: PASS``
  and lists no failures, and its identity equals the identity computed now;
* the three selected traces are recomputed from the FROZEN reference bank
  (the frozen problem list, the configured seed, the fixed longest/median/
  shortest rule) and must match the marker's (pid, prompt length, trace
  length, input hash) exactly -- so the smoke provably ran on the inputs it
  claims;
* the records cover the registered design exactly once each (no missing,
  duplicate or extra record);
* every record carries its family's required fields with strict types (ints
  are ints, not bools; floats are finite) and agrees with its trace's prompt
  length and input hash;
* the family's measurement predicates (``smoke_failures``) are re-applied and
  must return nothing.

Each family supplies its schema and predicates; the boundary is shared so the
two gates cannot drift apart.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NUM = "num"  # finite int or float, not bool
INT = "int"  # int, not bool
BOOL = "bool"
LIST = "list"
DICT = "dict"
OPT_INT = "opt_int"  # int or None
STR = "str"


@dataclass(frozen=True)
class SmokeDesign:
    """What one family's PASS marker must contain."""

    marker: str
    kinds: tuple[str, ...]  # record kinds required per (trace, press)
    identity_kind: str  # the kind run once per trace with ``identity_press``
    schemas: Mapping[str, Mapping[str, str]]  # kind -> field -> type
    failures: Callable[[list[dict[str, Any]]], list[str]]


def _typed(v: Any, t: str) -> bool:
    if t == NUM:
        return (
            isinstance(v, int | float)
            and not isinstance(v, bool)
            and math.isfinite(float(v))
        )
    if t == INT:
        return isinstance(v, int) and not isinstance(v, bool)
    if t == OPT_INT:
        return v is None or (isinstance(v, int) and not isinstance(v, bool))
    if t == BOOL:
        return isinstance(v, bool)
    if t == LIST:
        return isinstance(v, list)
    if t == DICT:
        return isinstance(v, dict)
    if t == STR:
        return isinstance(v, str) and bool(v)
    raise ValueError(f"unknown field type {t!r}")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reference_traces(
    runs_root: Path, smoke_cfg: Mapping[str, Any], *, root: Path
) -> list[dict[str, Any]]:
    """The three traces a smoke of ``smoke_cfg`` must use, with input hashes.

    Recomputed from the frozen reference bank under ``runs_root``: rows of the
    configured model and first seed on the frozen problem list, token sidecars
    for their exact ids, then ``select_smoke_traces``. The input hash covers the
    prompt ids plus the replayed trace prefix (``replay_tokens``).
    """
    from kvcache.presses.prompt_protect import select_smoke_traces
    from kvcache.provenance import inputs_sha256
    from kvcache.read_validation import frozen_pids

    ref = Path(runs_root) / str(smoke_cfg["reference_run_name"])
    if not ref.is_dir():
        raise ValueError(f"reference run {ref.name} is absent")
    pids = set(frozen_pids(int(smoke_cfg["subset"]["math500"]), root=root))
    model = str(smoke_cfg["models"][0])
    seed = int(smoke_cfg["seeds"][0])
    tokens: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted((ref / "token_ids").rglob("*.jsonl")):
        for rec in _jsonl(path):
            if rec.get("model") == model and int(rec.get("seed", -1)) == seed:
                key = (str(rec["pid"]), seed)
                if key in tokens:
                    raise ValueError(f"duplicate reference token record {key}")
                tokens[key] = rec
    candidates = []
    for path in sorted(ref.rglob("*.jsonl")):
        if "token_ids" in path.parts:
            continue
        for row in _jsonl(path):
            pid = str(row.get("pid"))
            if (
                pid in pids
                and row.get("model") == model
                and int(row["decoding"]["seed"]) == seed
            ):
                rec = tokens.get((pid, seed))
                if rec is None:
                    raise ValueError(f"no token sidecar for reference ({pid}, {seed})")
                candidates.append(
                    {
                        "pid": pid,
                        "prompt_len": len(rec["prompt_token_ids"]),
                        "trace_len": len(rec["trace_token_ids"]),
                        "_rec": rec,
                    }
                )
    chosen = select_smoke_traces(candidates, int(smoke_cfg["min_trace_tokens"]))
    n_replay = int(smoke_cfg["replay_tokens"])
    return [
        {
            "pid": c["pid"],
            "prompt_len": c["prompt_len"],
            "trace_len": c["trace_len"],
            "inputs_sha256": inputs_sha256(
                c["_rec"]["prompt_token_ids"],
                c["_rec"]["trace_token_ids"][:n_replay],
            ),
        }
        for c in chosen
    ]


def claim_problem(record: Any, m: str) -> str | None:
    """The marker's own declaration, checked before anything is recomputed."""
    if not isinstance(record, dict):
        return f"{m} is not a record"
    if record.get("passed") is not True or record.get("status") != "PASS":
        return f"{m} does not record a PASS"
    if record.get("failures") != []:
        return f"{m} lists failures or has no failures list"
    return None


def validate_marker(
    record: Any,
    *,
    design: SmokeDesign,
    want_identity: Mapping[str, Any],
    presses: Sequence[str],
    identity_press: str,
    expected_traces: Sequence[Mapping[str, Any]],
) -> tuple[bool, str]:
    """Accept a marker only if its measurements, not its claim, pass."""
    from kvcache.provenance import same_identity

    m = design.marker
    claim = claim_problem(record, m)
    if claim:
        return False, claim
    if not same_identity(record.get("identity"), want_identity):
        return False, f"{m} is stale: its identity differs from now"

    selected = record.get("selected")
    if not isinstance(selected, list) or len(selected) != 3:
        return False, f"{m}: selected must list exactly 3 traces"
    if len({str(s.get("pid")) for s in selected if isinstance(s, dict)}) != 3:
        return False, f"{m}: selected traces are not 3 distinct problems"
    want_sel = [
        {k: t[k] for k in ("pid", "prompt_len", "trace_len", "inputs_sha256")}
        for t in expected_traces
    ]
    got_sel = [
        {k: s.get(k) for k in ("pid", "prompt_len", "trace_len", "inputs_sha256")}
        for s in selected
    ]
    if got_sel != want_sel:
        return False, (
            f"{m}: selected traces or their input hashes differ from the frozen "
            "reference bank"
        )
    by_pid = {t["pid"]: t for t in expected_traces}

    records = record.get("records")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        return False, f"{m}: records must be a list of mappings"
    keys = Counter(
        (r.get("kind"), str(r.get("pid")), str(r.get("press"))) for r in records
    )
    need = {(k, p, f) for p in by_pid for f in presses for k in design.kinds}
    need |= {(design.identity_kind, p, identity_press) for p in by_pid}
    if set(keys) != need or any(n != 1 for n in keys.values()):
        missing = sorted(need - set(keys))
        extra = sorted(set(keys) - need)
        dup = sorted(k for k, n in keys.items() if n > 1)
        return False, (
            f"{m} does not cover the design exactly: missing {missing[:3]}, "
            f"extra {extra[:3]}, duplicated {dup[:3]}"
        )

    for r in records:
        tag = f"{r['kind']} {r['pid']} {r['press']}"
        if "raised" in r:
            return False, f"{m}: {tag} raised: {r['raised']}"
        schema = design.schemas[r["kind"]]
        for f, t in schema.items():
            if f not in r:
                return False, f"{m}: {tag} lacks measurement {f!r}"
            if not _typed(r[f], t):
                return False, f"{m}: {tag} field {f!r} is not a valid {t}: {r[f]!r}"
        trace = by_pid[r["pid"]]
        if "prompt_len" in schema and r["prompt_len"] != trace["prompt_len"]:
            return False, f"{m}: {tag} prompt length differs from its trace"
        if "inputs_sha256" in schema and r["inputs_sha256"] != trace["inputs_sha256"]:
            return False, f"{m}: {tag} replayed other inputs than its trace"

    try:
        fails = design.failures(records)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"{m}: a record is malformed for its predicates: {exc!r}"
    if fails:
        return False, f"{m}: measurements fail: {fails[:2]}"
    return True, "verified"


REPLAY_SCHEMA = {
    "prompt_len": INT,
    "n_replayed": INT,
    "n_layers": INT,
    "protect": DICT,
    "expected_target": INT,
    "first_effective_prune": OPT_INT,
    "pre_eviction_max_tv": NUM,
    "post_prune_sizes": LIST,
    "mean_tv": NUM,
    "inputs_sha256": STR,
}
GENERATION_SCHEMA = {"prompt_len": INT, "n_tokens": INT, "protect": DICT}


def gate(
    runs_root: Any,
    smoke_cfg: Mapping[str, Any],
    *,
    design: SmokeDesign,
    want_identity: Mapping[str, Any],
    presses: Sequence[str],
    identity_press: str,
    root: Path,
) -> tuple[bool, str]:
    """Read the marker under ``runs_root`` and validate it end to end."""
    marker = Path(runs_root) / design.marker
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, f"{design.marker} absent, empty or unreadable"
    claim = claim_problem(record, design.marker)
    if claim:
        return False, claim
    try:
        traces = reference_traces(Path(runs_root), smoke_cfg, root=root)
    except (OSError, ValueError, KeyError, SystemExit) as exc:
        return False, f"cannot recompute the smoke's frozen inputs: {exc}"
    return validate_marker(
        record,
        design=design,
        want_identity=want_identity,
        presses=presses,
        identity_press=identity_press,
        expected_traces=traces,
    )

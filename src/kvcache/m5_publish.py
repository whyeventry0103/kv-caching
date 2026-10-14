"""Atomic result/sidecar pair publication and recovery for the Meeting 5 runner
(review of 946c76a, issue 2; 2026-10-14). The historical generation paths are
untouched.

Scientific rationale. A free run produces two records that are only useful
together: the result row and its exact token-ID sidecar (the replay-ready
trace). The legacy runner appends the row, then the sidecar; an overnight stop
between the two leaves a row without its sidecar, the resume scan treats the
key as unfinished, regenerates it, and ``append_result`` refuses the duplicate
-- so the run can never complete. Regenerating would also discard the original
generation. Here a condition is published as a PAIR:

1. the row and the sidecar are built and validated first, then written together
   in one per-condition journal envelope (atomic write + rename);
2. publication appends each record only if it is absent; an identical record
   already present is accepted (idempotent), a different one is a CONFLICT and
   refuses; the envelope is then renamed ``*.published.json`` and kept as
   evidence;
3. on start, for the seed shards THIS process owns only: pending envelopes are
   re-validated against the run identity and published -- no new generation;
   a row or sidecar left without its partner and without an envelope (an
   orphan), and any torn line, is copied verbatim into a quarantine record
   (reason, time, source file) BEFORE it is removed from the live file, and
   that condition is then regenerated under control. Token IDs are never
   reconstructed from text;
4. completion is counted in complete pairs, so an orphan row can never make a
   job look finished.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

JOURNAL = "m5_journal"
QUARANTINE = "m5_quarantine"
PAIR_SCHEMA = "m5-pair-v1"
BENCH = "math500"

Key = tuple[str, str, str, int]  # (press, budget, pid, seed)


def row_key(rec: Mapping[str, Any]) -> Key:
    seed = rec["decoding"]["seed"] if "decoding" in rec else rec["seed"]
    return (str(rec["press"]), str(rec["budget"]), str(rec["pid"]), int(seed))


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def journal_path(runs_root: Path, key: Key) -> Path:
    press, budget, pid, seed = key
    name = hashlib.sha256(pid.encode()).hexdigest()[:24]
    return Path(runs_root) / JOURNAL / f"{press}-{budget}-s{seed}" / f"{name}.json"


def _live_paths(runs_root: Path, rec: Mapping[str, Any]) -> tuple[Path, Path]:
    from kvcache.io_schema import result_path
    from kvcache.token_ids import token_ids_path

    press, budget, _, seed = row_key(rec)
    model = str(rec["model"])
    return (
        result_path(runs_root, BENCH, model, press, budget, seed),
        token_ids_path(runs_root, BENCH, model, press, budget, seed),
    )


def write_journal(
    runs_root: Path, row: Mapping[str, Any], sidecar: Mapping[str, Any]
) -> Path:
    """Persist both records together before either is published."""
    if row_key(row) != row_key(sidecar):
        raise ValueError("row and sidecar are for different conditions")
    env = {
        "schema": PAIR_SCHEMA,
        "key": list(row_key(row)),
        "row": row,
        "sidecar": sidecar,
        "pair_sha256": hashlib.sha256(_canon([row, sidecar]).encode()).hexdigest(),
    }
    path = journal_path(runs_root, row_key(row))
    _atomic_write(path, json.dumps(env, sort_keys=True))
    return path


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _publish_one(path: Path, rec: Mapping[str, Any]) -> str:
    """Append ``rec`` unless an identical record for its pid exists; refuse a
    different one."""
    pid = str(rec["pid"])
    for line in _read_lines(path):
        existing = json.loads(line)
        if str(existing.get("pid")) == pid:
            if _canon(existing) == _canon(rec):
                return "present"
            raise SystemExit(
                f"REFUSED: conflicting record for {pid} already in {path.name}; "
                "the journal and the published evidence disagree"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return "appended"


def publish(runs_root: Path, journal: Path) -> dict[str, str]:
    """Publish one envelope's pair idempotently, then mark it published."""
    env = json.loads(journal.read_text(encoding="utf-8"))
    if env.get("schema") != PAIR_SCHEMA:
        raise SystemExit(f"REFUSED: {journal.name} is not an {PAIR_SCHEMA} envelope")
    row, side = env["row"], env["sidecar"]
    if hashlib.sha256(_canon([row, side]).encode()).hexdigest() != env.get(
        "pair_sha256"
    ):
        raise SystemExit(f"REFUSED: {journal.name} fails its own pair hash")
    rpath, spath = _live_paths(runs_root, row)
    out = {"row": _publish_one(rpath, row), "sidecar": _publish_one(spath, side)}
    os.replace(journal, journal.with_name(journal.stem + ".published.json"))
    return out


def _quarantine(runs_root: Path, source: Path, line: str, reason: str) -> None:
    rel = source.relative_to(runs_root)
    q = Path(runs_root) / QUARANTINE / rel
    q.parent.mkdir(parents=True, exist_ok=True)
    with q.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "reason": reason,
                    "quarantined_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "source": str(rel),
                    "line": line,
                },
                sort_keys=True,
            )
            + "\n"
        )


def _owned_files(runs_root: Path, sub: str, seeds: set[int]) -> list[Path]:
    base = Path(runs_root) / sub
    if not base.exists():
        return []
    out = []
    for p in sorted(base.rglob("*.jsonl")):
        tail = p.name.rsplit("-s", 1)
        if len(tail) == 2 and tail[1].removesuffix(".jsonl").isdigit():
            if int(tail[1].removesuffix(".jsonl")) in seeds:
                out.append(p)
    return out


def recover(
    runs_root: Path,
    *,
    seeds: Iterable[int],
    validate: Callable[[Mapping[str, Any]], list[str]],
) -> dict[str, int]:
    """Finish pending envelopes and quarantine orphans, owned shards only.

    ``validate`` returns why a row/sidecar does not belong to this run (the
    run-identity check); any problem in an envelope refuses the resume.
    """
    runs_root = Path(runs_root)
    owned = {int(s) for s in seeds}
    stats = {"published": 0, "quarantined": 0}
    jdir = runs_root / JOURNAL
    if jdir.exists():
        for env_path in sorted(jdir.rglob("*.json")):
            if env_path.name.endswith(".published.json"):
                continue
            env = json.loads(env_path.read_text(encoding="utf-8"))
            if int(env["key"][3]) not in owned:
                continue  # another process's shard: never touch it
            for rec in (env["row"], env["sidecar"]):
                probs = validate(rec)
                if probs:
                    raise SystemExit(
                        f"REFUSED: pending envelope {env_path.name}: {probs[0]}"
                    )
            publish(runs_root, env_path)
            stats["published"] += 1

    rows: dict[Key, tuple[Path, str]] = {}
    sides: dict[Key, tuple[Path, str]] = {}
    for sub, into in ((BENCH, rows), ("token_ids", sides)):
        for path in _owned_files(runs_root, sub, owned):
            keep = []
            for line in _read_lines(path):
                try:
                    rec = json.loads(line)
                    k = row_key(rec)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    _quarantine(runs_root, path, line, "torn or unreadable line")
                    stats["quarantined"] += 1
                    continue
                into[k] = (path, line)
                keep.append(line)
            if len(keep) != len(_read_lines(path)):
                _atomic_write(path, "".join(x + "\n" for x in keep))
    for k in sorted(set(rows) ^ set(sides)):
        path, line = rows[k] if k in rows else sides[k]
        reason = "row without sidecar" if k in rows else "sidecar without row"
        _quarantine(runs_root, path, line, reason)
        remaining = [x for x in _read_lines(path) if x != line]
        _atomic_write(path, "".join(x + "\n" for x in remaining))
        stats["quarantined"] += 1
    return stats


def complete_pairs(runs_root: Path, seeds: Iterable[int] | None = None) -> set[Key]:
    """Keys with BOTH a readable row and a sidecar (quarantine excluded)."""

    def keys(sub: str) -> set[Key]:
        base = Path(runs_root) / sub
        out: set[Key] = set()
        for p in sorted(base.rglob("*.jsonl")) if base.exists() else []:
            for line in _read_lines(p):
                try:
                    out.add(row_key(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return out

    both = keys(BENCH) & keys("token_ids")
    if seeds is not None:
        wanted = {int(s) for s in seeds}
        both = {k for k in both if k[3] in wanted}
    return both

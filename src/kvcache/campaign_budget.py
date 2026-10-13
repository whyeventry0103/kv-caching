"""Cumulative GPU-hour caps for approved campaigns (Meeting 5 review, G0.2).

Why this exists. Approvals are given per campaign (P1 = 20, P2 = 2, P3 = 6
allocated GPU-hours), but the nightly timer only bounds a single window: one
GPU can run almost nine hours in a night, more than P3's entire allowance. This
module keeps a persistent ledger, separate from scientific rows, and enforces
the cap BEFORE a launch.

Accounting rules.

* A launch reserves time per shard (one shard = one GPU = one process).
  Reservation = min(time to the nightly stop, fair share of what the campaign
  has left) + a termination grace (the launcher's 120 s kill delay, rounded
  up). The process is bounded by its reservation minus the grace, so a
  timer-stopped run cannot overrun unaccounted.
* Everything the process does is charged: start-up, model loading, execution,
  failed attempts, retries. Shards on two GPUs are charged twice.
* On exit the launcher writes an end record; the next scheduler tick settles
  the actual elapsed GPU time and releases the unused reservation.
* A shard that is no longer running and left no end record is UNRECONCILED and
  keeps its full reservation until a person reconciles it -- an uncertain crash
  is never treated as free credit.
* No allowance left (after grace and a minimum useful run) -> refuse, naming the
  deficit. Resuming a job never resets what was spent.

All ledger reads and writes happen under an exclusive file lock and are written
atomically, so concurrent scheduler ticks cannot both spend the same hours.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

GRACE_HOURS = 0.05  # 3 minutes: covers `timeout --kill-after=120s` plus exit
MIN_RUN_HOURS = 0.25  # below this a launch is not worth a model load
RESERVED = "reserved"
SETTLED = "settled"
UNRECONCILED = "unreconciled"


@dataclass(frozen=True)
class Shard:
    session: str
    gpu: int


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    with open(lock, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _load(path: Path, caps: Mapping[str, float]) -> dict[str, Any]:
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
    else:
        d = {"schema": "campaign-ledger-v1", "launches": {}}
    d["caps"] = {k: float(v) for k, v in caps.items()}
    return d


def _save(path: Path, d: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def charged_hours(ledger: Mapping[str, Any], campaign: str) -> float:
    """Settled hours plus every open or unreconciled reservation."""
    total = 0.0
    for rec in ledger["launches"].values():
        if rec["campaign"] != campaign:
            continue
        if rec["status"] == SETTLED:
            total += float(rec["settled_hours"])
        else:
            total += float(rec["reserved_hours"])
    return total


def remaining_hours(ledger: Mapping[str, Any], campaign: str) -> float:
    return float(ledger["caps"][campaign]) - charged_hours(ledger, campaign)


def plan_reservation(
    remaining: float, n_shards: int, window_hours: float
) -> tuple[float | None, str]:
    """Per-shard reservation and process bound, or None with the deficit."""
    if n_shards < 1:
        return None, "no shards"
    per = min(window_hours + GRACE_HOURS, remaining / n_shards)
    if per - GRACE_HOURS < MIN_RUN_HOURS:
        need = n_shards * (MIN_RUN_HOURS + GRACE_HOURS)
        return None, (
            f"remaining allowance {remaining:.3f} GPU-h cannot fund {n_shards} "
            f"shard(s) of at least {MIN_RUN_HOURS} h plus grace "
            f"(needs {need:.3f}; deficit {max(0.0, need - remaining):.3f})"
        )
    return per, "ok"


def reserve(
    path: Path,
    caps: Mapping[str, float],
    *,
    campaign: str,
    job: str,
    shards: Iterable[Shard],
    window_hours: float,
    now: datetime,
) -> tuple[dict[str, float] | None, str]:
    """Atomically reserve time for every shard of one launch.

    Returns ({launch_id: process_max_hours}, message) or (None, reason). Each
    shard's record is written before anything is launched.
    """
    shards = list(shards)
    with _locked(path):
        d = _load(path, caps)
        if campaign not in d["caps"]:
            return None, f"unknown campaign {campaign!r}"
        rem = remaining_hours(d, campaign)
        per, why = plan_reservation(rem, len(shards), window_hours)
        if per is None:
            return None, f"{campaign}: {why}"
        out = {}
        for sh in shards:
            lid = f"{campaign}-{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
            d["launches"][lid] = {
                "campaign": campaign,
                "job": job,
                "session": sh.session,
                "gpu": sh.gpu,
                "reserved_hours": round(per, 6),
                "process_max_hours": round(per - GRACE_HOURS, 6),
                "reserved_at": now.isoformat(),
                "status": RESERVED,
            }
            # floor, never round up: the bound must not exceed the reservation
            out[lid] = math.floor((per - GRACE_HOURS) * 100) / 100
        _save(path, d)
        return out, (
            f"{campaign}: reserved {len(shards)} x {per:.3f} GPU-h "
            f"(remaining before {rem:.3f})"
        )


def settle(
    path: Path,
    caps: Mapping[str, float],
    launch_id: str,
    *,
    started: datetime,
    ended: datetime,
    rc: int | None,
) -> float:
    """Charge the actual elapsed time of one shard; release the rest."""
    with _locked(path):
        d = _load(path, caps)
        rec = d["launches"][launch_id]
        hours = max(0.0, (ended - started).total_seconds() / 3600)
        rec.update(
            status=SETTLED,
            settled_hours=round(hours, 6),
            started_at=started.isoformat(),
            ended_at=ended.isoformat(),
            rc=rc,
            overrun=hours > float(rec["reserved_hours"]),
        )
        _save(path, d)
        return hours


def release_unlaunched(path: Path, caps: Mapping[str, float], launch_id: str) -> None:
    """A shard whose process never started (e.g. tmux refused): charge zero."""
    with _locked(path):
        d = _load(path, caps)
        d["launches"][launch_id].update(
            status=SETTLED, settled_hours=0.0, rc="not-started"
        )
        _save(path, d)


def read_end_record(path: Path) -> dict[str, Any] | None:
    """The launcher's start/end record for one launch (None if absent/torn)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and "start_epoch" in d else None


def reconcile(
    path: Path,
    caps: Mapping[str, float],
    *,
    live_sessions: Iterable[str],
    end_records_dir: Path,
) -> list[str]:
    """Settle every finished shard; mark crashed-without-record UNRECONCILED.

    A reserved shard whose session is still live is left open. Returns notes.
    """
    live = set(live_sessions)
    notes: list[str] = []
    with _locked(path):
        d = _load(path, caps)
        for lid, rec in d["launches"].items():
            if rec["status"] != RESERVED or rec["session"] in live:
                continue
            end = read_end_record(end_records_dir / f"{lid}.json")
            if end and end.get("end_epoch") is not None:
                started = datetime.fromtimestamp(float(end["start_epoch"]))
                ended = datetime.fromtimestamp(float(end["end_epoch"]))
                hours = max(0.0, (ended - started).total_seconds() / 3600)
                rec.update(
                    status=SETTLED,
                    settled_hours=round(hours, 6),
                    started_at=started.isoformat(),
                    ended_at=ended.isoformat(),
                    rc=end.get("rc"),
                    overrun=hours > float(rec["reserved_hours"]),
                )
                notes.append(f"settled {lid}: {hours:.3f} GPU-h (rc={end.get('rc')})")
            else:
                rec["status"] = UNRECONCILED
                notes.append(
                    f"UNRECONCILED {lid}: session gone without an end record; "
                    f"its {rec['reserved_hours']} GPU-h stay charged"
                )
        _save(path, d)
    return notes


def summary(path: Path, caps: Mapping[str, float]) -> dict[str, dict[str, float]]:
    with _locked(path):
        d = _load(path, caps)
    return {
        c: {
            "cap": float(cap),
            "charged": round(charged_hours(d, c), 4),
            "remaining": round(remaining_hours(d, c), 4),
        }
        for c, cap in caps.items()
    }


def record_historical(
    path: Path,
    caps: Mapping[str, float],
    *,
    launch_id: str,
    campaign: str,
    job: str,
    gpu: int,
    started: datetime,
    ended: datetime,
    evidence: str,
) -> float:
    """Charge a launch that ran before the ledger existed (idempotent by id)."""
    hours = max(0.0, (ended - started).total_seconds() / 3600)
    with _locked(path):
        d = _load(path, caps)
        d["launches"].setdefault(
            launch_id,
            {
                "campaign": campaign,
                "job": job,
                "session": f"{job}-g{gpu}",
                "gpu": gpu,
                "reserved_hours": round(hours, 6),
                "process_max_hours": None,
                "reserved_at": started.isoformat(),
                "status": SETTLED,
                "settled_hours": round(hours, 6),
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "rc": 0,
                "overrun": False,
                "historical": True,
                "evidence": evidence,
            },
        )
        _save(path, d)
    return hours

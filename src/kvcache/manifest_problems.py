"""Problem selection by frozen problem ID (Meeting 5 pid-filter adapter,
2026-11-10).

Scientific rationale. The Meeting 5 replication splits (GSM8K and BBH
calibration / confirmation / reserve) were frozen by hashed problem ID before
any outcome existed (``configs/meeting5/problem_manifests.json``). The existing
loaders select problems only by a count or a seeded sample, so a frozen ID list
cannot be expressed through them; selecting "the first n" or re-drawing a
sample would silently run a different population. This module resolves a
split to its exact IDs and selects exactly those problems, refusing anything
that would make the run differ from the frozen design:

* the manifest entry must name the same dataset revision as the human pin in
  ``configs/benchmarks.yaml`` (a different revision could renumber rows);
* the split must exist, be non-empty, hold no duplicate, and be disjoint from
  the benchmark's other splits (calibration never leaks into confirmation);
* the loaded pool must have the manifest's ``n_pool`` problems, and every ID
  in the split must be present in it exactly once.

Problems are returned in the manifest's order. Nothing here loads a model or a
dataset; the caller passes the loaded pool. Kept out of ``kvcache.datasets``,
which is on the generation-critical file list (``verify_generation_path_
stable.py``), so adding it changes no existing generation path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SPLITS = ("calibration", "confirmation", "reserve")


def frozen_split_pids(
    manifest: Mapping[str, Any],
    benchmark: str,
    split: str,
    *,
    pinned_revision: str,
) -> list[str]:
    """The frozen IDs of one split, after the manifest's own consistency checks."""
    entry = manifest.get(benchmark)
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest has no frozen entry for {benchmark!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r} (expected one of {SPLITS})")
    if str(entry.get("revision", "")) != str(pinned_revision):
        raise ValueError(
            f"{benchmark}: manifest revision {entry.get('revision')!r} differs "
            f"from the pinned dataset revision {pinned_revision!r}"
        )
    pids = entry.get(split)
    if not isinstance(pids, list) or not pids:
        raise ValueError(f"{benchmark}: split {split!r} is missing or empty")
    if len(set(pids)) != len(pids):
        raise ValueError(f"{benchmark}: split {split!r} repeats a problem ID")
    for other in SPLITS:
        if other == split:
            continue
        overlap = set(pids) & set(entry.get(other) or [])
        if overlap:
            raise ValueError(
                f"{benchmark}: {len(overlap)} IDs are in both {split!r} and "
                f"{other!r} (e.g. {sorted(overlap)[0]})"
            )
    return [str(p) for p in pids]


def select_frozen_split(
    problems: Sequence[Any], pids: Sequence[str], *, n_pool: int
) -> list[Any]:
    """Exactly the problems named by ``pids``, in that order, from the pool."""
    if len(problems) != int(n_pool):
        raise ValueError(
            f"loaded pool has {len(problems)} problems; the manifest froze "
            f"{n_pool} (a different revision or split was loaded)"
        )
    by_pid: dict[str, Any] = {}
    for p in problems:
        if p.pid in by_pid:
            raise ValueError(f"loaded pool repeats problem ID {p.pid!r}")
        by_pid[p.pid] = p
    missing = [pid for pid in pids if pid not in by_pid]
    if missing:
        raise ValueError(
            f"{len(missing)} frozen IDs are absent from the loaded pool "
            f"(e.g. {missing[0]!r})"
        )
    return [by_pid[pid] for pid in pids]

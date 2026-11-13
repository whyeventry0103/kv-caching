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

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
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


def manifest_key(bench: str, bench_cfg: Mapping[str, Any]) -> str:
    """The manifest entry a benchmark's frozen splits live under.

    BBH's rows, paths and grader registry use the short name ``bbh_lde5``;
    its manifest entry was frozen under the long subtask name (2026-10-14), so
    ``configs/benchmarks.yaml`` maps one to the other explicitly.
    """
    return str(bench_cfg.get("manifest_key", bench))


def load_frozen_pool(
    bench: str,
    bench_cfg: Mapping[str, Any],
    *,
    hf_loader: Callable[..., Sequence[Any]],
    data_root: str | Path | None = None,
) -> Sequence[Any]:
    """The pinned pool, offline: BBH from its blob-verified file under the data
    root (``kvcache.bbh.load_bbh``), every other benchmark through the
    caller's ``kvcache.datasets.load_benchmark`` (passed in, unchanged)."""
    if bench_cfg.get("loader") == "bbh_file":
        from kvcache import bbh

        if str(bench_cfg.get("revision")) != bbh.PINNED_COMMIT:
            raise ValueError(
                f"{bench}: configured revision {bench_cfg.get('revision')!r} is "
                f"not the pinned BBH commit {bbh.PINNED_COMMIT!r}"
            )
        root = (
            data_root if data_root is not None else os.environ.get("REMOTE_DATA_ROOT")
        )
        if not root:
            raise ValueError("BBH needs REMOTE_DATA_ROOT (the pinned file's root)")
        return bbh.load_bbh(bbh.data_path(Path(root)))
    return hf_loader(bench, dict(bench_cfg), subset=None, allow_download=False)

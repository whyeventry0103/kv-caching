"""Recoverable row + token-ID pair publication for manifest banks (GSM8K timing
pilot, 2026-11-10): the Meeting 5 runner's path, reused byte-for-byte.

Scientific rationale. A full-cache timing row is only useful with its exact
token-ID sidecar: the timing read needs the prompt length (to say whether a
budget would ever bind) and the trace token count, and any later replay needs
the IDs themselves. The legacy append-row-then-sidecar path can leave a row
without its sidecar after an overnight stop, and resume then refuses the
duplicate forever. ``kvcache.m5_publish`` already solves this -- one atomic
journal envelope per condition, idempotent publication, conflict refusal,
quarantine of orphans and torn lines, completion counted in pairs -- and is
covered by its interruption tests.

That module is hard-wired to MATH-500 (``m5_publish.BENCH``) and its bytes are
part of the ``m5_protsel`` and ``rkv_pp`` generation identities
(``m5_runs.M5_SOURCES_EXTRA``, ``presses/rkv_prompt_protect``), which the
completed Meeting 5 grids were generated and verified under. Editing it would
move those identities. So it is reused unchanged: :func:`bench_scope` points
its module-level ``BENCH`` at another benchmark for the duration of one call
and always restores it. Every m5_publish function reads ``BENCH`` at call time,
so the scoped call runs exactly the tested code against
``<runs_root>/<bench>/...`` and ``<runs_root>/token_ids/<bench>/...``. The
regression tests assert both the scoped layout and that MATH-500 is restored.
Run roots are per run (``launch.sh`` appends the run name), so the shared
``m5_journal`` / ``m5_quarantine`` directory names cannot collide across runs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from kvcache import m5_publish

Key = m5_publish.Key


@contextmanager
def bench_scope(bench: str) -> Iterator[None]:
    """Run m5_publish against ``bench`` for the duration of the block."""
    previous = m5_publish.BENCH
    if previous != "math500":
        raise RuntimeError(
            f"m5_publish.BENCH is {previous!r}; nested or leaked bench scope"
        )
    m5_publish.BENCH = str(bench)
    try:
        yield
    finally:
        m5_publish.BENCH = previous


def _same_bench(rec: Mapping[str, Any], bench: str) -> None:
    if str(rec.get("bench")) != bench:
        raise ValueError(
            f"record for {rec.get('pid')!r} is bench {rec.get('bench')!r}, "
            f"not {bench!r}"
        )


def publish_pair(
    runs_root: Path, bench: str, row: Mapping[str, Any], sidecar: Mapping[str, Any]
) -> dict[str, str]:
    """Journal both records, then publish them as one pair."""
    _same_bench(row, bench)
    _same_bench(sidecar, bench)
    journal = m5_publish.write_journal(runs_root, row, sidecar)
    with bench_scope(bench):
        return m5_publish.publish(runs_root, journal)


def recover_pairs(
    runs_root: Path,
    bench: str,
    *,
    seeds: Iterable[int],
    validate: Callable[[Mapping[str, Any]], list[str]],
) -> dict[str, int]:
    """Finish pending envelopes and quarantine orphans in the owned seeds."""
    with bench_scope(bench):
        return m5_publish.recover(runs_root, seeds=seeds, validate=validate)


def complete_pair_keys(
    runs_root: Path, bench: str, seeds: Iterable[int] | None = None
) -> set[Key]:
    """Keys with both a readable row and a sidecar."""
    with bench_scope(bench):
        return m5_publish.complete_pairs(runs_root, seeds)


def pair_identity_problems(
    rec: Mapping[str, Any],
    *,
    bench: str,
    model: str,
    pids: Iterable[str],
    seeds: Iterable[int],
    manifest_split: Mapping[str, str],
) -> list[str]:
    """Why a row or sidecar does not belong to this manifest bank (empty: it
    does). Applied to every record recovery would publish."""
    problems = []
    if str(rec.get("bench")) != bench:
        problems.append(f"bench {rec.get('bench')!r} != {bench!r}")
    if str(rec.get("model")) != model:
        problems.append(f"model {rec.get('model')!r} != {model!r}")
    if str(rec.get("press")) != "fullcache" or str(rec.get("budget")) != "full":
        problems.append("not a full-cache record")
    try:
        _, _, pid, seed = m5_publish.row_key(rec)
    except (KeyError, TypeError, ValueError):
        return [*problems, "record has no readable key"]
    if pid not in set(pids):
        problems.append(f"pid {pid!r} is not in the frozen split")
    if seed not in {int(s) for s in seeds}:
        problems.append(f"seed {seed} is not a design seed")
    if dict(rec.get("manifest_split") or {}) != dict(manifest_split):
        problems.append(
            f"manifest_split {rec.get('manifest_split')!r} != {dict(manifest_split)!r}"
        )
    return problems

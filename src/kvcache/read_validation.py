"""One validation boundary for the Amendment S and R.17 readers.

Scientific rationale. The Meeting #3 readers (R.14) learned the hard way that a
row count is not coverage: duplicated rows stand in for missing conditions,
undefined rows hide behind defined ones, and a failed control still counts. The
independent review of f69dce3 found the newer readers had slipped back to row
counts, accepted non-finite values, and could label a read from one row. This
module is the single place where a read earns the right to carry a label.

The contract: the EXPECTED key set is derived from committed configs and
frozen pid lists, never from the rows; every row's key is validated before it
is eligible for anything; duplicates are counted over every row (undefined and
ineligible included); required fields must exist and numeric fields must be
finite and in domain; provenance is checked where rows carry it and the
limitation is reported where they do not; and a read is ``FINAL`` only when
coverage is complete, every control passed and nothing was malformed.
``PARTIAL`` carries no scientific label; ``INVALID`` fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from kvcache.coverage import COMPLETE, INVALID, PARTIAL, check_coverage, expected_keys

FINAL = "FINAL"
GRID100_PIDS = "configs/meeting3/grid100_pids.json"
FROZEN_SUBSET = "configs/meeting3/q2_subset_pids.json"


# ------------------------------------------------------------------ design
def frozen_pids(n: int, *, root: Path) -> list[str]:
    """The committed pid list for a MATH-500 subset of size ``n`` (25 or 100)."""
    path = {25: FROZEN_SUBSET, 100: GRID100_PIDS}.get(int(n))
    if path is None:
        raise ValueError(f"no frozen pid list for a subset of {n}")
    d = json.loads((Path(root) / path).read_text(encoding="utf-8"))
    pids = list(d["pids"])
    if int(d["subset"]) != int(n) or len(pids) != int(n) or len(set(pids)) != n:
        raise ValueError(f"{path} is not a clean {n}-problem list")
    return pids


def design_keys(
    config_path: str,
    *,
    root: Path,
    presses: Sequence[str] | None = None,
) -> tuple[dict[str, Any], set[tuple]]:
    """Expected (press, budget, pid, seed) keys of one config, from frozen inputs."""
    import yaml

    cfg = yaml.safe_load((Path(root) / config_path).read_text(encoding="utf-8"))
    pids = frozen_pids(int(cfg["subset"]["math500"]), root=root)
    seeds = [int(s) for s in cfg["seeds"]]
    ps = list(presses) if presses is not None else list(cfg["presses"])
    budgets = [str(b) for b in cfg["budgets"]]
    keys = expected_keys(ps, budgets, pids, seeds)
    return {"pids": pids, "seeds": seeds, "presses": ps, "budgets": budgets}, keys


# ------------------------------------------------------------------ values
def is_finite_number(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def in_unit_interval(x: Any) -> bool:
    return is_finite_number(x) and 0.0 <= float(x) <= 1.0


def row_problems(
    row: Mapping[str, Any],
    *,
    required: Iterable[str] = (),
    unit_fields: Iterable[Callable[[Mapping[str, Any]], Any]] = (),
) -> list[str]:
    """Why one row is malformed (empty list = well formed)."""
    out = [f"missing field {f!r}" for f in required if _get(row, f) is None]
    for i, fn in enumerate(unit_fields):
        try:
            v = fn(row)
        except (KeyError, TypeError, IndexError):
            out.append(f"value {i} unreadable")
            continue
        if not in_unit_interval(v):
            out.append(f"value {i} not a finite number in [0, 1]: {v!r}")
    return out


def _get(row: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = row
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


# ------------------------------------------------------------------ status
def validate_population(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], tuple],
    expected: set[tuple],
    required: Iterable[str] = (),
    unit_fields: Iterable[Callable[[Mapping[str, Any]], Any]] = (),
) -> dict[str, Any]:
    """Coverage over EVERY row plus per-row malformation; one combined status."""
    cov = check_coverage(rows, key, expected)
    malformed: dict[str, list[str]] = {}
    for i, r in enumerate(rows):
        probs = row_problems(r, required=required, unit_fields=unit_fields)
        if probs:
            try:
                k = repr(key(r))
            except (KeyError, TypeError, ValueError):
                k = f"<row {i}>"
            malformed[k] = probs
    status = cov["status"]
    if malformed and status != INVALID:
        status = INVALID
    return {**cov, "status": status, "malformed": malformed}


def combined_status(*blocks: Mapping[str, Any]) -> str:
    """INVALID if any block is; else PARTIAL if any is; else FINAL."""
    statuses = [b.get("status") for b in blocks]
    if any(s == INVALID for s in statuses):
        return INVALID
    if any(s in (PARTIAL, None) for s in statuses):
        return PARTIAL
    if all(s in (COMPLETE, FINAL) for s in statuses):
        return FINAL
    return INVALID


def controls_passed(controls: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(controls) and all(bool(c.get("passed")) for c in controls.values())


def final_status(population: Mapping[str, Any], controls: Mapping[str, Any]) -> str:
    """FINAL only with complete coverage, no malformed rows and every control passed."""
    if population.get("status") == INVALID:
        return INVALID
    if controls and not controls_passed(controls):
        return INVALID
    if population.get("status") != COMPLETE:
        return PARTIAL
    return FINAL


# ------------------------------------------------------------------ provenance
def legacy_provenance_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What legacy (pre-provenance-v1) rows carry, and the stated limitation.

    The grid and readout rows of the S.1 cells use the unprotected runners'
    row schema: ``runmeta`` (commit, source tree hash) and ``versions`` per
    row, no experiment identity and no input manifest hash. Their reference
    inputs are therefore checked against the full-cache bank by key, not by
    hash. That limitation is reported, not papered over.
    """
    commits = sorted({str(_get(r, "runmeta.commit")) for r in rows})
    trees = sorted({str(_get(r, "runmeta.source_tree_hash")) for r in rows})
    versions = sorted(
        {json.dumps(r.get("versions"), sort_keys=True) for r in rows if "versions" in r}
    )
    missing = sum(1 for r in rows if _get(r, "runmeta.commit") is None)
    return {
        "schema": "legacy (runmeta + versions; no provenance-v1 identity)",
        "n_rows": len(rows),
        "n_without_runmeta": missing,
        "commits": commits,
        "source_tree_hashes": trees,
        "version_sets": len(versions),
        "limitation": "no per-row input-manifest hash; reference inputs are "
        "matched to the full-cache bank by (problem, seed) key only",
        "passed": missing == 0,
    }


def provenance_v1_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    config: str,
    runs_root: Path,
    key: Callable[[Mapping[str, Any]], tuple],
    root: Path,
) -> dict[str, Any]:
    """R.15 per-row identity check against the identity derived from committed specs."""
    from kvcache.gates import expected_run_identity, validate_rows

    expected, why = expected_run_identity(kind, config, runs_root, root=root)
    bad = validate_rows(rows, kind=kind, expected=expected, key=key)
    return {
        "schema": "provenance-v1",
        "expected_identity": expected,
        "expected_identity_reason": why,
        "n_rows": len(rows),
        "n_failing": len(bad),
        "failing": {repr(k): v for k, v in list(bad.items())[:20]},
        "passed": expected is not None and not bad,
    }


# ------------------------------------------------------------------ inputs and outputs
def files_sha256(paths: Iterable[Path]) -> str:
    """One hash over the sorted input files (name and bytes): the read's input id."""
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(str(p).encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def retire_stale(path: Path) -> Path | None:
    """Rename an existing output so a non-FINAL read never leaves a FINAL behind."""
    path = Path(path)
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.superseded-{int(time.time())}")
    os.replace(path, target)
    return target


def exit_code(status: str, *, partial_ok: bool) -> int:
    """0 FINAL; 0 PARTIAL only when the caller asked for a partial read, else 3;
    2 INVALID."""
    if status == FINAL:
        return 0
    if status == PARTIAL:
        return 0 if partial_ok else 3
    return 2

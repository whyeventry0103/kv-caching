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
    import yaml

    from kvcache.gates import expected_run_identity, validate_rows

    expected, why = expected_run_identity(kind, config, runs_root, root=root)
    # a config that names its own frozen manifest (a second model, R.17.3) is
    # checked against THAT manifest, as the gates do
    manifest_path = yaml.safe_load(
        (Path(root) / config).read_text(encoding="utf-8")
    ).get("input_manifest")
    bad = validate_rows(
        rows, kind=kind, expected=expected, key=key, manifest_path=manifest_path
    )
    return {
        "schema": "provenance-v1",
        "input_manifest": manifest_path or "configs/meeting3/input_manifest.json",
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


# ------------------------------------------------------------------ restoration rows
# Statuses each producer can emit (kvcache.restore_v2 / restore_v3, read from
# the producers' code 2026-10-03). Anything else -- a crash marker, a typo, an
# injected FAILED -- is not a legitimate exclusion and makes the read INVALID.
NO_EVICTION = "INELIGIBLE: no eviction at or after the trigger"
STASH_SMALL = "INELIGIBLE: stash smaller than the reinsertion count"
WINDOW_SHORT = "INELIGIBLE: the read window does not fit the trace"
NO_PROMPT = "INELIGIBLE: no prompt entry had been evicted"
RESTORE_STATUSES = {
    "restore_v2": ("READ", NO_EVICTION, STASH_SMALL),
    "restore_v3": ("READ", NO_EVICTION, WINDOW_SHORT, NO_PROMPT),
}


def prune_at_or_after(n_trace: int, trigger: int, interval: int) -> bool:
    """Whether a decode forward j with trigger <= j <= n_trace - 1 is a prune step.

    DecodingPress prunes when a layer's decode-step count reaches the interval,
    i.e. at forwards j = interval, 2 * interval, ...; at a trigger past the
    budget the cache always exceeds it, so every such step evicts.
    """
    last = int(n_trace) - 1
    first = -(-int(trigger) // int(interval)) * int(interval)  # ceil to a prune step
    return first <= last


def restoration_row_problems(
    row: Mapping[str, Any],
    *,
    family: str,
    trigger: int,
    interval: int,
    window: int,
) -> list[str]:
    """Why one restore-v2 / restore-v3 row cannot be used as it claims (empty = ok).

    Every status must be one the producer emits, and every exclusion must be
    legitimate given the row's own recorded trace length, cohort step and the
    config's trigger, interval and window. READ rows must carry passed gates
    and finite TV in [0, 1] for every quantity a reader consumes.
    """
    out: list[str] = []
    status = row.get("status")
    if status not in RESTORE_STATUSES[family]:
        return [f"status {status!r} is not one {family} emits"]
    n = row.get("n_trace_tokens")
    if not isinstance(n, int) or n < 1:
        return [f"n_trace_tokens {n!r} not a positive int"]
    if int(row.get("trigger_step", -1)) != int(trigger):
        out.append(f"trigger_step {row.get('trigger_step')!r} != config {trigger}")
    reaches = prune_at_or_after(n, trigger, interval)
    e = row.get("cohort_step")
    if status == NO_EVICTION:
        if reaches or e is not None:
            out.append(
                f"claims no eviction after the trigger, but a {n}-token trace "
                f"reaches a prune at or after {trigger}"
            )
        return out
    if not reaches or not isinstance(e, int):
        out.append("eligible status without a cohort the trace can reach")
        return out
    if e < trigger or e % int(interval) or e > n - 1:
        out.append(f"cohort_step {e} not a prune step in [{trigger}, {n - 1}]")
    fits = e + int(window) <= n - 1
    if family == "restore_v3":
        if status == WINDOW_SHORT:
            if fits:
                out.append(
                    f"claims the window does not fit, but {e}+{window} <= {n - 1}"
                )
            return out
        if status == NO_PROMPT:
            if row.get("n_prompt_return") != 0:
                out.append("claims no prompt eviction with a non-zero return count")
            return out
        if not fits:
            out.append(f"READ, but the window does not fit ({e}+{window} > {n - 1})")
        if (
            not isinstance(row.get("n_prompt_return"), int)
            or row["n_prompt_return"] < 1
        ):
            out.append("READ without a positive n_prompt_return")
        if not in_unit_interval(row.get("none_window_mean_tv")):
            out.append("none_window_mean_tv not a finite number in [0, 1]")
        arms = row.get("arms") or {}
        for arm in ("frozen", "prompt", "trace", "full"):
            blk = arms.get(arm)
            if not blk:
                out.append(f"arm {arm} missing")
                continue
            if not in_unit_interval(blk.get("window_mean_tv")):
                out.append(f"arm {arm}: TV {blk.get('window_mean_tv')!r} not in [0, 1]")
            if not (blk.get("gate") or {}).get("passed"):
                out.append(f"arm {arm}: gate not passed")
        return out
    # restore_v2
    if status == STASH_SMALL:
        if not (
            isinstance(row.get("stash_size"), int)
            and isinstance(row.get("n_reinsert"), int)
            and row["stash_size"] < row["n_reinsert"]
        ):
            out.append("claims a small stash, but stash_size >= n_reinsert")
        return out
    fs = row.get("full_restore") or {}
    if bool(fs.get("eligible")) != fits:
        out.append(
            f"full_restore.eligible {fs.get('eligible')!r} disagrees with "
            f"{e}+{window} <= {n - 1}"
        )
    if fs.get("eligible"):
        if not (fs.get("gate") or {}).get("passed"):
            out.append("full_restore gate not passed")
        for f in ("window_mean_tv", "none_window_mean_tv"):
            if not in_unit_interval(fs.get(f)):
                out.append(f"full_restore.{f} not a finite number in [0, 1]")
    for lag, blk in (row.get("lags") or {}).items():
        if not blk.get("eligible"):
            continue
        if not all(g.get("passed") for g in (blk.get("gates") or {}).values()):
            out.append(f"lag {lag}: an arm gate did not pass")
        for arm in ("none", "matched", "random", "relaxed"):
            if not in_unit_interval(blk.get(arm)):
                out.append(f"lag {lag}: {arm} TV not a finite number in [0, 1]")
    return out


# ------------------------------------------------------------------ capture rows
# Rounding tolerance for the attention-mass channels, per measurement. M is
# 1 - exp(lse_kept - lse_all); when almost all mass is kept (budget 2048) the
# two log-sum-exps agree to fp32 rounding and M can land a hair below 0. The
# real b2048 rows reach -2.7e-6 per measurement (counted 2026-10-03, BEFORE this
# value was set); 1e-5 is the registered layer-0 Delta tolerance (R.9), reused
# so there is one numerical tolerance for the attention channels, not two. Values
# are never clipped; anything beyond the tolerance is INVALID.
MASS_ROUNDING_TOL = 1e-5


def _mass_ok(v: Any, count: Any) -> bool:
    """A per-measurement-mean-in-[0, 1] check on a SUM over ``count`` measurements."""
    if not is_finite_number(v) or not isinstance(count, (int, float)):
        return False
    c = max(float(count), 1.0)
    return -MASS_ROUNDING_TOL * c <= float(v) <= (1.0 + MASS_ROUNDING_TOL) * c


def _mean_ok(v: Any) -> bool:
    return (
        is_finite_number(v)
        and -MASS_ROUNDING_TOL <= float(v) <= 1.0 + MASS_ROUNDING_TOL
    )


def _finite_nonneg_nested(x: Any) -> bool:
    if isinstance(x, list):
        return all(_finite_nonneg_nested(v) for v in x)
    return is_finite_number(x) and float(x) >= 0.0


def capture_metric_problems(row: Mapping[str, Any]) -> list[str]:
    """Domain checks on every capture field the R.17 reads consume.

    The strict per-row controls (layer-0 Delta, non-finite count) are
    ``report_capture.control_failures``, reused by the caller; this adds the
    metric values: M and Delta means in [0, 1] (within MASS_ROUNDING_TOL) when an
    event was measured and absent (unavailable, not zero) when none was;
    per-(layer, group) and per-(layer, head) sums finite and, as means over
    their counts, in [0, 1] within the tolerance; latent and nested latent sums
    finite and non-negative; omega finite and non-negative or absent.
    """
    out: list[str] = []
    n_ev = row.get("n_events")
    if not isinstance(n_ev, int) or n_ev < 0:
        return [f"n_events {n_ev!r} not a non-negative int"]
    for f in ("M_mean", "Delta_mean"):
        v = row.get(f)
        if n_ev >= 1 and not _mean_ok(v):
            out.append(f"{f} {v!r} not in [0, 1] with {n_ev} events")
        if n_ev == 0 and v is not None:
            out.append(f"{f} {v!r} reported with no event (must be unavailable)")
    sums = row.get("sum_by_layer_group") or {}
    counts = row.get("n_measured_by_layer_group")
    for f in ("M", "Delta"):
        s = sums.get(f)
        if (
            not isinstance(s, list)
            or not isinstance(counts, list)
            or len(s) != len(counts)
        ):
            out.append(f"sum_by_layer_group.{f} missing or misshapen")
            continue
        bad = [
            (li, gi)
            for li, (sl, cl) in enumerate(zip(s, counts, strict=True))
            for gi, (v, c) in enumerate(zip(sl, cl, strict=False))
            if not _mass_ok(v, c)
        ]
        if bad:
            out.append(f"sum_by_layer_group.{f} outside [0, count] at {bad[:3]}")
    per_layer = row.get("n_measured_by_layer")
    for f, v in (row.get("sum_by_layer_head") or {}).items():
        if (
            not isinstance(v, list)
            or not isinstance(per_layer, list)
            or len(v) != len(per_layer)
        ):
            out.append(f"sum_by_layer_head.{f} misshapen")
            continue
        bad = [
            (li, hi)
            for li, (sl, c) in enumerate(zip(v, per_layer, strict=True))
            for hi, x in enumerate(sl)
            if not _mass_ok(x, c)
        ]
        if bad:
            out.append(f"sum_by_layer_head.{f} outside [0, count] at {bad[:3]}")
    lat = row.get("latent") or {}
    for f in ("sum_omega", "sum_par", "sum_perp", "n_finite"):
        if f in lat and not _finite_nonneg_nested(lat[f]):
            out.append(f"latent.{f} non-finite or negative")
    om = lat.get("omega_trace")
    if om is not None and not (is_finite_number(om) and float(om) >= 0.0):
        out.append(f"latent.omega_trace {om!r} not finite and non-negative")
    nested = row.get("latent_nested")
    if nested:
        for f in ("sum_omega", "sum_par", "sum_perp"):
            if f in nested and not _finite_nonneg_nested(nested[f]):
                out.append(f"latent_nested.{f} non-finite or negative")
    return out


# ------------------------------------------------------------------ dependencies
def dependency_fingerprint(
    groups: Mapping[str, tuple[Path, Iterable[Path]]],
) -> dict[str, Any]:
    """One hash over everything a read consumes, by group, paths relative to a base.

    ``groups`` maps a name to (base directory, files). Each group's hash covers
    the relative path and bytes of every file, so a deleted, added or changed
    file -- a row file, a token sidecar, a comparator bank, a config, a frozen
    manifest or the analysis code -- changes it. A missing base contributes an
    explicit empty marker, never silence.
    """
    out: dict[str, Any] = {}
    total = hashlib.sha256()
    for name in sorted(groups):
        base, files = groups[name]
        base = Path(base)
        h = hashlib.sha256()
        n = 0
        for p in sorted(Path(f) for f in files):
            rel = p.relative_to(base) if p.is_absolute() else p
            h.update(str(rel).encode())
            h.update(b"\0")
            h.update(
                (base / rel).read_bytes() if not p.is_absolute() else p.read_bytes()
            )
            h.update(b"\0")
            n += 1
        out[name] = {"n_files": n, "sha256": h.hexdigest()}
        total.update(f"{name}:{n}:{h.hexdigest()}\n".encode())
    return {"groups": out, "sha256": total.hexdigest()}


def tree_files(root: Path, sub: str, *, token_ids: bool | None = False) -> list[Path]:
    """Every JSONL under ``root/sub``; ``token_ids`` True = only sidecars,
    False = only result rows, None = both."""
    base = Path(root) / sub
    if not base.exists():
        return []
    files = sorted(base.rglob("*.jsonl"))
    if token_ids is None:
        return files
    return [p for p in files if ("token_ids" in p.parts) == token_ids]

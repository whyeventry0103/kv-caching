"""Production identity for Meeting 5 protected-policy free runs (review of
6b409001, R2; prepared 2026-10-14).

Scientific rationale. The legacy grid runner attaches an experiment identity
and checks it on resume only on its onset-gated D_acc path. The protected
adapters reused its ungated path, so registering a source family did not put
that identity into any production row, nor refuse a resume into rows produced
by different code, a different config, model or reference input. New M5 rows
therefore go through ``scripts/run_grid_m5.py``, which uses this module:

* before the model loads, the expected identity (family generation sources,
  config bytes, pinned model revision, frozen input manifest) is computed and
  every existing row AND token sidecar in the run namespace must carry exactly
  that identity, a design key, and the frozen prompt hash -- otherwise the
  launch is refused, never resumed into;
* every new row and sidecar carries the identity plus the actual prompt-input
  hash (checked against the frozen manifest before writing) and an ``m5``
  block with the nominal and effective budget, prompt length, prune count,
  layers pruned, interval, the occupancy bound between prunes, and retained
  K/V bytes -- the protocol's per-arm reporting (section 3);
* the reader re-checks the same identity (``row_identity_problems``). A reader
  change does not touch generation identity, so it never forces regeneration.

The old generation paths and historical rows are untouched; nothing here
fabricates provenance for rows that never had it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

M5_SOURCES_EXTRA = (
    "scripts/run_grid_m5.py",
    "src/kvcache/m5_runs.py",
    "src/kvcache/m5_publish.py",
)
KIND_BY_FAMILY = {
    "rkv_pp_v1": "rkv_pp",
    "h2o_style_pp_v1": "m5_protsel",
    "uniform_fresh_pp_v1": "m5_protsel",
}


def _register() -> None:
    """``m5_protsel``: the selector family's generation sources plus the M5
    runner (registered here, not in provenance.py, whose bytes are part of
    stored reads' fingerprints)."""
    import kvcache.presses.rkv_prompt_protect  # noqa: F401  (registers rkv_pp)
    from kvcache.provenance import GENERATION_SOURCES

    src = tuple(GENERATION_SOURCES["protected_selectors"]) + M5_SOURCES_EXTRA
    if GENERATION_SOURCES.setdefault("m5_protsel", src) != src:
        raise RuntimeError("provenance family 'm5_protsel' registered twice")


_register()


def family_kind(presses: list[str]) -> str:
    kinds = {KIND_BY_FAMILY.get(str(p).lower()) for p in presses}
    if None in kinds or len(kinds) != 1:
        raise SystemExit(
            f"REFUSED: presses {presses} are not one registered M5 family "
            f"({sorted(KIND_BY_FAMILY)})"
        )
    return kinds.pop()


def expected_identity(
    config: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """The identity every row of this config must carry, from committed files."""
    import yaml

    from kvcache.provenance import PROJECT_ROOT, identity

    base = Path(root) if root is not None else PROJECT_ROOT
    models = yaml.safe_load((base / "configs/models.yaml").read_text())["models"]
    return identity(
        family_kind(list(config["presses"])),
        config,
        model_revision=str(models[config["models"][0]]["revision"]),
        reference_run=None,
        root=base,
        manifest_path=config.get("input_manifest"),
    )


def frozen_prompt_hash(manifest: Mapping[str, Any], pid: str) -> str:
    """The frozen prompt-input hash of a problem (free runs hash the prompt)."""
    try:
        return str(manifest["prompts"][pid])
    except KeyError:
        raise SystemExit(
            f"REFUSED: {pid} is not in the frozen input manifest"
        ) from None


def design_keys(config: Mapping[str, Any], pids: list[str]) -> set[tuple]:
    return {
        (str(p), str(b), pid, int(s))
        for p in config["presses"]
        for b in config["budgets"]
        for pid in pids
        for s in config["seeds"]
    }


def row_identity_problems(
    rec: Mapping[str, Any],
    *,
    ident: Mapping[str, Any],
    manifest: Mapping[str, Any],
    keys: set[tuple],
) -> list[str]:
    """Why one row/sidecar does not belong to this run (empty = it does)."""
    from kvcache.provenance import same_identity

    out = []
    prov = rec.get("provenance")
    if not isinstance(prov, dict) or not same_identity(prov, ident):
        out.append("identity differs from this run's expected identity")
    try:
        seed = rec["decoding"]["seed"] if "decoding" in rec else rec["seed"]
        key = (str(rec["press"]), str(rec["budget"]), str(rec["pid"]), int(seed))
    except (KeyError, TypeError, ValueError):
        return out + ["no readable design key"]
    if key not in keys:
        out.append(f"key {key} is outside the design")
    elif isinstance(prov, dict) and prov.get("inputs_sha256") != manifest.get(
        "prompts", {}
    ).get(key[2]):
        out.append("prompt input hash differs from the frozen manifest")
    return out


def refuse_foreign_namespace(
    runs_root: Path,
    *,
    ident: Mapping[str, Any],
    manifest: Mapping[str, Any],
    keys: set[tuple],
) -> int:
    """Refuse (SystemExit) unless every existing row and sidecar belongs to this
    run. A torn trailing line is skipped here; the resume scan heals it.
    Returns how many records were checked."""
    from kvcache.m5_publish import QUARANTINE

    n = 0
    for path in sorted(Path(runs_root).rglob("*.jsonl")):
        if QUARANTINE in path.relative_to(runs_root).parts:
            continue  # preserved evidence, not live records
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    continue
                raise SystemExit(f"REFUSED: {path} has a malformed line {i}") from None
            probs = row_identity_problems(
                rec, ident=ident, manifest=manifest, keys=keys
            )
            if probs:
                raise SystemExit(
                    f"REFUSED: {path} line {i}: {probs[0]}; start a new run "
                    "namespace instead of resuming"
                )
            n += 1
    return n


def retention_block(
    press: Any,
    *,
    family: str,
    nominal_budget: int,
    interval: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> dict[str, Any]:
    """The per-row capacity and retention diagnostics the protocol promises."""
    rep = press.protect.report()
    target = int(rep["target"])
    return {
        "family": family,
        "nominal_budget": int(nominal_budget),
        "effective_budget": target,
        "budget_relaxed": bool(rep["budget_relaxed"]),
        "prompt_len": int(rep["prompt_len"]),
        "n_prunes": int(rep["n_prunes"]),
        "n_layers_pruned": int(rep["n_layers_pruned"]),
        "compression_interval": int(interval),
        "peak_occupancy_bound": target + int(interval),
        "kv_bytes_after_prune": target
        * n_layers
        * n_kv_heads
        * head_dim
        * 2
        * dtype_bytes,
    }

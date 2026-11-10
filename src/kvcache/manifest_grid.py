"""Frozen-ID compressed free runs and forward readouts for the registered
Meeting 5 replication (M5-B, protocol §5; prepared 2026-11-10, before any
GSM8K compressed run). Design and cost stay PENDING the timing read.

Scientific rationale. The replication must run the SAME frozen problems
(``configs/meeting5/problem_manifests.json``, confirmation split) under each
compression policy, with the policies' own implementations and promotion
gates, and must score GSM8K answers with the registered read-time grader --
otherwise a GSM8K contrast would compare different problems, different press
code, or a grader known to mark GSM8K answers wrong. This module holds the
pieces both runners share, without changing any generation path:

* ``resolve_grid_design``: a config's frozen split (by ID, pinned revision,
  disjoint splits), seeds, presses, budgets and declared count, refusing
  anything a count- or sample-based ``subset`` could silently change;
* ``FAMILIES``: the existing press families, each with its own builder,
  eager-attention rule and promotion gate, imported unchanged. One family per
  config, as every existing runner requires, so each family's gate sees a
  config of its own kind;
* ``design_keys``: the expected (press, budget, pid, seed) keys, in order.

``design_status`` is ``pending`` until the timing read fixes the design; real
execution refuses until it is ``final`` (and the config is unheld, authorized
and its family gate passes).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kvcache.manifest_problems import frozen_split_pids

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESIGN_STATUSES = ("pending", "final")

Key = tuple[str, str, str, int]


@dataclass(frozen=True)
class Family:
    """One registered press family, resolved lazily (kvpress/torch stay out of
    the CPU gate)."""

    name: str
    flag: str | None
    module: str
    builder: str
    eager: str
    gate: str | None


FAMILIES: dict[str, Family] = {
    "base": Family(
        "base",
        None,
        "kvcache.presses",
        "build_press",
        "press_needs_eager_attention",
        None,
    ),
    "pp": Family(
        "pp",
        "prompt_protected",
        "kvcache.presses.prompt_protect",
        "build_prompt_protected_press",
        "",
        "require_pp_config",
    ),
    "protsel": Family(
        "protsel",
        "protected_selector",
        "kvcache.presses.protected_selectors",
        "build_protected_selector_press",
        "needs_eager_attention",
        "require_protected_config",
    ),
    "rkv_pp": Family(
        "rkv_pp",
        "rkv_prompt_protected",
        "kvcache.presses.rkv_prompt_protect",
        "build_rkv_prompt_protected_press",
        "needs_eager_attention",
        "require_rkv_pp_config",
    ),
}


def family_of(config: Mapping[str, Any]) -> Family:
    """The config's single press family (the flag it sets, or base)."""
    flagged = [f for f in FAMILIES.values() if f.flag and config.get(f.flag) is True]
    if len(flagged) > 1:
        raise ValueError(
            f"config sets {[f.flag for f in flagged]}: one press family per config"
        )
    return flagged[0] if flagged else FAMILIES["base"]


def _attr(family: Family, name: str) -> Any:
    import importlib

    return getattr(importlib.import_module(family.module), name)


def builder(family: Family) -> Callable[..., Any]:
    return _attr(family, family.builder)


def needs_eager(family: Family, press: str) -> bool:
    if family.name == "pp":
        # S.1 protects Random / StreamingLLM only; neither scores by attention.
        from kvcache.presses import press_needs_eager_attention
        from kvcache.presses.prompt_protect import pp_base_name

        return press_needs_eager_attention(pp_base_name(press))
    return bool(_attr(family, family.eager)(press))


def run_gate(
    family: Family,
    config: dict[str, Any],
    *,
    dry_run: bool,
    runs_root: Any,
    replay: bool,
) -> None:
    """The family's own promotion gate, unchanged (SystemExit on refusal)."""
    if family.gate is None:
        # base presses: eligibility tiers are enforced inside build_press
        # (get_press; R-KV promotes only via its smoke marker).
        from kvcache.presses import (
            DEFERRED,
            EXECUTABLE,
            PORT_REQUIRED,
            PREFILL_ONLY,
            get_press,
        )

        known = EXECUTABLE | PREFILL_ONLY | PORT_REQUIRED | DEFERRED
        for press in config["presses"]:
            name = str(press).lower()
            if name not in known:
                raise SystemExit(f"REFUSED: unknown press {press!r}")
            if not dry_run:
                try:
                    get_press(name)
                except NotImplementedError as exc:
                    raise SystemExit(f"REFUSED: {exc}") from exc
        return
    gate = _attr(family, family.gate)
    if family.name == "pp":
        gate(config, replay=replay)
        return
    gate(config, dry_run=dry_run, runs_root=runs_root, replay=replay)


def _yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_grid_design(
    config: Mapping[str, Any], *, kind: str, root: Path | None = None
) -> dict[str, Any]:
    """The frozen-ID design a compressed-run (``kind='grid'``) or readout
    (``kind='readout'``) config declares, or ValueError."""
    base = Path(root) if root is not None else PROJECT_ROOT
    if kind not in {"grid", "readout"}:
        raise ValueError(f"unknown kind {kind!r}")
    if config.get("design_status") not in DESIGN_STATUSES:
        raise ValueError(f"design_status must be one of {DESIGN_STATUSES}")
    if "subset" in config:
        raise ValueError("frozen-ID runs take no subset: the manifest is the set")
    if len(config["benchmarks"]) != 1:
        raise ValueError("one benchmark per manifest run")
    bench = str(config["benchmarks"][0])
    if bench == "math500":
        raise ValueError("MATH-500 cells run through their registered runners")
    presses = [str(p) for p in config["presses"]]
    if not presses or "fullcache" in [p.lower() for p in presses]:
        raise ValueError(
            "compressed presses only; the full cache is the manifest bank "
            "(scripts/run_bank_manifest.py)"
        )
    budgets = []
    for b in config["budgets"]:
        if isinstance(b, bool) or not isinstance(b, int) or b <= 0:
            raise ValueError(f"budget {b!r} is not a positive integer")
        budgets.append(b)
    spec = config.get("manifest") or {}
    if not spec.get("path") or not spec.get("split"):
        raise ValueError("config.manifest needs path and split")
    manifest = json.loads((base / spec["path"]).read_text(encoding="utf-8"))
    bench_cfg = _yaml(base / "configs/benchmarks.yaml")["benchmarks"][bench]
    revision = str(bench_cfg["revision"])
    pids = frozen_split_pids(manifest, bench, spec["split"], pinned_revision=revision)
    seeds = [int(s) for s in config["seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds repeat")
    n = len(pids) * len(seeds) * len(presses) * len(budgets)
    count_key = (
        "estimated_generation_count" if kind == "grid" else "estimated_replay_count"
    )
    if int(config.get(count_key, -1)) != n:
        raise ValueError(
            f"{count_key} {config.get(count_key)} != {len(pids)} problems x "
            f"{len(seeds)} seeds x {len(presses)} presses x {len(budgets)} "
            f"budgets = {n}"
        )
    if int(config.get("compression_interval", 0)) <= 0:
        raise ValueError("compression_interval must be a positive integer")
    design = {
        "kind": kind,
        "benchmark": bench,
        "split": str(spec["split"]),
        "pids": pids,
        "n_pool": int(manifest[bench]["n_pool"]),
        "revision": revision,
        "seeds": seeds,
        "presses": presses,
        "budgets": budgets,
        "n_conditions": n,
        "family": family_of(config).name,
        "model": str(config["models"][0]),
        "design_status": str(config["design_status"]),
    }
    if kind == "grid":
        if config.get("store_token_ids") is not True:
            raise ValueError("manifest grids publish row + token-ID pairs")
        design["max_new_tokens"] = int(config["max_new_tokens"][bench])
    else:
        if config.get("mode") != "readout":
            raise ValueError("manifest readout configs set mode: readout")
        if not config.get("reference_run_name"):
            raise ValueError("readout needs reference_run_name (the full-cache bank)")
        if int(config.get("max_answer_tokens", 0)) <= 0:
            raise ValueError("max_answer_tokens must be a positive integer")
        for key in ("matched_ratio", "restoration", "compression_start_step"):
            if key in config:
                raise ValueError(f"{key} is not registered for the replication")
        design["reference_run_name"] = str(config["reference_run_name"])
        design["max_answer_tokens"] = int(config["max_answer_tokens"])
    return design


def design_keys(design: Mapping[str, Any]) -> list[Key]:
    """Expected (press, budget, pid, seed) keys in run order."""
    return [
        (str(p), str(b), str(pid), int(s))
        for p in design["presses"]
        for b in design["budgets"]
        for s in design["seeds"]
        for pid in design["pids"]
    ]


def require_executable(config: Mapping[str, Any]) -> None:
    """The replication-level guard every real run passes first."""
    if config.get("held", True):
        raise SystemExit("REFUSED: this replication config is HELD")
    if config.get("design_status") != "final":
        raise SystemExit(
            "REFUSED: replication design is PENDING the GSM8K timing read; "
            "set design_status: final only with the registered design"
        )
    if not config.get("scale_authorized"):
        raise SystemExit("REFUSED: config must set scale_authorized: true")

"""Prompt-protected R-KV (Meeting 5, work package M5-A; prepared 2026-10-14,
HELD -- not registered, not run).

Scientific rationale. Amendment S.1 protected the prompt for Random and
StreamingLLM; the protected-selector module adds H2O-style and fresh-uniform
random. R-KV was the remaining policy without a protected arm, so its
accuracy/perturbation comparison under equal prompt protection could not be
made. This is a separate, versioned path: the parity-verified R-KV scorer
(``kvcache.presses.rkv``) and its kvpress press are reused unchanged.

Why this is NOT the H2O/uniform adapter. That adapter selects with a stable
Python sort and returns the kept slots in absolute-position order. R-KV's own
redundancy term (``cal_similarity``, retain-"last" link zeroing) and its window
query use SLOT order, and kvpress's ``ScorerPress.compress`` gathers survivors
in ``topk`` score order. Re-sorting chronologically would change R-KV's
behaviour. So this wrapper keeps the exact base contract:

1. absolute positions are tracked per layer and KV head (``prompt_protect``'s
   position map);
2. scores come from the unmodified R-KV scorer (window-query recompute,
   lambda 0.07, max-pooled importance, cosine redundancy); its recent-``8``
   window is padded with the score maximum, as in the base press;
3. the score of every prompt position is raised to the dtype maximum, so the
   mandatory set is prompt UNION recent-8 (an overlap counts once);
4. ``scores.topk(n_kept)`` and the K/V gather are exactly kvpress's -- with no
   prompt (``forced_prompt_len=0``) this is the base path line for line;
5. every prune checks that every prompt position and the 8 most recent
   positions survived in every layer and KV head, with no duplicates.

Budget rule: the S.1 total budget, relaxed to P + 64 when the prompt leaves
fewer than 64 trace slots (64 >= the 8-entry window).
"""

from __future__ import annotations

from typing import Any

from kvcache.presses.prompt_protect import (
    MIN_TRACE_SLOTS,
    ProtectState,
    current_positions,
    effective_budget,
)
from kvcache.presses.rkv import RKV_WINDOW_SIZE

FAMILY = "rkv_pp_v1"
SMOKE_MARKER = "rkv-pp-smoke-PASS.json"
SMOKE_KIND = "rkv_pp"
# Generation identity of an rkv_pp_v1 row: the wrapper, the unchanged R-KV
# scorer, the shared position tracking, and every module and runner a row passes
# through. Registered here, not in kvcache.provenance: provenance.py is part of
# the stored reads' analysis-code fingerprint, and editing it would stale every
# FINAL read for a family that has produced no row.
SOURCES = (
    "src/kvcache/presses/rkv_prompt_protect.py",
    "src/kvcache/presses/rkv.py",
    "src/kvcache/presses/prompt_protect.py",
    "src/kvcache/presses/__init__.py",
    "src/kvcache/generation.py",
    "src/kvcache/replay.py",
    "src/kvcache/models.py",
    "src/kvcache/bank.py",
    "scripts/run_grid.py",
    "scripts/run_replay.py",
    "scripts/run_grid_rkv_pp.py",
    "scripts/run_replay_rkv_pp.py",
    "scripts/rkv_pp_smoke.py",
)


def _register_sources() -> None:
    from kvcache.provenance import GENERATION_SOURCES

    if GENERATION_SOURCES.setdefault(SMOKE_KIND, SOURCES) != SOURCES:
        raise RuntimeError(f"provenance family {SMOKE_KIND!r} registered twice")


_register_sources()


def needs_eager_attention(name: str) -> bool:
    """R-KV recomputes its window queries from hidden states, so the protected
    family needs exactly the base policy's attention backend."""
    from kvcache.presses import press_needs_eager_attention

    return press_needs_eager_attention("rkv" if name.lower() == FAMILY else name)


def check_rkv_retained(
    state: ProtectState, before: Any, after: Any, layer: int, window: int
) -> None:
    """Raise unless prompt UNION the recent window survived in every head."""
    prompt_len = int(state.prompt_len or 0)
    for head in range(int(after.shape[0])):
        kept = set(after[head].tolist())
        had = sorted(before[head].tolist())
        if len(kept) != int(after.shape[1]):
            raise RuntimeError(f"layer {layer} head {head}: duplicated position")
        must = {p for p in had if p < prompt_len} | set(had[-window:])
        if not must <= kept:
            raise RuntimeError(
                f"{FAMILY} layer {layer} head {head}: a protected position "
                f"(prompt or recent-{window}) was evicted (prune {state.n_prunes})"
            )


def build_rkv_prompt_protected_press(
    name: str = FAMILY,
    *,
    budget: int,
    compression_interval: int = 64,
    seed: int | None = None,
    kvpress_module: Any | None = None,
    decoding_press_cls: type | None = None,
    instrumented: bool = False,
    forced_prompt_len: int | None = None,
    min_trace_slots: int = MIN_TRACE_SLOTS,
) -> Any:
    """A decoding press for prompt-protected R-KV (``build_press``'s signature).

    ``forced_prompt_len=0`` is the identity control: the press then IS the base
    R-KV path (same scores, same topk, same gather order).
    """
    from kvcache.presses import build_press

    if name.lower() != FAMILY:
        raise KeyError(f"not the protected R-KV family: {name!r}")
    press = build_press(
        "rkv",
        budget=budget,
        compression_interval=compression_interval,
        seed=seed,
        kvpress_module=kvpress_module,
        decoding_press_cls=decoding_press_cls,
        instrumented=instrumented,
    )
    scorer = press.base_press
    if "compress" in vars(scorer):
        raise RuntimeError("the base scorer already carries a patched compress")
    window = int(getattr(scorer, "window_size", RKV_WINDOW_SIZE))
    state = ProtectState(
        base=FAMILY,
        budget=int(budget),
        min_trace_slots=max(int(min_trace_slots), window),
    )
    press.protect = state

    def set_prompt(prompt_len: int) -> None:
        state.prompt_len = int(prompt_len)
        state.target, state.budget_relaxed = effective_budget(
            state.budget, state.prompt_len, state.min_trace_slots
        )
        press.target_size = state.target

    if forced_prompt_len is not None:
        state.prompt_len_forced = True
        set_prompt(int(forced_prompt_len))

    inner_hook = press.forward_hook

    def forward_hook(module: Any, input: Any, kwargs: dict, output: Any) -> Any:
        q_len = int(kwargs["hidden_states"].shape[1])
        prefill = int(kwargs["cache_position"][-1]) + 1 == q_len
        if state.prompt_len is None:
            if not prefill:
                raise RuntimeError("the first forward is not a pure prefill")
            set_prompt(q_len)
        elif prefill and q_len > 1 and not state.prompt_len_forced:
            if state.n_prunes or q_len != state.prompt_len:
                raise RuntimeError("a second prompt reached a used press")
        return inner_hook(module, input, kwargs, output)

    press.forward_hook = forward_hook

    def compress(
        module: Any,
        hidden_states: Any,
        keys: Any,
        values: Any,
        attentions: Any,
        kwargs: dict,
    ) -> tuple[Any, Any]:
        import torch

        if scorer.compression_ratio == 0:
            return keys, values
        if state.prompt_len is None:
            raise RuntimeError("prune before the prompt length was read")
        if int(keys.shape[0]) != 1:
            raise RuntimeError("protected R-KV is registered for batch size 1")
        layer = int(module.layer_idx)
        n_kv, k_len = int(keys.shape[1]), int(keys.shape[2])
        before = current_positions(state, layer, n_kv, k_len)
        scores = scorer.score(module, hidden_states, keys, values, attentions, kwargs)
        n_kept = int(k_len * (1 - scorer.compression_ratio))
        if state.prompt_len:
            protected = (before < state.prompt_len).to(scores.device)
            if int(protected.sum(-1).max()) + window > n_kept:
                raise RuntimeError(
                    f"target {n_kept} cannot hold the prompt plus the recent "
                    f"{window} entries"
                )
            scores = scores.clone()
            scores[0][protected] = torch.finfo(scores.dtype).max
        # kvpress ScorerPress.compress, verbatim from here on
        indices = scores.topk(n_kept, dim=-1).indices
        after = before.gather(1, indices[0].to(before.device))
        check_rkv_retained(state, before, after, layer, window)
        state.maps[layer] = after
        state.n_prunes += 1
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values

    scorer.compress = compress
    return press


def record_prune_outputs(press: Any) -> list[tuple[int, str]]:
    """Hash every prune's returned keys and values, per layer, in call order.

    Wraps the scorer's ``compress`` AFTER the press is built (the builder
    refuses an already-patched scorer). Used by the identity control: base
    ``rkv`` and ``rkv_pp_v1`` with nothing protected must produce identical
    logs.
    """
    import hashlib

    scorer = press.base_press
    inner = scorer.compress
    log: list[tuple[int, str]] = []

    def compress(
        module: Any,
        hidden_states: Any,
        keys: Any,
        values: Any,
        attentions: Any,
        kwargs: dict,
    ) -> tuple[Any, Any]:
        k, v = inner(module, hidden_states, keys, values, attentions, kwargs)
        h = hashlib.sha256()
        for t in (k, v):
            h.update(str(tuple(t.shape)).encode())
            h.update(t.detach().float().cpu().numpy().tobytes())
        log.append((int(module.layer_idx), h.hexdigest()))
        return k, v

    scorer.compress = compress
    return log


def first_mismatch(a: list[Any], b: list[Any]) -> int | None:
    """Index of the first differing entry (a length difference counts)."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def smoke_failures(records: list[dict[str, Any]]) -> list[str]:
    """Every registered rkv_pp_v1 check that did not hold (empty = PASS).

    identity: base ``rkv`` vs ``rkv_pp_v1`` with ``forced_prompt_len=0`` -- at
    least one prune, byte-identical K/V after every prune, identical TV.
    replay (prompt protected): the press read the true prompt length; target and
    every post-prune size equal the effective target; every layer pruned;
    pre-eviction TV exactly 0; the live retention check never raised.
    generation: prompt length read and at least one prune.
    """
    fails: list[str] = []
    if not records:
        return ["no smoke record"]
    for r in records:
        tag = f"{r['kind']} {r['pid']}"
        if "raised" in r:
            fails.append(f"{tag}: raised: {r['raised']}")
            continue
        if r["kind"] == "identity":
            if r["n_prunes_base"] < 1:
                fails.append(f"{tag}: identity control compared no prune")
            if r["first_kv_mismatch"] is not None:
                fails.append(f"{tag}: K/V differ from base at {r['first_kv_mismatch']}")
            if not r["tv_identical"]:
                fails.append(f"{tag}: TV profile differs from base")
            continue
        protect = r["protect"]
        if protect["prompt_len"] != r["prompt_len"]:
            fails.append(
                f"{tag}: press read prompt length {protect['prompt_len']}, "
                f"true length {r['prompt_len']}"
            )
        if r["kind"] == "generation":
            if protect["n_prunes"] < 1:
                fails.append(f"{tag}: no prune in {r['n_tokens']} generated tokens")
            continue
        if protect["target"] != r["expected_target"]:
            fails.append(f"{tag}: target {protect['target']} != {r['expected_target']}")
        if r["first_effective_prune"] is None:
            fails.append(f"{tag}: no effective prune")
            continue
        if protect["n_layers_pruned"] != r["n_layers"]:
            fails.append(
                f"{tag}: pruned {protect['n_layers_pruned']} of {r['n_layers']} layers"
            )
        if r["pre_eviction_max_tv"] != 0.0:
            fails.append(f"{tag}: pre-eviction TV {r['pre_eviction_max_tv']} != 0")
        if r["post_prune_sizes"] != [r["expected_target"]]:
            fails.append(
                f"{tag}: post-prune sizes {r['post_prune_sizes']} != "
                f"[{r['expected_target']}]"
            )
    return fails


def smoke_identity(smoke_cfg: dict[str, Any], *, root: Any = None) -> dict[str, Any]:
    """The identity a PASS must carry, computed from committed files now."""
    from pathlib import Path

    import yaml

    from kvcache.provenance import PROJECT_ROOT, identity

    base = Path(root) if root is not None else PROJECT_ROOT
    models = yaml.safe_load((base / "configs/models.yaml").read_text())["models"]
    return identity(
        SMOKE_KIND,
        smoke_cfg,
        model_revision=str(models[smoke_cfg["models"][0]]["revision"]),
        reference_run=smoke_cfg.get("reference_run_name"),
        root=base,
        manifest_path=smoke_cfg.get("input_manifest"),
    )


def smoke_coverage_problems(record: dict[str, Any]) -> list[str]:
    """The registered design: identity, replay and generation on each of the
    three selected traces (9 records)."""
    have = {(r.get("kind"), str(r.get("pid"))) for r in record.get("records") or []}
    pids = [str(t["pid"]) for t in record.get("selected") or []]
    if len(pids) != 3:
        return [f"{len(pids)} selected traces, expected 3"]
    need = [(k, p) for p in pids for k in ("identity", "replay", "generation")]
    return [f"missing {k}" for k in need if k not in have]


def smoke_record_ok(
    runs_root: Any, smoke_cfg: dict[str, Any], *, root: Any = None
) -> tuple[bool, str]:
    """Is the marker under ``runs_root`` a valid, current PASS for this config?"""
    import json
    from pathlib import Path

    from kvcache.provenance import same_identity

    marker = Path(runs_root) / SMOKE_MARKER
    try:
        d = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, f"{SMOKE_MARKER} absent, empty or unreadable"
    if not isinstance(d, dict):
        return False, f"{SMOKE_MARKER} is not a record"
    if d.get("passed") is not True or d.get("status") != "PASS":
        return False, f"{SMOKE_MARKER} does not record a PASS"
    if d.get("failures"):
        return False, f"{SMOKE_MARKER} lists failures"
    if not same_identity(d.get("identity"), smoke_identity(smoke_cfg, root=root)):
        return False, f"{SMOKE_MARKER} is stale: its identity differs from now"
    missing = smoke_coverage_problems(d)
    if missing:
        return False, f"{SMOKE_MARKER} does not cover the design: {missing[:3]}"
    return True, "verified"


def verify_smoke(
    runs_root: Any, run_config: dict[str, Any], *, root: Any = None
) -> tuple[bool, str]:
    """A valid PASS for the run's smoke config that covers the run's model,
    budgets and interval."""
    from pathlib import Path

    import yaml

    from kvcache.provenance import PROJECT_ROOT

    base = Path(root) if root is not None else PROJECT_ROOT
    if not run_config.get("smoke_config"):
        return False, "run config names no smoke_config"
    smoke_cfg = yaml.safe_load((base / run_config["smoke_config"]).read_text())
    ok, why = smoke_record_ok(runs_root, smoke_cfg, root=base)
    if not ok:
        return ok, why
    if run_config["models"] != smoke_cfg["models"]:
        return False, "the smoke tested a different model"
    if not {str(b) for b in run_config["budgets"]} <= {
        str(b) for b in smoke_cfg["budgets"]
    }:
        return False, "the smoke did not test every budget of this run"
    if int(run_config["compression_interval"]) != int(
        smoke_cfg["compression_interval"]
    ):
        return False, "the smoke used another compression interval"
    return True, "verified"


def require_rkv_pp_config(
    config: dict[str, Any],
    *,
    dry_run: bool,
    runs_root: Any = None,
    replay: bool = False,
) -> None:
    """Refuse a config that is not an rkv_pp_v1 cell, and refuse REAL execution
    while it is held or without a valid smoke PASS. Dry runs always allowed.

    This family is deliberately outside the protected-selector registry and the
    press eligibility tiers; this gate is its only promotion mechanism.
    """
    if config.get("rkv_prompt_protected") is not True:
        raise SystemExit("REFUSED: config must set rkv_prompt_protected: true")
    if [str(p).lower() for p in config["presses"]] != [FAMILY]:
        raise SystemExit(f"REFUSED: presses must be exactly [{FAMILY}]")
    for key in ("compression_start_step", "matched_ratio", "restoration"):
        if key in config:
            raise SystemExit(f"REFUSED: {key} is not registered for these cells")
    if replay and config.get("mode") != "readout":
        raise SystemExit("REFUSED: rkv_pp_v1 replay is registered for readout")
    if dry_run:
        return
    if config.get("held", True):
        raise SystemExit(
            "REFUSED: config is HELD (not authorized); remove `held` only with a "
            "dated registration and an approved budget"
        )
    if runs_root is None:
        raise SystemExit("REFUSED: no runs root to find the smoke marker in")
    ok, why = verify_smoke(runs_root, config)
    if not ok:
        raise SystemExit(f"REFUSED: smoke gate: {why}")

"""Additional prompt-protected selectors for the protected-selector extension
(execution plan WP4/WP5, prepared 2026-10-05; NOT registered, NOT run).

Scientific rationale. Amendment S.1 protected the prompt for Random and
StreamingLLM only. Two objections remain open, and these families answer them:

* ``h2o_style_pp_v1`` -- the H2O-style selector (kvpress
  ``ObservedAttentionPress``: observed current attention, NO recency window, NOT
  canonical accumulated H2O) with the prompt protected under the same S.1
  total-budget rule. Without it, the metric/accuracy mismatch on the capture
  problems (Random above H2O-style in accuracy while losing more mass) could be
  a prompt-protection artefact.
* ``uniform_fresh_pp_v1`` -- genuinely fresh uniform random retention. The
  legacy ``random`` / ``random_pp`` presses reseed one private generator to the
  same seed at every prune, so they evict a FIXED slot pattern (a soft-recency
  sieve), not a fresh uniform draw. Each draw here is keyed by (family, press
  seed, layer, KV head, prune index) through SHA-256 and drawn with a private
  ``random.Random``: stateless (a resumed sequence reproduces it), identical on
  every device, and it never touches torch's or Python's global RNG, so the
  language-model sampling stream is unaffected.
* ``uniform_fresh_pp_r64_v1`` -- the same, plus a 64-entry recency buffer that
  is never scored, matching the official Random Attention code's convention
  (``--token_budget K --residual_length 64``: after a prune, K entries in total,
  of which the 64 most recent are kept unconditionally and the prompt is pinned
  inside the other K - 64). This is OUR VARIANT of that convention, not a
  reproduction of the Random Attention method (their prune trigger, compaction
  order and kernels differ).

Selection is shared and pure Python (``select_keep``): protected entries
(prompt, and the recency buffer when present) are always kept; the remaining
slots go to the highest scores, ties broken by slot index (stable, so a run is
deterministic even where CUDA top-k is not); the kept slots are returned in
absolute-position order. Positions are tracked per layer and KV head exactly as
in ``kvcache.presses.prompt_protect``, whose bytes (generation identity of the
S.1 cells) are not modified: this module imports from it.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from kvcache.presses.prompt_protect import (
    MIN_TRACE_SLOTS,
    ProtectState,
    current_positions,
    effective_budget,
)


@dataclass(frozen=True)
class Family:
    name: str
    base: str  # build_press base whose DecodingPress machinery is reused
    fresh_uniform: bool
    recent: int  # recency buffer, never scored (0 = none)


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        Family("h2o_style_pp_v1", "h2o", fresh_uniform=False, recent=0),
        Family("uniform_fresh_pp_v1", "random", fresh_uniform=True, recent=0),
        Family("uniform_fresh_pp_r64_v1", "random", fresh_uniform=True, recent=64),
    )
}


def family(name: str) -> Family:
    try:
        return FAMILIES[name.lower()]
    except KeyError:
        raise KeyError(
            f"not a protected-selector family: {name!r}; known: {sorted(FAMILIES)}"
        ) from None


def draw_seed(
    family_name: str, press_seed: int, layer: int, head: int, prune: int
) -> int:
    """Stable 63-bit seed for one (sequence, layer, KV head, prune) draw."""
    msg = f"{family_name}|{int(press_seed)}|{int(layer)}|{int(head)}|{int(prune)}"
    return int.from_bytes(hashlib.sha256(msg.encode()).digest()[:8], "big") >> 1


def fresh_uniform_scores(n: int, seed: int) -> list[float]:
    """n i.i.d. Uniform(0, 1) scores from a private generator."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(int(n))]


def select_keep(
    scores: Sequence[float],
    positions: Sequence[int],
    *,
    n_keep: int,
    prompt_len: int,
    recent_from: int | None = None,
) -> list[int]:
    """Slot indices to keep, in absolute-position order.

    Protected: every slot whose position is < ``prompt_len`` and, when
    ``recent_from`` is given, every slot whose position is >= it. The rest of
    ``n_keep`` goes to the highest scores; ties go to the lower slot index.
    """
    n = len(scores)
    if len(positions) != n:
        raise ValueError("scores and positions differ in length")
    if len(set(positions)) != n:
        raise ValueError("duplicated position in the cache")
    if not 0 < n_keep <= n:
        raise ValueError(f"n_keep {n_keep} outside (0, {n}]")
    protected = [
        i
        for i, p in enumerate(positions)
        if p < prompt_len or (recent_from is not None and p >= recent_from)
    ]
    if len(protected) > n_keep:
        raise ValueError(
            f"{len(protected)} protected entries exceed the target {n_keep}"
        )
    pset = set(protected)
    rest = sorted(
        (i for i in range(n) if i not in pset), key=lambda i: (-float(scores[i]), i)
    )
    kept = protected + rest[: n_keep - len(protected)]
    return sorted(kept, key=lambda i: positions[i])


def needs_eager_attention(name: str) -> bool:
    """Attention backend for a press name, mapping each protected family to its
    base policy (review of 8e1e81a: ``h2o_style_pp_v1`` scores observed
    attention, so it needs eager attention exactly as ``h2o`` does; the
    registry's ``press_needs_eager_attention`` does not know the family)."""
    from kvcache.presses import press_needs_eager_attention

    key = name.lower()
    if key in FAMILIES:
        return press_needs_eager_attention(FAMILIES[key].base)
    return press_needs_eager_attention(name)


SMOKE_MARKER = "protected-selectors-smoke-PASS.json"
SMOKE_KIND = "protected_selectors"


def smoke_identity(smoke_cfg: dict[str, Any], *, root: Any = None) -> dict[str, Any]:
    """The identity a smoke PASS must carry: generation code of the family
    (``implementation_id``), the smoke config's bytes, the pinned model
    revision, the frozen input manifest and the reference run. Computed from
    committed files, never read from the marker."""
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


def verify_smoke(
    runs_root: Any, run_config: dict[str, Any], *, root: Any = None
) -> tuple[bool, str]:
    """Is there a VALID smoke PASS for this run config? (review of 8e1e81a)

    Required: the marker parses to a mapping that says ``passed: true``,
    ``status: "PASS"`` and has no failures; its identity equals the identity
    computed now (so a generation-code, model, smoke-config or input change
    makes it stale); and the run config is covered by what the smoke tested
    (every family, the model, every budget, the interval).
    """
    import json
    from pathlib import Path

    import yaml

    from kvcache.provenance import PROJECT_ROOT, same_identity

    base = Path(root) if root is not None else PROJECT_ROOT
    if not run_config.get("smoke_config"):
        return False, "run config names no smoke_config"
    smoke_cfg = yaml.safe_load((base / run_config["smoke_config"]).read_text())
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
    if not same_identity(d.get("identity"), smoke_identity(smoke_cfg, root=base)):
        return False, f"{SMOKE_MARKER} is stale: its identity differs from now"
    tested = set(d.get("families_tested") or [])
    if not {str(p).lower() for p in run_config["presses"]} <= tested:
        return False, "the smoke did not test every family of this run"
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


def require_protected_config(
    config: dict[str, Any],
    *,
    dry_run: bool,
    runs_root: Any = None,
    replay: bool = False,
) -> None:
    """Refuse a config that is not a protected-selector cell, and refuse REAL
    execution while the config is held or without a VALID smoke PASS.

    The families are outside ``kvcache.presses``'s eligibility tiers, so this is
    their promotion mechanism: a verified PASS record (``verify_smoke``) plus a
    config whose ``held`` flag was removed by an authorized registration. Dry
    runs are always allowed.
    """
    if config.get("protected_selector") is not True:
        raise SystemExit("REFUSED: config must set protected_selector: true")
    for name in config["presses"]:
        try:
            family(str(name))
        except KeyError as exc:
            raise SystemExit(f"REFUSED: {exc.args[0]}") from None
    for key in ("compression_start_step", "matched_ratio", "restoration"):
        if key in config:
            raise SystemExit(f"REFUSED: {key} is not registered for these cells")
    if replay and config.get("mode") != "readout":
        raise SystemExit("REFUSED: protected-selector replay is registered for readout")
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


def smoke_failures(records: list[dict[str, Any]]) -> list[str]:
    """Every registered WP4 smoke check that did not hold (empty = PASS).

    Replay records: the press read the true prompt length; target and every
    post-prune size equal the effective target; every layer pruned;
    pre-eviction TV exactly 0; the live retention check never raised. Identity
    records (h2o_style_pp_v1, nothing protected): at least one prune checked
    and zero heads whose selection differs from kvpress's top-k. Generation
    records: the prompt length was read and at least one prune happened.
    """
    fails: list[str] = []
    if not records:
        return ["no smoke record"]
    for r in records:
        tag = f"{r['kind']} {r['press']} {r['pid']}"
        if "raised" in r:
            fails.append(f"{tag}: raised: {r['raised']}")
            continue
        if r["kind"] == "identity":
            if r["identity_prunes"] < 1:
                fails.append(f"{tag}: identity control checked no prune")
            if r["identity_mismatched_heads"] != 0:
                fails.append(
                    f"{tag}: {r['identity_mismatched_heads']} head selections "
                    "differ from kvpress top-k"
                )
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


def build_protected_selector_press(
    name: str,
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
    """A decoding press for one protected-selector family (``build_press``'s
    signature, so a runner can swap it in as ``run_grid_pp`` does).

    The S.1 total-budget rule applies, with the recency buffer counted inside
    the budget; the minimum trace allowance is raised to cover the buffer.
    ``forced_prompt_len=0`` (with no recency buffer) is the identity control:
    the H2O-style family then keeps exactly the stable top-n of the base
    scores.
    """
    from kvcache.presses import build_press

    fam = family(name)
    if fam.fresh_uniform and seed is None:
        raise ValueError("fresh-uniform families need the per-sequence press seed")
    press = build_press(
        fam.base,
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
    min_slots = max(int(min_trace_slots), fam.recent)
    state = ProtectState(base=fam.name, budget=int(budget), min_trace_slots=min_slots)
    press.protect = state
    prunes: dict[int, int] = {}
    identity_check = {"prunes": 0, "mismatched_heads": 0}
    press.identity_check = identity_check

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
            raise RuntimeError("protected selectors are registered for batch size 1")
        layer = int(module.layer_idx)
        n_kv, k_len = int(keys.shape[1]), int(keys.shape[2])
        before = current_positions(state, layer, n_kv, k_len)
        n_keep = int(k_len * (1 - scorer.compression_ratio))
        prune = prunes.get(layer, 0)
        logical_len = int(before.max()) + 1
        recent_from = logical_len - fam.recent if fam.recent else None
        ref = None
        if fam.fresh_uniform:
            per_head = [
                fresh_uniform_scores(
                    k_len, draw_seed(fam.name, int(seed), layer, h, prune)
                )
                for h in range(n_kv)
            ]
        else:
            s = scorer.score(module, hidden_states, keys, values, attentions, kwargs)
            per_head = s[0].float().cpu().tolist()
            ref = (
                s[0].topk(n_keep, dim=-1).indices.cpu().tolist()
                if int(state.prompt_len) == 0 and not fam.recent
                else None
            )
        pos = before.tolist()
        idx = [
            select_keep(
                per_head[h],
                pos[h],
                n_keep=n_keep,
                prompt_len=int(state.prompt_len),
                recent_from=recent_from,
            )
            for h in range(n_kv)
        ]
        if not fam.fresh_uniform and ref is not None:
            # Identity control (smoke): nothing protected, so the selection
            # must equal kvpress's own top-k on the SAME live scores.
            identity_check["prunes"] += 1
            identity_check["mismatched_heads"] += sum(
                set(ref[h]) != set(idx[h]) for h in range(n_kv)
            )
        index = torch.tensor(idx, dtype=torch.long)
        after = before.gather(1, index)
        _check(state, fam, before, after, layer, recent_from)
        state.maps[layer] = after
        prunes[layer] = prune + 1
        state.n_prunes += 1
        g = index.to(keys.device).unsqueeze(0).unsqueeze(-1)
        g = g.expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, g).contiguous(), values.gather(2, g).contiguous()

    scorer.compress = compress
    return press


def _check(
    state: ProtectState,
    fam: Family,
    before: Any,
    after: Any,
    layer: int,
    recent_from: int | None,
) -> None:
    """Raise unless every head kept the prompt (and buffer), without duplicates."""
    prompt_len = int(state.prompt_len or 0)
    for head in range(int(after.shape[0])):
        kept = set(after[head].tolist())
        had = before[head].tolist()
        if len(kept) != int(after.shape[1]):
            raise RuntimeError(f"layer {layer} head {head}: duplicated position")
        must = {p for p in had if p < prompt_len}
        if recent_from is not None:
            must |= {p for p in had if p >= recent_from}
        if not must <= kept:
            raise RuntimeError(
                f"{fam.name} layer {layer} head {head}: a protected position "
                f"was evicted (prune {state.n_prunes})"
            )

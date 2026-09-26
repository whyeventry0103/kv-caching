"""R.17.2.1 capture-v2: the additions are exact, and v1's fields are untouched.

The pure parts (basis acceptance, the per-head and nested-k accumulators) run
everywhere. The replay parts need torch/transformers/kvpress and run on the GPU
host's CPU with the pinned stack, on the same tiny random-init Qwen2 as
``test_capture.py``.
"""

from __future__ import annotations

import math

import pytest

from kvcache import capture_v2 as c2


def _bases(rng, n_layers=2, d=12, k=4):
    import numpy as np

    q = np.stack(
        [np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(n_layers)]
    )
    frozen = {
        "mean": rng.standard_normal((n_layers, d)),
        "scale": rng.uniform(0.5, 2.0, (n_layers, d)),
        "basis": q[:, :, :k],
    }
    return frozen, q


def test_basis_acceptance_passes_an_exact_refit_and_a_rotation_within_the_subspace():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    frozen, q = _bases(rng)
    exact = {**frozen, "basis": q[:, :, :8]}
    assert c2.basis_acceptance(frozen, exact, k=4)["passed"]
    # leading 4 directions rotated among themselves (sign flips, mixing): same span
    rot = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    mixed = q[:, :, :8].copy()
    mixed[:, :, :4] = np.einsum("ldk,kj->ldj", q[:, :, :4], rot)
    o = c2.basis_acceptance(frozen, {**frozen, "basis": mixed}, k=4)
    assert o["passed"] and o["max_principal_angle_rad"] < 1e-6


def test_basis_acceptance_refuses_a_moved_scale_or_another_subspace():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    frozen, q = _bases(rng)
    moved = {**frozen, "scale": frozen["scale"] * (1 + 1e-4), "basis": q[:, :, :8]}
    assert not c2.basis_acceptance(frozen, moved, k=4)["passed"]
    other = {**frozen, "basis": q[:, :, 4:12]}
    o = c2.basis_acceptance(frozen, other, k=4)
    assert not o["passed"] and o["max_principal_angle_rad"] > 1.0


def test_summarise_heads_sums_per_layer_and_head():
    rows = [
        {
            "layer": 0,
            "M_head": [1.0, 2.0],
            "M_event_head": [0.1, 0.2],
            "Delta_head": [0.5, 0.0],
        },
        {
            "layer": 0,
            "M_head": [3.0, 4.0],
            "M_event_head": [0.3, 0.4],
            "Delta_head": [0.5, 1.0],
        },
        {"layer": 1, "n_eligible": [3]},  # an event row with no measurement
    ]
    o = c2.summarise_heads(rows, n_layers=2)
    assert o["sum_by_layer_head"]["M_head"][0] == [4.0, 6.0]
    assert o["sum_by_layer_head"]["Delta_head"][0] == [1.0, 1.0]
    assert o["n_measured_by_layer"] == [2, 0]


def test_nested_accumulator_applies_the_r6_omega_rule_per_k():
    acc = c2.NestedAccumulator(1, [2, 4])
    acc.add([[1.0, 2.0]], [[3.0, 2.0]])  # finite: 3.0 and 1.0
    acc.add([[0.0, 0.0]], [[0.0, 1.0]])  # undefined (0/0) and unbounded
    s = acc.summary()
    assert s["sum_omega"] == [[3.0, 1.0]] and s["n_finite"] == [[1, 1]]
    assert s["n_undefined"] == [[1, 0]] and s["n_unbounded"] == [[0, 1]]
    assert s["sum_par"] == [[1.0, 2.0]] and s["sum_perp"] == [[3.0, 3.0]]


# ------------------------------------------------------------- replay (torch)
def _tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("kvpress")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = transformers.Qwen2ForCausalLM._from_config(
        cfg, attn_implementation="eager"
    ).eval()
    prompt = torch.randint(3, 97, (12,)).tolist()
    trace = torch.randint(3, 97, (60,)).tolist()
    return torch, model, prompt, trace


def _press(name="random", budget=16):
    from kvcache.presses import build_press

    return build_press(
        name, budget=budget, compression_interval=4, seed=7, instrumented=True
    )


def _basis(torch, model, k=4):
    n, d = model.config.num_hidden_layers, model.config.hidden_size
    g = torch.Generator().manual_seed(3)
    q, _ = torch.linalg.qr(torch.randn(n, d, 8, generator=g))
    return {"scale": torch.ones(n, d), "basis": q[:, :, :k]}, q


@pytest.mark.parametrize("policy", ["random", "streamingllm", "h2o"])
def test_v2_keeps_every_v1_field_and_adds_exact_head_and_nested_sums(policy):
    torch, model, prompt, trace = _tiny()
    from kvcache.capture import record_full_branch, replay_condition

    full = record_full_branch(model, prompt, trace, interval=4)
    basis, q8 = _basis(torch, model)
    kw = dict(budget=16, interval=4, tau=0.01, basis=basis, trigger_step=20)
    v1 = replay_condition(model, _press(policy), full, prompt, trace, **kw)
    nested = {"scale": basis["scale"], "basis": q8}
    v2 = c2.replay_condition_v2(
        model,
        _press(policy),
        full,
        prompt,
        trace,
        nested_basis=nested,
        nested_ks=(2, 4, 8),
        **kw,
    )
    # every v1 field identical (v1 code runs verbatim inside v2)
    for key, val in v1.items():
        if key == "latent_sampled":
            continue
        assert v2[key] == val, key
    # per-head means over each group reproduce the stored group sums
    heads = v2["sum_by_layer_head"]
    grp = v2["sum_by_layer_group"]
    rep = model.config.num_attention_heads // model.config.num_key_value_heads
    for layer in range(model.config.num_hidden_layers):
        for g in range(model.config.num_key_value_heads):
            for f_h, f_g in (("M_head", "M"), ("Delta_head", "Delta")):
                hs = heads[f_h][layer][g * rep : (g + 1) * rep]
                assert sum(hs) / rep == pytest.approx(grp[f_g][layer][g], abs=1e-9)
    assert v2["n_measured_by_layer"] == [
        sum(x) // model.config.num_key_value_heads
        for x in v2["n_measured_by_layer_group"]
    ]
    # nested k = 4 on a basis whose first 4 columns ARE the v1 basis: v1's sums
    lat, nl = v2["latent"], v2["latent_nested"]
    assert nl["ks"] == [2, 4, 8]
    for layer in range(model.config.num_hidden_layers):
        assert nl["sum_par"][layer][1] == pytest.approx(lat["sum_par"][layer], rel=1e-5)
        assert nl["sum_perp"][layer][1] == pytest.approx(
            lat["sum_perp"][layer], rel=1e-5
        )
        # par grows and perp shrinks with k (nested subspaces)
        assert nl["sum_par"][layer][0] <= nl["sum_par"][layer][2] + 1e-6
        assert nl["sum_perp"][layer][2] <= nl["sum_perp"][layer][0] + 1e-6
        assert all(math.isfinite(x) for x in nl["sum_omega"][layer])


def test_v2_without_a_nested_basis_matches_v1_exactly():
    torch, model, prompt, trace = _tiny()
    from kvcache.capture import record_full_branch, replay_condition

    full = record_full_branch(model, prompt, trace, interval=4)
    basis, _ = _basis(torch, model)
    kw = dict(budget=16, interval=4, tau=0.01, basis=basis, trigger_step=20)
    v1 = replay_condition(model, _press(), full, prompt, trace, **kw)
    v2 = c2.replay_condition_v2(model, _press(), full, prompt, trace, **kw)
    assert "latent_nested" not in v2
    assert all(v2[k] == v for k, v in v1.items() if k != "latent_sampled")

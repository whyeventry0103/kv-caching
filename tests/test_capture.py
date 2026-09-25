"""Meeting #3 capture (Amendments R.4, R.6, R.9).

CPU-only checks pin the pure pieces. The torch checks (skipped without torch /
transformers / kvpress; run on the remote CPU with the pinned stack) pin the
tensor implementations against the pure-Python references and run the whole
capture on a tiny random-init Qwen2 with a real kvpress press, where the live
invariants (topk gather equality, layer-0 key identity at mapped positions,
bitwise pre-eviction equality) must hold.
"""

from __future__ import annotations

import random

import pytest

from kvcache.capture import (
    LatentAccumulator,
    armed_steps,
    omega_trace_value,
    pooled_fraction,
    sampled_decode_positions,
)


def test_sampled_positions_are_uniform_with_endpoints_and_skip_the_prefill():
    assert sampled_decode_positions(1) == []
    assert sampled_decode_positions(2) == [1]
    assert sampled_decode_positions(6, k=256) == [1, 2, 3, 4, 5]
    pos = sampled_decode_positions(10001, k=256)
    assert len(pos) == 256 and pos[0] == 1 and pos[-1] == 10000
    assert pos == sorted(set(pos))


def test_armed_steps_cover_the_prune_and_the_forward_after():
    assert [s for s in range(0, 140) if armed_steps(s, 64)] == [1, 64, 65, 128, 129]


def test_latent_accumulator_counts_omega_reasons_and_samples():
    acc = LatentAccumulator(2, sampled=[3])
    acc.add(1, [0.0, 0.0], [0.0, 1.0])  # undefined, unbounded
    acc.add(3, [2.0, 1.0], [1.0, 3.0])  # 0.5, 3.0
    assert acc.n_undefined == [1, 0] and acc.n_unbounded == [0, 1]
    assert acc.n_finite == [1, 1]
    assert acc.sum_omega == [0.5, 3.0]
    assert acc.sampled_par == [[2.0, 1.0]]
    v = omega_trace_value({"sum_omega": acc.sum_omega, "n_finite": acc.n_finite})
    assert v == pytest.approx(1.75)  # equal layer weights
    assert omega_trace_value({"sum_omega": [0.0], "n_finite": [0]}) is None


def test_pooled_fraction_is_unavailable_not_zero_without_eligible_positions():
    assert pooled_fraction([1, 2], [4, 4]) == pytest.approx(3 / 8)
    assert pooled_fraction([0], [0]) is None


def test_moment_fit_recovers_a_planted_direction_and_validates():
    np = pytest.importorskip("numpy")
    from kvcache.capture import MomentAccumulator, validate_basis_arrays

    rng = np.random.default_rng(0)
    acc = MomentAccumulator(n_layers=1, dim=6)
    direction = np.array([1.0, 1.0, 0, 0, 0, 0]) / np.sqrt(2)
    for _ in range(4):
        x = rng.normal(size=(1, 200, 6)) * 0.1
        x[0] += np.outer(rng.normal(size=200) * 5.0, direction)
        acc.add(x, weight=1.0 / 200)
    out = acc.solve(k=2)
    # standardisation keeps the planted plane dominant
    top = out["basis"][0][:, 0]
    assert abs(top[0]) > 0.6 and abs(top[1]) > 0.6
    info = validate_basis_arrays(
        out["mean"].astype("float32"),
        out["scale"].astype("float32"),
        out["basis"].astype("float32"),
    )
    assert info["k"] == 2 and info["max_orthonormality_error"] < 1e-6
    bad = out["basis"].copy()
    bad[0][:, 1] = bad[0][:, 0]
    with pytest.raises(ValueError, match="orthonormal"):
        validate_basis_arrays(out["mean"], out["scale"], bad)


def test_moment_fit_refuses_fewer_usable_directions_than_k():
    np = pytest.importorskip("numpy")
    from kvcache.capture import MomentAccumulator

    acc = MomentAccumulator(1, 3)
    x = np.zeros((1, 50, 3))
    x[0, :, 0] = np.arange(50)
    x[0, :, 1] = 2 * np.arange(50) + 1
    x[0, :, 2] = -np.arange(50)
    acc.add(x, 1.0)
    with pytest.raises(ValueError, match="usable directions"):
        acc.solve(k=2)


# ---------------------------------------------------------------- torch parity
def test_collateral_tensor_version_matches_the_reference():
    torch = pytest.importorskip("torch")
    from kvcache.capture import collateral_stats
    from kvcache.metrics.attn_channels import collateral_event

    rng = random.Random(3)
    heads, groups, length, budget = 4, 2, 20, 8
    logits = torch.tensor(
        [[rng.gauss(0, 2) for _ in range(length)] for _ in range(heads)]
    )
    cands = torch.tensor([sorted(rng.sample(range(length), 12)) for _ in range(groups)])
    ev = torch.stack([c[torch.randperm(12)[:4]] for c in cands])
    got = collateral_stats(logits, cands, ev, budget=budget, tau=0.05)
    probs = torch.softmax(logits.double(), -1)
    rep = heads // groups
    for g in range(groups):
        rows = [
            [float(probs[h, int(p)]) for p in cands[g]]
            for h in range(g * rep, (g + 1) * rep)
        ]
        ref = collateral_event(
            rows,
            [int(p) for p in cands[g]],
            [int(p) for p in ev[g]],
            budget=budget,
            tau=0.05,
        )
        assert got["n_eligible"][g] == ref["n_evicted_eligible"]
        assert got["n_wanted"][g] == ref["n_wanted_any_head"]
        assert got["n_collateral"][g] == ref["n_collateral"]


def test_channel_stats_match_the_log_space_reference():
    torch = pytest.importorskip("torch")
    from kvcache.capture import channel_stats
    from kvcache.metrics.attn_channels import restricted_from_logits

    rng = random.Random(5)
    full = torch.tensor([[rng.gauss(0, 3) for _ in range(10)] for _ in range(2)])
    pos = torch.tensor([[7, 2, 9, 4]])  # one group, score order
    comp = torch.tensor([[rng.gauss(0, 3) for _ in range(4)] for _ in range(2)])
    ev = torch.tensor([[0, 5]])
    st = channel_stats(full, comp, pos, ev)
    ms, ds, mes = [], [], []
    for h in range(2):
        m, a_s = restricted_from_logits(full[h].tolist(), [7, 2, 9, 4])
        a_c = torch.softmax(comp[h].double(), -1).tolist()
        ms.append(m)
        ds.append(0.5 * sum(abs(x - y) for x, y in zip(a_c, a_s, strict=True)))
        p = torch.softmax(full[h].double(), -1)
        mes.append(float(p[0] + p[5]))
    assert float(st["M"][0]) == pytest.approx(sum(ms) / 2, abs=1e-5)
    assert float(st["Delta"][0]) == pytest.approx(sum(ds) / 2, abs=1e-5)
    assert float(st["M_event"][0]) == pytest.approx(sum(mes) / 2, abs=1e-5)
    # same logits restricted = the algebra identity -> Delta 0
    same = channel_stats(full, full[:, [7, 2, 9, 4]], pos, ev)
    assert float(same["Delta"].max()) < 1e-6


def test_position_tracker_follows_gathers_and_appends():
    torch = pytest.importorskip("torch")
    from kvcache.capture import PositionTracker

    tr = PositionTracker(n_kv_heads=2)
    idx = torch.tensor([[4, 0, 2], [1, 3, 5]])
    cand, kept = tr.prune(0, 6, idx)
    assert cand.tolist() == [list(range(6))] * 2
    assert kept.tolist() == [[4, 0, 2], [1, 3, 5]]
    # two new tokens appended (positions 6, 7), then prune again
    cur = tr.current(0, 5)
    assert cur.tolist() == [[4, 0, 2, 6, 7], [1, 3, 5, 6, 7]]
    cand, kept = tr.prune(0, 5, torch.tensor([[3, 0], [4, 2]]))
    assert kept.tolist() == [[6, 4], [7, 5]]
    assert tr.current(0, 3).tolist() == [[6, 4, 8], [7, 5, 8]]


# ---------------------------------------------------------------- tiny model
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


def _basis(torch, model):
    n, d = model.config.num_hidden_layers, model.config.hidden_size
    q, _ = torch.linalg.qr(torch.randn(n, d, 4))
    return {"scale": torch.ones(n, d), "basis": q}


@pytest.mark.parametrize("policy", ["random", "streamingllm", "h2o", "rkv"])
def test_capture_end_to_end_holds_its_live_invariants(policy, tmp_path, monkeypatch):
    torch, model, prompt, trace = _tiny()
    if policy == "rkv":
        # Test fixture only: the production tier gate reads the real smoke
        # marker under KVCACHE_BASE_RUNS_ROOT; this exercises the capture path.
        from kvcache.presses import RKV_SMOKE_MARKER_ENV

        marker = tmp_path / "rkv-gen-smoke-PASS.json"
        marker.write_text("{}")
        monkeypatch.setenv(RKV_SMOKE_MARKER_ENV, str(marker))
    from kvcache.capture import record_full_branch, replay_condition

    full = record_full_branch(model, prompt, trace, interval=4)
    out = replay_condition(
        model,
        _press(policy),
        full,
        prompt,
        trace,
        budget=16,
        interval=4,
        tau=0.01,
        basis=_basis(torch, model),
        trigger_step=20,
    )
    assert out["n_events"] > 3
    assert out["n_measured"] > 0
    assert out["checks"]["n_nonfinite"] == 0
    # layer 0 sees the same q and the same retained keys: the algebra identity
    assert out["checks"]["layer0_delta_max"] < 1e-5
    assert 0.0 <= out["M_mean"] <= 1.0
    assert out["collateral_pooled"] is not None
    lat = out["latent"]
    # pre-eviction steps have zero displacement -> undefined, never finite
    assert sum(lat["n_undefined"]) > 0 and sum(lat["n_finite"]) > 0
    assert out["slices"]["trigger"]["step"] >= 20


def test_full_full_control_is_exactly_zero():
    torch, model, prompt, trace = _tiny()
    from kvcache.capture import record_full_branch, replay_condition

    full = record_full_branch(model, prompt, trace, interval=4)
    out = replay_condition(
        model, None, full, prompt, trace, budget=None, interval=4, tau=0.01
    )
    assert out["checks"]["fullfull_max"] == 0.0
    assert out["n_events"] == 0


def test_a_wrong_position_map_is_caught_by_the_layer0_key_check(monkeypatch):
    torch, model, prompt, trace = _tiny()
    from kvcache import capture
    from kvcache.capture import record_full_branch, replay_condition

    real = capture.PositionTracker.current

    def shifted(self, layer, cache_len, device=None):
        cur = real(self, layer, cache_len, device)
        return cur.flip(-1) if layer in self.maps else cur

    monkeypatch.setattr(capture.PositionTracker, "current", shifted)
    full = record_full_branch(model, prompt, trace, interval=4)
    with pytest.raises(RuntimeError, match="position map is wrong"):
        replay_condition(
            model, _press(), full, prompt, trace, budget=16, interval=4, tau=0.01
        )


# ------------------------------------------------ collateral at real dimensions
def _logits_from_probs(torch, probs):
    return torch.log(torch.tensor(probs, dtype=torch.float64).clamp_min(1e-300))


def test_collateral_positive_control_at_seven_heads_b256():
    """Review of f68d3f1: at B=256 collateral is NOT structurally zero.

    One KV group of 7 query heads over 320 candidates (a real B=256 event:
    budget + interval). Head 0 gives 0.02 to evicted position 319 and 0.98 to
    position 0; heads 1-6 spread their mass over positions 0..255. Position 319
    is wanted (0.02 > tau) but its group mean ranks below all 256 others.
    Production tensor path and the pure-Python reference must both count it.
    """
    torch = pytest.importorskip("torch")
    from kvcache.capture import collateral_stats
    from kvcache.metrics.attn_channels import collateral_event

    n = 320
    h0 = [0.0] * n
    h0[0], h0[319] = 0.98, 0.02
    other = [1 / 256 if j < 256 else 0.0 for j in range(n)]
    probs = [h0] + [other] * 6
    cands = torch.arange(n).unsqueeze(0)
    ev = torch.tensor([[319]])
    got = collateral_stats(
        _logits_from_probs(torch, probs), cands, ev, budget=256, tau=0.01
    )
    assert got["n_collateral"] == [1] and got["n_wanted"] == [1]
    ref = collateral_event(probs, list(range(n)), [319], budget=256, tau=0.01)
    assert ref["n_collateral"] == 1 and ref["collateral"] == 1.0


def test_collateral_is_structurally_zero_at_b2048_with_seven_heads():
    """A wanted position has group mean > tau/7 = 1/700; group means sum to 1,
    so fewer than 700 positions can exceed it and all lie inside the top 2048.
    Checked on adversarial random attention at the real B=2048 event size."""
    torch = pytest.importorskip("torch")
    from kvcache.capture import collateral_stats

    g = torch.Generator().manual_seed(0)
    n = 2112
    for sharp in (0.5, 2.0, 8.0):
        logits = torch.randn(7, n, generator=g, dtype=torch.float64) * sharp
        ev = torch.randperm(n, generator=g)[:64].unsqueeze(0)
        got = collateral_stats(
            logits, torch.arange(n).unsqueeze(0), ev, budget=2048, tau=0.01
        )
        assert got["n_collateral"] == [0]

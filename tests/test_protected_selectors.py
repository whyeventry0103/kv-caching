"""Protected-selector families (WP4): selection rule, fresh-uniform draws, and
the production press on a synthetic cache (torch/kvpress parts skip on the
CPU-only gate and run where the stack is installed)."""

from __future__ import annotations

import pytest

from kvcache.presses import protected_selectors as ps


def test_select_keep_protects_prompt_and_ranks_the_rest_stably():
    scores = [0.0, 0.0, 0.9, 0.1, 0.9, 0.5]
    positions = [0, 1, 2, 3, 4, 5]
    kept = ps.select_keep(scores, positions, n_keep=4, prompt_len=2)
    # prompt {0,1}; then the tie 0.9 at slots 2 and 4: both fit
    assert kept == [0, 1, 2, 4]
    kept = ps.select_keep(scores, positions, n_keep=3, prompt_len=2)
    assert kept == [0, 1, 2]  # tie broken by the lower slot index


def test_select_keep_returns_position_order_and_honours_recency():
    scores = [5.0, 0.0, 9.0, 1.0, 0.0]
    positions = [0, 40, 10, 30, 20]  # slots out of position order
    kept = ps.select_keep(scores, positions, n_keep=3, prompt_len=1, recent_from=40)
    # protected: pos 0 (slot 0) and pos 40 (slot 1); best other: slot 2 (pos 10)
    assert kept == [0, 2, 1]
    assert [positions[i] for i in kept] == [0, 10, 40]


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"n_keep": 1, "prompt_len": 2}, "exceed the target"),
        ({"n_keep": 0, "prompt_len": 0}, "outside"),
    ],
)
def test_select_keep_refuses_impossible_targets(kw, match):
    with pytest.raises(ValueError, match=match):
        ps.select_keep([0.1, 0.2, 0.3], [0, 1, 2], **kw)


def test_select_keep_refuses_duplicate_positions():
    with pytest.raises(ValueError, match="duplicated"):
        ps.select_keep([0.1, 0.2], [3, 3], n_keep=1, prompt_len=0)


def test_draw_seed_is_stable_and_keyed_on_every_field():
    base = ps.draw_seed("uniform_fresh_pp_v1", 123, 4, 1, 7)
    assert base == ps.draw_seed("uniform_fresh_pp_v1", 123, 4, 1, 7)
    variants = {
        ps.draw_seed("uniform_fresh_pp_r64_v1", 123, 4, 1, 7),
        ps.draw_seed("uniform_fresh_pp_v1", 124, 4, 1, 7),
        ps.draw_seed("uniform_fresh_pp_v1", 123, 5, 1, 7),
        ps.draw_seed("uniform_fresh_pp_v1", 123, 4, 2, 7),
        ps.draw_seed("uniform_fresh_pp_v1", 123, 4, 1, 8),
    }
    assert base not in variants and len(variants) == 5
    assert 0 <= base < 2**63


def test_fresh_draws_differ_across_prunes_and_leave_global_rng_alone():
    import random

    random.seed(99)
    expected_next = random.random()
    random.seed(99)
    a = ps.fresh_uniform_scores(50, ps.draw_seed("uniform_fresh_pp_v1", 1, 0, 0, 0))
    b = ps.fresh_uniform_scores(50, ps.draw_seed("uniform_fresh_pp_v1", 1, 0, 0, 1))
    assert a != b  # a new draw at every prune, not a fixed slot pattern
    assert random.random() == expected_next  # the global stream was not consumed
    assert a == ps.fresh_uniform_scores(
        50, ps.draw_seed("uniform_fresh_pp_v1", 1, 0, 0, 0)
    )  # stateless: a resumed sequence redraws the same values


def test_fresh_uniform_inclusion_is_approximately_uniform():
    # 10 candidates after a 2-token prompt, keep 5 -> 3 random of 8, p = 3/8.
    # Broad prespecified bound: every inclusion frequency within 0.375 +- 0.05.
    n_trials, counts = 4000, [0] * 10
    for t in range(n_trials):
        sc = ps.fresh_uniform_scores(
            10, ps.draw_seed("uniform_fresh_pp_v1", 7, 0, 0, t)
        )
        for i in ps.select_keep(sc, list(range(10)), n_keep=5, prompt_len=2):
            counts[i] += 1
    assert counts[0] == counts[1] == n_trials
    for c in counts[2:]:
        assert abs(c / n_trials - 3 / 8) < 0.05


def test_family_registry_and_refusals():
    assert ps.family("H2O_STYLE_PP_V1").base == "h2o"
    assert ps.family("uniform_fresh_pp_r64_v1").recent == 64
    with pytest.raises(KeyError, match="not a protected-selector family"):
        ps.family("random_pp")


# ------------------------------------------------------------ production press
def _fake_module(layer: int, head_dim: int):
    class M:
        pass

    m = M()
    m.layer_idx = layer
    m.head_dim = head_dim
    return m


def test_press_keeps_prompt_draws_fresh_and_is_resume_stable():
    torch = pytest.importorskip("torch")
    kvpress = pytest.importorskip("kvpress")

    def run():
        press = ps.build_protected_selector_press(
            "uniform_fresh_pp_v1",
            budget=12,
            seed=5,
            kvpress_module=kvpress,
            forced_prompt_len=3,
            min_trace_slots=4,
        )
        press.base_press.compression_ratio = 0.5
        k = torch.randn(1, 2, 24, 8)
        mod = _fake_module(0, 8)
        out = []
        for _ in range(2):
            k2, _v = press.base_press.compress(mod, None, k, k.clone(), None, {})
            out.append(press.protect.maps[0].clone())
            k = torch.cat([k2, torch.randn(1, 2, 12, 8)], dim=2)
        return out

    first, second = run(), run()
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)  # deterministic / resume-equivalent
    for m in first:
        for h in range(2):
            assert set(range(3)) <= set(m[h].tolist())  # prompt kept in every head
    assert not torch.equal(first[0][0], first[0][1])  # heads draw independently


def test_h2o_style_identity_control_matches_stable_topk():
    torch = pytest.importorskip("torch")
    kvpress = pytest.importorskip("kvpress")
    press = ps.build_protected_selector_press(
        "h2o_style_pp_v1", budget=8, kvpress_module=kvpress, forced_prompt_len=0
    )
    scores = torch.tensor([[[0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.6, 0.5]]])
    press.base_press.score = lambda *a, **k: scores
    press.base_press.compression_ratio = 0.5
    k = torch.randn(1, 1, 8, 4)
    press.base_press.compress(_fake_module(0, 4), None, k, k.clone(), None, {})
    kept = sorted(press.protect.maps[0][0].tolist())
    assert kept == sorted(scores[0, 0].topk(4).indices.tolist())  # {1, 3, 5, 6}


# ------------------------------------------------------------ promotion gate
def _cfg(**over):
    cfg = {
        "protected_selector": True,
        "presses": ["h2o_style_pp_v1", "uniform_fresh_pp_v1"],
        "mode": "readout",
    }
    cfg.update(over)
    return cfg


def test_gate_allows_dry_runs_of_held_configs():
    ps.require_protected_config(_cfg(held=True), dry_run=True)


def test_gate_refuses_real_runs_while_held_or_without_smoke(tmp_path):
    with pytest.raises(SystemExit, match="HELD"):
        ps.require_protected_config(_cfg(held=True), dry_run=False, runs_root=tmp_path)
    with pytest.raises(SystemExit, match="HELD"):  # absent flag means held
        ps.require_protected_config(_cfg(), dry_run=False, runs_root=tmp_path)
    with pytest.raises(SystemExit, match="smoke"):
        ps.require_protected_config(_cfg(held=False), dry_run=False, runs_root=tmp_path)
    (tmp_path / ps.SMOKE_MARKER).write_text("{}")
    ps.require_protected_config(_cfg(held=False), dry_run=False, runs_root=tmp_path)


@pytest.mark.parametrize(
    "over, match",
    [
        ({"protected_selector": False}, "protected_selector"),
        ({"presses": ["random_pp"]}, "not a protected-selector family"),
        ({"restoration": {}}, "not registered"),
    ],
)
def test_gate_refuses_non_protected_configs(over, match):
    with pytest.raises(SystemExit, match=match):
        ps.require_protected_config(_cfg(**over), dry_run=True)

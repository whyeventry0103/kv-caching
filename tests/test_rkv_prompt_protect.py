"""Prompt-protected R-KV (Meeting 5, M5-A): base parity with protection off,
prompt + recent-8 retention, K/V alignment across repeated prunes, layers and
heads, ties, and determinism. Needs torch and kvpress (runs on the GPU host's
CPU; skipped on the CPU-only gate). The scorer is replaced by a deterministic
function of the keys IN BOTH ARMS, so these tests isolate the wrapper; the
unchanged R-KV scorer is covered by its own parity harness and the real-model
check in scripts/rkv_pp_cpu_parity.py."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
kvpress = pytest.importorskip("kvpress")

from kvcache.presses import build_press  # noqa: E402
from kvcache.presses import rkv_prompt_protect as rpp  # noqa: E402

W = 8


@pytest.fixture(autouse=True)
def _rkv_promoted(monkeypatch, tmp_path):
    """R-KV's eligibility gate needs its smoke marker; point it at a temporary
    PASS file exactly as tests/test_presses.py does (the gate itself is tested
    there)."""
    from kvcache.presses import RKV_SMOKE_MARKER_ENV

    marker = tmp_path / "rkv-gen-smoke-PASS.json"
    marker.write_text('{"passed": true}')
    monkeypatch.setenv(RKV_SMOKE_MARKER_ENV, str(marker))


class _Mod:
    def __init__(self, layer: int, head_dim: int) -> None:
        self.layer_idx = layer
        self.head_dim = head_dim


def _fake_score(module, hidden_states, keys, values, attentions, kwargs):
    # R-KV-shaped: a key-dependent score with the recent W slots padded to max,
    # exactly as RKVScorerPress pads its window.
    s = keys.float().sum(-1)  # [1, H, L]
    s[..., -W:] = s.max()
    return s.to(keys.dtype)


def _presses(prompt_len, budget=32, ties=False):
    base = build_press("rkv", budget=budget, kvpress_module=kvpress)
    prot = rpp.build_rkv_prompt_protected_press(
        budget=budget, kvpress_module=kvpress, forced_prompt_len=prompt_len
    )
    score = _fake_score
    if ties:

        def score(module, hidden_states, keys, values, attentions, kwargs):
            s = torch.zeros(keys.shape[:-1], dtype=keys.dtype)
            s[..., -W:] = 1.0
            return s

    for p in (base, prot):
        p.base_press.score = score
        p.base_press.compression_ratio = 0.5
    return base, prot


def _stream(press, steps=3, heads=2, layers=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    outs = []
    caches = {
        layer: torch.randn(1, heads, 64, 4, generator=g) for layer in range(layers)
    }
    for _ in range(steps):
        for layer in range(layers):
            k = caches[layer]
            v = k * 2 + 1  # V is a known function of K: alignment is checkable
            k2, v2 = press.base_press.compress(_Mod(layer, 4), None, k, v, None, {})
            assert torch.equal(v2, k2 * 2 + 1)  # K and V gathered with one index
            outs.append(k2.clone())
            caches[layer] = torch.cat(
                [k2, torch.randn(1, heads, 32, 4, generator=g)], 2
            )
    return outs


def test_protection_off_is_the_base_path_exactly():
    base, prot = _presses(prompt_len=0)
    for a, b in zip(_stream(base), _stream(prot), strict=True):
        assert torch.equal(a, b)  # same scores, same topk, same gather order


def test_protection_off_with_ties_matches_base_too():
    base, prot = _presses(prompt_len=0, ties=True)
    for a, b in zip(_stream(base), _stream(prot), strict=True):
        assert torch.equal(a, b)


def test_prompt_and_recent_window_survive_every_prune():
    _, prot = _presses(prompt_len=5)
    _stream(prot, steps=4)  # the live check raises on any violation
    st = prot.protect
    assert st.n_prunes == 8  # 4 steps x 2 layers
    for m in st.maps.values():
        for h in range(m.shape[0]):
            kept = set(m[h].tolist())
            assert set(range(5)) <= kept
            assert len(kept) == m.shape[1]


def test_deterministic_and_resume_equivalent():
    _, a = _presses(prompt_len=5)
    _, b = _presses(prompt_len=5)
    for x, y in zip(_stream(a, seed=3), _stream(b, seed=3), strict=True):
        assert torch.equal(x, y)


def test_refuses_a_target_that_cannot_hold_prompt_plus_window():
    _, prot = _presses(prompt_len=30, budget=32)
    prot.protect.target = 32
    with pytest.raises(RuntimeError, match="cannot hold the prompt"):
        _stream(prot, steps=1)


def test_budget_relaxation_covers_the_window():
    prot = rpp.build_rkv_prompt_protected_press(
        budget=256, kvpress_module=kvpress, forced_prompt_len=300
    )
    assert prot.protect.target == 300 + 64
    assert prot.protect.budget_relaxed is True


def test_wrong_family_name_is_refused():
    with pytest.raises(KeyError):
        rpp.build_rkv_prompt_protected_press("rkv", budget=256, kvpress_module=kvpress)

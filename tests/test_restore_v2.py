"""R.17.2.3 restoration redesign, on a tiny random-init Qwen2 (skipped without
torch; runs on the GPU host's CPU with the pinned stack).

Pins: the cohort arms are EXACTLY exact-lag's (same cohort, snapshots and
reinsertion RNG); full self-restoration returns every evicted entry (the gate
checks every absolute position is present once) so M is ~0 afterwards, while the
none arm still misses mass; and the gate fails when an entry is withheld.
"""

from __future__ import annotations

import pytest

from kvcache.restore_v2 import measure_steps


def test_measure_steps_start_after_the_trigger_prune():
    assert measure_steps(100, 200, 16) == list(range(101, 301, 16))
    assert measure_steps(100, 200, 16)[0] == 101 and len(measure_steps(100, 200)) == 13


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
        max_position_embeddings=512,
    )
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    prompt = torch.randint(3, 97, (12,)).tolist()
    trace = torch.randint(3, 97, (90,)).tolist()
    return model, prompt, trace


def _press(policy):
    from kvcache.presses import build_press

    return build_press(
        policy, budget=16, compression_interval=4, seed=3, instrumented=True
    )


KW = dict(n_reinsert=4, rng_key="t", trigger_step=20, window=10, lags=(1, 2, 4))


@pytest.mark.parametrize("policy", ["random", "streamingllm"])
def test_cohort_arms_equal_exact_lag_and_full_restoration_removes_missing_mass(
    policy,
):
    model, prompt, trace = _tiny()
    from kvcache.exact_lag import run_exact_lag
    from kvcache.restore_v2 import run_restore_v2

    out = run_restore_v2(model, _press(policy), prompt, trace, every=3, **KW)
    ref = run_exact_lag(model, _press(policy), prompt, trace, **KW)
    assert out["status"] == "READ" and out["cohort_step"] == ref["cohort_step"]
    assert out["lags"] == ref["lags"]
    fs = out["full_restore"]
    assert fs["eligible"] and fs["gate"]["passed"]
    e = out["cohort_step"]
    # every position once at restoration; then the window's tokens are appended
    assert fs["cache_len_at_restore"] == len(prompt) + e
    assert fs["cache_len_after"] == len(prompt) + e + 10
    steps = [str(s) for s in measure_steps(e, 10, 3)]
    assert sorted(out["channels"]["full_restore"], key=int) == steps
    assert sorted(out["channels"]["none"], key=int) == steps
    for s in steps:
        assert out["channels"]["full_restore"][s]["M"] < 1e-5
        assert out["channels"]["full_restore"][s]["n_layers"] == 2
    assert max(out["channels"]["none"][s]["M"] for s in steps) > 1e-3
    # non-vacuous: restoring changes the prediction relative to none
    assert fs["window_mean_tv"] != fs["none_window_mean_tv"]


def test_full_restore_gate_fails_when_an_entry_is_withheld():
    torch = pytest.importorskip("torch")
    from kvcache.capture import PositionTracker
    from kvcache.restore_v2 import full_restore

    class _Layer:
        def __init__(self, k, v):
            self.keys, self.values = k, v

    class _Cache:
        def __init__(self, layers):
            self.layers = layers

    def build(withhold):
        n_kv, d = 2, 3
        live_pos = torch.tensor([[0, 3, 5], [1, 2, 5]])
        ev_pos = torch.tensor([[1, 2, 4], [0, 3, 4]])
        if withhold:
            ev_pos = ev_pos[:, :2]
        k = torch.randn(1, n_kv, 3, d)
        cache = _Cache([_Layer(k.clone(), k.clone())])
        tr = PositionTracker(n_kv)
        tr.maps[0], tr.next_abs[0] = live_pos, 6
        ek = torch.randn(n_kv, ev_pos.shape[1], d)
        return cache, {0: [(5, ev_pos, ek, ek.clone())]}, tr

    cache, stash, tr = build(withhold=False)
    assert full_restore(cache, stash, tr, 6)["passed"]
    assert cache.layers[0].keys.shape[2] == 6
    cache, stash, tr = build(withhold=True)
    assert not full_restore(cache, stash, tr, 6)["passed"]

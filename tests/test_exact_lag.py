"""Amendment R.5 exact-lag restoration.

The reader arithmetic runs on hand-built rows (CPU). The protocol itself runs
on a tiny random-init Qwen2 with a real kvpress press (skipped without torch):
the stash must be the true evicted complement (layer-0 keys equal the full
keys at the same positions; the driver raises otherwise), reinsertion must
land exactly at e + lag, arms must be deterministic from their snapshot, and
the retained-set gate must pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_exact_lag import read_rows  # noqa: E402


def _row(pid, seed, none, matched, rnd, relaxed, lags=(1, 4)):
    return {
        "pid": pid,
        "seed": seed,
        "press": "random",
        "status": "READ",
        "lags": {
            str(lag): {
                "eligible": True,
                "achieved_lag": lag,
                "none": none,
                "matched": matched - 0.01 * i,
                "random": rnd,
                "relaxed": relaxed,
            }
            for i, lag in enumerate(lags)
        },
    }


def test_reader_computes_effects_per_lag_and_the_sweep_verdict():
    rows = [
        _row(p, s, 0.30, 0.20 + 0.001 * k, 0.29, 0.10)
        for k, (p, s) in enumerate((p, s) for p in "abcde" for s in (1, 2))
    ]
    rows.append({"pid": "z", "seed": 1, "press": "random", "status": "INELIGIBLE"})
    out = read_rows(rows, n_resamples=200, seed=0)["random"]
    assert out["n_ineligible"] == 1
    m1 = out["per_lag"]["1"]["arms"]["matched"]
    assert m1["effect_vs_no_intervention"] == pytest.approx(0.30 - 0.2045)
    assert m1["excludes_zero"]
    # matched effect grows with lag here, so not DECAY
    assert out["sweep"]["verdict"] == "NON-MONOTONE"


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


def _run(model, prompt, trace, policy="random"):
    from kvcache.exact_lag import run_exact_lag
    from kvcache.presses import build_press

    press = build_press(
        policy, budget=16, compression_interval=4, seed=3, instrumented=True
    )
    return run_exact_lag(
        model,
        press,
        prompt,
        trace,
        n_reinsert=4,
        rng_key="t",
        trigger_step=10,
        window=8,
        lags=(1, 4, 16),
    )


@pytest.mark.parametrize("policy", ["random", "streamingllm"])
def test_protocol_runs_every_arm_at_every_lag_deterministically(policy):
    model, prompt, trace = _tiny()
    out = _run(model, prompt, trace, policy)
    assert out["status"] == "READ"
    e = out["cohort_step"]
    assert e >= 10 and e % 4 == 0
    assert out["stash_size"] >= 4
    for lag in (1, 4, 16):
        res = out["lags"][str(lag)]
        assert res["eligible"] and res["applied_step"] == e + lag
        # non-vacuous: reinsertion changes what the model predicts
        assert res["relaxed"] != res["none"]
    again = _run(model, prompt, trace, policy)
    assert again["lags"] == out["lags"]  # arms replay from an exact snapshot


def test_a_trace_too_short_for_the_window_is_ineligible_at_that_lag():
    model, prompt, trace = _tiny()
    out = _run(model, prompt, trace[:30])  # e=12: lag 16 needs forward 35 > 29
    assert out["status"] == "READ"
    assert out["lags"]["16"] == {"eligible": False}


def test_the_old_tail_stash_bug_is_refused(monkeypatch):
    """Mutation: stash keys[:, :, -dropped:] (the pre-R.5 bug) -> must raise."""
    model, prompt, trace = _tiny()
    from kvcache import capture

    real_end = capture.EventTap.end

    def tail_end(self, module, keys_before, keys_after, step, values_before=None):
        real_end(self, module, keys_before, keys_after, step, values_before)
        entries = self.stash.get(int(module.layer_idx))
        if entries and entries[-1][0] == step:
            s, pos, k, v = entries[-1]
            dropped = k.shape[1]
            entries[-1] = (s, pos, keys_before[0][:, -dropped:], v)

    monkeypatch.setattr(capture.EventTap, "end", tail_end)
    with pytest.raises(RuntimeError, match=r"stashed keys differ .*\(layer 0\)"):
        _run(model, prompt, trace)


def test_exact_lag_row_checks_reject_the_reviewed_counterexample():
    """Review of cfa30cd: status READ, lags {}, provenance {anything: same},
    stash_identity_checks {passed: false} was FINAL."""
    import report_exact_lag as rel

    kw = dict(lags=[1, 4, 16, 64, 256], window=200, trigger=130)
    probe = {
        "status": "READ",
        "lags": {},
        "stash_identity_checks": {"passed": False},
        "cohort_step": 192,
        "n_trace_tokens": 1000,
    }
    probs = rel.row_problems(probe, **kw)
    assert any("stash identity" in p for p in probs)
    assert any("lag set" in p for p in probs)
    assert (
        rel.row_problems(
            {"status": "INELIGIBLE: no eviction at or after the trigger"}, **kw
        )
        == []
    )
    assert rel.row_problems({"status": "INELIGIBLE: because"}, **kw)
    gate = {"passed": True, "checked": "keys AND values"}
    ok_lag = {
        "eligible": True,
        "requested_lag": 1,
        "achieved_lag": 1,
        "applied_step": 193,
        "none": 0.3,
        "matched": 0.2,
        "random": 0.3,
        "relaxed": 0.1,
        "gates": {a: dict(gate) for a in ("matched", "random", "relaxed")},
    }
    row = {
        "status": "READ",
        "cohort_step": 192,
        "n_trace_tokens": 1000,
        "stash_identity_checks": {"n_entries_checked": 5},
        "lags": {"1": ok_lag},
    }
    assert rel.row_problems(row, lags=[1], window=200, trigger=130) == []
    bad = {**row, "lags": {"1": {**ok_lag, "achieved_lag": 4}}}
    assert rel.row_problems(bad, lags=[1], window=200, trigger=130)
    nan = {**row, "lags": {"1": {**ok_lag, "matched": float("nan")}}}
    assert rel.row_problems(nan, lags=[1], window=200, trigger=130)
    short = {**row, "n_trace_tokens": 300, "lags": {"1": {"eligible": False}}}
    assert rel.row_problems(short, lags=[1], window=200, trigger=130) == []
    fake_inel = {**row, "lags": {"1": {"eligible": False}}}  # window fits
    assert rel.row_problems(fake_inel, lags=[1], window=200, trigger=130)


def test_a_corrupted_deep_layer_stash_value_is_refused(monkeypatch):
    """Mutation (review of 4e38224): values, not only keys, and beyond layer 0."""
    model, prompt, trace = _tiny()
    from kvcache import capture

    real_end = capture.EventTap.end

    def corrupt(self, module, keys_before, keys_after, step, values_before=None):
        real_end(self, module, keys_before, keys_after, step, values_before)
        entries = self.stash.get(int(module.layer_idx))
        if int(module.layer_idx) == 1 and entries and entries[-1][0] == step:
            s, pos, k, v = entries[-1]
            entries[-1] = (s, pos, k, v + 1.0)

    monkeypatch.setattr(capture.EventTap, "end", corrupt)
    with pytest.raises(RuntimeError, match=r"values differ .*layer 1"):
        _run(model, prompt, trace)


def test_corrupted_reinserted_values_fail_the_retained_set_gate(monkeypatch):
    model, prompt, trace = _tiny()
    from kvcache.metrics import restoration

    real = restoration.apply_reinsertion

    def bad(*a, **k):
        kk, vv = real(*a, **k)
        return kk, vv + 1.0

    monkeypatch.setattr(restoration, "apply_reinsertion", bad)
    with pytest.raises(RuntimeError, match="retained-set gate failed"):
        _run(model, prompt, trace)


def test_gate_evidence_and_identity_checks_are_recorded():
    model, prompt, trace = _tiny()
    out = _run(model, prompt, trace)
    assert out["stash_identity_checks"]["n_entries_checked"] > 0
    g = out["lags"]["1"]["gates"]["matched"]
    assert g["passed"] and "VALUES" in g["checked"].upper()

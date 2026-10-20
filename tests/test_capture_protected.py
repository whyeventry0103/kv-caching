"""Protected M / Delta / omega capture (M5A-CAPTURE-PP, prepared 2026-10-20).

A tiny eager Qwen2 runs the validated capture replay with every protected
press (``kvcache.capture_protected``). The protected selection is read from
the press's own position record and verified bitwise; the replay's live
controls (pre-eviction bitwise equality, layer-0 keys at mapped positions,
layer-0 Delta) all still hold; no prompt position is ever evicted; the
identity control (prompt length forced to 0) reproduces the unprotected
capture EXACTLY; a corrupted record is refused; and the unprotected EventTap
would refuse a protected press (which is why this module exists). torch,
transformers and kvpress are required; the CPU gate without them skips."""

from __future__ import annotations

import pytest

from kvcache import capture_protected as cp

# interval >= the R-KV recent window (8): the registered configs use 64
BUDGET, INTERVAL, SLOTS = 24, 8, 8


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


def _basis(torch, model):
    n, d = model.config.num_hidden_layers, model.config.hidden_size
    q, _ = torch.linalg.qr(torch.randn(n, d, 4))
    return {"scale": torch.ones(n, d), "basis": q}


def _rkv_marker(tmp_path, monkeypatch):
    # test fixture only: the production tier gate reads the real smoke marker
    from kvcache.presses import RKV_SMOKE_MARKER_ENV

    marker = tmp_path / "rkv-gen-smoke-PASS.json"
    marker.write_text("{}")
    monkeypatch.setenv(RKV_SMOKE_MARKER_ENV, str(marker))


def _pp(name, **kw):
    return cp.build_protected_capture_press(
        name,
        budget=BUDGET,
        compression_interval=INTERVAL,
        seed=7,
        min_trace_slots=SLOTS,
        **kw,
    )


def _run(model, press, full, prompt, trace, torch, protected=True, basis=None):
    from kvcache.capture import replay_condition

    fn = cp.replay_condition_protected if protected else replay_condition
    return fn(
        model,
        press,
        full,
        prompt,
        trace,
        budget=BUDGET,
        interval=INTERVAL,
        tau=0.01,
        basis=basis if basis is not None else _basis(torch, model),
        trigger_step=20,
    )


@pytest.mark.parametrize("policy", cp.PROTECTED_POLICIES)
def test_protected_capture_holds_every_live_invariant(policy, tmp_path, monkeypatch):
    torch, model, prompt, trace = _tiny()
    _rkv_marker(tmp_path, monkeypatch)
    from kvcache.capture import record_full_branch

    full = record_full_branch(model, prompt, trace, interval=INTERVAL)
    torch.manual_seed(1)
    press = _pp(policy)
    out = _run(model, press, full, prompt, trace, torch)
    assert out["n_events"] > 3 and out["n_measured"] > 0
    assert out["checks"]["n_nonfinite"] == 0
    assert out["checks"]["layer0_delta_max"] < 1e-5
    assert 0.0 <= out["M_mean"] <= 1.0 and out["Delta_mean"] is not None
    assert out["latent"]["omega_trace"] is not None
    assert out["protect"]["prompt_len"] == len(prompt)
    assert out["protect"]["n_prunes"] > 0
    # no prompt position is ever among the evicted, at any event or layer
    events = press.event_tap.events
    assert events and all(int(ev.evicted.min()) >= len(prompt) for ev in events)


def test_identity_control_random_pp_at_zero_reproduces_random_exactly():
    torch, model, prompt, trace = _tiny()
    from kvcache.capture import record_full_branch
    from kvcache.presses import build_press

    full = record_full_branch(model, prompt, trace, interval=INTERVAL)
    base = build_press(
        "random",
        budget=BUDGET,
        compression_interval=INTERVAL,
        seed=7,
        instrumented=True,
    )
    basis = _basis(torch, model)  # ONE basis: omega must be comparable
    ref = _run(model, base, full, prompt, trace, torch, protected=False, basis=basis)
    pp = _run(
        model,
        _pp("random_pp", forced_prompt_len=0),
        full,
        prompt,
        trace,
        torch,
        basis=basis,
    )
    pp.pop("protect")
    assert ref["n_events"] > 3
    for k in ("sum_by_layer_group", "n_measured_by_layer_group", "event_steps"):
        assert pp[k] == ref[k], k
    for k in ("sum_omega", "n_finite", "sum_par", "sum_perp"):
        assert pp["latent"][k] == ref["latent"][k], k
    assert (pp["M_mean"], pp["Delta_mean"]) == (ref["M_mean"], ref["Delta_mean"])


def test_identity_control_rkv_pp_at_zero_reproduces_rkv_exactly(tmp_path, monkeypatch):
    torch, model, prompt, trace = _tiny()
    _rkv_marker(tmp_path, monkeypatch)
    from kvcache.capture import record_full_branch
    from kvcache.presses import build_press

    full = record_full_branch(model, prompt, trace, interval=INTERVAL)
    base = build_press(
        "rkv", budget=BUDGET, compression_interval=INTERVAL, seed=7, instrumented=True
    )
    basis = _basis(torch, model)
    ref = _run(model, base, full, prompt, trace, torch, protected=False, basis=basis)
    pp = _run(
        model,
        _pp("rkv_pp_v1", forced_prompt_len=0),
        full,
        prompt,
        trace,
        torch,
        basis=basis,
    )
    assert ref["n_events"] > 3
    for k in ("sum_by_layer_group", "event_steps"):
        assert pp[k] == ref[k], k
    assert pp["latent"]["sum_omega"] == ref["latent"]["sum_omega"]


def test_a_corrupted_selection_record_is_refused(monkeypatch):
    torch, model, prompt, trace = _tiny()
    from kvcache.capture import record_full_branch

    real = cp.local_indices
    monkeypatch.setattr(cp, "local_indices", lambda c, k: real(c, k).flip(-1))
    full = record_full_branch(model, prompt, trace, interval=INTERVAL)
    with pytest.raises(RuntimeError, match="does not reproduce the retained cache"):
        _run(model, _pp("random_pp"), full, prompt, trace, torch)


def test_the_unprotected_tap_would_refuse_a_protected_press():
    """Why the protected tap exists: EventTap re-derives top-k from the base
    scores, which a protected selection is not."""
    torch, model, prompt, trace = _tiny()
    from kvcache.capture import record_full_branch

    full = record_full_branch(model, prompt, trace, interval=INTERVAL)
    # either refusal: the protected press bypasses the score wrapper, or the
    # re-derived top-k does not reproduce the protected selection
    with pytest.raises(RuntimeError, match="event tap: "):
        _run(model, _pp("random_pp"), full, prompt, trace, torch, protected=False)


def test_the_tap_swap_is_scoped_and_restored_on_error(monkeypatch):
    torch, model, prompt, trace = _tiny()
    from kvcache import capture
    from kvcache.capture import record_full_branch

    original = capture.EventTap
    full = record_full_branch(model, prompt, trace, interval=INTERVAL)
    _run(model, _pp("random_pp"), full, prompt, trace, torch)
    assert capture.EventTap is original
    real = cp.local_indices
    monkeypatch.setattr(cp, "local_indices", lambda c, k: real(c, k).flip(-1))
    with pytest.raises(RuntimeError):
        _run(model, _pp("random_pp"), full, prompt, trace, torch)
    assert capture.EventTap is original


def test_the_protected_tap_refuses_an_unprotected_press():
    pytest.importorskip("torch")
    pytest.importorskip("kvpress")
    from kvcache.capture import PositionTracker
    from kvcache.presses import build_press

    press = build_press(
        "random",
        budget=BUDGET,
        compression_interval=INTERVAL,
        seed=7,
        instrumented=True,
    )
    with pytest.raises(RuntimeError, match="needs a protected press"):
        cp.ProtectedEventTap(press, PositionTracker(2))


def test_runner_config_contract():
    """CPU-only: the runner refuses non-protected presses, a missing cohort and
    other budgets; the predeclared cohort is the 23 strict-fit capture problems."""
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import run_capture_pp as rc
    import yaml

    cfg = yaml.safe_load(
        (root / "configs/experiments/capture_pp_math500_b256_m5.yaml").read_text()
    )
    assert rc.validate_config(cfg) == [] and cfg["held"] is True
    bad = {**cfg, "presses": ["random"], "budgets": [1024]}
    bad.pop("cohort_source")
    assert len(rc.validate_config(bad)) == 3
    capture25 = json.loads((root / "configs/meeting3/q2_subset_pids.json").read_text())[
        "pids"
    ]
    cohort = rc.cohort_pids(cfg, capture25)
    assert len(capture25) == 25 and len(cohort) == 23
    assert "test/algebra/1349.json" not in cohort
    assert "test/precalculus/1199.json" not in cohort


def test_campaign_refuses_without_a_same_identity_smoke_pass(tmp_path):
    """CPU-only: missing, failed or stale smoke markers all refuse."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_capture_pp as rc

    ident = {
        "kind": "capture_pp",
        "implementation_id": "a" * 64,
        "model_revision": "r",
        "basis_npz_sha256": "b",
    }
    assert "no capture-pp-smoke-PASS.json" in rc.smoke_gate_problem(tmp_path, ident)
    marker = tmp_path / rc.SMOKE_PASS
    marker.write_text(json.dumps({"passed": False, "identity": ident}))
    assert "does not say passed" in rc.smoke_gate_problem(tmp_path, ident)
    marker.write_text(json.dumps({"passed": True, "identity": ident}))
    assert rc.smoke_gate_problem(tmp_path, ident) is None
    stale = {**ident, "implementation_id": "c" * 64}
    marker.write_text(json.dumps({"passed": True, "identity": stale}))
    assert "implementation_id differs" in rc.smoke_gate_problem(tmp_path, ident)

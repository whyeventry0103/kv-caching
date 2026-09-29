"""Amendment S.1 prompt-protected presses.

The rule tests run anywhere. The press tests need torch, transformers and
kvpress (skipped without them; they run on the GPU host's CPU with the pinned
stack) and use a tiny random-init Qwen2.

Pins: the prompt is never evicted in any layer or KV head; the press reads the
prompt length from the prefill forward; with the protected length forced to
what the unprotected press already keeps, the two presses are the same press
(identity control); the long-prompt rule raises the target and flags it; and
the live check fails when a prompt position is missing.
"""

from __future__ import annotations

import pytest

from kvcache.presses.prompt_protect import (
    ProtectState,
    effective_budget,
    pp_base_name,
    require_pp_config,
)


def test_names_map_to_registered_bases_only():
    assert pp_base_name("random_pp") == "random"
    assert pp_base_name("StreamingLLM_pp") == "streamingllm"
    for bad in ("random", "h2o_pp", "rkv_pp", "_pp"):
        with pytest.raises(KeyError):
            pp_base_name(bad)


def test_budget_includes_the_prompt_and_is_relaxed_only_for_long_prompts():
    assert effective_budget(256, 15) == (256, False)
    assert effective_budget(256, 192) == (256, False)
    assert effective_budget(256, 193) == (257, True)
    assert effective_budget(256, 353) == (417, True)
    assert effective_budget(1024, 353) == (1024, False)
    with pytest.raises(ValueError):
        effective_budget(0, 10)


def test_the_three_flagged_prompts_of_the_grid_at_256():
    # prompt lengths of the 100-problem grid, from the token-ID sidecars
    longest = [154, 155, 180, 183, 216, 344, 353]
    assert [p for p in longest if effective_budget(256, p)[1]] == [216, 344, 353]
    assert not any(effective_budget(1024, p)[1] for p in longest)


def _cfg(**kw):
    base = {"presses": ["random_pp", "streamingllm_pp"], "prompt_protected": True}
    return {**base, **kw}


def test_config_guard_refuses_anything_not_registered():
    require_pp_config(_cfg())
    require_pp_config(_cfg(mode="readout"), replay=True)
    for bad in (
        _cfg(prompt_protected=False),
        _cfg(presses=["random"]),
        _cfg(presses=["random_pp", "h2o_pp"]),
        _cfg(compression_start_step=512),
        _cfg(matched_ratio=0.04),
        _cfg(restoration={"arms": ["none"]}),
    ):
        with pytest.raises(SystemExit):
            require_pp_config(bad)
    with pytest.raises(SystemExit):
        require_pp_config(_cfg(mode="compression_replay"), replay=True)


# ------------------------------------------------------------- tiny model
PROMPT, TRACE, BUDGET, INTERVAL = 12, 80, 24, 4


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
    prompt = torch.randint(3, 97, (PROMPT,)).tolist()
    trace = torch.randint(3, 97, (TRACE,)).tolist()
    return torch, model, prompt, trace


def _replay(model, press, prompt, trace):
    import torch

    from kvcache.replay import TransformersReplayBackend, paired_teacher_forced_replay

    backend = TransformersReplayBackend(model)
    with torch.inference_mode(), press(model):
        return paired_teacher_forced_replay(
            prompt,
            trace,
            cache_factory=backend.cache_factory,
            forward=backend.forward,
            press=press,
        )


def _pp(name, **kw):
    from kvcache.presses.prompt_protect import build_prompt_protected_press

    kw.setdefault("min_trace_slots", 8)
    return build_prompt_protected_press(
        name,
        budget=BUDGET,
        compression_interval=INTERVAL,
        seed=3,
        instrumented=True,
        **kw,
    )


@pytest.mark.parametrize("name", ["random_pp", "streamingllm_pp"])
def test_the_prompt_survives_every_prune_in_every_layer_and_head(name):
    torch, model, prompt, trace = _tiny()
    press = _pp(name)
    profile = _replay(model, press, prompt, trace)
    st = press.protect
    assert st.prompt_len == PROMPT and st.target == BUDGET and not st.budget_relaxed
    assert st.n_prunes >= 2 and len(st.maps) == model.config.num_hidden_layers
    assert profile.max_after_prune == BUDGET
    assert all(v <= 1e-6 for v in profile.estimate[: profile.onset_step])
    want = torch.arange(PROMPT)
    for positions in st.maps.values():
        assert positions.shape == (2, BUDGET)
        for head in positions:
            assert torch.equal(torch.sort(head[head < PROMPT]).values, want)


def test_streamingllm_pp_keeps_the_prompt_and_the_most_recent_trace():
    torch, model, prompt, trace = _tiny()
    press = _pp("streamingllm_pp")
    _replay(model, press, prompt, trace)
    for positions in press.protect.maps.values():
        head = torch.sort(positions[0]).values
        recent = head[PROMPT:]
        assert torch.equal(recent, torch.arange(int(recent[0]), int(recent[-1]) + 1))
        assert int(recent.numel()) == BUDGET - PROMPT


def test_unprotected_random_does_evict_the_prompt_on_the_same_trace():
    # the control for the test above: without protection the prompt is lost
    torch, model, prompt, trace = _tiny()
    press = _pp("random_pp", forced_prompt_len=0)
    _replay(model, press, prompt, trace)
    kept = [int((m < PROMPT).sum()) for m in press.protect.maps.values()]
    assert all(k < 2 * PROMPT for k in kept)


def test_identity_control_random_with_nothing_protected_is_the_unprotected_press():
    _torch, model, prompt, trace = _tiny()
    from kvcache.presses import build_press

    base = build_press(
        "random",
        budget=BUDGET,
        compression_interval=INTERVAL,
        seed=3,
        instrumented=True,
    )
    a = _replay(model, base, prompt, trace)
    b = _replay(model, _pp("random_pp", forced_prompt_len=0), prompt, trace)
    assert a.estimate == b.estimate
    assert a.compression_events == b.compression_events
    # and protection changes the trajectory, so the control can fail
    c = _replay(model, _pp("random_pp"), prompt, trace)
    assert max(abs(x - y) for x, y in zip(c.estimate, a.estimate, strict=True)) > 1e-3


def test_identity_control_streamingllm_with_four_sinks_is_sinks_plus_recency():
    # The TV comparison against kvpress's own StreamingLLMPress is made by the
    # GPU smoke, not here: that press scores by SLOT, so it is recency only
    # where tied top-k returns slots in position order. Probed on the host: it
    # does at every production size (k_len >= 100, CUDA) and does not at this
    # test's size, on CPU or CUDA. What is pinned here is the definition.
    torch, model, prompt, trace = _tiny()
    press = _pp("streamingllm_pp", forced_prompt_len=4)
    _replay(model, press, prompt, trace)
    assert press.base_press.n_sink == 4 and press.protect.target == BUDGET
    for positions in press.protect.maps.values():
        for head in positions:
            kept = torch.sort(head).values
            assert torch.equal(kept[:4], torch.arange(4))
            last = int(kept[-1])
            assert torch.equal(kept[4:], torch.arange(last - (BUDGET - 5), last + 1))


def test_a_long_prompt_raises_the_target_and_is_flagged():
    _torch, model, prompt, trace = _tiny()
    press = _pp("random_pp", min_trace_slots=16)  # 12 + 16 > 24
    profile = _replay(model, press, prompt, trace)
    st = press.protect
    assert (st.target, st.budget_relaxed) == (PROMPT + 16, True)
    assert profile.max_after_prune == PROMPT + 16


def test_prompt_length_is_read_from_free_generation_too():
    torch, model, prompt, _trace = _tiny()
    from kvcache.presses.prompt_protect import build_prompt_protected_press

    press = build_prompt_protected_press(
        "random_pp",
        budget=BUDGET,
        compression_interval=INTERVAL,
        seed=3,
        min_trace_slots=8,
    )
    ids = torch.tensor([prompt])
    with torch.inference_mode(), press(model):
        model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=40,
            min_new_tokens=40,
            use_cache=True,
            pad_token_id=0,
        )
    assert press.protect.prompt_len == PROMPT and press.protect.n_prunes >= 2


def test_the_live_check_fails_when_a_prompt_position_is_missing():
    torch = pytest.importorskip("torch")
    from kvcache.presses.prompt_protect import check_retained

    st = ProtectState(base="random", budget=8, prompt_len=4)
    before = torch.arange(12).unsqueeze(0)
    check_retained(st, before, torch.tensor([[0, 1, 2, 3, 7, 9, 10, 11]]), 0)
    with pytest.raises(RuntimeError, match="prompt position was evicted"):
        check_retained(st, before, torch.tensor([[0, 1, 2, 5, 7, 9, 10, 11]]), 0)
    slm = ProtectState(base="streamingllm", budget=8, prompt_len=4)
    check_retained(slm, before, torch.tensor([[0, 1, 2, 3, 8, 9, 10, 11]]), 0)
    with pytest.raises(RuntimeError, match="not the most recent"):
        check_retained(slm, before, torch.tensor([[0, 1, 2, 3, 7, 9, 10, 11]]), 0)


def test_a_press_is_refused_a_second_sequence():
    _torch, model, prompt, trace = _tiny()
    press = _pp("random_pp")
    _replay(model, press, prompt, trace)
    with pytest.raises(RuntimeError, match="second prompt"):
        _replay(model, press, prompt, trace)


# ------------------------------------------------------- smoke gate and runners
def test_smoke_traces_are_picked_by_rule_not_by_hand():
    from kvcache.presses.prompt_protect import select_smoke_traces

    cands = [
        {"pid": f"p{i}", "prompt_len": n, "trace_len": t}
        for i, (n, t) in enumerate(
            [(353, 5000), (15, 900), (19, 2000), (52, 1500), (90, 1024), (216, 800)]
        )
    ]
    got = select_smoke_traces(cands, 1024)
    assert [c["prompt_len"] for c in got] == [353, 90, 19]
    with pytest.raises(SystemExit):
        select_smoke_traces(cands[:2], 1024)


def _replay_record(**kw):
    rec = {
        "kind": "replay",
        "pid": "p",
        "press": "random_pp",
        "prompt_len": 50,
        "n_layers": 28,
        "expected_target": 256,
        "first_effective_prune": 256,
        "pre_eviction_max_tv": 0.0,
        "post_prune_sizes": [256],
        "identity_max_abs_diff": 0.0,
        "protection_max_abs_diff": 0.2,
        "protect": {
            "prompt_len": 50,
            "target": 256,
            "n_prunes": 560,
            "n_layers_pruned": 28,
        },
    }
    rec.update(kw)
    return rec


def test_smoke_passes_only_when_every_registered_check_holds():
    from kvcache.presses.prompt_protect import smoke_failures

    gen = {
        "kind": "generation",
        "pid": "p",
        "press": "random_pp",
        "prompt_len": 50,
        "n_tokens": 700,
        "protect": {"prompt_len": 50, "n_prunes": 28},
    }
    assert smoke_failures([_replay_record(), gen]) == []
    assert smoke_failures([]) == ["no smoke record"]
    broken = [
        _replay_record(raised="layer 3 head 0: a prompt position was evicted"),
        _replay_record(pre_eviction_max_tv=1e-7),
        _replay_record(post_prune_sizes=[256, 320]),
        _replay_record(identity_max_abs_diff=1e-9),
        _replay_record(protection_max_abs_diff=0.0),
        _replay_record(first_effective_prune=None),
        _replay_record(expected_target=417),
        _replay_record(protect={**_replay_record()["protect"], "prompt_len": 49}),
        _replay_record(protect={**_replay_record()["protect"], "n_layers_pruned": 27}),
        {**gen, "protect": {"prompt_len": 50, "n_prunes": 0}},
        {**gen, "prompt_len": 51},
    ]
    for rec in broken:
        assert smoke_failures([_replay_record(), rec]), rec


def test_noop_press_calls_are_not_counted_as_prunes():
    from kvcache.presses.prompt_protect import effective_prunes

    events = [
        {"before": 79, "after": 79, "step": 64},
        {"before": 320, "after": 256, "step": 320},
    ]
    assert effective_prunes(events) == [events[1]]


def _load_script(name):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location(
        f"{name}_t", root / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "script,config,attr",
    [
        ("run_grid_pp", "grid_math500_pp256_s1", "build_press"),
        ("run_grid_pp", "grid_math500_pp1024_s1", "build_press"),
        ("run_replay_pp", "readout_math500_pp256_s1", "build_replay_press"),
        ("run_replay_pp", "readout_math500_pp1024_s1", "build_replay_press"),
    ],
)
def test_protected_runners_swap_only_the_press_factory(
    script, config, attr, monkeypatch, capsys
):
    module = _load_script(script)
    inner = getattr(module, script.removesuffix("_pp"))
    before = getattr(inner, attr)
    monkeypatch.setattr(inner, attr, before)  # restored after the test
    monkeypatch.setattr(
        "sys.argv",
        [script, "--config", f"configs/experiments/{config}.yaml", "--dry-run"],
    )
    assert module.main() == 0
    assert "DRY RUN" in capsys.readouterr().out
    after = getattr(inner, attr)
    assert after is not before and "prompt_protect" in after.__module__ + str(
        after.__code__.co_names
    )


@pytest.mark.parametrize("script", ["run_grid_pp", "run_replay_pp"])
def test_protected_runners_refuse_an_unprotected_config(script, monkeypatch):
    module = _load_script(script)
    monkeypatch.setattr(
        "sys.argv",
        [
            script,
            "--config",
            "configs/experiments/readout_math500_diss1024.yaml",
            "--dry-run",
        ],
    )
    with pytest.raises(SystemExit, match="prompt_protected"):
        module.main()


def test_pp_smoke_gate_accepts_only_this_identity(tmp_path):
    import json

    from kvcache.gates import _revision, _yaml, gate_status
    from kvcache.provenance import PROJECT_ROOT, identity

    config = "configs/experiments/pp_smoke_s1.yaml"
    cfg = _yaml(PROJECT_ROOT / config)
    ident = identity(
        "pp",
        cfg,
        model_revision=_revision(cfg, PROJECT_ROOT),
        reference_run=cfg["reference_run_name"],
    )
    marker = tmp_path / "pp-smoke-PASS.json"
    assert not gate_status("pp_smoke", tmp_path, config)[0]
    marker.write_text(json.dumps({"passed": True, "identity": ident}))
    assert gate_status("pp_smoke", tmp_path, config)[0]
    marker.write_text(json.dumps({"passed": False, "identity": ident}))
    assert not gate_status("pp_smoke", tmp_path, config)[0]
    stale = {**ident, "implementation_id": "0" * 64}
    marker.write_text(json.dumps({"passed": True, "identity": stale}))
    assert not gate_status("pp_smoke", tmp_path, config)[0]


def test_queue_gates_every_protected_cell_on_the_smoke():
    import yaml

    from kvcache.provenance import PROJECT_ROOT

    jobs = {
        j["name"]: j
        for j in yaml.safe_load(
            (PROJECT_ROOT / "configs/nightly_queue.yaml").read_text()
        )["jobs"]
    }
    assert jobs["pp-smoke"]["gate"] == "pp_smoke"
    for name in ("grid-pp256", "readout-pp256", "grid-pp1024", "readout-pp1024"):
        job, seen = jobs[name], set()
        assert job["command"] in ("grid_pp", "replay_pp")
        while job.get("requires"):
            seen.add(job["requires"])
            job = jobs[job["requires"]]
        assert "pp-smoke" in seen, name
    assert jobs["reverse-readout-pp256"]["command"] == "replay"

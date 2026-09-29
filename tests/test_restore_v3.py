"""Amendment S.2 restoration by origin.

The planning rules run anywhere. The replay tests use a tiny random-init Qwen2
(skipped without torch; they run on the GPU host's CPU with the pinned stack).

Pins: the prompt arm returns every evicted prompt entry in every head, padded
to a common count with that head's most recent trace evictions; the trace arm
returns the same number and never a prompt entry; the full arm and the
no-intervention read reproduce restore-v2 exactly (the registered determinism
control); and the gate fails when a prompt entry is withheld.
"""

from __future__ import annotations

import random

import pytest

from kvcache.restore_v3 import ARMS, plan_origin_return, prompt_return_count

# one head's stash, in eviction order; prompt length 4
HEAD = [2, 7, 0, 9, 3, 11, 12]


def test_arms_are_the_registered_four():
    assert ARMS == ("frozen", "prompt", "trace", "full")


def test_return_count_is_the_largest_evicted_prompt_count():
    positions = {0: [HEAD, [1, 8, 9]], 1: [[5, 6], [0, 1, 2, 3, 9]]}
    assert prompt_return_count(positions, 4) == 4
    assert prompt_return_count({0: [[5, 6]]}, 4) == 0
    assert prompt_return_count({}, 4) == 0


def test_prompt_arm_returns_every_prompt_entry_and_pads_with_recent_trace():
    rng = random.Random(0)
    idx, pad = plan_origin_return(HEAD, prompt_len=4, n=3, arm="prompt", rng=rng)
    assert ([HEAD[i] for i in idx], pad) == ([2, 0, 3], 0)
    idx, pad = plan_origin_return(HEAD, prompt_len=4, n=5, arm="prompt", rng=rng)
    assert ([HEAD[i] for i in idx], pad) == ([2, 0, 3, 11, 12], 2)
    with pytest.raises(ValueError, match="smaller"):
        plan_origin_return(HEAD, prompt_len=4, n=2, arm="prompt", rng=rng)
    with pytest.raises(ValueError, match="pad"):
        plan_origin_return(HEAD, prompt_len=4, n=8, arm="prompt", rng=rng)


def test_trace_arm_returns_the_same_count_and_never_a_prompt_entry():
    a, pad = plan_origin_return(
        HEAD, prompt_len=4, n=3, arm="trace", rng=random.Random("k|0|trace")
    )
    b, _ = plan_origin_return(
        HEAD, prompt_len=4, n=3, arm="trace", rng=random.Random("k|0|trace")
    )
    assert a == b and pad == 0 and len(set(a)) == 3
    assert all(HEAD[i] >= 4 for i in a)
    with pytest.raises(ValueError, match="draw"):
        plan_origin_return(HEAD, prompt_len=4, n=5, arm="trace", rng=random.Random(0))


def test_plans_exist_only_for_the_partial_arms_and_clean_stashes():
    for arm in ("frozen", "full", "matched"):
        with pytest.raises(ValueError, match="no return plan"):
            plan_origin_return(HEAD, prompt_len=4, n=3, arm=arm, rng=random.Random(0))
    with pytest.raises(ValueError, match="twice"):
        plan_origin_return(
            [2, 2, 9], prompt_len=4, n=1, arm="prompt", rng=random.Random(0)
        )


# ------------------------------------------------------------- tiny model
PROMPT = 12
KW = dict(rng_key="t", trigger_step=20, window=10)


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
    trace = torch.randint(3, 97, (90,)).tolist()
    return model, prompt, trace


def _press(policy):
    from kvcache.presses import build_press

    return build_press(
        policy, budget=16, compression_interval=4, seed=3, instrumented=True
    )


@pytest.mark.parametrize("policy", ["random", "streamingllm"])
def test_full_arm_and_the_none_read_reproduce_restore_v2_exactly(policy):
    model, prompt, trace = _tiny()
    from kvcache.restore_v2 import run_restore_v2
    from kvcache.restore_v3 import run_restore_v3

    v3 = run_restore_v3(model, _press(policy), prompt, trace, **KW)
    v2 = run_restore_v2(
        model, _press(policy), prompt, trace, n_reinsert=4, lags=(1, 2, 4), **KW
    )
    assert v3["status"] == "READ" and v2["status"] == "READ"
    assert v3["cohort_step"] == v2["cohort_step"]
    assert v3["stash_size"] == v2["stash_size"]
    fs = v2["full_restore"]
    assert v3["arms"]["full"]["window_mean_tv"] == fs["window_mean_tv"]
    assert v3["none_window_mean_tv"] == fs["none_window_mean_tv"]
    assert v3["arms"]["full"]["cache_len_after"] == fs["cache_len_after"]


@pytest.mark.parametrize("policy", ["random", "streamingllm"])
def test_every_arm_is_gated_and_holds_its_cache_for_the_window(policy):
    model, prompt, trace = _tiny()
    from kvcache.restore_v3 import run_restore_v3

    out = run_restore_v3(model, _press(policy), prompt, trace, **KW)
    arms = out["arms"]
    assert set(arms) == set(ARMS) and out["prompt_len"] == PROMPT
    n = out["n_prompt_return"]
    assert 1 <= n <= PROMPT
    for arm in ARMS:
        a = arms[arm]
        assert a["gate"]["passed"]
        # eviction is off: the cache only grows, by one entry per forward
        assert a["cache_len_after"] == a["cache_len_at_restore"] + KW["window"]
    frozen = arms["frozen"]["cache_len_at_restore"]
    assert arms["prompt"]["cache_len_at_restore"] == frozen + n
    assert arms["trace"]["cache_len_at_restore"] == frozen + n
    assert arms["full"]["cache_len_at_restore"] == frozen + out["stash_size"]
    assert arms["prompt"]["gate"]["n_returned_per_head"] == n
    if policy == "streamingllm":
        # four sinks are never evicted; every head evicts the same positions
        assert n == PROMPT - 4 and arms["prompt"]["gate"]["pad_max"] == 0
    tvs = {arm: arms[arm]["window_mean_tv"] for arm in ARMS}
    assert len(set(tvs.values())) == 4, tvs


def test_the_gate_fails_when_a_prompt_entry_is_withheld(monkeypatch):
    model, prompt, trace = _tiny()
    import kvcache.restore_v3 as rv3

    real = rv3.plan_origin_return

    def withhold(head_positions, *, prompt_len, n, arm, rng):
        if arm == "prompt":  # hand back trace entries under the prompt label
            return real(
                head_positions, prompt_len=prompt_len, n=n, arm="trace", rng=rng
            )
        return real(head_positions, prompt_len=prompt_len, n=n, arm=arm, rng=rng)

    monkeypatch.setattr(rv3, "plan_origin_return", withhold)
    with pytest.raises(RuntimeError, match="restoration gate failed: prompt"):
        rv3.run_restore_v3(model, _press("streamingllm"), prompt, trace, **KW)


def test_a_short_trace_is_ineligible_not_read():
    model, prompt, trace = _tiny()
    from kvcache.restore_v3 import run_restore_v3

    out = run_restore_v3(model, _press("random"), prompt, trace[:24], **KW)
    assert out["status"].startswith("INELIGIBLE") and out["arms"] == {}

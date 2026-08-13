"""Hand-constructed checks for the restoration test (Amendment P.5)."""

import random

import pytest

from kvcache.metrics.restoration import (
    classify_restoration,
    plan_reinsertion,
    restoration_read,
)


def _rng():
    return random.Random(0)


def test_matched_arm_reinserts_the_most_recent_evictions_and_stays_budget_matched():
    plan = plan_reinsertion(
        evicted_order=[1, 2, 3, 4, 5, 6],
        retained=[10, 11, 12, 13],
        n_reinsert=2,
        arm="matched",
        rng=_rng(),
    )
    assert plan["reinsert"] == [5, 6]  # most recent
    assert len(plan["evict"]) == 2  # budget preserved
    assert set(plan["evict"]) <= {10, 11, 12, 13}


def test_relaxed_arm_grows_the_cache_and_evicts_nothing():
    plan = plan_reinsertion(
        evicted_order=[1, 2, 3],
        retained=[10, 11],
        n_reinsert=2,
        arm="relaxed",
        rng=_rng(),
    )
    assert plan["reinsert"] == [2, 3]
    assert plan["evict"] == []


def test_random_arm_draws_from_the_whole_evicted_pool_not_the_recent_tail():
    """Otherwise the control is the intervention and cannot separate anything."""
    picks = set()
    for seed in range(30):
        plan = plan_reinsertion(
            evicted_order=list(range(20)),
            retained=[100, 101, 102, 103],
            n_reinsert=2,
            arm="random",
            rng=random.Random(seed),
        )
        picks.update(plan["reinsert"])
    assert min(picks) < 10, "random arm never sampled an older eviction"


def test_the_eviction_set_is_drawn_from_the_whole_live_cache():
    """Replaces an assertion that encoded the index-space bug.

    The old test passed `evicted_order=[1,2,3]` and `retained=[1,2,3,10,11,12]`
    -- "overlap on purpose" -- and asserted the two sets never intersect. That
    premise is the defect: the real caller passes `range(stash_size)` and
    `range(cache_size)`, two independent 0-based spaces in which a shared number
    means nothing. Enforcing disjointness there removed real cache positions
    from the eviction pool purely because of their numeric value, which is how
    `matched` came to be barred from evicting recent entries. The invariant that
    actually matters is the one asserted here: every retained position stays
    eligible, so the eviction set is an unbiased sample of the live cache.
    """
    retained = list(range(64))
    seen: set[int] = set()
    # 200 seeds: a given position is missed with probability (1-8/64)^200 ~ 3e-12,
    # so the coverage assertion below is deterministic in practice rather than flaky.
    for seed in range(200):
        plan = plan_reinsertion(
            evicted_order=list(range(64)),
            retained=retained,
            n_reinsert=8,
            arm="matched",
            rng=random.Random(seed),
        )
        assert set(plan["evict"]) <= set(retained)
        assert len(set(plan["evict"])) == 8
        seen |= set(plan["evict"])
    # Nothing is structurally excluded: across seeds every position is reachable.
    assert seen == set(retained)


def test_none_arm_does_nothing():
    plan = plan_reinsertion(
        evicted_order=[1, 2], retained=[3], n_reinsert=1, arm="none", rng=_rng()
    )
    assert plan == {"reinsert": [], "evict": []}


def test_refuses_an_unknown_arm_and_an_impossible_request():
    with pytest.raises(ValueError):
        plan_reinsertion(
            evicted_order=[1], retained=[2], n_reinsert=1, arm="bogus", rng=_rng()
        )
    with pytest.raises(ValueError):
        plan_reinsertion(
            evicted_order=[1], retained=[2], n_reinsert=5, arm="matched", rng=_rng()
        )


def _series(value, n=400):
    return [value] * n


def test_read_requires_the_no_intervention_arm():
    with pytest.raises(ValueError, match="no-intervention arm is required"):
        restoration_read({"matched": {"t1": _series(0.1)}}, trigger_step=10, window=10)


def test_read_measures_the_effect_against_no_intervention_not_before_after():
    arms = {
        "none": {f"t{i}": _series(0.20) for i in range(8)},
        "matched": {f"t{i}": _series(0.12) for i in range(8)},
        "random": {f"t{i}": _series(0.19) for i in range(8)},
    }
    out = restoration_read(arms, trigger_step=10, window=50, n_resamples=200)
    assert out["arms"]["matched"]["effect_vs_no_intervention"] == pytest.approx(0.08)
    assert out["arms"]["matched"]["excludes_zero"] is True
    assert classify_restoration(out) == "RESTORES"


def test_a_control_that_helps_as_much_reads_as_non_specific():
    arms = {
        "none": {f"t{i}": _series(0.20) for i in range(8)},
        "matched": {f"t{i}": _series(0.12) for i in range(8)},
        "random": {f"t{i}": _series(0.11) for i in range(8)},
    }
    out = restoration_read(arms, trigger_step=10, window=50, n_resamples=200)
    assert classify_restoration(out) == "NON-SPECIFIC"


def test_no_effect_reads_as_no_restoration():
    arms = {
        "none": {f"t{i}": _series(0.20) for i in range(8)},
        "matched": {f"t{i}": _series(0.20) for i in range(8)},
    }
    out = restoration_read(arms, trigger_step=10, window=50, n_resamples=200)
    assert classify_restoration(out) == "NO RESTORATION"


def test_arms_must_cover_the_same_traces():
    arms = {
        "none": {"t1": _series(0.2), "t2": _series(0.2)},
        "matched": {"t1": _series(0.1)},
    }
    with pytest.raises(ValueError, match="same traces"):
        restoration_read(arms, trigger_step=10, window=10)


def test_budget_matched_surgery_preserves_cache_size_and_is_bitwise_exact():
    """The instrument gate's own test: same size, and the entries really moved."""
    torch = pytest.importorskip("torch")
    from kvcache.metrics.restoration import apply_reinsertion, retained_set_report

    keys = torch.arange(2 * 3 * 8 * 4, dtype=torch.float32).reshape(2, 3, 8, 4)
    values = keys * -1.0
    evicted_k = torch.full((2, 3, 5, 4), 7.0)
    evicted_v = torch.full((2, 3, 5, 4), -7.0)

    new_k, new_v = apply_reinsertion(
        keys,
        values,
        evicted_keys=evicted_k,
        evicted_values=evicted_v,
        reinsert_local=[0, 2],
        evict_local=[1, 3],
    )
    assert new_k.shape[2] == keys.shape[2]  # budget preserved
    assert new_v.shape[2] == values.shape[2]

    expected = evicted_k.index_select(2, torch.tensor([0, 2]))
    report = retained_set_report(new_k, expected, n_reinserted=2)
    assert report["passed"] is True
    assert report["max_abs_difference"] == 0.0


def test_relaxed_surgery_grows_the_cache_by_exactly_the_reinsertion():
    torch = pytest.importorskip("torch")
    from kvcache.metrics.restoration import apply_reinsertion

    keys = torch.zeros(1, 2, 6, 3)
    values = torch.zeros(1, 2, 6, 3)
    ek = torch.ones(1, 2, 4, 3)
    new_k, _ = apply_reinsertion(
        keys,
        values,
        evicted_keys=ek,
        evicted_values=ek,
        reinsert_local=[0, 1, 2],
        evict_local=[],
    )
    assert new_k.shape[2] == 9


def test_gate_fails_when_the_reinserted_entries_were_altered():
    """A surgery that changes values it claimed only to move must be caught."""
    torch = pytest.importorskip("torch")
    from kvcache.metrics.restoration import retained_set_report

    new_k = torch.ones(1, 1, 4, 2)
    expected = torch.ones(1, 1, 2, 2) * 1.5  # not what is actually in the tail
    report = retained_set_report(new_k, expected, n_reinserted=2)
    assert report["passed"] is False
    assert report["max_abs_difference"] == pytest.approx(0.5)


def test_no_intervention_arm_passes_the_gate_trivially_and_says_so():
    pytest.importorskip("torch")
    from kvcache.metrics.restoration import retained_set_report

    report = retained_set_report(None, None, n_reinserted=0)
    assert report["passed"] is True
    assert "no-intervention" in report["note"]


def test_the_read_anchors_on_the_step_the_intervention_actually_landed_on():
    """The nominal trigger and the applied step differ, and by a trace-dependent gap.

    A press can only intervene during a compression event with a full stash, so
    a nominal trigger of 130 landed at 256 in the smoke. Anchoring on 130 would
    fold pre-intervention steps -- where every arm is identical, so the paired
    difference is exactly zero -- into the window, diluting the effect by a
    different factor per trace. Here trace "a" applies at 10 and trace "b" at
    50; both show a clean 0.4 reduction over their own next 5 steps, so a
    correctly anchored read recovers 0.4 and an unanchored one does not.
    """
    from kvcache.metrics.restoration import restoration_read

    def series(applied):
        values = [0.5] * 100
        return values, [0.5] * applied + [0.1] * (100 - applied)

    none_a, treated_a = series(10)
    none_b, treated_b = series(50)
    arms = {
        "none": {"a": none_a, "b": none_b},
        "matched": {"a": treated_a, "b": treated_b},
    }
    anchored = restoration_read(
        arms,
        applied_steps={"a": 10, "b": 50},
        window=5,
        n_resamples=200,
    )
    assert anchored["anchored_on"] == "applied_step"
    assert anchored["applied_step_range"] == [10, 50]
    assert anchored["arms"]["matched"]["effect_vs_no_intervention"] == pytest.approx(
        0.4
    )

    unanchored = restoration_read(arms, trigger_step=10, window=5, n_resamples=200)
    assert unanchored["anchored_on"] == "trigger_step"
    # Trace "b" has not been treated yet at step 10, so its difference is 0 and
    # the pooled effect is halved -- the dilution this anchoring exists to avoid.
    assert unanchored["arms"]["matched"]["effect_vs_no_intervention"] == pytest.approx(
        0.2
    )


def test_the_read_refuses_a_trace_with_no_applied_step():
    from kvcache.metrics.restoration import restoration_read

    arms = {
        "none": {"a": [0.5] * 50, "b": [0.5] * 50},
        "matched": {"a": [0.1] * 50, "b": [0.1] * 50},
    }
    with pytest.raises(ValueError, match="applied step"):
        restoration_read(arms, applied_steps={"a": 10}, window=5, n_resamples=10)


def test_the_eviction_pool_is_not_intersected_across_index_spaces():
    """Regression: `retained` indexes the live cache, `reinsert` indexes the stash.

    Excluding one from the other barred `matched` -- which reinserts the highest
    stash indices -- from ever evicting the highest CACHE indices, i.e. the most
    recent tokens, while `random` evicted them at the uniform rate. The arms then
    differed in two ways at once and the control could not isolate which tokens
    mattered. On the powered cell this showed up as random = -0.248 against
    matched = -0.001.
    """
    import random as _random

    from kvcache.metrics.restoration import plan_reinsertion

    stash = cache = 256
    n = 64
    top_quarter = {}
    for arm in ("matched", "random"):
        plan = plan_reinsertion(
            evicted_order=list(range(stash)),
            retained=list(range(cache)),
            n_reinsert=n,
            arm=arm,
            rng=_random.Random(0),
            evict_rng=_random.Random(7),
        )
        assert len(plan["evict"]) == n
        top_quarter[arm] = sum(1 for p in plan["evict"] if p >= 192)
    # Both arms must now be free to evict recent entries; before the fix
    # `matched` was structurally pinned to exactly zero.
    assert top_quarter["matched"] > 0
    assert top_quarter["matched"] == top_quarter["random"]


def test_arms_sharing_a_seed_evict_the_same_entries():
    """What makes `random` a control: only WHICH entries return may differ."""
    import random as _random

    from kvcache.metrics.restoration import plan_reinsertion

    def plan(arm):
        return plan_reinsertion(
            evicted_order=list(range(256)),
            retained=list(range(256)),
            n_reinsert=32,
            arm=arm,
            rng=_random.Random(11),
            evict_rng=_random.Random(11),
        )

    matched, chance = plan("matched"), plan("random")
    assert matched["evict"] == chance["evict"], "arms must evict the same entries"
    assert matched["reinsert"] != chance["reinsert"], "and reinsert different ones"

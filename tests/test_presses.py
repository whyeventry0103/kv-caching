import pytest

from kvcache.presses import (
    build_press,
    build_replay_press,
    get_press,
    press_needs_eager_attention,
    press_seed,
)


def test_press_seed_is_deterministic_and_decorrelated():
    # Stable across calls (reproducible).
    assert press_seed("p1", 11) == press_seed("p1", 11)
    # Varies by problem and by seed (decorrelates random eviction).
    assert press_seed("p1", 11) != press_seed("p2", 11)
    assert press_seed("p1", 11) != press_seed("p1", 23)
    assert isinstance(press_seed("p1", 11), int)


def test_h2o_needs_eager_attention_others_do_not():
    assert press_needs_eager_attention("h2o") is True
    assert press_needs_eager_attention("streamingllm") is False
    assert press_needs_eager_attention("random") is False


def test_rkv_refuses_without_smoke_marker(monkeypatch, tmp_path):
    # R-KV is parity-verified but must not enter analysis until the real-generation
    # smoke marker exists (Amendment E). No marker -> refuse.
    monkeypatch.delenv("KVCACHE_RKV_SMOKE_MARKER", raising=False)
    monkeypatch.delenv("KVCACHE_BASE_RUNS_ROOT", raising=False)
    with pytest.raises(NotImplementedError, match="smoke"):
        get_press("rkv")


def test_rkv_promoted_by_smoke_marker(monkeypatch, tmp_path):
    marker = tmp_path / "rkv-gen-smoke-PASS.json"
    marker.write_text('{"passed": true}')
    monkeypatch.setenv("KVCACHE_RKV_SMOKE_MARKER", str(marker))
    assert get_press("rkv") == "rkv"


def test_rkv_promoted_by_base_runs_root_marker(monkeypatch, tmp_path):
    # launch.sh exports KVCACHE_BASE_RUNS_ROOT; the smoke writes the marker there.
    monkeypatch.delenv("KVCACHE_RKV_SMOKE_MARKER", raising=False)
    monkeypatch.setenv("KVCACHE_BASE_RUNS_ROOT", str(tmp_path))
    with pytest.raises(NotImplementedError):
        get_press("rkv")
    (tmp_path / "rkv-gen-smoke-PASS.json").write_text('{"passed": true}')
    assert get_press("rkv") == "rkv"


@pytest.mark.parametrize(
    "name",
    [
        "snapkv",
        "expected_attention",
        "rkv",
        "kivi2",
        "triattention",
    ],
)
def test_unimplemented_or_deferred_presses_refuse(name, monkeypatch):
    monkeypatch.delenv("KVCACHE_RKV_SMOKE_MARKER", raising=False)
    monkeypatch.delenv("KVCACHE_BASE_RUNS_ROOT", raising=False)
    with pytest.raises(NotImplementedError):
        get_press(name)


def test_unknown_press_is_rejected():
    with pytest.raises(KeyError):
        get_press("not-a-press")


class _FakeStreaming:
    pass


class _FakeRandom:
    def __init__(self, seed=None):
        self.seed = seed


class _FakeDecoding:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeObservedAttention:
    pass


class _FakeKVPress:
    StreamingLLMPress = _FakeStreaming
    RandomPress = _FakeRandom
    ObservedAttentionPress = _FakeObservedAttention
    DecodingPress = _FakeDecoding


@pytest.mark.parametrize("name", ["streamingllm", "random", "h2o"])
def test_build_press_constructs_decoding_press(name):
    press = build_press(
        name,
        budget=256,
        compression_interval=64,
        seed=11,
        kvpress_module=_FakeKVPress,
    )
    assert isinstance(press, _FakeDecoding)
    assert press.kwargs["target_size"] == 256
    assert press.kwargs["compression_interval"] == 64
    assert press.kwargs["hidden_states_buffer_size"] == 0
    if name == "random":
        assert press.kwargs["base_press"].seed == 11


def test_build_press_accepts_instrumented_decoding_class():
    class _InstrumentedDecoding(_FakeDecoding):
        pass

    press = build_press(
        "random",
        budget=256,
        seed=11,
        kvpress_module=_FakeKVPress,
        decoding_press_cls=_InstrumentedDecoding,
    )
    assert isinstance(press, _InstrumentedDecoding)


def test_fullcache_press_is_noop():
    assert build_press("fullcache", budget=256) is None


def test_replay_press_enabled_flag_scopes_forward_hook():
    class _ReplayDecoding:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.__post_init__()

        def __post_init__(self):
            self.parent_calls = 0

        def forward_hook(self, module, input, kwargs, output):
            self.parent_calls += 1
            return "compressed"

    class _ReplayKVPress(_FakeKVPress):
        DecodingPress = _ReplayDecoding

    press = build_replay_press(
        "streamingllm",
        budget=256,
        kvpress_module=_ReplayKVPress,
    )
    assert press.forward_hook(None, None, None, "full") == "full"
    press.enabled = True
    assert press.forward_hook(None, None, None, "full") == "compressed"
    assert press.parent_calls == 1


def test_replay_press_onset_is_step_after_prune():
    class _ReplayDecoding:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.__post_init__()

        def __post_init__(self):
            pass

        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            return keys, values

    class _ReplayKVPress(_FakeKVPress):
        DecodingPress = _ReplayDecoding

    class _Keys:
        shape = (1, 1, 320, 1)

    class _Module:
        layer_idx = 0

    press = build_replay_press(
        "streamingllm",
        budget=256,
        kvpress_module=_ReplayKVPress,
    )
    press.profile_step = 64
    press.compress(_Module(), None, _Keys(), _Keys(), None, {})
    assert press.onset_step == 65


def test_rkv_scorer_overrides_snapkv_dataclass_defaults():
    # SnapKVPress is a dataclass; a plain subclass's annotated class attributes are
    # shadowed by the generated __init__ (instance gets window_size=64/kernel_size=5).
    # The live smoke caught this (Jaccard 0.805). Lock the override behavior.
    from dataclasses import dataclass, field
    from types import SimpleNamespace

    from kvcache.presses.rkv import (
        RKV_KERNEL_SIZE,
        RKV_MIX_LAMBDA,
        RKV_WINDOW_SIZE,
        build_rkv_scorer,
    )

    @dataclass
    class FakeSnapKVPress:  # mirrors kvpress's dataclass defaults
        compression_ratio: float = 0.0
        window_size: int = 64
        kernel_size: int = 5
        extra: dict = field(default_factory=dict)

    fake_kvpress = SimpleNamespace(SnapKVPress=FakeSnapKVPress)
    scorer = build_rkv_scorer(fake_kvpress)
    assert scorer.window_size == RKV_WINDOW_SIZE == 8
    assert scorer.kernel_size == RKV_KERNEL_SIZE == 7
    assert scorer.mix_lambda == RKV_MIX_LAMBDA


def test_rope_inv_freq_unscaled_matches_the_hand_rolled_formula():
    # The scaled-RoPE fix replaced a hand-rolled inv_freq with transformers'
    # rope-init. Every number already measured (Qwen-7B, Llama-8B on MATH-500)
    # came from the hand-rolled path, so the unscaled branch must reproduce it
    # exactly or those results silently change meaning. Bitwise equality, not
    # approx: this is the equivalence the previous runs are certified under.
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")

    from kvcache.presses.rkv import rope_inv_freq

    head_dim = 128
    config = SimpleNamespace(rope_theta=10000.0, rope_scaling=None)
    inv_freq, attention_scaling = rope_inv_freq(config, head_dim, torch.device("cpu"))
    expected = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    assert attention_scaling == 1.0
    assert torch.equal(inv_freq, expected)


def test_rope_inv_freq_delegates_scaled_families_to_the_framework():
    # llama3-scaled RoPE (R1-Distill-Llama-8B) must go to the framework's own
    # schedule rather than the unscaled formula -- the R-KV published-configuration
    # cell crashed on every launch until it did.
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")

    from kvcache.presses.rkv import rope_inv_freq

    head_dim = 8
    sentinel = torch.arange(1, head_dim // 2 + 1, dtype=torch.float32)
    calls = []

    def fake_llama3(config, device):
        calls.append((config, device))
        return sentinel, 2.0

    config = SimpleNamespace(
        rope_theta=500000.0,
        rope_scaling={"rope_type": "llama3", "factor": 8.0},
    )
    inv_freq, attention_scaling = rope_inv_freq(
        config,
        head_dim,
        torch.device("cpu"),
        rope_init_functions={"llama3": fake_llama3},
    )
    assert len(calls) == 1
    assert torch.equal(inv_freq, sentinel)
    assert attention_scaling == 2.0


def test_rope_inv_freq_still_refuses_an_unknown_scaling_family():
    # Narrowing the guard must not remove it: a family the installed transformers
    # has no schedule for still refuses rather than silently mis-rotating.
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")

    from kvcache.presses.rkv import rope_inv_freq

    config = SimpleNamespace(
        rope_theta=10000.0, rope_scaling={"rope_type": "not_a_real_family"}
    )
    with pytest.raises(NotImplementedError, match="not_a_real_family"):
        rope_inv_freq(
            config, 8, torch.device("cpu"), rope_init_functions={"llama3": None}
        )


def test_rope_inv_freq_rejects_a_head_dim_mismatch():
    # A config whose rope dim disagrees with the query head_dim would broadcast
    # into a wrong rotation instead of failing; make it fail.
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")

    from kvcache.presses.rkv import rope_inv_freq

    config = SimpleNamespace(rope_theta=1.0, rope_scaling={"rope_type": "llama3"})
    with pytest.raises(RuntimeError, match="head_dim"):
        rope_inv_freq(
            config,
            128,
            torch.device("cpu"),
            rope_init_functions={"llama3": lambda c, d: (torch.ones(4), 1.0)},
        )


class _Keys:
    def __init__(self, n):
        self.shape = (1, 1, n, 8)


class _EvictingFakeBase:
    """A fake that ACTUALLY evicts down to target_size.

    The previous fake returned its input unchanged, so the cache never shrank
    and the ratio-vs-cache-size feedback loop could not occur. That is exactly
    how the collapse below reached a 200-profile GPU run undetected.
    """

    target_size = 0

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        n = keys.shape[2]
        if n <= self.target_size:
            return keys, values
        return _Keys(self.target_size), _Keys(self.target_size)


def test_ratio_matched_press_scales_with_the_logical_context_not_the_cache(monkeypatch):
    """Amendment M.3: the ratio is taken against the context the model would have.

    Regression for the feedback loop that made the first M.3 run a constant-32
    cell: target computed off `keys.shape[2]` (the previous compression's
    output) collapses to `min_budget` and never leaves it.
    """
    import kvcache.presses as presses_module

    base = _EvictingFakeBase()
    monkeypatch.setattr(presses_module, "build_press", lambda *a, **k: base)
    press = presses_module.build_ratio_matched_press(
        "random", ratio=0.25, min_budget=32
    )

    # Prefill of 1000, then repeated 64-token intervals -- the real shape.
    press.compress(None, None, _Keys(1000), _Keys(1000), None, {})
    assert press.target_size == 250
    targets = [press.target_size]
    for _ in range(6):
        cache = press.target_size + 64
        press.compress(None, None, _Keys(cache), _Keys(cache), None, {})
        targets.append(press.target_size)

    # The logical context grows by 64 each interval, so the target must GROW.
    assert targets == sorted(targets), targets
    assert targets[-1] > targets[0], "target collapsed instead of tracking context"
    assert not any(e["at_floor"] for e in press.ratio_trace)
    # And the realized ratio stays at the requested value, which is the point.
    last = press.ratio_trace[-1]
    assert last["target_size"] == round(0.25 * last["logical_context"])


def test_ratio_matched_press_still_honours_the_floor_on_a_short_context(monkeypatch):
    import kvcache.presses as presses_module

    base = _EvictingFakeBase()
    monkeypatch.setattr(presses_module, "build_press", lambda *a, **k: base)
    press = presses_module.build_ratio_matched_press(
        "random", ratio=0.25, min_budget=32
    )
    press.compress(None, None, _Keys(64), _Keys(64), None, {})  # 64*.25=16 -> floor
    assert press.target_size == 32
    assert press.ratio_trace[-1]["at_floor"] is True


def test_ratio_matched_press_rejects_impossible_ratios():
    import pytest as _pytest

    from kvcache.presses import build_ratio_matched_press

    for bad in (0.0, 1.0, -0.1, 1.5):
        with _pytest.raises(ValueError):
            build_ratio_matched_press("random", ratio=bad)


def _fake_kvpress_base(monkeypatch):
    """A base press that evicts the oldest entries down to target_size."""
    import kvcache.presses as presses_module

    class FakeBase:
        target_size = 4
        profile_step = 0

        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            n = keys.shape[2]
            if n <= self.target_size:
                return keys, values
            return (
                keys[:, :, -self.target_size :, :],
                values[:, :, -self.target_size :, :],
            )

    base = FakeBase()
    monkeypatch.setattr(presses_module, "build_press", lambda *a, **k: base)
    return base


def test_restoration_press_none_arm_never_intervenes(monkeypatch):
    """The temporal null must run the same code path and change nothing."""
    torch = pytest.importorskip("torch")
    from kvcache.presses import build_restoration_press

    base = _fake_kvpress_base(monkeypatch)
    press = build_restoration_press(
        "random", budget=4, arm="none", trigger_step=0, reinsert_fraction=0.5
    )
    keys = torch.arange(1 * 1 * 8 * 2, dtype=torch.float32).reshape(1, 1, 8, 2)
    base.profile_step = 10
    out_k, _ = press.compress(None, None, keys, keys.clone(), None, {})
    assert out_k.shape[2] == 4  # plain compression only
    assert press.restoration_report is None
    assert press.intervened is False


def test_restoration_press_matched_arm_holds_the_budget_and_passes_its_gate(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from kvcache.presses import build_restoration_press

    base = _fake_kvpress_base(monkeypatch)
    press = build_restoration_press(
        "random", budget=4, arm="matched", trigger_step=1, reinsert_fraction=0.5
    )
    keys = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2)
    base.profile_step = 0
    press.compress(None, None, keys, keys.clone(), None, {})  # builds the stash
    base.profile_step = 5
    out_k, _ = press.compress(None, None, keys, keys.clone(), None, {})

    report = press.restoration_report
    assert report is not None, "matched arm should have intervened"
    assert report["passed"] is True
    assert report["bitwise_identical"] is True
    assert report["budget_matched"] is True
    assert report["cache_after"] == 4  # unchanged: the whole point of this arm


def test_restoration_press_relaxed_arm_grows_the_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    from kvcache.presses import build_restoration_press

    base = _fake_kvpress_base(monkeypatch)
    press = build_restoration_press(
        "random", budget=4, arm="relaxed", trigger_step=1, reinsert_fraction=0.5
    )
    keys = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2)
    base.profile_step = 0
    press.compress(None, None, keys, keys.clone(), None, {})
    base.profile_step = 5
    out_k, _ = press.compress(None, None, keys, keys.clone(), None, {})
    assert out_k.shape[2] > 4
    assert press.restoration_report["budget_matched"] is False


def test_restoration_press_intervenes_exactly_once(monkeypatch):
    torch = pytest.importorskip("torch")
    from kvcache.presses import build_restoration_press

    base = _fake_kvpress_base(monkeypatch)
    press = build_restoration_press(
        "random", budget=4, arm="matched", trigger_step=1, reinsert_fraction=0.5
    )
    keys = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2)
    for step in (0, 5, 6, 7):
        base.profile_step = step
        press.compress(None, None, keys, keys.clone(), None, {})
    assert press.restoration_report["applied_at_step"] == 5


def test_restoration_press_rejects_a_bad_arm_or_fraction():
    from kvcache.presses import build_restoration_press

    for kwargs in (
        {"arm": "bogus", "reinsert_fraction": 0.25},
        {"arm": "matched", "reinsert_fraction": 0.0},
        {"arm": "matched", "reinsert_fraction": 1.5},
    ):
        with pytest.raises(ValueError):
            build_restoration_press("random", budget=4, trigger_step=1, **kwargs)


def _fake_kvpress_module():
    """A stand-in kvpress whose DecodingPress records what it was told to do."""
    from types import SimpleNamespace

    class DecodingPress:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.__post_init__()

        def __post_init__(self):
            pass

        def forward_hook(self, module, input, kwargs, output):
            return output

        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            n = keys.shape[2]
            if n <= self.target_size:
                return keys, values
            return (
                keys[:, :, -self.target_size :, :],
                values[:, :, -self.target_size :, :],
            )

    class RandomPress:
        def __init__(self, seed=None):
            self.seed = seed

    return SimpleNamespace(DecodingPress=DecodingPress, RandomPress=RandomPress)


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        ("build_replay_press", {"budget": 8}),
        ("build_ratio_matched_press", {"ratio": 0.25, "min_budget": 8}),
        (
            "build_restoration_press",
            {"budget": 8, "arm": "matched", "trigger_step": 1},
        ),
    ],
)
def test_every_replay_press_carries_the_replay_instrumentation(builder, kwargs):
    """Regression: a press used in replay MUST expose `enabled` and `profile_step`.

    Both are written by the replay loop. `enabled` is what keeps the full-cache
    reference forward uncompressed, and `profile_step` is what the restoration
    trigger reads. A press built on a bare DecodingPress has neither, so the
    reference arm would be silently compressed and the restoration trigger would
    never fire -- every arm degenerating to the null while still producing a
    plausible divergence curve. The unit tests above could not catch this
    because they inject a fake base with `profile_step` already set.
    """
    import kvcache.presses as presses_module

    press = getattr(presses_module, builder)(
        "random", kvpress_module=_fake_kvpress_module(), **kwargs
    )
    assert hasattr(press, "enabled"), f"{builder} lacks the enabled toggle"
    assert hasattr(press, "profile_step"), f"{builder} lacks profile_step"
    assert press.enabled is False, "a replay press must start disabled"
    assert isinstance(press.compressions, list)


# --------------------------------------------------------------------------
# Amendment Q.2: onset-staggered replay. The gate that makes exposure a
# manipulable variable at fixed realised ratio.
#
# Landmine 12 is why these tests are written the way they are: two builders were
# once put on a bare DecodingPress that lacked `enabled`/`profile_step`, so the
# restoration trigger silently never fired and every arm degenerated to the
# null. That invalidated a powered cell. A gate that silently does not gate is
# worse than no gate, so each test below asserts the attribute EXISTS before
# asserting it works.


class _OnsetReplayDecoding:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.__post_init__()

    def __post_init__(self):
        self.parent_calls = 0

    def forward_hook(self, module, input, kwargs, output):
        self.parent_calls += 1
        return "compressed"


class _OnsetKVPress(_FakeKVPress):
    DecodingPress = _OnsetReplayDecoding


def _onset_press():
    return build_replay_press("streamingllm", budget=256, kvpress_module=_OnsetKVPress)


def test_instrumented_press_exposes_the_gate_attributes():
    """Landmine 12: assert the attributes exist, not merely that nothing crashed.

    A bare DecodingPress has no `enabled`, so `press.enabled = True` would just
    create the attribute and the gate would never have run. Absence must fail
    loudly here rather than silently at cell time.
    """
    press = _onset_press()
    for attr in ("enabled", "profile_step", "onset_step", "compression_start_step"):
        assert hasattr(press, attr), (
            f"instrumented press is missing {attr!r}; it was probably built on a "
            "bare DecodingPress (landmine 12) and the onset gate would silently "
            "never fire"
        )
    assert press.compression_start_step is None, "default must be deployed behaviour"


def test_onset_gate_none_is_bit_identical_to_always_on():
    """The default must not perturb any existing cell."""
    press = _onset_press()
    press.enabled = True
    press.profile_step = 0
    assert press.compression_active() is True
    assert press.forward_hook(None, None, None, "full") == "compressed"


def test_onset_gate_suppresses_compression_before_the_start_step():
    press = _onset_press()
    press.enabled = True
    press.compression_start_step = 100
    for step in (0, 1, 99):
        press.profile_step = step
        assert press.compression_active() is False
        assert press.forward_hook(None, None, None, "full") == "full"
    assert press.parent_calls == 0, "compression ran before its onset"


def test_onset_gate_admits_compression_from_the_start_step_onward():
    press = _onset_press()
    press.enabled = True
    press.compression_start_step = 100
    for step in (100, 101, 5000):
        press.profile_step = step
        assert press.compression_active() is True
        assert press.forward_hook(None, None, None, "full") == "compressed"
    assert press.parent_calls == 3


def test_onset_gate_is_subordinate_to_enabled():
    """`enabled` still wins: the full-cache reference must never compress."""
    press = _onset_press()
    press.enabled = False
    press.compression_start_step = 0
    press.profile_step = 10_000
    assert press.forward_hook(None, None, None, "full") == "full"
    assert press.parent_calls == 0


def test_onset_gate_without_profile_step_refuses_rather_than_compressing():
    """A staggered cell that silently ran always-on would be undetectable later."""
    press = _onset_press()
    press.enabled = True
    press.compression_start_step = 100
    press.profile_step = None
    with pytest.raises(RuntimeError, match="onset gate cannot be evaluated"):
        press.compression_active()


def test_onset_gate_does_not_collide_with_measured_onset_step():
    """compression_start_step is an INPUT; onset_step is a MEASURED OUTPUT.

    Conflating them would make the gate overwrite its own measurement, which is
    the bug this separation exists to prevent.
    """

    class _CompressingDecoding(_OnsetReplayDecoding):
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            return keys, values

    class _KV(_FakeKVPress):
        DecodingPress = _CompressingDecoding

    class _Keys:
        shape = (1, 1, 320, 1)

    class _Module:
        layer_idx = 0

    press = build_replay_press("streamingllm", budget=256, kvpress_module=_KV)
    press.enabled = True
    press.compression_start_step = 128
    press.profile_step = 128
    press.compress(_Module(), None, _Keys(), _Keys(), None, {})
    assert press.compression_start_step == 128, "the gate input was overwritten"
    assert press.onset_step == 129, "the measured onset must still be recorded"

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from kvcache.io_schema import ResultRow, validate_row
from kvcache.metrics.tdp import tdp, tdp_percentile
from kvcache.presses import NoOpReplayPress
from kvcache.replay import (
    ReplayProfile,
    build_replay_row,
    exact_tv_from_logits,
    paired_teacher_forced_replay,
    reasoning_boundary_char,
    reasoning_boundary_token,
    teacher_forced_readout_tokens,
    write_replay_profile,
)


class FakePress:
    enabled = False
    profile_step = None
    onset_step = None
    compression_interval = 64
    compressions = []


def test_paired_replay_alignment_and_independent_caches():
    caches = []

    def cache_factory():
        cache = []
        caches.append(cache)
        return cache

    press = FakePress()

    def forward(token_ids, cache):
        cache.extend(token_ids)
        if press.enabled:
            cache.append(99)
        return tuple(cache), cache

    profile = paired_teacher_forced_replay(
        [1, 2],
        [3, 4, 5],
        cache_factory=cache_factory,
        forward=forward,
        press=press,
        tv_fn=lambda full, test: float(full != tuple(x for x in test if x != 99)),
    )
    assert len(profile.estimate) == 3
    assert caches[0] == [1, 2, 3, 4]
    assert caches[1] == [1, 2, 99, 3, 99, 4, 99]
    assert caches[0] is not caches[1]


def test_paired_replay_rejects_replaced_cache():
    cache = []
    with pytest.raises(RuntimeError, match="replaced"):
        paired_teacher_forced_replay(
            [1],
            [2],
            cache_factory=lambda: cache.copy(),
            forward=lambda token_ids, current: (token_ids, current.copy()),
            press=NoOpReplayPress(),
            tv_fn=lambda full, test: 0.0,
        )


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def float(self):
        return self

    def __sub__(self, other):
        return FakeTensor(a - b for a, b in zip(self.values, other.values, strict=True))

    def __rmul__(self, value):
        return FakeTensor(value * item for item in self.values)

    def sum(self):
        return FakeTensor([sum(self.values)])

    def item(self):
        return self.values[0]


def test_exact_tv_identical_and_known_logits(monkeypatch):
    fake_torch = SimpleNamespace(
        softmax=lambda tensor, dim: tensor,
        abs=lambda tensor: FakeTensor(abs(item) for item in tensor.values),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    identical = FakeTensor([0.5, 0.5])
    assert exact_tv_from_logits(identical, identical) == 0.0
    assert exact_tv_from_logits(
        FakeTensor([0.75, 0.25]), FakeTensor([0.25, 0.75])
    ) == pytest.approx(0.5)


def test_no_crossing_tdp_plots_at_endpoint():
    profile = [0.0, 0.01]
    assert tdp(profile, 0.05) is None
    assert tdp_percentile(profile, 0.05) == 1.0


class CharTokenizer:
    def decode(self, token_ids, *, skip_special_tokens):
        return "".join(chr(token) for token in token_ids)


def test_readout_boundary_prefers_think_end_then_answer_marker():
    tokenizer = CharTokenizer()
    trace = "<think>x</think>\\nFinal Answer: 2"
    ids = [ord(char) for char in trace]
    boundary, kind = reasoning_boundary_char(trace)
    assert kind == "think_close"
    assert trace[:boundary].endswith("</think>")
    assert reasoning_boundary_token(tokenizer, ids, trace) == (boundary, "think_close")
    # No </think> but an answer marker -> defined via fallback.
    marker_trace = "reasoning... answer is 2"
    char, mkind = reasoning_boundary_char(marker_trace)
    assert (char, mkind) == (marker_trace.index("answer"), "answer_marker")


def test_readout_boundary_last_think_and_undefined():
    tokenizer = CharTokenizer()
    # Multiple </think>: use the LAST close (answer follows the final one).
    multi = "<think>a</think>more<think>b</think>FINAL"
    char, kind = reasoning_boundary_char(multi)
    assert kind == "think_close"
    assert char == multi.rfind("</think>") + len("</think>")
    # Neither </think> nor a marker -> undefined (excluded + counted, never dropped).
    undefined = "just plain reasoning with no close and no boxed answer"
    assert reasoning_boundary_char(undefined) == (None, "undefined")
    ids = [ord(c) for c in undefined]
    assert reasoning_boundary_token(tokenizer, ids, undefined) == (None, "undefined")


def test_teacher_forced_readout_uses_reference_prefix_then_samples():
    press = FakePress()
    caches = []

    def cache_factory():
        cache = []
        caches.append(cache)
        return cache

    def forward(token_ids, cache):
        assert press.enabled
        cache.extend(token_ids)
        return len(cache), cache

    sampled = iter([9])
    result = teacher_forced_readout_tokens(
        [1, 2],
        [3, 4, 5],
        boundary=2,
        cache_factory=cache_factory,
        forward=forward,
        press=press,
        seed=11,
        temperature=0.6,
        top_p=0.95,
        eos_token_ids={9},
        max_answer_tokens=4,
        sampler=lambda *args, **kwargs: next(sampled),
        generator_factory=lambda seed: object(),
    )
    assert result == [9]
    assert caches == [[1, 2, 3, 4]]


def _reference_row():
    return asdict(
        ResultRow(
            bench="math500",
            pid="p1",
            model="model",
            press="fullcache",
            budget="full",
            compression_ratio=1.0,
            decoding={"temp": 0.6, "top_p": 0.95, "max_new": 16, "seed": 11},
            trace=r"\boxed{2}",
            answer_raw="2",
            answer_canon="2",
            correct=True,
            n_tokens=4,
            truncated=False,
            wallclock_s=1.0,
            runmeta={
                "commit": "abc",
                "dirty": False,
                "config_hash": "config",
                "host": "host",
                "gpu": "gpu",
                "cuda_visible": "0",
            },
            versions={"torch": "x", "transformers": "y", "kvpress": "z"},
            ts="2026-06-14T00:00:00Z",
            has_answer_marker=True,
            finish_reason="marker",
        )
    )


def test_replay_row_and_scalar_profile_are_valid(tmp_path):
    profile = ReplayProfile(
        estimate=(0.0, 0.1),
        lower=(0.0, 0.1),
        upper=(0.0, 0.1),
        reference_token_ids=(1, 2),
        onset_step=None,
        compression_interval=64,
        compression_events=0,
        max_before_prune=None,
        max_after_prune=None,
    )
    profile_path = write_replay_profile(
        tmp_path,
        reference_key=("math500", "p1", "model", "fullcache", "full", 11),
        press_name="streamingllm",
        budget=2048,
        pid_hash="pid",
        profile=profile,
    )
    text = profile_path.read_text()
    assert "logits" not in text
    assert "probabilities" not in text
    row = build_replay_row(
        _reference_row(),
        press_name="streamingllm",
        budget=2048,
        compression_ratio=1.0,
        replay_of="reference",
        tv_profile_path=str(profile_path.relative_to(tmp_path)),
        profile=profile,
    )
    validate_row(row, replay=True)
    assert row["onset_step"] is None


def test_profile_writer_rejects_out_of_range_tv(tmp_path):
    profile = ReplayProfile(
        estimate=(1.1,),
        lower=(1.1,),
        upper=(1.1,),
        reference_token_ids=(1,),
        onset_step=None,
        compression_interval=64,
        compression_events=0,
        max_before_prune=None,
        max_after_prune=None,
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        write_replay_profile(
            tmp_path,
            reference_key=("math500", "p1", "model", "fullcache", "full", 11),
            press_name="fullcache",
            budget="full",
            pid_hash="pid",
            profile=profile,
        )


def test_extended_measures_are_emitted_and_bump_the_schema_version(tmp_path):
    """P.4: JSD/top-1/rank ride along with TV, and v2 marks their presence."""
    pytest.importorskip("torch")
    from dataclasses import replace

    from kvcache.metrics.divergence import measures_from_logits
    from kvcache.replay import ReplayProfile, write_replay_profile

    n = 4
    profile = ReplayProfile(
        estimate=(0.1,) * n,
        lower=(0.1,) * n,
        upper=(0.1,) * n,
        reference_token_ids=(1, 2, 3, 4),
        onset_step=1,
        compression_interval=2,
        compression_events=2,
        max_before_prune=8,
        max_after_prune=4,
        jsd=(0.05,) * n,
        top1_agree=(True, True, False, False),
        rank_displacement=(0, 0, 1, 3),
    )
    path = write_replay_profile(
        tmp_path,
        reference_key=["math500", "p1", "m", "x", "y", 11],
        press_name="random",
        budget=256,
        pid_hash="abc",
        profile=profile,
    )
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "tv-profile-v2"
    assert payload["jsd"] == [0.05] * n
    assert payload["rank_displacement"] == [0, 0, 1, 3]

    # A TV-only profile must still declare v1, so the two cannot be pooled.
    v1 = replace(profile, jsd=(), top1_agree=(), rank_displacement=())
    p1 = write_replay_profile(
        tmp_path,
        reference_key=["math500", "p2", "m", "x", "y", 11],
        press_name="random",
        budget=256,
        pid_hash="def",
        profile=v1,
    )
    assert json.loads(p1.read_text())["schema_version"] == "tv-profile-v1"
    assert measures_from_logits is not None


def test_extended_arrays_must_match_the_tv_length(tmp_path):
    from kvcache.replay import ReplayProfile, write_replay_profile

    bad = ReplayProfile(
        estimate=(0.1, 0.1),
        lower=(0.1, 0.1),
        upper=(0.1, 0.1),
        reference_token_ids=(1, 2),
        onset_step=1,
        compression_interval=2,
        compression_events=1,
        max_before_prune=4,
        max_after_prune=2,
        jsd=(0.05,),
        top1_agree=(True, True),
        rank_displacement=(0, 1),
    )
    with pytest.raises(ValueError, match="extended measure arrays"):
        write_replay_profile(
            tmp_path,
            reference_key=["math500", "p", "m", "x", "y", 11],
            press_name="random",
            budget=256,
            pid_hash="h",
            profile=bad,
        )

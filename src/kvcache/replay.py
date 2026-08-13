"""Paired teacher-forced replay with stepwise total variation.

Full-vocabulary distributions are consumed one step at a time and never stored.
For the top-k fallback, probabilities are queried on the union of both top-k
supports and the remaining mass is collapsed into an unknown tail. Its TV
contribution is between `abs(tail_p-tail_q)/2` and `(tail_p+tail_q)/2`. The
midpoint estimate is returned with those lower and upper bounds.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kvcache.canonicalize import extract_answer
from kvcache.grading import is_equiv
from kvcache.io_schema import validate_row
from kvcache.metrics.tdp import summarize_profile


@dataclass(frozen=True)
class BoundedTV:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ReplayProfile:
    estimate: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    reference_token_ids: tuple[int, ...]
    onset_step: int | None
    compression_interval: int | None
    compression_events: int
    max_before_prune: int | None
    max_after_prune: int | None
    tv_method: str = "exact"
    # Amendment P.4 extended measures, computed in the SAME forward pass as TV.
    # Empty tuples mean a v1-style TV-only profile; the writer emits
    # schema_version accordingly so old and new profiles cannot be silently
    # mixed in an analysis.
    jsd: tuple[float, ...] = ()
    top1_agree: tuple[bool, ...] = ()
    rank_displacement: tuple[int, ...] = ()


_ANSWER_BOUNDARY = re.compile(r"\\boxed\{|(?:final\s+)?answer\s*(?:is|:)", re.I)


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("distributions must have equal support")
    if abs(sum(p) - 1.0) > 1e-6 or abs(sum(q) - 1.0) > 1e-6:
        raise ValueError("distributions must sum to one")
    # TV is mathematically in [0,1]; clamp the tiny float overshoot (e.g. 1.0000001)
    # that otherwise trips the strict [0,1] profile validation and crashes the replay.
    return min(1.0, 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True)))


def topk_tv_bounds(
    p: dict[int, float],
    q: dict[int, float],
    *,
    p_tail: float,
    q_tail: float,
) -> BoundedTV:
    if p.keys() != q.keys():
        raise ValueError("query both distributions on the union top-k support")
    if min(p_tail, q_tail, *p.values(), *q.values()) < 0:
        raise ValueError("probabilities cannot be negative")
    if abs(sum(p.values()) + p_tail - 1.0) > 1e-6:
        raise ValueError("p visible probabilities and tail must sum to one")
    if abs(sum(q.values()) + q_tail - 1.0) > 1e-6:
        raise ValueError("q visible probabilities and tail must sum to one")
    visible = 0.5 * sum(abs(p[i] - q[i]) for i in p)
    tail_lower = 0.5 * abs(p_tail - q_tail)
    tail_upper = 0.5 * (p_tail + q_tail)
    # All bounded by 1 (TV in [0,1]); clamp float overshoot so the profile validation
    # downstream never crashes the replay on a 1.0000001.
    lower = min(1.0, visible + tail_lower)
    upper = min(1.0, visible + tail_upper)
    return BoundedTV(estimate=min(1.0, (lower + upper) / 2), lower=lower, upper=upper)


def replay_tv_profile(
    full_steps: Iterable[Sequence[float]],
    compressed_steps: Iterable[Sequence[float]],
) -> list[float]:
    """Compute and retain only one scalar TV per paired replay step."""
    profile = []
    for full, compressed in zip(full_steps, compressed_steps, strict=True):
        profile.append(total_variation(full, compressed))
    return profile


def exact_tv_from_logits(full_logits: Any, test_logits: Any) -> float:
    """Compute exact TV from one live pair of logits without retaining vectors."""
    import torch

    full = torch.softmax(full_logits.float(), dim=-1)
    test = torch.softmax(test_logits.float(), dim=-1)
    return min(1.0, float(0.5 * torch.abs(full - test).sum().item()))


@contextmanager
def press_enabled(press: Any, enabled: bool):
    previous = bool(getattr(press, "enabled", False))
    press.enabled = enabled
    try:
        yield
    finally:
        press.enabled = previous


ForwardFn = Callable[[Sequence[int], Any], tuple[Any, Any]]


class TransformersReplayBackend:
    """Stateful adapter that preserves absolute positions after cache pruning."""

    def __init__(self, model: Any):
        self.model = model
        self._positions: dict[int, int] = {}

    def cache_factory(self) -> Any:
        from transformers import DynamicCache

        cache = DynamicCache()
        self._positions[id(cache)] = 0
        return cache

    def forward(self, token_ids: Sequence[int], cache: Any) -> tuple[Any, Any]:
        import torch

        if id(cache) not in self._positions:
            raise ValueError("unknown replay cache")
        start = self._positions[id(cache)]
        input_ids = torch.tensor(
            [list(token_ids)], dtype=torch.long, device=self.model.device
        )
        cache_position = torch.arange(
            start,
            start + len(token_ids),
            dtype=torch.long,
            device=self.model.device,
        )
        output = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        self._positions[id(cache)] += len(token_ids)
        return output.logits[0, -1], output.past_key_values


def paired_teacher_forced_replay(
    prompt_token_ids: Sequence[int],
    trace_token_ids: Sequence[int],
    *,
    cache_factory: Callable[[], Any],
    forward: ForwardFn,
    press: Any,
    tv_fn: Callable[[Any, Any], float] = exact_tv_from_logits,
    extended_measures: Callable[[Any, Any], dict[str, Any]] | None = None,
) -> ReplayProfile:
    """Replay one fixed trace through independent full and test cache states.

    ``extended_measures`` (Amendment P.4) computes JSD, top-1 agreement and rank
    displacement from the SAME logits pair as TV, in the same forward pass --
    the measures are free once the pair exists, and a second pass would not be
    guaranteed to reproduce the same cache states. When it is supplied the TV it
    returns is used, so TV and the new measures cannot come from different
    computations.
    """
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids cannot be empty")
    if not trace_token_ids:
        raise ValueError("trace_token_ids cannot be empty")
    full_cache = cache_factory()
    test_cache = cache_factory()
    if full_cache is test_cache:
        raise ValueError("replay requires independent cache objects")

    press.profile_step = 0
    with press_enabled(press, False):
        full_logits, returned_full = forward(prompt_token_ids, full_cache)
    with press_enabled(press, True):
        test_logits, returned_test = forward(prompt_token_ids, test_cache)
    if returned_full is not full_cache or returned_test is not test_cache:
        raise RuntimeError("backend replaced a replay cache object")

    profile: list[float] = []
    extended: list[dict[str, Any]] = []
    for step, token in enumerate(trace_token_ids):
        if extended_measures is not None:
            measures = extended_measures(full_logits, test_logits)
            extended.append(measures)
            profile.append(float(measures["tv"]))
        else:
            profile.append(tv_fn(full_logits, test_logits))
        if step == len(trace_token_ids) - 1:
            break
        press.profile_step = step + 1
        with press_enabled(press, False):
            full_logits, returned_full = forward([int(token)], full_cache)
        with press_enabled(press, True):
            test_logits, returned_test = forward([int(token)], test_cache)
        if returned_full is not full_cache or returned_test is not test_cache:
            raise RuntimeError("backend replaced a replay cache object")

    events = list(getattr(press, "compressions", []))
    return ReplayProfile(
        estimate=tuple(profile),
        lower=tuple(profile),
        upper=tuple(profile),
        jsd=tuple(float(m["jsd"]) for m in extended),
        top1_agree=tuple(bool(m["top1_agree"]) for m in extended),
        rank_displacement=tuple(int(m["rank_displacement"]) for m in extended),
        reference_token_ids=tuple(int(token) for token in trace_token_ids),
        onset_step=getattr(press, "onset_step", None),
        compression_interval=getattr(press, "compression_interval", None),
        compression_events=len(events),
        max_before_prune=max((item["before"] for item in events), default=None),
        max_after_prune=max((item["after"] for item in events), default=None),
    )


def reasoning_boundary_char(trace: str) -> tuple[int | None, str]:
    """Char boundary after reasoning + a deterministic, logged ``kind``.

    Rule (per-row logged, for the powered set):
    - ``think_close``: the LAST ``</think>`` closes reasoning. Using the last close
      is robust to multiple ``</think>`` and to ``\\boxed{}`` emitted mid-reasoning
      (the answer always follows the final close).
    - ``answer_marker``: no ``</think>`` at all, but a recognized answer marker — fall
      back to its start (covers non-thinking traces).
    - ``undefined``: neither — there is no clean reasoning/answer split, so the readout
      estimand is ill-defined. The row is EXCLUDED from readout-accuracy and COUNTED
      separately (never silently dropped or mis-segmented).
    """
    last_think = trace.rfind("</think>")
    if last_think >= 0:
        return last_think + len("</think>"), "think_close"
    marker = _ANSWER_BOUNDARY.search(trace)
    if marker:
        return marker.start(), "answer_marker"
    return None, "undefined"


def token_boundary_for_text(
    tokenizer: Any,
    trace_token_ids: Sequence[int],
    trace: str,
    char_boundary: int,
) -> int:
    """Map a decoded-text boundary back to the smallest exact token prefix."""
    if not 0 <= char_boundary <= len(trace):
        raise ValueError("character boundary outside trace")
    for stop in range(len(trace_token_ids) + 1):
        decoded = tokenizer.decode(
            list(trace_token_ids[:stop]), skip_special_tokens=True
        )
        if len(decoded) >= char_boundary:
            if not trace.startswith(decoded):
                raise ValueError("token prefix does not match stored trace")
            return stop
    if char_boundary == len(trace):
        return len(trace_token_ids)
    raise ValueError("could not map text boundary to token IDs")


def reasoning_boundary_token(
    tokenizer: Any,
    trace_token_ids: Sequence[int],
    trace: str,
) -> tuple[int | None, str]:
    """Return ``(token_boundary, kind)``; ``token_boundary`` is None when undefined."""
    char_boundary, kind = reasoning_boundary_char(trace)
    if char_boundary is None:
        return None, kind
    return (
        token_boundary_for_text(tokenizer, trace_token_ids, trace, char_boundary),
        kind,
    )


def sample_top_p(
    logits: Any,
    *,
    temperature: float,
    top_p: float,
    generator: Any,
) -> int:
    """Sample one token with temperature and nucleus filtering."""
    import torch

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    scores = logits.float() / temperature
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    probabilities = torch.softmax(sorted_scores, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative - probabilities > top_p
    sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
    filtered = torch.softmax(sorted_scores, dim=-1)
    sampled = torch.multinomial(filtered, 1, generator=generator)
    return int(sorted_indices.gather(-1, sampled).item())


def teacher_forced_readout_tokens(
    prompt_token_ids: Sequence[int],
    trace_token_ids: Sequence[int],
    *,
    boundary: int,
    cache_factory: Callable[[], Any],
    forward: ForwardFn,
    press: Any,
    seed: int,
    temperature: float,
    top_p: float,
    eos_token_ids: set[int],
    max_answer_tokens: int,
    sampler: Callable[..., int] = sample_top_p,
    generator_factory: Callable[[int], Any] | None = None,
) -> list[int]:
    """Teacher-force reference reasoning, then sample an answer under compression."""
    if not 0 <= boundary <= len(trace_token_ids):
        raise ValueError("boundary outside reference token sequence")
    if max_answer_tokens <= 0:
        raise ValueError("max_answer_tokens must be positive")
    cache = cache_factory()
    press.profile_step = 0
    with press_enabled(press, True):
        logits, returned_cache = forward(prompt_token_ids, cache)
    if returned_cache is not cache:
        raise RuntimeError("backend replaced the readout cache object")

    for step, token in enumerate(trace_token_ids[:boundary]):
        press.profile_step = step + 1
        with press_enabled(press, True):
            logits, returned_cache = forward([int(token)], cache)
        if returned_cache is not cache:
            raise RuntimeError("backend replaced the readout cache object")

    if generator_factory is None:
        import torch

        device = getattr(logits, "device", "cpu")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = generator_factory(seed)

    generated: list[int] = []
    for offset in range(max_answer_tokens):
        token = sampler(
            logits,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        generated.append(token)
        if token in eos_token_ids:
            break
        press.profile_step = boundary + offset + 1
        with press_enabled(press, True):
            logits, returned_cache = forward([token], cache)
        if returned_cache is not cache:
            raise RuntimeError("backend replaced the readout cache object")
    return generated


def score_readout_answer(gold_answer: str, generated_text: str) -> bool:
    answer = extract_answer(generated_text)
    return bool(answer) and is_equiv(gold_answer, answer)


def is_profile_file(path: str | Path) -> bool:
    """True for a stored TV profile, False for a sidecar written beside it.

    Profiles are ``<pid_hash>.json``. Sidecars added alongside them carry a
    compound suffix -- ``<pid_hash>.restoration.json``,
    ``<pid_hash>.ratio_trace.json`` -- so a bare ``rglob("*.json")`` over a run's
    profile tree picks them up too. That silently doubled a progress count and
    would have made every profile reader fail on a payload with no ``estimate``
    key. One predicate, used by every reader, so the next sidecar cannot
    reintroduce it.
    """
    return Path(path).suffixes[-2:] == [".json"]


def write_replay_profile(
    root: str | Path,
    *,
    reference_key: Sequence[Any],
    press_name: str,
    budget: int | str,
    pid_hash: str,
    profile: ReplayProfile,
    overwrite: bool = False,
) -> Path:
    lengths = {
        len(profile.estimate),
        len(profile.lower),
        len(profile.upper),
        len(profile.reference_token_ids),
    }
    if len(lengths) != 1:
        raise ValueError("replay profile arrays must have equal length")
    extended = bool(profile.jsd or profile.top1_agree or profile.rank_displacement)
    if extended:
        extra = {
            len(profile.jsd),
            len(profile.top1_agree),
            len(profile.rank_displacement),
        }
        if extra != lengths:
            raise ValueError("extended measure arrays must match the TV profile length")
        if any(not 0.0 <= value <= 1.0 for value in profile.jsd):
            raise ValueError("JSD values must be in [0, 1] (base-2)")
        if any(value < 0 for value in profile.rank_displacement):
            raise ValueError("rank displacement cannot be negative")
    if any(
        not 0.0 <= value <= 1.0
        for values in (profile.estimate, profile.lower, profile.upper)
        for value in values
    ):
        raise ValueError("TV profile values must be in [0, 1]")
    if any(
        not lower <= estimate <= upper
        for lower, estimate, upper in zip(
            profile.lower, profile.estimate, profile.upper, strict=True
        )
    ):
        raise ValueError("TV estimate must lie within its bounds")
    bench, _, model, _, _, seed = reference_key
    path = (
        Path(root)
        / "profiles"
        / str(bench)
        / str(model)
        / f"{press_name}-{budget}-s{seed}"
        / f"{pid_hash}.json"
    )
    if path.exists() and not overwrite:
        raise FileExistsError(f"replay profile already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # v2 carries the Amendment P.4 measures; v1 is TV-only. Versioning is
        # load-bearing: an analysis that pooled the two would silently average
        # profiles that do not contain the same fields.
        "schema_version": "tv-profile-v2" if extended else "tv-profile-v1",
        "reference_key": list(reference_key),
        "condition": {"press": press_name, "budget": budget},
        **asdict(profile),
        "cache_stats": {
            "events": profile.compression_events,
            "max_before_prune": profile.max_before_prune,
            "max_after_prune": profile.max_after_prune,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_replay_row(
    reference_row: dict[str, Any],
    *,
    press_name: str,
    budget: int | str,
    compression_ratio: float,
    replay_of: str,
    tv_profile_path: str,
    profile: ReplayProfile,
    runmeta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = json.loads(json.dumps(reference_row))
    row["press"] = press_name
    row["budget"] = budget
    row["compression_ratio"] = compression_ratio
    if runmeta is not None:
        row["runmeta"] = {
            "commit": runmeta["commit"],
            "dirty": runmeta["dirty"],
            "config_hash": runmeta["config_hash"],
            "host": runmeta["host"],
            "gpu": runmeta["gpu"],
            "cuda_visible": runmeta["cuda_visible"],
            "sync_method": runmeta.get("sync_method", "git"),
            "source_tree_hash": runmeta.get("source_tree_hash", ""),
            "status_hash": runmeta.get("status_hash", ""),
            "synced_at": runmeta.get("synced_at", ""),
        }
        row["ts"] = str(runmeta["timestamp"])
    summary = summarize_profile(profile.estimate)
    row.update(
        {
            "replay_of": replay_of,
            "tv_profile_path": tv_profile_path,
            "auc_tv": summary["auc_tv"],
            "frac_above": summary["frac_above"],
            "tdp": summary["tdp"],
            "onset_step": profile.onset_step,
        }
    )
    validate_row(row, replay=True)
    return row

"""Frozen-ID compressed-run and readout paths for the registered GSM8K
replication (M5-B), CPU only. The design is pending; these tests pin the
paths, not a design:

* the template configs resolve to the 200 frozen confirmation IDs, disjoint
  from the timing calibration split, and every departure refuses;
* each press family resolves to its own unchanged builder and gate;
* real execution refuses while held or while the design is pending;
* the compressed runner publishes recoverable pairs (a stop after the result
  write resumes to exactly one pair per key, no regeneration) and its read
  scores stored traces with the registered GSM8K scorer;
* the readout runner drives ``run_replay``'s unchanged readout loop over a
  frozen-ID full-cache bank (stubbed model and kernels), and its read covers
  every expected key, excludes undefined boundaries, and scores a capped
  answer span as truncated -- never using the record's MATH-graded field.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_bank_manifest as rbm  # noqa: E402
import run_grid_manifest as rgm  # noqa: E402
import run_readout_manifest as rrm  # noqa: E402
import run_replay  # noqa: E402

from kvcache import m5_publish as mp  # noqa: E402
from kvcache import manifest_grid as mg  # noqa: E402
from kvcache.datasets import Problem  # noqa: E402
from kvcache.manifest_reads import score_grid, score_readouts  # noqa: E402
from kvcache.task_scoring import grader_sha256  # noqa: E402

GRID = ROOT / "configs/experiments/grid_gsm8k_template_m5b.yaml"
READOUT = ROOT / "configs/experiments/readout_gsm8k_template_m5b.yaml"
TIMING = ROOT / "configs/experiments/bank_gsm8k_timing_m5.yaml"
GSM = json.loads((ROOT / "configs/meeting5/problem_manifests.json").read_text())[
    "gsm8k"
]
CONF = GSM["confirmation"]
GOLDS = {p: str(10 + i) for i, p in enumerate(CONF)}
POOL = [
    Problem(p, f"q {p}", GOLDS.get(p, "0"), {})
    for p in GSM["calibration"] + GSM["confirmation"] + GSM["reserve"]
]


class Interrupt(Exception):
    pass


class _Tok:
    eos_token_id = 0

    def __init__(self):
        self.texts: dict[tuple[int, ...], str] = {}

    def decode(self, ids, skip_special_tokens=False):
        return self.texts[tuple(ids)]


def _cfg(path: Path, tmp_path: Path, **over) -> Path:
    cfg = {**yaml.safe_load(path.read_text()), **over}
    out = tmp_path / path.name
    out.write_text(yaml.safe_dump(cfg))
    return out


def _runmeta(real):
    return lambda *a, **k: {**real(*a, **k), "source_tree_hash": "tree-test"}


def _generation(tok, pid, seed, n, trace, salt):
    i = CONF.index(pid) if pid in CONF else 0
    base = 1_000_000 * (salt + 1) + 10_000 * seed + 3_000 * i
    prompt_ids = tuple(base + j for j in range(40 + i % 7))
    trace_ids = tuple(base + 1_000 + j for j in range(n))
    prompt = f"prompt {pid}"
    tok.texts[prompt_ids], tok.texts[trace_ids] = prompt, trace
    return SimpleNamespace(
        prompt=prompt,
        trace=trace,
        answer_raw="x",
        answer_canon="x",
        correct=False,
        n_tokens=n,
        truncated=False,
        wallclock_s=0.01 * n,
        hit_token_cap=False,
        has_answer_marker=True,
        finish_reason="marker",
        prompt_token_ids=prompt_ids,
        trace_token_ids=trace_ids,
    )


# ---------------------------------------------------------------- design


def test_templates_resolve_to_the_frozen_confirmation_split():
    g = mg.resolve_grid_design(yaml.safe_load(GRID.read_text()), kind="grid")
    r = mg.resolve_grid_design(yaml.safe_load(READOUT.read_text()), kind="readout")
    for d in (g, r):
        assert d["pids"] == CONF and len(d["pids"]) == 200
        assert d["split"] == "confirmation" and d["design_status"] == "pending"
        assert d["revision"] == GSM["revision"] and d["family"] == "pp"
        assert not set(d["pids"]) & set(GSM["calibration"])  # timing split
    keys = mg.design_keys(g)
    assert len(keys) == len(set(keys)) == 800
    assert keys[0] == ("random_pp", "256", CONF[0], 11)
    assert r["max_answer_tokens"] == 2048


def _base(path):
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize(
    "path, kind, over, match",
    [
        (GRID, "grid", {"subset": {"gsm8k": 200}}, "no subset"),
        (GRID, "grid", {"estimated_generation_count": 799}, "200 problems"),
        (GRID, "grid", {"presses": ["fullcache"]}, "compressed presses only"),
        (GRID, "grid", {"budgets": ["256"]}, "positive integer"),
        (GRID, "grid", {"benchmarks": ["math500"]}, "registered runners"),
        (GRID, "grid", {"design_status": None}, "design_status"),
        (GRID, "grid", {"store_token_ids": False}, "pairs"),
        (GRID, "grid", {"manifest": {"path": "x"}}, "path and split"),
        (READOUT, "readout", {"reference_run_name": ""}, "reference_run_name"),
        (READOUT, "readout", {"restoration": {"arms": ["none"]}}, "not registered"),
        (READOUT, "readout", {"mode": "self_replay"}, "mode: readout"),
    ],
)
def test_design_refusals(path, kind, over, match):
    with pytest.raises(ValueError, match=match):
        mg.resolve_grid_design({**_base(path), **over}, kind=kind)


def test_one_family_per_config():
    cfg = {**_base(GRID), "protected_selector": True}
    with pytest.raises(ValueError, match="one press family"):
        mg.family_of(cfg)


def test_families_use_the_registered_builders_and_gates():
    from kvcache import presses
    from kvcache.presses import prompt_protect, protected_selectors
    from kvcache.presses import rkv_prompt_protect as rkvpp

    fam = mg.FAMILIES
    assert mg.builder(fam["base"]) is presses.build_press
    assert mg.builder(fam["pp"]) is prompt_protect.build_prompt_protected_press
    assert (
        mg.builder(fam["protsel"]) is protected_selectors.build_protected_selector_press
    )
    assert mg.builder(fam["rkv_pp"]) is rkvpp.build_rkv_prompt_protected_press
    assert mg.needs_eager(fam["base"], "h2o") is True
    assert mg.needs_eager(fam["pp"], "random_pp") is False
    assert mg.needs_eager(fam["protsel"], "h2o_style_pp_v1") is (
        protected_selectors.needs_eager_attention("h2o_style_pp_v1")
    )


def test_family_gates_refuse_real_execution_of_held_or_uncertified_cells(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("KVCACHE_RKV_SMOKE_MARKER", raising=False)
    monkeypatch.setenv("KVCACHE_BASE_RUNS_ROOT", str(tmp_path))
    base = {**_base(GRID), "prompt_protected": False, "presses": ["rkv"]}
    mg.run_gate(mg.FAMILIES["base"], base, dry_run=True, runs_root=None, replay=0)
    with pytest.raises(SystemExit, match="REFUSED: rkv"):
        mg.run_gate(mg.FAMILIES["base"], base, dry_run=False, runs_root=None, replay=0)
    sel = {
        **_base(GRID),
        "prompt_protected": False,
        "protected_selector": True,
        "presses": ["h2o_style_pp_v1"],
    }
    with pytest.raises(SystemExit, match="HELD"):
        mg.run_gate(
            mg.FAMILIES["protsel"], sel, dry_run=False, runs_root=tmp_path, replay=0
        )


@pytest.mark.parametrize(
    "script, path", [(rgm, GRID), (rrm, READOUT)], ids=["grid", "readout"]
)
@pytest.mark.parametrize(
    "over, match",
    [({}, "HELD"), ({"held": False}, "PENDING")],
    ids=["held", "pending"],
)
def test_real_execution_refuses_while_held_or_pending(
    tmp_path, monkeypatch, script, path, over, match
):
    monkeypatch.setenv("KVCACHE_REMOTE", "1")
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(tmp_path / "out"))
    cfg = _cfg(path, tmp_path, **over)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--i-understand"])
    with pytest.raises(SystemExit, match=match):
        script.main()
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------- compressed


def _grid_setup(tmp_path, monkeypatch):
    out = tmp_path / "runs" / "grid-gsm8k"
    out.mkdir(parents=True)
    cfg = _cfg(GRID, tmp_path, held=False, design_status="final", scale_authorized=True)
    tok, n = _Tok(), {"generations": 0}

    def generate(model, tokenizer, problem, *a, seed, press, **k):
        n["generations"] += 1
        gold = int(GOLDS[problem.pid])
        value = gold + (1 if press.name == "streamingllm_pp" and seed == 23 else 0)
        trace = f"steps</think> so \\boxed{{{value}}}"
        return _generation(tok, problem.pid, seed, 60, trace, salt=n["generations"])

    def build(name, **kw):
        report = {"base": name, "budget": kw["budget"], "budget_relaxed": False}
        return SimpleNamespace(
            name=name, protect=SimpleNamespace(report=lambda: report)
        )

    monkeypatch.setattr(rgm, "builder", lambda family: build)
    monkeypatch.setattr(rgm, "generate_compressed", generate)
    monkeypatch.setattr(rgm, "load_model_and_tokenizer", lambda *a, **k: (None, tok))
    monkeypatch.setattr(rgm, "load_benchmark", lambda *a, **k: POOL)
    monkeypatch.setattr(rgm, "capture_runmeta", _runmeta(rgm.capture_runmeta))
    monkeypatch.setenv("KVCACHE_REMOTE", "1")
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(out))
    return cfg, out, n


def _run(monkeypatch, module, cfg, *extra):
    argv = ["x", "--config", str(cfg), "--i-understand", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return module.main()


def test_compressed_runner_resumes_a_stop_after_the_result_write(tmp_path, monkeypatch):
    cfg, out, n = _grid_setup(tmp_path, monkeypatch)
    real, state = mp._publish_one, {"fired": False}

    def stop(path, rec):
        if not state["fired"] and "token_ids" in path.parts:
            state["fired"] = True
            raise Interrupt
        return real(path, rec)

    monkeypatch.setattr(mp, "_publish_one", stop)
    with pytest.raises(Interrupt):
        _run(monkeypatch, rgm, cfg)
    assert mp.BENCH == "math500"
    assert _run(monkeypatch, rgm, cfg, "--resume") == 0
    assert n["generations"] == 800  # the interrupted pair was finished, not redone

    design = mg.resolve_grid_design(yaml.safe_load(cfg.read_text()), kind="grid")
    read = score_grid(out, design, GOLDS)
    assert read["validated_pairs"] == 800
    assert read["grader_sha256"] == grader_sha256()
    assert read["cells"]["random_pp-256"]["accuracy"] == 1.0
    assert read["cells"]["streamingllm_pp-256"]["accuracy"] == pytest.approx(0.5)
    rows = [
        json.loads(x)
        for p in sorted((out / "gsm8k").rglob("*.jsonl"))
        for x in p.read_text().splitlines()
    ]
    assert all(r["correct"] is False for r in rows)  # the field the read ignores
    r0 = rows[0]
    assert r0["press_family"] == "pp" and r0["protection"]["budget"] == 256
    assert r0["manifest_split"] == {
        "split": "confirmation",
        "revision": GSM["revision"],
    }
    n_prompt = 40 + CONF.index(r0["pid"]) % 7
    assert r0["compression_ratio"] == pytest.approx(min(1.0, 256 / (n_prompt + 60)))


def test_grid_read_refuses_a_missing_pair(tmp_path, monkeypatch):
    cfg, out, _ = _grid_setup(tmp_path, monkeypatch)
    assert _run(monkeypatch, rgm, cfg) == 0
    side = sorted((out / "token_ids").rglob("*.jsonl"))[0]
    side.write_text("".join(x + "\n" for x in side.read_text().splitlines()[1:]))
    design = mg.resolve_grid_design(yaml.safe_load(cfg.read_text()), kind="grid")
    with pytest.raises(ValueError, match="lack sidecars"):
        score_grid(out, design, GOLDS)


# ---------------------------------------------------------------- readout


def _reference_bank(tmp_path, monkeypatch, base):
    """A frozen-ID full-cache confirmation bank made by the real bank runner."""
    bank_cfg = _cfg(
        TIMING,
        tmp_path,
        manifest={
            "path": "configs/meeting5/problem_manifests.json",
            "split": "confirmation",
        },
        estimated_generation_count=400,
        held=False,
        scale_authorized=True,
    )
    tok = _Tok()
    count = {"n": 0}

    def generate(model, tokenizer, problem, *a, seed, **k):
        count["n"] += 1
        trace = f"reasoning for {problem.pid}</think> \\boxed{{{GOLDS[problem.pid]}}}"
        return _generation(tok, problem.pid, seed, 30, trace, salt=10_000 + count["n"])

    monkeypatch.setattr(rbm, "load_model_and_tokenizer", lambda *a, **k: (None, tok))
    monkeypatch.setattr(rbm, "generate_full_cache", generate)
    monkeypatch.setattr(rbm, "load_benchmark", lambda *a, **k: POOL)
    monkeypatch.setattr(rbm, "capture_runmeta", _runmeta(rbm.capture_runmeta))
    bank = base / "bank-gsm8k-confirmation-m5b"
    bank.mkdir(parents=True)
    monkeypatch.setenv("KVCACHE_REMOTE", "1")
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(bank))
    assert _run(monkeypatch, rbm, bank_cfg) == 0
    # a quarantined record must never be read as a reference row
    q = bank / mp.QUARANTINE / "gsm8k" / "x.jsonl"
    q.parent.mkdir(parents=True)
    q.write_text(json.dumps({"reason": "orphan", "line": "{}"}) + "\n")
    return tok


def test_readout_runner_drives_run_replay_over_the_frozen_split(tmp_path, monkeypatch):
    base = tmp_path / "runs"
    tok = _reference_bank(tmp_path, monkeypatch, base)
    out = base / "readout-gsm8k"
    out.mkdir()
    cfg = _cfg(
        READOUT, tmp_path, held=False, design_status="final", scale_authorized=True
    )
    undefined_pid, capped_pid = CONF[0], CONF[1]
    calls = {"loaded": None}

    def boundary(tokenizer, trace_ids, trace):
        undefined = trace.startswith(f"reasoning for {undefined_pid}</think>")
        return (None, "no_think_close") if undefined else (5, "think")

    def readout(prompt_ids, trace_ids, *, max_answer_tokens, press, seed, **k):
        pid = tok.texts[tuple(prompt_ids)].removeprefix("prompt ")
        cap = pid == capped_pid and press.name == "random_pp"
        ids = tuple(
            900_000_000 + 7_000 * CONF.index(pid) + 10 * seed + j
            for j in range(max_answer_tokens if cap else 3)
        )
        value = int(GOLDS[pid]) + (1 if press.name == "streamingllm_pp" else 0)
        tok.texts[ids] = f"\\boxed{{{value}}}"
        return list(ids)

    real_loader = rrm.frozen_loader

    def counting_loader(design):
        load = real_loader(design)

        def wrapped(*a, **k):
            got = load(*a, **k)
            calls["loaded"] = [p.pid for p in got]
            return got

        return wrapped

    import run_replay_pp

    monkeypatch.setattr(rrm, "frozen_loader", counting_loader)
    monkeypatch.setattr(rrm.datasets, "load_benchmark", lambda *a, **k: POOL)

    class _Press(SimpleNamespace):
        def __call__(self, model):
            return contextlib.nullcontext()

    monkeypatch.setattr(
        run_replay_pp, "build_replay_press_pp", lambda name, **kw: _Press(name=name)
    )
    monkeypatch.setattr(
        run_replay, "load_model_and_tokenizer", lambda *a, **k: (None, tok)
    )
    monkeypatch.setattr(
        run_replay,
        "TransformersReplayBackend",
        lambda model: SimpleNamespace(cache_factory=None, forward=None),
    )
    monkeypatch.setattr(run_replay, "reasoning_boundary_token", boundary)
    monkeypatch.setattr(run_replay, "teacher_forced_readout_tokens", readout)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(inference_mode=contextlib.nullcontext)
    )
    for name in ("load_benchmark", "_load_jsonl", "build_replay_press"):
        monkeypatch.setattr(run_replay, name, getattr(run_replay, name))
    monkeypatch.setattr(
        run_replay,
        "press_needs_eager_attention",
        run_replay.press_needs_eager_attention,
    )
    monkeypatch.setenv("KVCACHE_BASE_RUNS_ROOT", str(base))
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(out))
    assert _run(monkeypatch, rrm, cfg) == 0
    assert calls["loaded"] == CONF  # exactly the frozen IDs, in manifest order

    design = mg.resolve_grid_design(yaml.safe_load(cfg.read_text()), kind="readout")
    read = score_readouts(out, design, GOLDS)
    assert read["expected"] == 800
    assert read["undefined"] == 4  # one pid x 2 seeds x 2 presses, counted
    assert read["defined"] == 796
    rp = read["cells"]["random_pp-256"]
    # random_pp: all correct except the capped span on capped_pid (2 seeds)
    assert rp["n"] == 398 and rp["accuracy"] == pytest.approx(396 / 398)
    assert rp["truncation_rate"] == pytest.approx(2 / 398)
    assert read["cells"]["streamingllm_pp-256"]["accuracy"] == 0.0
    assert read["grader_sha256"] == grader_sha256()


def test_readout_refuses_an_incomplete_reference_bank(tmp_path, monkeypatch):
    base = tmp_path / "runs"
    _reference_bank(tmp_path, monkeypatch, base)
    bank = base / "bank-gsm8k-confirmation-m5b"
    side = sorted((bank / "token_ids").rglob("*.jsonl"))[0]
    side.write_text("".join(x + "\n" for x in side.read_text().splitlines()[1:]))
    cfg = _cfg(
        READOUT, tmp_path, held=False, design_status="final", scale_authorized=True
    )
    monkeypatch.setenv("KVCACHE_BASE_RUNS_ROOT", str(base))
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(base / "readout-x"))
    for name in ("load_benchmark", "_load_jsonl", "build_replay_press"):
        monkeypatch.setattr(run_replay, name, getattr(run_replay, name))
    with pytest.raises(SystemExit, match="REFUSED: reference bank: .*lack sidecars"):
        _run(monkeypatch, rrm, cfg)


def test_readout_loaders_select_frozen_ids_and_skip_quarantine(tmp_path, monkeypatch):
    design = mg.resolve_grid_design(_base(READOUT), kind="readout")
    monkeypatch.setattr(rrm.datasets, "load_benchmark", lambda *a, **k: POOL)
    got = rrm.frozen_loader(design)("gsm8k", {}, subset={"count": 5})
    assert [p.pid for p in got] == CONF
    with pytest.raises(SystemExit, match="REFUSED"):
        rrm.frozen_loader(design)("math500", {})
    (tmp_path / "gsm8k" / "m").mkdir(parents=True)
    (tmp_path / "gsm8k" / "m" / "a.jsonl").write_text('{"pid": "p"}\n')
    for sub in (mp.QUARANTINE, "token_ids", mp.JOURNAL):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "z.jsonl").write_text('{"pid": "q"}\n')
    assert rrm.live_rows_loader("gsm8k")(tmp_path) == [{"pid": "p"}]


def test_readout_factories_are_the_family_wrappers():
    import run_replay_pp
    import run_replay_protected
    import run_replay_rkv_pp

    f = mg.FAMILIES
    assert rrm.replay_factory(f["pp"]) is run_replay_pp.build_replay_press_pp
    assert (
        rrm.replay_factory(f["protsel"])
        is run_replay_protected.build_replay_press_protected
    )
    assert (
        rrm.replay_factory(f["rkv_pp"]) is run_replay_rkv_pp.build_replay_press_rkv_pp
    )
    assert rrm.replay_factory(f["base"]) is run_replay.build_replay_press

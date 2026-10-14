"""M5 production runner (review of 6b409001, R2) on CPU stubs for the model,
generation and press: identity and prompt-input hashes in every row and
sidecar, resume refused into foreign rows or after a config change (before the
model loads), the smoke gate enforced, and no write on an input mismatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_grid_m5 as m5  # noqa: E402
import smoke_fixtures as sfx  # noqa: E402

from kvcache.datasets import Problem  # noqa: E402
from kvcache.provenance import inputs_sha256  # noqa: E402
from kvcache.read_validation import frozen_pids  # noqa: E402

GRID = "configs/experiments/grid_math500_rkvpp256_m5.yaml"
SMOKE = "configs/experiments/rkvpp_smoke_m5.yaml"
PIDS = frozen_pids(100, root=ROOT)


def _prompt_ids(pid):
    return tuple(1000 + (sum(map(ord, pid)) + i) % 97 for i in range(12))


class _Press:
    def __init__(self, budget):
        self.protect = SimpleNamespace(
            report=lambda: {
                "target": budget,
                "budget_relaxed": False,
                "prompt_len": 12,
                "n_prunes": 3,
                "n_layers_pruned": 28,
            }
        )


def _setup(tmp_path, monkeypatch, *, cfg_over=None, bad_pid=None, smoke=True):
    base, out = tmp_path / "base", tmp_path / "base" / "grid-rkvpp256-m5"
    out.mkdir(parents=True)
    smoke_cfg = yaml.safe_load((ROOT / SMOKE).read_text())
    if smoke:
        sfx.reference_bank(base, smoke_cfg)
        sfx.measured_marker(base, smoke_cfg, "rkvpp")
    cfg = yaml.safe_load((ROOT / GRID).read_text())
    cfg.update({"seeds": [11], **(cfg_over or {})})
    cfg_path = tmp_path / "grid.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    calls = {"model_loads": 0}

    def load_model(*a, **k):
        calls["model_loads"] += 1
        c = SimpleNamespace(
            num_attention_heads=28,
            num_key_value_heads=4,
            num_hidden_layers=28,
            hidden_size=3584,
            head_dim=128,
        )
        return SimpleNamespace(config=c, dtype=SimpleNamespace(itemsize=2)), None

    def generate(model, tok, problem, *a, **k):
        ids = _prompt_ids(problem.pid)
        if problem.pid == bad_pid:
            ids = ids[:-1]
        return SimpleNamespace(
            prompt="p",
            trace="t \\boxed{1}",
            answer_raw="1",
            answer_canon="1",
            correct=False,
            n_tokens=50,
            truncated=False,
            wallclock_s=0.1,
            hit_token_cap=False,
            has_answer_marker=True,
            finish_reason="marker",
            prompt_token_ids=ids,
            trace_token_ids=tuple(range(50)),
        )

    def sidecar(*, result_row, **kw):
        return {
            k: result_row[k] for k in ("bench", "pid", "model", "press", "budget")
        } | {"seed": result_row["decoding"]["seed"]}

    manifest = {"prompts": {p: inputs_sha256(_prompt_ids(p)) for p in PIDS}}
    monkeypatch.setattr(m5, "load_model_and_tokenizer", load_model)
    monkeypatch.setattr(m5, "generate_compressed", generate)
    monkeypatch.setattr(m5, "build_token_ids_record", sidecar)
    monkeypatch.setattr(m5, "load_input_manifest", lambda **k: manifest)
    monkeypatch.setattr(
        m5,
        "load_benchmark",
        lambda *a, **k: [Problem(p, "q", "1", {}) for p in PIDS],
    )
    monkeypatch.setattr(m5, "render_prompt", lambda *a, **k: "p")
    monkeypatch.setattr(
        m5, "tokenize_rendered_prompt", lambda *a, **k: {"input_ids": [0] * 12}
    )
    real_api = m5.family_api

    def api(kind):
        _, eager, gate = real_api(kind)
        return (lambda name, budget, **kw: _Press(budget)), eager, gate

    monkeypatch.setattr(m5, "family_api", api)
    monkeypatch.setenv("KVCACHE_REMOTE", "1")
    monkeypatch.setenv("KVCACHE_BASE_RUNS_ROOT", str(base))
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(out))
    return cfg_path, out, calls


def _run(monkeypatch, cfg_path, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_grid_m5", "--config", str(cfg_path), "--i-understand", *extra],
    )
    return m5.main()


def _rows(out, sub="math500"):
    return [
        json.loads(x)
        for p in sorted((out / sub).rglob("*.jsonl"))
        for x in p.read_text().splitlines()
    ]


def test_rows_and_sidecars_carry_identity_inputs_and_retention(tmp_path, monkeypatch):
    cfg_path, out, calls = _setup(tmp_path, monkeypatch)
    assert _run(monkeypatch, cfg_path) == 0
    rows = _rows(out)
    assert len(rows) == 100 and calls["model_loads"] == 1
    want = m5.expected_identity(yaml.safe_load(cfg_path.read_text()))
    for r in rows:
        assert {k: r["provenance"][k] for k in want} == want
        assert r["provenance"]["inputs_sha256"] == inputs_sha256(_prompt_ids(r["pid"]))
        assert r["m5"]["effective_budget"] == 256
        assert r["m5"]["kv_bytes_after_prune"] == 256 * 28 * 4 * 128 * 2 * 2
        assert r["m5"]["peak_occupancy_bound"] == 256 + 64
    side = _rows(out, "token_ids")
    assert len(side) == 100 and all(
        s["provenance"] == r["provenance"] for s, r in zip(side, rows, strict=True)
    )
    # a valid resume checks every record and skips everything
    assert _run(monkeypatch, cfg_path, "--resume") == 0
    assert len(_rows(out)) == 100


def test_resume_refused_after_a_config_change_before_model_load(tmp_path, monkeypatch):
    cfg_path, out, calls = _setup(tmp_path, monkeypatch)
    assert _run(monkeypatch, cfg_path) == 0
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["max_new_tokens"] = {"math500": 8192}
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="identity differs"):
        _run(monkeypatch, cfg_path, "--resume")
    assert calls["model_loads"] == 1  # refused before loading again


def test_foreign_row_refused(tmp_path, monkeypatch):
    cfg_path, out, calls = _setup(tmp_path, monkeypatch)
    legacy = out / "math500" / "r1_distill_qwen_7b" / "rkv_pp_v1-256-s11.jsonl"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "pid": PIDS[0],
                "press": "rkv_pp_v1",
                "budget": 256,
                "decoding": {"seed": 11},
            }
        )
        + "\n"
    )
    with pytest.raises(SystemExit, match="identity differs"):
        _run(monkeypatch, cfg_path, "--resume")
    assert calls["model_loads"] == 0


def test_prompt_input_mismatch_writes_nothing_for_that_problem(tmp_path, monkeypatch):
    cfg_path, out, _ = _setup(tmp_path, monkeypatch, bad_pid=PIDS[3])
    with pytest.raises(SystemExit, match="differ from the frozen manifest"):
        _run(monkeypatch, cfg_path)
    assert [r["pid"] for r in _rows(out)] == PIDS[:3]


def test_smoke_gate_and_hold_enforced(tmp_path, monkeypatch):
    cfg_path, _, calls = _setup(tmp_path, monkeypatch, smoke=False)
    with pytest.raises(SystemExit, match="smoke gate"):
        _run(monkeypatch, cfg_path)
    cfg_path, _, calls = _setup(tmp_path / "b", monkeypatch, cfg_over={"held": True})
    with pytest.raises(SystemExit, match="HELD"):
        _run(monkeypatch, cfg_path)
    assert calls["model_loads"] == 0


def test_dry_run_needs_no_smoke(tmp_path, monkeypatch):
    cfg_path, _, calls = _setup(tmp_path, monkeypatch, smoke=False)
    assert _run(monkeypatch, cfg_path, "--dry-run") == 0
    assert calls["model_loads"] == 0

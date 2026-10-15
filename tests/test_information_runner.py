"""Information-calibration runner on CPU (review of d50aa40, item 4): the REAL
``execute`` path of scripts/run_information_probe.py with stubs only for the
model, tokenizer, filler and presses (tests/test_information_probe.py), then
the REAL reader CLI with the frozen calibration criteria. A working channel
reads FINAL and PASSES; a channel whose eviction never reaches the facts FAILS
on the frozen criteria; resume skips finished blocks and refuses a changed
design; the held flag and the smoke gates refuse real execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_information_probe as rip  # noqa: E402
import test_information_probe as tip  # noqa: E402

import kvcache.models as km  # noqa: E402
from kvcache import information_calibration as ic  # noqa: E402
from kvcache import information_probe as ip  # noqa: E402

REAL = yaml.safe_load((ROOT / "configs/meeting5/info_calibration.yaml").read_text())


def _cfg(tmp_path, budget=40):
    cfg = {
        "name": "info_calibration_test",
        "held": False,
        "models": ["r1_distill_qwen_7b"],
        "design_seed": 3,
        "n_blocks": 2,
        "values": tip.VALUES,
        "aliases": tip.ALIASES,
        "spec": {"body_tokens": 120, "early_offset": 10, "late_offset": 90},
        "policies": ["full", "recency"],
        "budget": budget,
        "compression_interval": 8,
        "eviction_seeds": [0],
        "queries": ["s", "n"],
        "paired_contrasts": [],
        "calibration_criteria": REAL["calibration_criteria"],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return cfg, path


def _stubs(monkeypatch, budget):
    backend = tip.PressedStub()
    tok = SimpleNamespace(encode=lambda t, add_special_tokens=False: tip.encode(t))

    def policies(model, cfg):
        def full(seed):
            backend.press = None
            return ip.NoPress()

        def recency(seed):
            p = tip.Recency(None, 0, budget)
            backend.press = p
            return p

        return {"full": full, "recency": recency}

    monkeypatch.setattr(rip, "load_passages", lambda cfg, base: tip.PASSAGES)
    monkeypatch.setattr(rip, "smoke_gates", lambda cfg, base: [])
    monkeypatch.setattr(rip, "make_backend", lambda model: backend)
    monkeypatch.setattr(rip, "policy_factories", policies)
    monkeypatch.setattr(km, "load_tokenizer", lambda cfg: tok)
    monkeypatch.setattr(
        km, "load_model_and_tokenizer", lambda cfg, **k: (object(), tok)
    )


def _read(tmp_path, out, cfg_path):
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/report_information_probe.py"),
            "--rows",
            str(out / "readout"),
            "--design",
            str(out / "design.json"),
            "--out",
            str(tmp_path / "read.json"),
            "--calibration-criteria",
            str(cfg_path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return p.returncode, json.loads((tmp_path / "read.json").read_text())


def test_working_channel_reads_final_and_passes(tmp_path, monkeypatch):
    cfg, path = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    res = rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    assert res == {"rows_written": 2 * 2 * 16 * 2, "blocks": 2, "skipped": 0}
    rc, read = _read(tmp_path, out, path)
    assert (rc, read["status"]) == (0, "FINAL"), read.get("problems")
    cal = read["calibration"]
    assert cal["verdict"] == "PASS", cal
    assert cal["criteria"]["C4_end_to_end"]["value"] == 0.0
    assert cal["criteria"]["C5_eviction_reaches_facts"]["value"] == 0.0
    assert cal["criteria"]["C6_dynamic_range"]["value"] == pytest.approx(1.0, abs=1e-3)


def test_no_eviction_fails_the_frozen_criteria(tmp_path, monkeypatch):
    cfg, path = _cfg(tmp_path, budget=10_000)
    _stubs(monkeypatch, budget=10_000)  # the "policy" never evicts anything
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    rc, read = _read(tmp_path, out, path)
    assert read["status"] == "FINAL"
    assert read["calibration"]["verdict"] == "FAIL"
    assert read["calibration"]["failed"] == [
        "C5_eviction_reaches_facts",
        "C6_dynamic_range",
    ]


def test_resume_skips_finished_blocks_and_refuses_a_changed_design(
    tmp_path, monkeypatch
):
    cfg, _ = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    again = rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    assert again["rows_written"] == 0 and again["skipped"] == 2
    changed = {**cfg, "design_seed": 4}
    with pytest.raises(SystemExit, match="design.json differs"):
        rip.execute(changed, out_dir=out, base_runs=tmp_path)


def test_real_execution_refuses_while_held_or_without_smokes(tmp_path, monkeypatch):
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_information_probe.py"),
            "--i-understand",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"KVCACHE_REMOTE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert p.returncode != 0 and "HELD" in p.stderr
    cfg, _ = _cfg(tmp_path)
    monkeypatch.setattr(rip, "smoke_gates", lambda cfg, base: ["no PASS marker"])
    with pytest.raises(SystemExit, match="smoke gate"):
        rip.execute(cfg, out_dir=tmp_path / "run", base_runs=tmp_path)


def test_evaluator_refuses_a_non_final_read():
    out = ic.evaluate({"status": "PARTIAL"}, [], REAL["calibration_criteria"])
    assert out["verdict"] == "FAIL" and not out["criteria"]["C3_read_final"]["passed"]

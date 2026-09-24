"""End-to-end glue of run_basis_fit.py and run_capture.py on a tiny model.

The real model and dataset are swapped for a tiny random-init Qwen2 and a
hand-built reference run tree; everything else (reference loading, the frozen
basis hash check, resume, row and npz writing, the smoke gate) runs as it will
on the GPU. Skipped without torch/transformers/kvpress.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _setup(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("kvpress")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_basis_fit
    import run_capture

    import kvcache.models as models

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
    model = transformers.Qwen2ForCausalLM._from_config(
        cfg, attn_implementation="eager"
    ).eval()
    monkeypatch.setattr(
        models, "load_model_and_tokenizer", lambda *a, **k: (model, None)
    )
    pids = ["p1", "p2"]
    fake = [SimpleNamespace(pid=p) for p in pids]
    for mod in (run_basis_fit, run_capture):
        monkeypatch.setattr(mod, "load_benchmark", lambda *a, **k: fake)
    ref = tmp_path / "runs" / "ref"
    for seed in (11, 23):
        rows, toks = [], []
        for p in pids:
            g = torch.Generator().manual_seed(hash((p, seed)) % 1000)
            rows.append(
                {
                    "bench": "math500",
                    "pid": p,
                    "model": "r1_distill_qwen_7b",
                    "press": "fullcache",
                    "budget": "full",
                    "decoding": {"seed": seed},
                }
            )
            toks.append(
                {
                    "bench": "math500",
                    "pid": p,
                    "model": "r1_distill_qwen_7b",
                    "press": "fullcache",
                    "budget": "full",
                    "seed": seed,
                    "prompt_token_ids": torch.randint(
                        3, 97, (10,), generator=g
                    ).tolist(),
                    "trace_token_ids": torch.randint(
                        3, 97, (70,), generator=g
                    ).tolist(),
                }
            )
        (ref / "math500").mkdir(parents=True, exist_ok=True)
        (ref / "token_ids").mkdir(parents=True, exist_ok=True)
        (ref / "math500" / f"s{seed}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        (ref / "token_ids" / f"s{seed}.jsonl").write_text(
            "".join(json.dumps(t) + "\n" for t in toks)
        )
    monkeypatch.setenv("KVCACHE_REMOTE", "1")
    monkeypatch.setenv("KVCACHE_BASE_RUNS_ROOT", str(tmp_path / "runs"))
    return run_basis_fit, run_capture


def _cfg(tmp_path, name, body):
    p = tmp_path / f"{name}.yaml"
    p.write_text(body)
    return p


COMMON = """models: [r1_distill_qwen_7b]
benchmarks: [math500]
seeds: [11, 23]
subset: {math500: 2}
reference_run_name: ref
"""


def test_fit_then_capture_then_smoke(tmp_path, monkeypatch):
    run_basis_fit, run_capture = _setup(tmp_path, monkeypatch)
    fit_cfg = _cfg(
        tmp_path,
        "fit",
        COMMON + "name: fit\npresses: [fullcache]\nbudgets: [full]\nk: 4\n"
        "n_positions: 16\n",
    )
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(tmp_path / "runs" / "fit"))
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(fit_cfg), "--i-understand"])
    assert run_basis_fit.main() == 0
    header = json.loads((tmp_path / "runs/fit/basis/q6_basis.json").read_text())
    assert header["n_traces"] == 4 and header["k"] == 4

    cap_body = COMMON + (
        "name: cap\npresses: [random, streamingllm]\nbudgets: [16]\n"
        "compression_interval: 4\ntau: 0.01\ntrigger_step: 20\nbasis_run: fit\n"
    )
    cap = _cfg(tmp_path, "cap", cap_body)
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(tmp_path / "runs" / "cap"))
    argv = ["x", "--config", str(cap), "--i-understand", "--resume", "--seeds", "11"]
    monkeypatch.setattr(sys, "argv", argv)
    assert run_capture.main() == 0
    rows = [
        json.loads(line)
        for p in (tmp_path / "runs/cap/math500").rglob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert len(rows) == 4  # 2 problems x 2 policies, seed 11 shard only
    r = rows[0]
    assert r["schema_version"] == "capture-v1"
    assert r["basis_npz_sha256"] == header["npz_sha256"]
    assert r["latent"]["omega_trace"] is not None
    assert r["checks"]["layer0_delta_max"] < 1e-5
    assert len(list((tmp_path / "runs/cap/latent").rglob("*.npz"))) == 4
    # resume: nothing redone
    assert run_capture.main() == 0
    again = sum(
        1
        for p in (tmp_path / "runs/cap/math500").rglob("*.jsonl")
        for _ in p.read_text().splitlines()
    )
    assert again == 4

    smoke = _cfg(
        tmp_path,
        "smoke",
        cap_body.replace("name: cap", "name: smoke")
        + "reference_limit: 1\nfullfull_control: true\nsmoke: true\n",
    )
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(tmp_path / "runs" / "smoke"))
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(smoke), "--i-understand"])
    assert run_capture.main() == 0
    rep = json.loads((tmp_path / "runs/capture-smoke-PASS.json").read_text())
    assert rep["passed"] and len(rep["rows"]) == 3


def test_capture_refuses_a_basis_whose_bytes_changed(tmp_path, monkeypatch):
    run_basis_fit, run_capture = _setup(tmp_path, monkeypatch)
    bdir = tmp_path / "runs" / "fit" / "basis"
    bdir.mkdir(parents=True)
    (bdir / "q6_basis.npz").write_bytes(b"tampered")
    (bdir / "q6_basis.json").write_text(json.dumps({"npz_sha256": "0" * 64}))
    cap = _cfg(
        tmp_path,
        "cap",
        COMMON + "name: cap\npresses: [random]\nbudgets: [16]\n"
        "compression_interval: 4\ntau: 0.01\nbasis_run: fit\n",
    )
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(tmp_path / "runs" / "cap"))
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cap), "--i-understand"])
    with pytest.raises(SystemExit, match="frozen hash"):
        run_capture.main()

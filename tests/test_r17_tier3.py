"""R.17.3 cross-architecture plumbing: a second model's frozen input manifest.

The default manifest (and so every existing Meeting #3 identity) is unchanged;
a config that names ``input_manifest`` gets that manifest in gates, readers and
-- through KVCACHE_INPUT_MANIFEST exported by the scheduler -- in the
unmodified runners. The capture-v2 replay must also hold its live invariants on
a Llama-architecture model (torch; runs on the GPU host).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kvcache import provenance as pv  # noqa: E402

LLAMA_MANIFEST = "configs/meeting3/input_manifest_llama8b.json"
QWEN_SHA = hashlib.sha256((ROOT / pv.INPUT_MANIFEST).read_bytes()).hexdigest()
LLAMA_SHA = hashlib.sha256((ROOT / LLAMA_MANIFEST).read_bytes()).hexdigest()


def test_manifest_precedence_is_explicit_then_env_then_default(monkeypatch):
    monkeypatch.delenv("KVCACHE_INPUT_MANIFEST", raising=False)
    assert pv.input_manifest_sha256() == QWEN_SHA
    assert pv.input_manifest_sha256(path=LLAMA_MANIFEST) == LLAMA_SHA
    monkeypatch.setenv("KVCACHE_INPUT_MANIFEST", str(ROOT / LLAMA_MANIFEST))
    assert pv.input_manifest_sha256() == LLAMA_SHA
    assert pv.input_manifest_sha256(path=pv.INPUT_MANIFEST) == QWEN_SHA
    assert LLAMA_SHA != QWEN_SHA


def test_default_identity_is_unchanged_by_the_optional_path(monkeypatch):
    monkeypatch.delenv("KVCACHE_INPUT_MANIFEST", raising=False)
    cfg = {"name": "x"}
    a = pv.identity("capture", cfg, model_revision="r")
    b = pv.identity("capture", cfg, model_revision="r", manifest_path=None)
    assert a == b and a["input_manifest_sha256"] == QWEN_SHA
    c = pv.identity("capture", cfg, model_revision="r", manifest_path=LLAMA_MANIFEST)
    assert c["input_manifest_sha256"] == LLAMA_SHA


def test_llama_manifest_covers_the_frozen_subset():
    m = json.loads((ROOT / LLAMA_MANIFEST).read_text())
    pids = json.loads((ROOT / "configs/meeting3/q2_subset_pids.json").read_text())[
        "pids"
    ]
    assert m["model"] == "r1_distill_llama_8b" and m["reference_run"] == "fc-llama50"
    assert set(m["prompts"]) == set(pids)
    assert set(m["traces_prompt_plus_trace"]) == {
        f"{p}|{s}" for p in pids for s in (11, 23, 47, 89)
    }


def test_the_llama_basis_is_verified_against_its_own_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("KVCACHE_INPUT_MANIFEST", raising=False)
    from kvcache.gates import _revision, _yaml, verify_basis

    cfg_path = "configs/experiments/basis_fit_q6_llama8b_r17.yaml"
    cfg = _yaml(ROOT / cfg_path)

    def write(manifest_path):
        ident = pv.identity(
            "basis",
            cfg,
            model_revision=_revision(cfg, ROOT),
            reference_run=cfg["reference_run_name"],
            manifest_path=manifest_path,
        )
        d = tmp_path / cfg["basis_run_name"] / "basis"
        d.mkdir(parents=True, exist_ok=True)
        (d / "q6_basis.npz").write_bytes(b"b")
        (d / "q6_basis.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "status": "FROZEN",
                    "npz": "q6_basis.npz",
                    "npz_sha256": hashlib.sha256(b"b").hexdigest(),
                    "identity": ident,
                }
            )
        )

    write(LLAMA_MANIFEST)
    ok, why, _ = verify_basis(tmp_path, cfg_path, run_name=cfg["basis_run_name"])
    assert ok, why
    write(None)  # a basis stamped with the Qwen manifest is not this fit
    ok, _why, _ = verify_basis(tmp_path, cfg_path, run_name=cfg["basis_run_name"])
    assert not ok


def test_the_scheduler_exports_a_configs_own_manifest_only():
    spec = importlib.util.spec_from_file_location(
        "na_t3", ROOT / "scripts" / "nightly_autorun.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["na_t3"] = m
    spec.loader.exec_module(m)
    base = dict(
        name="x",
        command="capture2",
        run_name="x",
        total=1,
        seeds=[11],
        min_free_gb=1,
        progress_kind="grid",
        requires=None,
    )
    llama = m.Job(
        config="configs/experiments/capture_math500_llama8b_b256_r17.yaml", **base
    )
    qwen = m.Job(config="configs/experiments/capture_math500_b2048_r17.yaml", **base)
    (s1,) = m.build_launch_specs(llama, {0: [11]}, 1.0)
    (s2,) = m.build_launch_specs(qwen, {0: [11]}, 1.0)
    assert s1.env["KVCACHE_INPUT_MANIFEST"] == str(ROOT / LLAMA_MANIFEST)
    assert "KVCACHE_INPUT_MANIFEST" not in s2.env


# ------------------------------------------------------- Llama architecture
def _tiny_llama():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("kvpress")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    model = transformers.LlamaForCausalLM._from_config(
        cfg, attn_implementation="eager"
    ).eval()
    prompt = torch.randint(3, 97, (12,)).tolist()
    trace = torch.randint(3, 97, (60,)).tolist()
    return torch, model, prompt, trace


@pytest.mark.parametrize("policy", ["random", "streamingllm", "h2o"])
def test_capture_v2_holds_its_live_invariants_on_llama(policy):
    torch, model, prompt, trace = _tiny_llama()
    from kvcache.capture import record_full_branch
    from kvcache.capture_v2 import replay_condition_v2
    from kvcache.presses import build_press

    n, d = model.config.num_hidden_layers, model.config.hidden_size
    q, _ = torch.linalg.qr(torch.randn(n, d, 4))
    basis = {"scale": torch.ones(n, d), "basis": q}
    full = record_full_branch(model, prompt, trace, interval=4)
    press = build_press(
        policy, budget=16, compression_interval=4, seed=7, instrumented=True
    )
    out = replay_condition_v2(
        model,
        press,
        full,
        prompt,
        trace,
        budget=16,
        interval=4,
        tau=0.01,
        basis=basis,
        trigger_step=20,
    )
    assert out["n_events"] >= 1 and out["checks"]["n_nonfinite"] == 0
    assert out["checks"]["layer0_delta_max"] <= 1e-5
    assert out["latent"]["omega_trace"] is not None
    assert any(out["n_measured_by_layer"])

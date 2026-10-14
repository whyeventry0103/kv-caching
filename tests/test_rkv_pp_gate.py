"""Gate, provenance and smoke checks for prompt-protected R-KV (rkv_pp_v1;
Meeting 5 M5-A). CPU-only and torch-free, so these run in the local gate; the
press itself is tested in tests/test_rkv_prompt_protect.py on the GPU host."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from kvcache.presses import rkv_prompt_protect as rpp
from kvcache.presses.protected_selectors import FAMILIES
from kvcache.provenance import GENERATION_SOURCES, implementation_id

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CFG = "configs/experiments/rkvpp_smoke_m5.yaml"
GRID_CFG = "configs/experiments/grid_math500_rkvpp256_m5.yaml"
READ_CFG = "configs/experiments/readout_math500_rkvpp256_m5.yaml"
PIDS = ["p/a", "p/b", "p/c"]


def _run_cfg(path=GRID_CFG, **over):
    cfg = yaml.safe_load((ROOT / path).read_text())
    cfg.update(over)
    return cfg


sys.path.insert(0, str(ROOT / "tests"))
import smoke_fixtures as sfx  # noqa: E402


def _valid_marker(tmp_path, **over):
    """A fully measured PASS on a fixture reference bank (review of 6b409001)."""
    smoke = yaml.safe_load((ROOT / SMOKE_CFG).read_text())
    if not (tmp_path / smoke["reference_run_name"]).exists():
        sfx.reference_bank(tmp_path, smoke)
    rec = sfx.measured_marker(tmp_path, smoke, "rkvpp")
    rec.update(over)
    sfx.write(tmp_path, rpp.SMOKE_MARKER, rec)
    return rec


def test_own_provenance_family_and_not_in_the_selector_registry():
    assert GENERATION_SOURCES["rkv_pp"] == rpp.SOURCES
    for rel in rpp.SOURCES:
        assert (ROOT / rel).is_file(), rel
    assert "src/kvcache/presses/rkv.py" in rpp.SOURCES
    assert len(implementation_id("rkv_pp")) == 64
    assert rpp.FAMILY not in FAMILIES  # review: not the H2O/uniform adapter


def test_attention_backend_is_the_base_policys():
    from kvcache.presses import press_needs_eager_attention

    assert rpp.needs_eager_attention("rkv_pp_v1") == press_needs_eager_attention("rkv")


def test_first_mismatch():
    assert rpp.first_mismatch([1, 2], [1, 2]) is None
    assert rpp.first_mismatch([1, 2], [1, 3]) == 1
    assert rpp.first_mismatch([1, 2], [1, 2, 3]) == 2
    assert rpp.first_mismatch([], []) is None


def _replay(**over):
    r = {
        "kind": "replay",
        "pid": "p",
        "prompt_len": 344,
        "n_layers": 28,
        "expected_target": 408,
        "first_effective_prune": 128,
        "pre_eviction_max_tv": 0.0,
        "post_prune_sizes": [408],
        "protect": {"prompt_len": 344, "target": 408, "n_layers_pruned": 28},
    }
    r.update(over)
    return r


def _identity(**over):
    r = {
        "kind": "identity",
        "pid": "p",
        "n_prunes_base": 112,
        "n_prunes_protected_off": 112,
        "first_kv_mismatch": None,
        "tv_identical": True,
    }
    r.update(over)
    return r


def test_smoke_failures_pass_and_each_failure():
    gen = {
        "kind": "generation",
        "pid": "p",
        "prompt_len": 50,
        "n_tokens": 700,
        "protect": {"prompt_len": 50, "n_prunes": 3},
    }
    assert rpp.smoke_failures([_identity(), _replay(), gen]) == []
    assert rpp.smoke_failures([]) == ["no smoke record"]
    cases = [
        (_identity(n_prunes_base=0), "compared no prune"),
        (_identity(first_kv_mismatch=7), "K/V differ from base at 7"),
        (_identity(tv_identical=False), "TV profile differs"),
        (_identity(n_prunes_protected_off=111), "different count"),
        (_replay(pre_eviction_max_tv=1e-6), "pre-eviction TV"),
        (_replay(post_prune_sizes=[407, 408]), "post-prune sizes"),
        (_replay(expected_target=256), "target 408 != 256"),
        (_replay(first_effective_prune=None), "no effective prune"),
        (
            _replay(protect={"prompt_len": 1, "target": 408, "n_layers_pruned": 28}),
            "press read prompt length 1",
        ),
        (
            _replay(protect={"prompt_len": 344, "target": 408, "n_layers_pruned": 3}),
            "pruned 3 of 28 layers",
        ),
        (dict(gen, protect={"prompt_len": 50, "n_prunes": 0}), "no prune in 700"),
        ({"kind": "replay", "pid": "p", "raised": "evicted"}, "raised: evicted"),
    ]
    for rec, why in cases:
        fails = rpp.smoke_failures([rec])
        assert fails and why in fails[0], (why, fails)


def test_verify_smoke_accepts_a_valid_pass(tmp_path):
    _valid_marker(tmp_path)
    assert rpp.verify_smoke(tmp_path, _run_cfg()) == (True, "verified")
    assert rpp.verify_smoke(tmp_path, _run_cfg(READ_CFG)) == (True, "verified")


@pytest.mark.parametrize(
    "content, why",
    [
        ("", "absent, empty or unreadable"),
        ('{"passed": false, "status": "FAIL"}', "does not record a PASS"),
        ('{"passed": true}', "does not record a PASS"),
        ("[1, 2]", "not a record"),
    ],
)
def test_verify_smoke_refuses_bad_markers(tmp_path, content, why):
    (tmp_path / rpp.SMOKE_MARKER).write_text(content)
    ok, msg = rpp.verify_smoke(tmp_path, _run_cfg())
    assert not ok and why in msg


def test_verify_smoke_refuses_stale_failed_partial_and_uncovered(tmp_path):
    smoke = yaml.safe_load((ROOT / SMOKE_CFG).read_text())
    _valid_marker(tmp_path, failures=["x"])
    assert "lists failures" in rpp.verify_smoke(tmp_path, _run_cfg())[1]
    stale = dict(rpp.smoke_identity(smoke), implementation_id="0" * 64)
    _valid_marker(tmp_path, identity=stale)
    assert "stale" in rpp.verify_smoke(tmp_path, _run_cfg())[1]
    rec = _valid_marker(tmp_path)
    _valid_marker(tmp_path, records=rec["records"][:-1])
    assert "does not cover the design" in rpp.verify_smoke(tmp_path, _run_cfg())[1]
    _valid_marker(tmp_path, selected=rec["selected"][:1])
    assert "exactly 3 traces" in rpp.verify_smoke(tmp_path, _run_cfg())[1]
    _valid_marker(tmp_path)
    assert "every budget" in rpp.verify_smoke(tmp_path, _run_cfg(budgets=[1024]))[1]
    assert (
        "interval" in rpp.verify_smoke(tmp_path, _run_cfg(compression_interval=32))[1]
    )
    assert "different model" in rpp.verify_smoke(tmp_path, _run_cfg(models=["x"]))[1]
    assert (
        "no smoke_config" in rpp.verify_smoke(tmp_path, _run_cfg(smoke_config=None))[1]
    )


@pytest.mark.parametrize(
    "mutate, why",
    [
        (
            lambda r: r.update(
                records=[
                    {k: x[k] for k in ("kind", "pid", "press")} for x in r["records"]
                ]
            ),
            "lacks measurement",
        ),
        (lambda r: r["records"][0].update(raised="injected control failure"), "raised"),
        (
            lambda r: next(x for x in r["records"] if x["kind"] == "identity").update(
                first_kv_mismatch=4
            ),
            "measurements fail",
        ),
        (
            lambda r: next(x for x in r["records"] if x["kind"] == "identity").update(
                tv_identical="True"
            ),
            "not a valid bool",
        ),
        (lambda r: r["records"][0].update(post_prune_sizes=[255]), "measurements fail"),
        (lambda r: r["records"][0].update(inputs_sha256="0" * 64), "other inputs"),
    ],
)
def test_rkv_runtime_gate_validates_measurements(tmp_path, mutate, why):
    """Review of 6b409001 R2: names-only and failed-record markers verified."""
    rec = _valid_marker(tmp_path)
    assert rpp.verify_smoke(tmp_path, _run_cfg()) == (True, "verified")
    mutate(rec)
    sfx.write(tmp_path, rpp.SMOKE_MARKER, rec)
    ok, msg = rpp.verify_smoke(tmp_path, _run_cfg())
    assert not ok and why in msg, msg


def test_require_config_refusals(tmp_path):
    rpp.require_rkv_pp_config(_run_cfg(), dry_run=True)  # dry runs always allowed
    for over, why in [
        ({"rkv_prompt_protected": None}, "rkv_prompt_protected: true"),
        ({"presses": ["rkv"]}, "exactly [rkv_pp_v1]"),
        ({"presses": ["rkv_pp_v1", "h2o_style_pp_v1"]}, "exactly [rkv_pp_v1]"),
        ({"matched_ratio": 0.5}, "matched_ratio"),
    ]:
        with pytest.raises(SystemExit, match=why.replace("[", r"\[")):
            rpp.require_rkv_pp_config(_run_cfg(**over), dry_run=True)
    with pytest.raises(SystemExit, match="registered for readout"):
        rpp.require_rkv_pp_config(_run_cfg(), dry_run=True, replay=True)
    rpp.require_rkv_pp_config(_run_cfg(READ_CFG), dry_run=True, replay=True)
    with pytest.raises(SystemExit, match="HELD"):
        rpp.require_rkv_pp_config(
            _run_cfg(held=True), dry_run=False, runs_root=tmp_path
        )
    with pytest.raises(SystemExit, match="smoke gate"):
        rpp.require_rkv_pp_config(
            _run_cfg(held=False), dry_run=False, runs_root=tmp_path
        )
    _valid_marker(tmp_path)
    rpp.require_rkv_pp_config(_run_cfg(held=False), dry_run=False, runs_root=tmp_path)


def test_committed_config_holds_match_the_owner_decision():
    """2026-10-14: the owner approved the smoke (1 GPU-h) and the grid (28 GPU-h);
    the forward readout is not approved and stays held."""
    for path in (SMOKE_CFG, GRID_CFG):
        cfg = _run_cfg(path)
        assert cfg["held"] is False and cfg["scale_authorized"] is True, path
        assert "APPROVED by the owner 2026-10-14" in (ROOT / path).read_text()
    read = _run_cfg(READ_CFG)
    assert read["held"] is True and read["scale_authorized"] is False
    for path in (SMOKE_CFG, GRID_CFG, READ_CFG):
        assert _run_cfg(path)["presses"] == ["rkv_pp_v1"]


@pytest.mark.parametrize(
    "script, cfg",
    [
        ("scripts/run_grid_m5.py", GRID_CFG),
        ("scripts/run_replay_rkv_pp.py", READ_CFG),
        ("scripts/rkv_pp_smoke.py", SMOKE_CFG),
    ],
)
def test_real_execution_refused_while_held(script, cfg, tmp_path):
    held = _run_cfg(cfg, held=True)
    path = tmp_path / "held.yaml"
    path.write_text(yaml.safe_dump(held))
    p = subprocess.run(
        [sys.executable, str(ROOT / script), "--config", str(path), "--i-understand"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={
            **os.environ,
            "KVCACHE_BASE_RUNS_ROOT": str(tmp_path),
            "KVCACHE_REMOTE": "1",
        },
    )
    assert p.returncode != 0
    assert "HELD" in p.stderr

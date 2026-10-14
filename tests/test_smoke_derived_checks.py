"""Review of 946c76a, issue 1: both runtime smoke gates must refuse measurements
that agree with each other but not with the experiment -- a NaN prune count, a
coordinated wrong target, zero layers -- by deriving the expected values from
the frozen config, the frozen trace and the pinned model's metadata."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import smoke_fixtures as sfx  # noqa: E402

from kvcache import smoke_validation as sv  # noqa: E402
from kvcache.presses import protected_selectors as ps  # noqa: E402
from kvcache.presses import rkv_prompt_protect as rp  # noqa: E402

FAMILIES = {
    "protsel": (ps, "protsel_smoke_wp4.yaml", "grid_math500_h2opp256_m5.yaml"),
    "rkvpp": (rp, "rkvpp_smoke_m5.yaml", "grid_math500_rkvpp256_m5.yaml"),
}


def _each(kind, fn):
    def go(marker):
        for r in marker["records"]:
            if r["kind"] == kind:
                fn(r)

    return go


MUTATIONS = {
    # the review's three reproductions
    "nan_prunes": (
        _each("generation", lambda r: r["protect"].update(n_prunes=float("nan"))),
        "protect.n_prunes is not a valid count",
    ),
    "wrong_target": (
        _each(
            "replay",
            lambda r: (
                r.update(expected_target=255, post_prune_sizes=[255]),
                r["protect"].update(target=255),
            ),
        ),
        "derived effective budget 256",
    ),
    "zero_layers": (
        _each(
            "replay",
            lambda r: (r.update(n_layers=0), r["protect"].update(n_layers_pruned=0)),
        ),
        "pinned model's 28",
    ),
    # neighbours of the same defect class
    "bool_prunes": (
        _each("generation", lambda r: r["protect"].update(n_prunes=True)),
        "protect.n_prunes is not a valid count",
    ),
    "zero_generation_prunes": (
        _each("generation", lambda r: r["protect"].update(n_prunes=0)),
        "no prune recorded",
    ),
    "missing_nested": (
        _each("replay", lambda r: r["protect"].pop("n_layers_pruned")),
        "protect.n_layers_pruned",
    ),
    "float_size": (
        _each("replay", lambda r: r.update(post_prune_sizes=[256.0])),
        "not exact counts",
    ),
    "replay_length": (
        _each("replay", lambda r: r.update(n_replayed=r["n_replayed"] - 1)),
        "not the configured",
    ),
    "over_cap_generation": (
        _each("generation", lambda r: r.update(n_tokens=10_000)),
        "outside [1, 700]",
    ),
    "press_prompt_len": (
        _each("replay", lambda r: r["protect"].update(prompt_len=1)),
        "another prompt length",
    ),
}


def _setup(tmp_path, family):
    mod, smoke_path, grid_path = FAMILIES[family]
    smoke = yaml.safe_load((ROOT / "configs/experiments" / smoke_path).read_text())
    grid = yaml.safe_load((ROOT / "configs/experiments" / grid_path).read_text())
    sfx.reference_bank(tmp_path, smoke)
    return mod, grid, sfx.measured_marker(tmp_path, smoke, family)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_consistent_measured_control_verifies(tmp_path, family):
    mod, grid, _ = _setup(tmp_path, family)
    assert mod.verify_smoke(tmp_path, grid) == (True, "verified")


@pytest.mark.parametrize("name", sorted(MUTATIONS))
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_self_consistent_wrong_measurements_are_refused(tmp_path, family, name):
    mod, grid, marker = _setup(tmp_path, family)
    mutate, why = MUTATIONS[name]
    bad = copy.deepcopy(marker)
    mutate(bad)
    sfx.write(tmp_path, mod.SMOKE_MARKER, bad)
    ok, msg = mod.verify_smoke(tmp_path, grid)
    assert not ok and why in msg, msg


def test_relaxed_prompt_derives_the_relaxed_target():
    smoke = yaml.safe_load(
        (ROOT / "configs/experiments/protsel_smoke_wp4.yaml").read_text()
    )
    assert ps.smoke_expected_target("h2o_style_pp_v1", 353, smoke) == 417
    assert ps.smoke_expected_target("uniform_fresh_pp_r64_v1", 10, smoke) == 256
    rk = yaml.safe_load((ROOT / "configs/experiments/rkvpp_smoke_m5.yaml").read_text())
    assert rp.smoke_expected_target("rkv_pp_v1", 344, rk) == 408


def test_layer_count_comes_from_frozen_metadata_for_the_pinned_revision(
    tmp_path, monkeypatch
):
    smoke = yaml.safe_load(
        (ROOT / "configs/experiments/rkvpp_smoke_m5.yaml").read_text()
    )
    assert sv.model_layers(smoke, root=ROOT) == 28
    meta = json.loads((ROOT / sv.MODEL_METADATA).read_text())
    assert meta["r1_distill_qwen_7b"]["revision"] == (
        "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
    )
    # a stale or missing record refuses rather than trusting the marker
    fake = tmp_path / "root"
    (fake / "configs/meeting5").mkdir(parents=True)
    (fake / "configs/models.yaml").write_text(
        (ROOT / "configs/models.yaml").read_text()
    )
    meta["r1_distill_qwen_7b"]["revision"] = "0" * 40
    (fake / sv.MODEL_METADATA).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="another revision"):
        sv.model_layers(smoke, root=fake)

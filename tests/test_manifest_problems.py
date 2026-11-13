"""Frozen-ID problem selection (kvcache.manifest_problems) and the held
manifest-bank runner (scripts/run_bank_manifest.py), CPU only.

Hand-built manifests check every refusal; the committed manifest and the held
GSM8K timing config check that the real calibration split resolves to its 24
frozen IDs, disjoint from confirmation and reserve, at the pinned revision."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from kvcache.manifest_problems import frozen_split_pids, select_frozen_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/bank_gsm8k_timing_m5.yaml"
MANIFEST = ROOT / "configs/meeting5/problem_manifests.json"
REV = "rev-a"


@dataclass(frozen=True)
class P:
    pid: str
    question: str = "q"


def _manifest(**over):
    entry = {
        "revision": REV,
        "n_pool": 6,
        "calibration": ["t/3", "t/1"],
        "confirmation": ["t/0", "t/4"],
        "reserve": ["t/2", "t/5"],
    }
    entry.update(over)
    return {"bench": entry}


def test_split_resolves_in_manifest_order():
    assert frozen_split_pids(
        _manifest(), "bench", "calibration", pinned_revision=REV
    ) == ["t/3", "t/1"]


@pytest.mark.parametrize(
    "manifest, split, pinned, match",
    [
        (_manifest(), "calibration", "rev-b", "differs from the pinned"),
        (_manifest(), "holdout", REV, "unknown split"),
        (_manifest(calibration=[]), "calibration", REV, "missing or empty"),
        (_manifest(calibration=["t/3", "t/3"]), "calibration", REV, "repeats"),
        (_manifest(calibration=["t/3", "t/0"]), "calibration", REV, "both"),
        ({}, "calibration", REV, "no frozen entry"),
    ],
)
def test_split_refusals(manifest, split, pinned, match):
    with pytest.raises(ValueError, match=match):
        frozen_split_pids(manifest, "bench", split, pinned_revision=pinned)


def test_selection_returns_exactly_the_ids_in_their_order():
    pool = [P(f"t/{i}") for i in range(6)]
    got = select_frozen_split(pool, ["t/3", "t/1"], n_pool=6)
    assert [p.pid for p in got] == ["t/3", "t/1"]


def test_selection_refuses_a_different_pool_size():
    pool = [P(f"t/{i}") for i in range(5)]
    with pytest.raises(ValueError, match="froze 6"):
        select_frozen_split(pool, ["t/3"], n_pool=6)


def test_selection_refuses_a_missing_id_or_a_repeated_pool_id():
    pool = [P(f"t/{i}") for i in range(6)]
    with pytest.raises(ValueError, match="absent"):
        select_frozen_split(pool, ["t/3", "t/9"], n_pool=6)
    dup = [*pool[:5], P("t/0")]
    with pytest.raises(ValueError, match="repeats"):
        select_frozen_split(dup, ["t/3"], n_pool=6)


def test_committed_manifest_splits_partition_their_pools():
    m = json.loads(MANIFEST.read_text())
    for bench in ("gsm8k", "bbh_logical_deduction_five_objects"):
        e = m[bench]
        ids = [*e["calibration"], *e["confirmation"], *e["reserve"]]
        assert len(ids) == len(set(ids)) == e["n_pool"], bench
    assert len(m["gsm8k"]["calibration"]) == 24 and m["gsm8k"]["n_pool"] == 1319


def _runner():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_bank_manifest

    return run_bank_manifest


def _config():
    import yaml

    return yaml.safe_load(CONFIG.read_text())


def test_gsm8k_timing_config_resolves_to_the_frozen_calibration_split():
    cfg = _config()
    # Released 2026-11-13 under campaign M5B-GSM8K-TIMING (owner approval).
    assert cfg["held"] is False and cfg["scale_authorized"] is True
    d = _runner().resolve_design(cfg)
    m = json.loads(MANIFEST.read_text())["gsm8k"]
    assert d["pids"] == m["calibration"] and len(d["pids"]) == 24
    assert d["n_generations"] == 48 and d["seeds"] == [11, 23]
    assert d["revision"] == m["revision"] and d["max_new_tokens"] == 8192


@pytest.mark.parametrize(
    "over, match",
    [
        ({"estimated_generation_count": 47}, "estimated_generation_count"),
        ({"presses": ["random"]}, "full-cache banks only"),
        (
            {"manifest": {"path": "configs/meeting5/problem_manifests.json"}},
            "path and split",
        ),
        ({"benchmarks": ["gsm8k", "math500"]}, "one benchmark"),
    ],
)
def test_runner_refuses_a_config_that_departs_from_the_design(over, match):
    with pytest.raises(ValueError, match=match):
        _runner().resolve_design({**_config(), **over})


def _run(config: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_bank_manifest.py"),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_runner_refuses_to_generate_while_held(tmp_path):
    import yaml

    held = tmp_path / "held.yaml"
    held.write_text(yaml.safe_dump({**_config(), "held": True}))
    r = _run(held)
    assert r.returncode != 0
    assert "HELD" in (r.stderr + r.stdout)


def test_released_runner_still_refuses_without_i_understand():
    r = _run(CONFIG)
    assert r.returncode != 0
    assert "--i-understand" in (r.stderr + r.stdout)

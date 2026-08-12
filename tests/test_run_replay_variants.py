"""Config-to-press wiring in scripts/run_replay.py (Amendments M.3 and P.5).

The runner is what turns a config block into an actual press. That step had no
coverage: the press builders were unit-tested against fake bases while nothing
checked that a config saying ``restoration:`` produced restoration presses at
all. These tests pin the selection, the arm product, and the two refusals.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_replay", ROOT / "scripts" / "run_replay.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

CONFIGS = ROOT / "configs" / "experiments"


def _config(name):
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_a_plain_replay_config_has_exactly_one_unnamed_arm():
    assert MODULE._condition_arms(_config("replay_ext_math500_diss256.yaml")) == [None]


def test_the_restoration_configs_run_all_four_arms():
    for name in ("restoration_smoke.yaml", "restoration_diss256.yaml"):
        assert MODULE._condition_arms(_config(name)) == [
            "none",
            "relaxed",
            "matched",
            "random",
        ], name


def test_the_arm_product_matches_each_config_s_declared_replay_count():
    """The declared count is the pilot guard's sanity number; arms multiply it.

    A mismatch here means the runner would either refuse a correct config or
    silently run a different number of conditions than the queue was sized for.
    """
    for name, problems in (
        ("restoration_smoke.yaml", 3),
        ("restoration_diss256.yaml", 25),
        ("replay_ratio_matched_m3.yaml", 25),
    ):
        config = _config(name)
        arms = MODULE._condition_arms(config)
        expected = (
            problems
            * len(config["seeds"])
            * len(config["presses"])
            * len(config["budgets"])
            * len(arms)
        )
        assert expected == config["estimated_replay_count"], name


def test_the_temporal_null_cannot_be_omitted():
    config = _config("restoration_diss256.yaml")
    config["restoration"]["arms"] = ["relaxed", "matched", "random"]
    with pytest.raises(SystemExit, match="none"):
        MODULE._condition_arms(config)


def test_restoration_and_ratio_matching_cannot_be_combined():
    """Two interventions in one cell make neither attributable."""
    config = _config("restoration_diss256.yaml")
    config["matched_ratio"] = 0.038
    with pytest.raises(SystemExit, match="confounded"):
        MODULE._condition_arms(config)


def test_each_config_selects_the_press_builder_it_asks_for(monkeypatch):
    """The selection itself -- the step that had no test and was wrong."""
    calls = []
    for builder in (
        "build_replay_press",
        "build_ratio_matched_press",
        "build_restoration_press",
    ):
        monkeypatch.setattr(
            MODULE,
            builder,
            lambda *a, _b=builder, **k: calls.append((_b, k)) or object(),
        )

    MODULE._build_condition_press(
        "random",
        budget=256,
        config=_config("replay_ext_math500_diss256.yaml"),
        seed=1,
        arm=None,
    )
    assert calls[-1][0] == "build_replay_press"

    MODULE._build_condition_press(
        "random",
        budget=256,
        config=_config("replay_ratio_matched_m3.yaml"),
        seed=1,
        arm=None,
    )
    assert calls[-1][0] == "build_ratio_matched_press"
    assert calls[-1][1]["ratio"] == 0.038

    MODULE._build_condition_press(
        "random",
        budget=256,
        config=_config("restoration_diss256.yaml"),
        seed=1,
        arm="matched",
    )
    assert calls[-1][0] == "build_restoration_press"
    assert calls[-1][1]["arm"] == "matched"
    assert calls[-1][1]["trigger_step"] == 130

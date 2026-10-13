"""Meeting 5 inventory / ledger (kvcache.meeting5_inventory): coverage against
expected keys, strict-fit cohort, mean-based rate band, cost arithmetic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kvcache import meeting5_inventory as mi

ROOT = Path(__file__).resolve().parents[1]


def _row(pid, seed, press="rkv", wall=10.0, **kw):
    return {
        "model": "m",
        "bench": "math500",
        "press": press,
        "budget": 256,
        "pid": pid,
        "decoding": {"seed": seed},
        "wallclock_s": wall,
        **kw,
    }


def test_inventory_coverage_duplicates_and_foreign_keys():
    rows = [_row(p, s) for p in ("a", "b") for s in (1, 2)]
    (cell,) = mi.inventory_run(
        "r", rows, expected_pids=["a", "b"], expected_seeds=[1, 2]
    )
    assert cell["complete"] and cell["n_covered"] == 4 and cell["n_rows"] == 4
    (cell,) = mi.inventory_run(
        "r", rows + rows[:1], expected_pids=["a", "b"], expected_seeds=[1, 2]
    )
    assert not cell["complete"] and cell["n_duplicates"] == 1
    (cell,) = mi.inventory_run(
        "r", rows[1:], expected_pids=["a", "b"], expected_seeds=[1, 2]
    )
    assert not cell["complete"] and cell["n_covered"] == 3
    (cell,) = mi.inventory_run(
        "r", rows + [_row("z", 1)], expected_pids=["a", "b"], expected_seeds=[1, 2]
    )
    assert cell["complete"] and cell["n_foreign"] == 1  # extra keys are reported


def test_rate_band_uses_the_mean_for_skewed_times():
    band = mi.rate_band([10, 10, 10, 10, 110])
    assert band["mean"] == pytest.approx(30.0)
    assert band["median"] == 10
    assert mi.rate_band([1, 2, 3]) is None


def test_strict_fit_and_relaxed_budgets():
    sf = mi.strict_fit({"a": 192, "b": 193, "c": 344}, 256)
    assert sf["strict_fit_pids"] == ["a"]  # 192 + 64 = 256 fits exactly
    assert sf["relaxed_effective_budgets"] == {"b": 257, "c": 408}


def test_comparable_rates_and_cost():
    cells = [
        {
            "run": "x",
            "press": p,
            "model": "m",
            "bench": "math500",
            "budget": "256",
            "complete": True,
            "wallclock_s": {"mean": s},
        }
        for p, s in (("random", 100.0), ("h2o", 200.0))
    ] + [
        {
            "run": "y",
            "press": "rkv",
            "model": "m",
            "bench": "math500",
            "budget": "256",
            "complete": False,
            "wallclock_s": {"mean": 999.0},
        }
    ]
    r = mi.comparable_rates(cells, model="m", bench="math500", budget="256")
    assert (r["lower_s"], r["central_s"], r["upper_s"]) == (100.0, 150.0, 200.0)
    # 360 rows x 100 s = 10 h, + 2 launches x 0.05 h = 10.1 h, x 1.10 = 11.11
    assert mi.cost_hours(360, r, n_launches=2)["lower"] == pytest.approx(11.11)
    with pytest.raises(ValueError):
        mi.comparable_rates(cells, model="m", bench="gsm8k", budget="256")


def test_committed_ledger_matches_manifest_and_inventory(tmp_path):
    out = tmp_path / "ledger.json"
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/plan_meeting5_cost.py"),
            "--ledger",
            "--inventory-json",
            str(ROOT / "results/diag/meeting5/inventory.json"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert p.returncode == 0, p.stderr
    fresh = json.loads(out.read_text())
    committed = json.loads((ROOT / "results/diag/meeting5/ledger.json").read_text())
    assert fresh == committed  # the archived ledger is reproducible
    assert fresh["strict_fit"]["256"]["n_strict_fit"] == 97
    assert len(fresh["reuse"]) == 7


def test_hashed_split_is_disjoint_stable_and_outcome_free():
    ids = [f"x/{i}" for i in range(50)]
    a = mi.hashed_split(
        ids, salt="s", n_calibration=5, n_confirmation=20, exclude=ids[:10]
    )
    assert a == mi.hashed_split(
        ids, salt="s", n_calibration=5, n_confirmation=20, exclude=ids[:10]
    )
    parts = [set(a[k]) for k in ("calibration", "confirmation", "reserve")]
    assert sum(len(p) for p in parts) == 40 and len(set().union(*parts)) == 40
    assert not set(ids[:10]) & set().union(*parts)
    b = mi.hashed_split(ids, salt="t", n_calibration=5, n_confirmation=20)
    assert b["confirmation"] != a["confirmation"]  # the salt is the only lever
    with pytest.raises(ValueError):
        mi.hashed_split(ids, salt="s", n_calibration=40, n_confirmation=20)


def test_committed_problem_manifests():
    d = json.loads((ROOT / "configs/meeting5/problem_manifests.json").read_text())
    for name in ("math500", "gsm8k", "bbh_logical_deduction_five_objects"):
        m = d[name]
        parts = [set(m[k]) for k in ("calibration", "confirmation", "reserve")]
        assert sum(map(len, parts)) == len(set().union(*parts)) == m["n_pool"]
        assert len(m["confirmation"]) == 200
    cohort = set(
        json.loads((ROOT / "configs/meeting3/grid100_pids.json").read_text())["pids"]
    )
    assert not cohort & set(
        d["math500"]["confirmation"]
    )  # discovery is not confirmation
    bbh = d["bbh_logical_deduction_five_objects"]
    again = mi.hashed_split(
        [f"bbh_lde5/{i}" for i in range(250)],
        salt=bbh["salt"],
        n_calibration=25,
        n_confirmation=200,
    )
    assert {k: bbh[k] for k in again} == again  # reproducible without the data
    assert bbh["revision"].startswith("TODO(human)")


def test_readout_inventory_keys_on_the_reference_trace():
    rows = [
        {
            "reference_key": ["aime24", p, "m", "fullcache", "full", s],
            "condition": {"press": "rkv", "budget": 2048},
            "readout_defined": p != "b",
        }
        for p in ("a", "b")
        for s in (1, 2)
    ]
    (c,) = mi.inventory_readout_run(
        "r", rows, expected_pids=["a", "b"], expected_seeds=[1, 2]
    )
    assert c["complete"] and c["n_defined"] == 2 and c["press"] == "rkv"

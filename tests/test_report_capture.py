"""R.14 capture read: exact coverage, controls, identity, equal-problem weights."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_capture import POLICIES, read_capture, trigger_ratio  # noqa: E402

from kvcache.coverage import (  # noqa: E402
    equal_problem_bootstrap,
    paired_equal_problem,
)
from kvcache.stats import cluster_bootstrap_mean  # noqa: E402

PIDS, SEEDS, B = ["a", "b", "c", "d", "e", "f"], [1, 2], 256
PROV = {"kind": "capture", "implementation_id": "x", "config_sha256": "y"}


def _row(pid, seed, press, i=0, **kw):
    coll = {"rkv": 0.10, "h2o": 0.30}.get(press, 0.2) + 0.01 * i
    omega = {"random": 2.0, "streamingllm": 1.0}.get(press, 1.5) + 0.1 * i
    r = {
        "pid": pid,
        "seed": seed,
        "press": press,
        "budget": B,
        "provenance": dict(PROV),
        "checks": {"n_nonfinite": 0, "layer0_delta_max": 1e-7},
        "collateral_pooled": coll,
        "wanted_any_head_pooled": 0.5,
        "M_mean": 0.2,
        "M_event_mean": 0.01,
        "Delta_mean": 0.1,
        "slices": {"trigger": {"M": [[0.2, 0.4]], "Delta": [[0.1, 0.1]]}},
        "latent": {"omega_trace": omega},
    }
    r.update(kw)
    return r


def _grid():
    return [
        _row(p, s, press, i)
        for i, p in enumerate(PIDS)
        for s in SEEDS
        for press in POLICIES
    ]


def _read(rows):
    return read_capture(rows, pids=PIDS, seeds=SEEDS, budget=B, n_resamples=200)


def test_complete_valid_grid_is_final_with_verdicts():
    out = _read(_grid())
    assert out["status"] == "FINAL"
    assert out["q3_registered_contrast"]["mean"] == pytest.approx(-0.2)
    assert out["q3_registered_contrast"]["verdict"].startswith("SUPPORTS")
    assert out["q6_registered_contrast"]["verdict"] == "PREDICTED"
    assert len(out["sufficient_statistics"]) == 48


def test_duplicates_cannot_substitute_for_missing_conditions():
    rows = _grid()
    victim = next(r for r in rows if r["pid"] == "a" and r["seed"] == 2)
    rows.remove(victim)
    rows.append(copy.deepcopy(rows[0]))  # same count, one key twice
    out = _read(rows)
    assert out["coverage"]["status"] == "INVALID"
    assert out["coverage"]["missing"] and out["coverage"]["duplicated"]
    assert out["status"] != "FINAL"
    assert out["q3_registered_contrast"]["verdict"] is None


def test_a_missing_seed_is_partial():
    rows = [r for r in _grid() if not (r["pid"] == "b" and r["seed"] == 2)]
    out = _read(rows)
    assert out["coverage"]["status"] == "PARTIAL" and out["status"] == "PARTIAL"
    assert len(out["coverage"]["missing"]) == 4


def test_an_unexpected_problem_and_a_second_budget_are_invalid():
    rows = _grid() + [_row("zz", 1, "rkv")]
    assert _read(rows)["coverage"]["unexpected"] == [["zz", 1, "rkv", "256"]]
    mixed = _grid() + [_row("a", 1, "rkv", budget=2048)]
    out = _read(mixed)
    assert out["coverage"]["status"] == "INVALID" and out["status"] == "INVALID"


def test_failed_and_non_finite_controls_invalidate_and_are_listed():
    rows = _grid()
    rows[0]["checks"]["layer0_delta_max"] = 1e-3
    rows[1]["checks"]["layer0_delta_max"] = float("nan")
    rows[2]["checks"]["n_nonfinite"] = 3
    out = _read(rows)
    assert out["controls"] == "INVALID" and out["status"] == "INVALID"
    assert len(out["control_failures"]) == 3
    assert out["q6_registered_contrast"]["verdict"] is None


def test_mixed_provenance_is_not_final():
    rows = _grid()
    rows[5]["provenance"]["config_sha256"] = "other"
    out = _read(rows)
    assert not out["identity_consistent"] and out["status"] != "FINAL"


def test_an_undefined_direction_is_an_exclusion_not_a_failed_instrument():
    rows = _grid()
    for r in rows:
        if r["press"] == "random" and r["pid"] == "a" and r["seed"] == 1:
            r["latent"]["omega_trace"] = None
    out = _read(rows)
    assert out["status"] == "FINAL" and out["controls"] == "VALID"
    ex = out["eligibility_exclusions"]["omega"]["random"]
    assert sum(ex.values()) == 1
    assert out["q6_registered_contrast"]["n_pairs"] == 11
    # problem "a" keeps its one remaining seed, weighted as one problem
    assert out["q6_registered_contrast"]["seeds_per_problem"]["a"] == 1


def test_four_versus_one_seed_gives_the_equal_problem_mean():
    a = {("A", s): 1.0 for s in range(4)} | {("B", 0): 0.0}
    b = {k: 0.0 for k in a}
    out = paired_equal_problem(a, b, n_resamples=200)
    assert out["mean"] == pytest.approx(0.5)  # not 0.8 (observation-weighted)
    assert out["seeds_per_problem"] == {"A": 4, "B": 1}


def test_balanced_grids_match_the_historical_estimator():
    vals = {(p, s): float(i + s) for i, p in enumerate("abcd") for s in (1, 2, 3)}
    new = equal_problem_bootstrap(vals, n_resamples=300, seed=0)
    per = {p: [v for (q, _), v in vals.items() if q == p] for p in "abcd"}
    old = cluster_bootstrap_mean(per, n_resamples=300, seed=0)
    assert new["mean"] == pytest.approx(old["mean"])


def test_trigger_ratio_is_mean_delta_over_mean_m():
    r = _row("a", 1, "rkv")
    assert trigger_ratio(r) == pytest.approx(0.1 / 0.3)
    r["slices"] = {}
    assert trigger_ratio(r) is None


def test_cli_final_exits_nonzero_when_incomplete(tmp_path, monkeypatch):
    import report_capture as rc

    run = tmp_path / "cap" / "math500" / "m"
    run.mkdir(parents=True)
    rows = _grid()[:-1]
    (run / "x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(rc, "_design", lambda _c: (PIDS, SEEDS, B))
    out = tmp_path / "o.json"
    argv = [
        "x",
        "--runs-root",
        str(tmp_path),
        "--run",
        "cap",
        "--config",
        "c.yaml",
        "--out",
        str(out),
        "--final",
        "--n-resamples",
        "50",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert rc.main() == 3
    d = json.loads(out.read_text())
    assert d["status"] == "PARTIAL" and d["sufficient_statistics_sha256"]
    assert (tmp_path / "o.conditions.tsv").exists()

"""Replication cost model (kvcache.replication_cost): hand-computed cases."""

from __future__ import annotations

import pytest

from kvcache.replication_cost import (
    grid_cost,
    plan_stages,
    press_rates,
    readout_cost,
    readout_rate,
)


def test_press_rates_by_hand():
    rows = [
        {
            "press": "p",
            "pid": "a",
            "seed": 1,
            "n_tokens": 100,
            "wallclock_s": 2.0,
            "truncated": False,
        },
        {
            "press": "p",
            "pid": "b",
            "seed": 1,
            "n_tokens": 300,
            "wallclock_s": 4.0,
            "truncated": True,
        },
    ]
    got = press_rates(rows, {("a", 1): 50, ("b", 1): 150})["p"]
    assert got["s_per_token"] == pytest.approx(6 / 400)
    assert got["inflation"] == pytest.approx(400 / 200)
    assert got["mean_tokens"] == pytest.approx(200)
    assert got["truncation_rate"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no matched"):
        press_rates(rows, {})


def test_grid_cost_caps_inflated_traces_and_adds_retry_allowance():
    # per trace: min(1500, 2*100)=200 and min(1500, 2*1000)=1500 -> mean 850;
    # 10 rows * 850 tok * 0.01 s = 85 s = 0.02361 h, x1.1 = 0.026 -> 0.03
    got = grid_cost(
        [100, 1000],
        n_rows=10,
        s_per_token=0.01,
        inflation=2.0,
        cap=1500,
        n_launches=0,
    )
    assert got["mean_tokens_central"] == pytest.approx(850)
    assert got["central"] == 0.03
    # upper: 10 * 1500 * 0.01 = 150 s = 0.04167 h, x1.1 = 0.0458 -> 0.05
    assert got["upper"] == 0.05
    # two launches add 2 * 0.05 h before the allowance: (0.02361+0.1)*1.1
    two = grid_cost(
        [100, 1000],
        n_rows=10,
        s_per_token=0.01,
        inflation=2.0,
        cap=1500,
        n_launches=2,
    )
    assert two["central"] == pytest.approx(0.14)


def test_readout_rate_and_cost_by_hand():
    assert readout_rate(1.0, 10, 300, 60) == pytest.approx(1.0)
    # central: 100 * (1000 + 100) * 0.01 s = 1100 s = 0.3056 h, x1.1 -> 0.34
    # upper:   100 * (1000 + 2048) * 0.01 s = 3048 s = 0.8467 h, x1.1 -> 0.93
    got = readout_cost(
        [1000],
        n_records=100,
        s_per_token=0.01,
        mean_answer=100,
        max_answer_tokens=2048,
        n_launches=0,
    )
    assert (got["central"], got["upper"]) == (0.34, 0.93)
    with pytest.raises(ValueError):
        readout_rate(1.0, 0, 1, 1)


RATES = {
    "source": "test",
    "presses": {"p": {"s_per_token": 0.01, "inflation": 1.0}},
    "readout": {
        "forward": {"s_per_token": 0.01, "mean_answer_tokens": 100.0},
        "reverse": {"s_per_token": 0.005},
    },
}
DESIGN = {
    "stages": [
        {
            "id": "S",
            "benchmark": "x",
            "answers": "q",
            "n_problems": 3,
            "n_seeds": 2,
            "max_new_tokens": 1000,
            "max_answer_tokens": 2048,
            "cells": [{"name": "g", "kind": "grid", "presses": ["p"]}],
        },
        {
            "id": "U",
            "benchmark": "unpriced",
            "answers": "q",
            "n_problems": 1,
            "n_seeds": 1,
            "max_new_tokens": 1,
            "max_answer_tokens": 1,
            "cells": [],
        },
    ]
}


def test_plan_stages_counts_rows_and_labels_cap_bound_prices():
    own = plan_stages(
        DESIGN,
        RATES,
        {"x": {"conditions": [{"n_tokens": 500}], "seconds_per_token": 0.02}},
    )
    s = own["stages"][0]
    assert s["rows"] == 6 and s["priced_from"] == "own calibration bank"
    assert own["stages"][1]["status"].startswith("UNPRICED")
    capped = plan_stages(
        DESIGN, RATES, {"x": {"cap_bound": True, "seconds_per_token": 0.02}}
    )["stages"][0]
    assert capped["priced_from"].startswith("cap bound")
    # a cap-bound price is the upper bound: central == upper
    assert capped["gpu_hours_central"] == capped["gpu_hours_upper"]
    assert capped["gpu_hours_upper"] >= s["gpu_hours_upper"]

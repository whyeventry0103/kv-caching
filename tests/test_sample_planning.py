"""Sample planning (plan §9.2): hand-checked SD and normal N, and the simulator's
size/power behaviour on a table whose answer is known."""

from __future__ import annotations

import pytest

from kvcache.metrics import sample_planning as sp


def test_per_problem_values_and_problem_sd_by_hand():
    pp = sp.per_problem_values({("a", 1): 1.0, ("a", 2): 0.0, ("b", 1): 1.0})
    assert pp == {"a": [1.0, 0.0], "b": [1.0]}
    # problem means 0.5 and 1.0 -> sample SD = 0.3536
    assert sp.problem_sd(pp) == pytest.approx(0.35355, abs=1e-4)


def test_normal_n():
    # (1.96 * 0.5 / 0.025)^2 = 1536.64 -> 1537
    assert sp.normal_n(0.5, 0.025) == 1537


def test_simulator_size_power_and_seed_subsampling():
    pp = {f"p{i}": [float(i % 3 - 1), float((i + 1) % 3 - 1)] * 2 for i in range(60)}
    null = sp.simulate(pp, n_problems=100, n_seeds=2, effect=0.0, reps=300)
    big = sp.simulate(pp, n_problems=100, n_seeds=2, effect=0.5, reps=300)
    assert null["share_excluding_zero"] < 0.12  # about 5% nominal
    assert big["share_excluding_zero"] > 0.95
    more = sp.simulate(pp, n_problems=400, n_seeds=2, effect=0.0, reps=100)
    assert more["mean_half_width"] < null["mean_half_width"]
    with pytest.raises(ValueError, match="no discovery"):
        sp.simulate({}, n_problems=10, n_seeds=2, effect=0.0)

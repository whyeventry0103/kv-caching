"""Protocol audit (execution plan Sections 4 and 9): hand-computed 2 x 2 cases,
worst-case bounds, practical margins, the shrinkage ratio and joint contrasts.

The tables are small enough to verify every number by hand; the bootstrap is
checked against the registered single-series estimator it must reproduce.
"""

from __future__ import annotations

import pytest

from kvcache.coverage import equal_problem_bootstrap
from kvcache.metrics import protocol_audit as pa


def _keys(n_problems: int, seeds=(1, 2)) -> list[tuple[str, int]]:
    return [(f"p{i}", s) for i in range(n_problems) for s in seeds]


def test_single_series_draws_match_registered_estimator():
    vals = {("a", 1): 1.0, ("a", 2): 0.0, ("b", 1): 1.0, ("c", 1): 0.0}
    ref = equal_problem_bootstrap(vals, n_resamples=500, seed=0)
    point, reps = pa.joint_draws({"x": vals}, n_resamples=500, seed=0)
    s = pa.summarize(point["x"], [r["x"] for r in reps], n_resamples=500, seed=0)
    assert s["mean"] == pytest.approx(ref["mean"])  # (0.5 + 1 + 0) / 3
    assert s["mean"] == pytest.approx(0.5)
    assert s["ci95"] == pytest.approx(ref["ci95"])


def test_joint_draws_refuse_mismatched_series():
    with pytest.raises(ValueError, match="different conditions"):
        pa.joint_draws({"x": {("a", 1): 1.0}, "y": {("b", 1): 1.0}})
    with pytest.raises(ValueError, match="empty"):
        pa.joint_draws({"x": {}})


def _toy():
    # Two problems x two seeds. p0: FF all right; p1: FF one right.
    pop = [("p0", 1), ("p0", 2), ("p1", 1), ("p1", 2)]
    ff = {("p0", 1): True, ("p0", 2): True, ("p1", 1): True, ("p1", 2): False}
    cc = {("p0", 1): False, ("p0", 2): True, ("p1", 1): False, ("p1", 2): False}
    # forward readout undefined at (p1, 2); reverse undefined at (p0, 1)
    fc = {("p0", 1): True, ("p0", 2): True, ("p1", 1): False}
    cf = {("p0", 2): True, ("p1", 1): True, ("p1", 2): False}
    return pop, ff, cc, fc, cf


def test_two_by_two_supports_and_hand_values():
    pop, ff, cc, fc, cf = _toy()
    out = pa.two_by_two(ff=ff, fc=fc, cc=cc, cf=cf, population=pop, n_resamples=200)
    p = out["population"]
    assert (p["n_forward_defined"], p["n_reverse_defined"]) == (3, 3)
    assert p["n_common_support"] == 2
    assert p["forward_only"] == [["p0", 1]]
    assert p["reverse_only"] == [["p1", 2]]
    assert p["neither"] == []
    # full population: A_FF = (1 + 0.5)/2 = 0.75, A_CC = (0.5 + 0)/2 = 0.25
    free = out["free_run_full_population"]
    assert free["D_total"]["mean"] == pytest.approx(-0.5)
    # common support {(p0,2), (p1,1)}: FF 1,1  FC 1,0  CC 1,0  CF 1,1
    cs = out["common_support"]
    assert cs["accuracy"]["ff"]["mean"] == pytest.approx(1.0)
    assert cs["accuracy"]["fc"]["mean"] == pytest.approx(0.5)
    assert cs["accuracy"]["cc"]["mean"] == pytest.approx(0.5)
    assert cs["accuracy"]["cf"]["mean"] == pytest.approx(1.0)
    assert cs["readout_on_reference_trace"]["mean"] == pytest.approx(-0.5)
    assert cs["residual_forward"]["mean"] == pytest.approx(0.0)
    assert cs["trajectory_under_full_readout"]["mean"] == pytest.approx(0.0)
    assert cs["readout_on_compressed_trace"]["mean"] == pytest.approx(-0.5)
    # I = residual_forward - trajectory_under_full_readout = 0 - 0
    assert cs["interaction"]["mean"] == pytest.approx(0.0)
    # both decompositions add up to D on the same support, exactly
    for blk in (cs,):
        assert blk["readout_on_reference_trace"]["mean"] + blk["residual_forward"][
            "mean"
        ] == (pytest.approx(blk["D_total"]["mean"]))
        assert blk["trajectory_under_full_readout"]["mean"] + blk[
            "readout_on_compressed_trace"
        ]["mean"] == (pytest.approx(blk["D_total"]["mean"]))


def test_worst_case_bounds_by_hand():
    pop, ff, cc, fc, cf = _toy()
    # residual_forward = A_CC - A_FC over the population; FC unknown at (p1, 2).
    # FC filled 0: p0 FC 1, p1 FC (0+0)/2 = 0 -> A_FC 0.5; filled 1: p1 0.5 -> 0.75
    # A_CC = 0.25 -> residual_forward in [0.25 - 0.75, 0.25 - 0.5] = [-0.5, -0.25]
    lo, hi = pa.worst_case_bounds(
        {"cc": cc, "fc": fc}, pop, pa.CONTRASTS["residual_forward"]
    )
    assert (lo, hi) == (pytest.approx(-0.5), pytest.approx(-0.25))
    # a fully defined contrast has zero-width bounds
    lo, hi = pa.worst_case_bounds({"cc": cc, "ff": ff}, pop, pa.CONTRASTS["D_total"])
    assert lo == pytest.approx(hi) == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"fc": {("zz", 1): True}}, "outside the population"),
        ({"cc": {("p0", 1): True}}, "does not cover"),
        ({"cf": {("p0", 2): None}}, "non-bool"),
    ],
)
def test_two_by_two_refuses_malformed_inputs(bad, match):
    pop, ff, cc, fc, cf = _toy()
    arms = {"ff": ff, "fc": fc, "cc": cc, "cf": cf, **bad}
    with pytest.raises(ValueError, match=match):
        pa.two_by_two(population=pop, n_resamples=10, **arms)


def test_practical_reading_false_reassurance_and_lossy_readout():
    keys = _keys(60)
    ff = {k: True for k in keys}
    cc = {k: False for k in keys}
    fine = {k: True for k in keys}
    out = pa.practical_reading(ff=ff, cc=cc, fc=fine, n_resamples=200)
    assert out["F_loss"]["mean"] == pytest.approx(1.0)
    assert out["R_loss"]["mean"] == pytest.approx(0.0)
    for m in ("0.01", "0.03", "0.05"):
        assert out["margins"][m]["false_reassurance"] is True
        assert out["margins"][m]["readout_already_lossy"] is False
        assert out["margins"][m]["residual_equivalent_within_margin"] is False
    lossy = {k: False for k in keys}
    out = pa.practical_reading(ff=ff, cc=cc, fc=lossy, n_resamples=200)
    assert out["margins"]["0.03"]["false_reassurance"] is False
    assert out["margins"]["0.03"]["readout_already_lossy"] is True
    # residual exactly 0 on every condition -> equivalent within every margin
    assert out["margins"]["0.01"]["residual_equivalent_within_margin"] is True
    assert "SENSITIVITY" in out["margin_status"]


def test_practical_reading_requires_defined_readouts_inside_free_runs():
    with pytest.raises(ValueError, match="no defined"):
        pa.practical_reading(ff={("a", 1): True}, cc={("a", 1): True}, fc={})
    with pytest.raises(ValueError, match="do not cover"):
        pa.practical_reading(ff={}, cc={}, fc={("a", 1): True})


def test_shrinkage_ratio_point_and_absolute_change():
    keys = _keys(4)
    base = {k: -1.0 for k in keys}
    pp = {k: (-0.5 if k[1] == 1 else 0.0) for k in keys}  # mean -0.25
    out = pa.shrinkage_ratio(pp, base, n_resamples=200)
    assert out["ratio"] == pytest.approx(0.75)  # 1 - 0.25 / 1
    assert out["absolute_change"]["mean"] == pytest.approx(0.75)
    assert out["share_base_near_zero"] == 0.0
    assert out["share_sign_flipped"] == 0.0
    assert out["ci95"][0] <= 0.75 <= out["ci95"][1]
    with pytest.raises(ValueError, match="different conditions"):
        pa.shrinkage_ratio({("a", 1): 0.0}, {("b", 1): 0.0})


def test_policy_contrasts_joint_labels():
    keys = _keys(40)
    acc = {"a": {k: 1.0 for k in keys}, "b": {k: 0.0 for k in keys}}
    loss = {"a": {k: 0.9 for k in keys}, "b": {k: 0.1 for k in keys}}
    out = pa.policy_contrasts(acc, loss, [("a", "b")], n_resamples=200)
    blk = out["a - b"]
    assert blk["accuracy"]["mean"] == pytest.approx(1.0)
    assert blk["metric"]["mean"] == pytest.approx(0.8)
    assert blk["joint"].startswith("MISMATCH ESTABLISHED")
    flat = {"a": {k: 0.5 for k in keys}, "b": {k: 0.5 for k in keys}}
    out = pa.policy_contrasts(flat, loss, [("a", "b")], n_resamples=200)
    assert out["a - b"]["joint"] == "NOT ESTABLISHED"
    out = pa.policy_contrasts(acc, None, [("a", "b")], n_resamples=50)
    assert "metric" not in out["a - b"]


def test_by_key_refuses_duplicates_and_drops_undefined():
    rows = [{"k": ("a", 1), "v": True}, {"k": ("b", 1), "v": None}]
    got = pa.by_key(rows, key=lambda r: r["k"], value=lambda r: r["v"])
    assert got == {("a", 1): True}
    with pytest.raises(ValueError, match="duplicate"):
        pa.by_key(rows + rows[:1], key=lambda r: r["k"], value=lambda r: r["v"])


def test_protection_by_direction_pairs_on_the_four_arm_support():
    pop, ff, cc, fc, cf = _toy()
    base = {"ff": ff, "fc": fc, "cc": cc, "cf": cf}
    # protected cell: every arm right where defined; reverse undefined at (p1, 1)
    prot = {
        "ff": ff,
        "fc": {k: True for k in pop},
        "cc": {k: True for k in pop},
        "cf": {k: True for k in pop if k != ("p1", 1)},
    }
    out = pa.protection_by_direction(prot, base, n_resamples=100)
    # base support {(p0,2),(p1,1)} minus the protected hole (p1,1) -> {(p0,2)}
    assert out["n_conditions"] == 1
    # at (p0,2) every arm of both cells is right: all contrasts 0, ratio undefined
    assert out["residual_forward"]["absolute_change"]["mean"] == pytest.approx(0.0)
    assert out["residual_forward"]["ratio"] is None
    with pytest.raises(ValueError, match="all four arms"):
        pa.protection_by_direction(prot, {**base, "cf": {}}, n_resamples=10)

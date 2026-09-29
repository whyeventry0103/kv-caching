"""Amendment S reads: hand-computed cases for the registered estimands.

The branch labels are lookups on intervals, so each threshold is pinned at its
boundary; the paired contrasts and the ratios are checked against values
computed by hand on tables small enough to do so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kvcache.metrics import s_reads as s

ROOT = Path(__file__).resolve().parents[1]


def test_residual_and_effect_labels_at_their_boundaries():
    assert s.residual_label([-0.30, -0.01]) == "RESIDUAL PERSISTS"
    assert s.residual_label([-0.30, 0.0]) == "RESIDUAL NOT DETECTED"
    assert s.residual_label([-0.05, 0.05]) == "RESIDUAL NOT DETECTED"
    assert s.residual_label([0.01, 0.05]).startswith("EXCLUDES ZERO, POSITIVE")
    assert s.protection_effect_label([0.01, 0.2]) == "SHRINKS"
    assert s.protection_effect_label([-0.2, -0.01]) == "GROWS"
    assert s.protection_effect_label([0.0, 0.2]) == "UNCHANGED"


def test_rho_labels_at_their_boundaries():
    assert s.rho_label([0.50, 0.90]) == "PROMPT CARRIES IT"
    assert s.rho_label([0.49, 0.90]) == "INDETERMINATE"
    assert s.rho_label([0.00, 0.20]) == "PROMPT MINOR"
    assert s.rho_label([0.00, 0.21]) == "INDETERMINATE"
    assert s.rho_label([0.21, 0.49]) == "PARTIAL"
    assert s.rho_label([0.20, 0.49]) == "INDETERMINATE"


def _tables():
    # two problems, two seeds each; readout undefined for ("b", 2)
    keys = [("a", 1), ("a", 2), ("b", 1), ("b", 2)]
    free_pp = dict(zip(keys, [True, True, False, True], strict=True))
    free_base = dict(zip(keys, [False, True, False, False], strict=True))
    ro_keys = keys[:3]
    ro_pp = dict(zip(ro_keys, [True, True, True], strict=True))
    ro_base = dict(zip(ro_keys, [True, False, True], strict=True))
    return free_pp, free_base, ro_pp, ro_base


def test_protection_contrasts_match_hand_computation():
    free_pp, free_base, ro_pp, ro_base = _tables()
    out = s.protection_contrasts(free_pp, free_base, ro_pp, ro_base, n_resamples=200)
    # free: a = (1+1)/2, b = (0+1)/2 -> 0.75 ; base a = 0.5, b = 0 -> 0.25
    assert out["free_accuracy_pp"]["mean"] == pytest.approx(0.75)
    assert out["delta_free"]["mean"] == pytest.approx(0.5)
    # residual per condition = free - readout on the 3 defined conditions
    #   pp:   a1 0, a2 0, b1 -1  -> a 0, b -1   -> -0.5
    #   base: a1 -1, a2 +1, b1 -1 -> a 0, b -1  -> -0.5
    assert out["residual_pp"]["mean"] == pytest.approx(-0.5)
    assert out["residual_base"]["mean"] == pytest.approx(-0.5)
    assert out["delta_residual"]["mean"] == pytest.approx(0.0)
    # readout: pp a 1, b 1 ; base a 0.5, b 1 -> +0.25
    assert out["delta_readout"]["mean"] == pytest.approx(0.25)
    assert (out["n_conditions"], out["n_readout_defined"]) == (4, 3)


def test_sensitivity_read_drops_the_problem_from_both_arms():
    free_pp, free_base, ro_pp, ro_base = _tables()
    out = s.protection_contrasts(
        free_pp, free_base, ro_pp, ro_base, exclude_pids=["b"], n_resamples=50
    )
    assert out["n_conditions"] == 2 and out["excluded_problems"] == ["b"]
    assert out["residual_pp"]["mean"] == pytest.approx(0.0)
    assert out["delta_free"]["mean"] == pytest.approx(0.5)


def test_mismatched_populations_are_refused():
    free_pp, free_base, ro_pp, ro_base = _tables()
    with pytest.raises(ValueError, match="different conditions"):
        s.protection_contrasts(
            {k: v for k, v in free_pp.items() if k != ("b", 2)},
            free_base,
            ro_pp,
            ro_base,
        )
    with pytest.raises(ValueError, match="answer boundary"):
        s.protection_contrasts(free_pp, free_base, {**ro_pp, ("b", 2): True}, ro_base)


def test_flip_masses_are_over_the_whole_cell():
    ref = {("a", 1): True, ("a", 2): True, ("b", 1): False, ("b", 2): False}
    cell = {("a", 1): False, ("a", 2): True, ("b", 1): True, ("b", 2): False}
    assert s.flip_masses(ref, cell) == {"harmful": 0.25, "beneficial": 0.25}


def test_paper_consequence_covers_every_registered_branch():
    P, N = "RESIDUAL PERSISTS", "RESIDUAL NOT DETECTED"

    def both(a, b):
        return {
            "random_pp": {256: a, 1024: b},
            "streamingllm_pp": {256: a, 1024: b},
        }

    assert s.paper_consequence(both(P, P)).startswith("A:")
    assert s.paper_consequence(both(P, N)).startswith("B:")
    assert s.paper_consequence(both(N, N)).startswith("C:")
    assert s.paper_consequence(both(N, P)).startswith("FITS NO BRANCH")
    mixed = {"random_pp": {256: P, 1024: P}, "streamingllm_pp": {256: N, 1024: N}}
    assert s.paper_consequence(mixed).startswith("PRESSES DISAGREE")
    assert s.paper_consequence({"random_pp": {256: P}}).startswith("INCOMPLETE")


def _tv(frozen, prompt, trace, full):
    keys = [(p, sd) for p in "abcdefgh" for sd in (1, 2)]
    return {
        "frozen": dict.fromkeys(keys, frozen),
        "prompt": dict.fromkeys(keys, prompt),
        "trace": dict.fromkeys(keys, trace),
        "full": dict.fromkeys(keys, full),
    }


def test_origin_ratios_match_hand_computation():
    out = s.origin_ratios(_tv(0.10, 0.04, 0.09, 0.02), n_resamples=200)
    # gap 0.08 ; prompt recovers 0.06 -> 0.75 ; trace recovers 0.01 -> 0.125
    assert out["rho_prompt"]["ratio"] == pytest.approx(0.75)
    assert out["rho_trace"]["ratio"] == pytest.approx(0.125)
    assert out["rho_prompt_minus_trace"]["mean"] == pytest.approx(0.625)
    assert out["label"] == "PROMPT CARRIES IT"
    assert out["specificity"] == "PROMPT-SPECIFIC"
    assert (out["n_traces"], out["n_problems"]) == (16, 8)


def test_origin_ratios_other_branches():
    minor = s.origin_ratios(_tv(0.10, 0.09, 0.09, 0.02), n_resamples=200)
    assert minor["label"] == "PROMPT MINOR" and minor["specificity"] == "NON-SPECIFIC"
    nothing = s.origin_ratios(_tv(0.10, 0.04, 0.09, 0.10), n_resamples=200)
    assert nothing["label"].startswith("UNDEFINED") and "rho_prompt" not in nothing
    with pytest.raises(ValueError, match="exactly the arms"):
        s.origin_ratios({"frozen": {}, "full": {}})
    bad = _tv(0.10, 0.04, 0.09, 0.02)
    bad["trace"].pop(("a", 1))
    with pytest.raises(ValueError, match="same, non-empty"):
        s.origin_ratios(bad)


def test_determinism_control_stops_on_any_difference_or_missing_row():
    v2 = {
        ("a", 1): {"full": 0.03, "none": 0.10},
        ("b", 1): {"full": 0.02, "none": 0.09},
    }
    assert s.determinism_control(dict(v2), v2)["passed"]
    off = {**v2, ("a", 1): {"full": 0.03 + 2e-6, "none": 0.10}}
    res = s.determinism_control(off, v2)
    assert not res["passed"] and res["max_abs_diff"]["full"] == pytest.approx(2e-6)
    extra = {**v2, ("c", 1): {"full": 0.01, "none": 0.05}}
    assert not s.determinism_control(extra, v2)["passed"]
    assert not s.determinism_control({}, v2)["passed"]


# ------------------------------------------------------------- the script
def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _script():
    import importlib.util

    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "report_s_t", ROOT / "scripts" / "report_s.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_runs(tmp: Path, n_pp: int) -> Path:
    runs = tmp / "runs"
    keys = [(f"p{i:03d}", sd) for i in range(100) for sd in (11, 23, 47, 89)]

    def grid(name, presses, budget, correct, n):
        rows, toks = [], []
        for press in presses:
            for pid, sd in keys[:n]:
                rows.append(
                    {
                        "pid": pid,
                        "press": press,
                        "budget": budget,
                        "correct": correct(press, pid),
                        "decoding": {"seed": sd},
                    }
                )
                plen = 353 if pid == "p000" else 50
                toks.append({"pid": pid, "prompt_token_ids": [0] * plen})
        _write(runs / name / "math500" / "m" / "rows.jsonl", rows)
        _write(runs / name / "token_ids" / "math500" / "m" / "t.jsonl", toks)

    def readout(name, presses, budget, correct, n):
        rows = [
            {
                "reference_key": ["math500", pid, "m", "fullcache", "full", sd],
                "condition": {"press": press, "budget": budget},
                "readout_correct": None if pid == "p099" else correct(press, pid),
            }
            for press in presses
            for pid, sd in keys[:n]
        ]
        _write(runs / name / "readout" / "r.jsonl", rows)

    _write(
        runs / "math500-fc-powered-merged" / "math500" / "fc.jsonl",
        [{"pid": p, "correct": True, "decoding": {"seed": sd}} for p, sd in keys],
    )
    base, pp = ["random", "streamingllm"], ["random_pp", "streamingllm_pp"]

    def half(_press, pid):
        return int(pid[1:]) % 2 == 0

    grid("grid-diss256-tok", base, 256, lambda *_: False, 400)
    grid("grid-pp256", pp, 256, half, n_pp)
    readout("readout-diss256-g0", base, 256, lambda *_: True, 200)
    readout("readout-diss256-g1", base, 256, lambda *_: True, 400)
    readout("readout-pp256", pp, 256, lambda *_: True, n_pp)
    return runs


def test_protection_read_end_to_end_on_synthetic_rows(tmp_path, monkeypatch):
    mod = _script()
    monkeypatch.setattr(mod, "OUT", tmp_path / "out")
    runs = _synthetic_runs(tmp_path, 400)
    # readout-diss256-g0 and -g1 overlap on the first 200 conditions
    monkeypatch.setattr(
        "sys.argv",
        ["report_s", "--runs-root", str(runs), "protection", "--budget", "256"],
    )
    with pytest.raises(ValueError, match="duplicate readout row"):
        mod.main()
    (runs / "readout-diss256-g0" / "readout" / "r.jsonl").write_text("")
    assert mod.main() == 0
    d = json.loads((tmp_path / "out" / "protection_b256.json").read_text())
    assert d["status"] == "FINAL" and d["budget_relaxed_problems"] == ["p000"]
    block = d["per_press"]["random_pp"]
    # readout is defined on p000..p098 and always right; free is right on the
    # 50 even problems and wrong on the 49 odd ones: residual = -49/99
    assert block["primary"]["residual_pp"]["mean"] == pytest.approx(-49 / 99)
    assert block["primary"]["residual_label"] == "RESIDUAL PERSISTS"
    assert block["primary"]["delta_free"]["mean"] == pytest.approx(0.5)
    assert block["primary"]["protection_effect"] == "SHRINKS"
    assert block["sensitivity_without_budget_relaxed"]["n_conditions"] == 396
    assert block["flip_mass_pp"] == {"harmful": 0.5, "beneficial": 0.0}


def test_an_incomplete_cell_is_refused_and_a_partial_read_has_no_label(
    tmp_path, monkeypatch
):
    mod = _script()
    monkeypatch.setattr(mod, "OUT", tmp_path / "out")
    runs = _synthetic_runs(tmp_path, 300)
    (runs / "readout-diss256-g0" / "readout" / "r.jsonl").write_text("")
    argv = ["report_s", "--runs-root", str(runs), "protection", "--budget", "256"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit, match="incomplete"):
        mod.main()
    monkeypatch.setattr("sys.argv", [*argv, "--partial"])
    assert mod.main() == 0
    d = json.loads((tmp_path / "out" / "protection_b256_partial.json").read_text())
    assert d["status"] == "PARTIAL"
    assert (
        d["per_press"]["random_pp"]["primary"]["residual_label"] == "PARTIAL: no label"
    )
    monkeypatch.setattr(
        "sys.argv", ["report_s", "--runs-root", str(runs), "consequence"]
    )
    with pytest.raises(SystemExit, match="has not been read"):
        mod.main()

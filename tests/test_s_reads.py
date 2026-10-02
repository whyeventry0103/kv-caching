"""Amendment S reads: hand-computed cases for the registered estimands.

The branch labels are lookups on intervals, so each threshold is pinned at its
boundary; the paired contrasts and the ratios are checked against values
computed by hand on tables small enough to do so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kvcache.metrics import s_reads as s

ROOT = Path(__file__).resolve().parents[1]


def test_residual_and_effect_labels_at_their_boundaries():
    assert s.residual_label([-0.30, -0.01]) == "RESIDUAL PERSISTS"
    assert s.residual_label([-0.30, 0.0]) == s.NOT_DETECTED
    assert s.residual_label([-0.05, 0.05]) == s.NOT_DETECTED
    assert "not evidence of absence" in s.NOT_DETECTED
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


REF = {("a", 1): True, ("a", 2): True, ("b", 1): True, ("b", 2): True}


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
    out = s.protection_contrasts(
        free_pp, free_base, ro_pp, ro_base, reference=REF, n_resamples=200
    )
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
        free_pp,
        free_base,
        ro_pp,
        ro_base,
        reference=REF,
        exclude_pids=["b"],
        n_resamples=50,
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
            reference=REF,
        )
    with pytest.raises(ValueError, match="answer boundary"):
        s.protection_contrasts(
            free_pp, free_base, {**ro_pp, ("b", 2): True}, ro_base, reference=REF
        )


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
    assert "not established" in s.paper_consequence(both(N, N))
    U = "UNINFORMATIVE: floor (both channels at or below 0.05)"
    assert s.paper_consequence(both(U, P)).startswith("UNINFORMATIVE")
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


# The script-level tests moved to tests/test_s_acceptance.py, which drives the
# real CLI on complete fixtures and the reviewed mutations (2026-10-02).


def test_informativeness_rule_is_explicit_and_exploratory():
    assert s.informativeness(0.0, 0.0, 0.9).startswith("UNINFORMATIVE: floor")
    assert s.informativeness(0.05, 0.05, 0.9).startswith("UNINFORMATIVE: floor")
    assert s.informativeness(0.06, 0.0, 0.9) == "INFORMATIVE"
    assert s.informativeness(0.9, 0.95, 0.9).startswith("UNINFORMATIVE: ceiling")
    assert s.informativeness(0.4, 0.7, 0.9) == "INFORMATIVE"
    assert s.UNINFORMATIVE_FLOOR == 0.05

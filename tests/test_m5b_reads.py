"""M5-B per-dataset read (kvcache.m5b_reads): hand-computed cases."""

from __future__ import annotations

import pytest

from kvcache.m5b_reads import exposure, read_dataset

EXPECTED = {("a", 11), ("a", 23), ("b", 11), ("b", 23)}
EQUIV = {"alpha": 0.05, "primary_margin": 0.03, "sensitivity_margin": 0.05}


def _rows(correct: dict, *, trunc=(), prompt=60, n_tokens=300, ratio=0.7):
    return {
        k: {
            "correct": correct[k],
            "truncated": k in trunc,
            "compression_ratio": ratio,
            "prompt_tokens": prompt,
            "n_tokens": n_tokens,
        }
        for k in EXPECTED
    }


ALL = dict.fromkeys(EXPECTED, True)
NONE = dict.fromkeys(EXPECTED, False)
HALF = {("a", 11): True, ("a", 23): True, ("b", 11): False, ("b", 23): False}


def _read(arms, readouts=None):
    return read_dataset(
        _rows(ALL),
        arms,
        readouts or {},
        expected=EXPECTED,
        budget=256,
        reference="random",
        pairings=("h2o", "streamingllm"),
        descriptive=("rkv",),
        equivalence=EQUIV,
        n_resamples=200,
    )


def test_protected_only_stage_reports_contrasts_but_no_interaction():
    arms = {
        ("random", True): _rows(ALL),
        ("streamingllm", True): _rows(HALF),
        ("h2o", True): _rows(NONE, trunc={("b", 23)}),
        ("rkv", True): _rows(ALL),
    }
    got = _read(arms)
    assert got["full_cache"]["accuracy"]["mean"] == pytest.approx(1.0)
    assert got["arms"]["streamingllm+P"]["loss_vs_full_cache"]["mean"] == (
        pytest.approx(-0.5)
    )
    assert got["arms"]["h2o+P"]["truncation_rate"] == pytest.approx(0.25)
    side = got["sides"]["protected"]
    assert side["contrasts"]["random-streamingllm"]["mean"] == pytest.approx(0.5)
    assert side["contrasts"]["random-h2o"]["mean"] == pytest.approx(1.0)
    assert side["contrasts"]["random-rkv"]["role"] == "descriptive"
    fam = side["registered_inference"]
    assert set(fam["members"]) == {"random-h2o", "random-streamingllm"}
    assert got["sides"]["unprotected"]["status"] == "NOT RUN"
    assert got["protection_interaction"]["status"] == "NOT ESTIMABLE"


def test_both_arms_give_the_registered_interaction():
    arms = {}
    for p in ("random", "streamingllm", "h2o", "rkv"):
        arms[(p, True)] = _rows(ALL)
        arms[(p, False)] = _rows(ALL if p == "random" else HALF)
    inter = _read(arms)["protection_interaction"]
    # protected gap 0, unprotected gap 0.5 -> interaction -0.5 per pairing
    assert inter["contrasts"]["random-streamingllm"]["interaction"]["mean"] == (
        pytest.approx(-0.5)
    )
    assert "registered_inference" in inter


def test_readout_uses_only_keys_defined_for_every_compared_arm():
    arms = {("random", True): _rows(ALL), ("streamingllm", True): _rows(HALF)}

    def ro(correct, undefined=()):
        return {
            k: {"defined": k not in undefined, "correct": correct[k]} for k in EXPECTED
        }

    readouts = {
        ("random", True): ro(HALF, undefined={("b", 23)}),
        ("streamingllm", True): ro(HALF),
    }
    got = _read(arms, readouts)["sides"]["protected"]["readout"]
    assert got["matched_defined_keys"] == 3
    assert got["undefined_by_arm"] == {"random": 1, "streamingllm": 0}
    rnd = got["arms"]["random"]
    # keys a11, a23, b11: free 1,1,1; readout 1,1,0 -> residual (free - ro)
    # per key 0,0,1; problem means a=0, b=1 -> 0.5
    assert rnd["residual"]["mean"] == pytest.approx(0.5)
    # readout change vs FC (all correct): 0,0,-1 -> a=0, b=-1 -> -0.5
    assert rnd["readout_change"]["mean"] == pytest.approx(-0.5)
    gap = got["contrasts"]["random-streamingllm"]
    # free gap: random 1,1,1 vs slm 1,1,0 -> 0,0,1 -> 0.5; readout gap 0
    assert gap["free"]["mean"] == pytest.approx(0.5)
    assert gap["readout"]["mean"] == pytest.approx(0.0)
    assert gap["residual"]["mean"] == pytest.approx(0.5)


def test_exposure_by_hand():
    rows = {
        ("a", 11): {
            "compression_ratio": 0.5,
            "prompt_tokens": 60,
            "n_tokens": 452,
            "n_prunes": 4,
        },
        ("a", 23): {"compression_ratio": 1.0, "prompt_tokens": 60, "n_tokens": 100},
    }
    got = exposure(rows, budget=256)
    # 60+452-256 = 256 > 0 reached; 60+100-256 < 0 not reached
    assert got["overflow_reached_rate"] == pytest.approx(0.5)
    assert got["overflow_tokens_when_reached"]["median"] == pytest.approx(256)
    assert got["n_with_prune_record"] == 1


def test_incomplete_arm_is_refused():
    arms = {("random", True): {("a", 11): _rows(ALL)[("a", 11)]}}
    with pytest.raises(ValueError, match="does not cover"):
        _read(arms)


def test_cell_status_classification():
    from kvcache.m5b_reads import cell_status

    assert cell_status(None, exists=False) == "ABSENT"
    assert cell_status(None, exists=True) == "FINAL"
    missing = ValueError(
        "3 expected keys lack result rows: x; 3 expected keys lack sidecars: x"
    )
    assert cell_status(missing, exists=True) == "PARTIAL"
    mixed = ValueError(
        "3 expected keys lack result rows: x; 1 result rows outside the design: y"
    )
    assert cell_status(mixed, exists=True) == "INVALID"
    assert (
        cell_status(ValueError("torn or unreadable lines: z"), exists=True) == "INVALID"
    )


def test_grid_and_readout_assembly_map_presses_to_arms(monkeypatch):
    import kvcache.gsm8k_timing as gt
    import kvcache.manifest_reads as mr
    from kvcache.m5b_reads import grid_arms, readout_arms

    design = {
        "benchmark": "gsm8k",
        "split": "confirmation",
        "revision": "r",
        "pids": ["a"],
        "seeds": [11],
        "presses": ["random_pp", "h2o"],
        "budgets": [256],
    }
    pairs = [
        (
            {
                "press": p,
                "pid": "a",
                "decoding": {"seed": 11},
                "compression_ratio": 0.5,
                "n_tokens": 400,
                **({"protection": {"n_prunes": 3}} if p == "random_pp" else {}),
            },
            {"prompt_token_ids": [0] * 60},
        )
        for p in ("random_pp", "h2o")
    ]
    monkeypatch.setattr(gt, "validate_pairs", lambda *a, **k: pairs)
    monkeypatch.setattr(
        gt,
        "score_rows",
        lambda rows, *a, **k: {
            "scores": [
                {"correct": True, "truncated": False},
                {"correct": False, "truncated": True},
            ]
        },
    )
    got = grid_arms("root", design, {"a": "1"})
    assert got[("random", True)][("a", 11)]["n_prunes"] == 3
    assert got[("h2o", False)][("a", 11)] == {
        "correct": False,
        "truncated": True,
        "compression_ratio": 0.5,
        "prompt_tokens": 60,
        "n_tokens": 400,
        "n_prunes": None,
    }
    monkeypatch.setattr(
        mr,
        "score_readouts",
        lambda *a, **k: {
            "scores": [{"press": "random_pp", "pid": "a", "seed": 11, "correct": True}]
        },
    )
    ro = readout_arms("root", design, {"a": "1"})
    assert ro[("random", True)][("a", 11)] == {"defined": True, "correct": True}
    assert ro[("h2o", False)][("a", 11)] == {"defined": False, "correct": None}

"""M5-A protection-by-policy read: exact coverage or PARTIAL, hand-computed
contrasts and interaction, TOST labels, registered Holm family."""

from __future__ import annotations

import pytest

from kvcache.metrics import policy_replication as pr
from kvcache.metrics.protocol_audit import NOT_TESTABLE

PIDS = [f"p{i}" for i in range(10)]
SEEDS = [1, 2]
EXPECTED = {(p, s) for p in PIDS for s in SEEDS}


def _rows(rate_by_pid):
    """Rows where problem i is correct on both seeds iff rate_by_pid(i)."""
    return [
        {"pid": p, "decoding": {"seed": s}, "correct": bool(rate_by_pid(i))}
        for i, p in enumerate(PIDS)
        for s in SEEDS
    ]


def _arms(over=None):
    base = {
        ("random", False): _rows(lambda i: i < 6),  # 0.6
        ("random", True): _rows(lambda i: i < 8),  # 0.8
        ("h2o", False): _rows(lambda i: i < 2),  # 0.2
        ("h2o", True): _rows(lambda i: i < 7),  # 0.7
        ("streamingllm", False): _rows(lambda i: i < 6),  # 0.6
        ("streamingllm", True): _rows(lambda i: i < 8),  # 0.8
    }
    base.update(over or {})
    return base


def _read(arms):
    return pr.replication_read(
        arms, expected=EXPECTED, reference="random", pairings=["h2o", "streamingllm"]
    )


def test_final_read_hand_computed():
    d = _read(_arms())
    assert d["status"] == "FINAL"
    h = d["contrasts"]["random-h2o"]
    assert h["protected"]["mean"] == pytest.approx(0.1)  # 0.8 - 0.7
    assert h["unprotected"]["mean"] == pytest.approx(0.4)  # 0.6 - 0.2
    assert h["interaction"]["mean"] == pytest.approx(-0.3)
    s = d["contrasts"]["random-streamingllm"]
    assert s["protected"]["mean"] == 0.0 and s["interaction"]["mean"] == 0.0
    assert s["protected"]["tost"]["label"] == "within margin"  # identical arms
    fam = d["registered_inference"]
    assert fam["family_size"] == 2
    assert fam["members"]["random-streamingllm"]["status"] == NOT_TESTABLE
    assert fam["members"]["random-h2o"]["status"] == "TESTED"
    assert d["accuracy"]["h2o+P"]["mean"] == pytest.approx(0.7)


def test_missing_duplicate_or_absent_arm_is_partial_without_numbers():
    rows = _rows(lambda i: True)
    for arms in (
        _arms({("h2o", True): None}),
        _arms({("h2o", True): rows[1:]}),
        _arms({("h2o", True): rows + rows[:1]}),
    ):
        d = _read(arms)
        assert d["status"] == "PARTIAL" and "contrasts" not in d
        assert d["missing_arms"] == ["h2o+P"]


def test_tost_labels():
    assert pr.tost([0.01, -0.01, 0.0, 0.02, -0.02] * 20)["label"] == "within margin"
    assert pr.tost([0.3, 0.2, 0.25, 0.35] * 10)["label"] == "outside margin"
    assert pr.tost([1.0, -1.0, 0.5, -0.5])["label"] == "inconclusive"
    assert pr.tost([0.1])["label"] == "NOT_TESTABLE"

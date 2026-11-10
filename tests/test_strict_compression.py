"""Strict-cohort nominal ratios (kvcache.strict_compression, 2026-11-10).

Hand-built rows: the arm is summarized on exactly the strict keys (rows for
other problems are ignored, a missing or duplicated key refuses), the stored
ratio must equal min(1, B / (prompt + n_tokens)), a relaxed protected row is an
error, and the cohort/prompt tables are accepted only against their reads'
hashes."""

from __future__ import annotations

import hashlib

import pytest

from kvcache import strict_compression as sc


def _row(pid, seed, press, n_tokens, prompt, budget=256, ratio=None):
    nom = min(1.0, budget / (prompt + n_tokens))
    return {
        "pid": pid,
        "decoding": {"seed": seed},
        "press": press,
        "budget": budget,
        "n_tokens": n_tokens,
        "compression_ratio": nom if ratio is None else ratio,
    }


ARM = sc.Arm("random+P", "random", True, "grid-pp256", "random_pp")
KEYS = {("a", 11), ("a", 23), ("b", 11), ("b", 23)}
PLEN = {"a": 50, "b": 100, "c": 40}


def _rows():
    return [
        _row("a", 11, "random_pp", 974, 50),  # 256 / 1024 = 0.25
        _row("a", 23, "random_pp", 462, 50),  # 256 / 512 = 0.5
        _row("b", 11, "random_pp", 924, 100),  # 0.25
        _row("b", 23, "random_pp", 100, 100),  # 256 / 200 -> 1
        _row("c", 11, "random_pp", 216, 40),  # not in the cohort: ignored
        _row("a", 11, "streamingllm_pp", 200, 50),  # other press: ignored
    ]


def test_arm_is_summarized_on_exactly_the_strict_keys():
    out = sc.arm_ratios(_rows(), ARM, KEYS, PLEN)
    assert out["n_rows"] == 4
    assert out["median"] == pytest.approx(0.375)
    assert out["mean"] == pytest.approx((0.25 + 0.5 + 0.25 + 1.0) / 4)
    assert out["n_ratio_one"] == 1 and out["n_budget_relaxed"] == 0
    assert out["press"] == "random_pp" and out["protected"] is True


def test_a_missing_or_duplicated_strict_key_refuses():
    rows = [r for r in _rows() if (r["pid"], r["decoding"]["seed"]) != ("b", 23)]
    with pytest.raises(ValueError, match="do not match"):
        sc.arm_ratios(rows, ARM, KEYS, PLEN)
    with pytest.raises(ValueError, match="do not match"):
        sc.arm_ratios(_rows() + [_rows()[0]], ARM, KEYS, PLEN)


def test_a_stored_ratio_that_disagrees_with_the_tokens_refuses():
    rows = _rows()
    rows[0] = _row("a", 11, "random_pp", 974, 50, ratio=0.3)
    with pytest.raises(ValueError, match="stored ratio"):
        sc.arm_ratios(rows, ARM, KEYS, PLEN)


def test_a_relaxed_protected_row_is_an_error_on_the_strict_cohort():
    plen = {**PLEN, "a": 220}  # 256 - 220 < 64 trace slots -> relaxed
    rows = [
        _row("a", 11, "random_pp", 804, 220),
        _row("a", 23, "random_pp", 804, 220),
        *_rows()[2:4],
    ]
    with pytest.raises(ValueError, match="relaxed rows"):
        sc.arm_ratios(rows, ARM, KEYS, plen)


def test_an_unprotected_arm_never_counts_relaxation():
    arm = sc.Arm("random-P", "random", False, "grid-diss256-tok", "random")
    rows = [_row("a", 11, "random", 804, 220), _row("a", 23, "random", 804, 220)]
    out = sc.arm_ratios(rows, arm, {("a", 11), ("a", 23)}, {"a": 220})
    assert out["n_budget_relaxed"] == 0 and out["n_rows"] == 2


def _tsv(pairs):
    body = "".join(f"{p}\t{s}\t1\n" for p, s in pairs)
    return ("pid\tseed\tx\n" + body).encode()


def test_strict_keys_require_a_final_read_and_a_matching_hash(monkeypatch):
    monkeypatch.setattr(sc, "N_PROBLEMS", 2)
    monkeypatch.setattr(sc, "N_SEEDS", 2)
    tsv = _tsv(sorted(KEYS))
    read = {"status": "FINAL", "sufficient_statistics_sha256": ""}
    read["sufficient_statistics_sha256"] = hashlib.sha256(tsv).hexdigest()
    assert sc.strict_keys(read, tsv) == KEYS
    with pytest.raises(ValueError, match="not FINAL"):
        sc.strict_keys({**read, "status": "PARTIAL"}, tsv)
    with pytest.raises(ValueError, match="hash"):
        sc.strict_keys(read, tsv + b"x\t1\t1\n")
    small = _tsv(sorted(KEYS)[:3])
    with pytest.raises(ValueError, match="expected 2 / 4"):
        sc.strict_keys(
            {**read, "sufficient_statistics_sha256": hashlib.sha256(small).hexdigest()},
            small,
        )


def test_prompt_lengths_are_accepted_only_against_the_s1_hash():
    tsv = b"pid\tprompt_len\na\t50\nb\t100\n"
    read = {
        "status": "FINAL",
        "sufficient_statistics_sha256": hashlib.sha256(tsv).hexdigest(),
    }
    assert sc.verified_prompt_lengths(read, tsv) == {"a": 50, "b": 100}
    with pytest.raises(ValueError, match="hash"):
        sc.verified_prompt_lengths(read, tsv.replace(b"50", b"51"))


def test_arms_cover_the_eight_strict_read_arms_once():
    labels = [a.label for a in sc.ARMS]
    assert len(labels) == len(set(labels)) == 8
    assert {a.policy for a in sc.ARMS} == {"random", "streamingllm", "h2o", "rkv"}
    assert sum(a.protected for a in sc.ARMS) == 4

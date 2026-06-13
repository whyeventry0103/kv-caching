import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "diagnose_math_failures",
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_math_failures.py",
)
diagnose = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnose)


def _row(pid, *, n_tokens, max_new, trace, answer_raw, truncated, correct, seed=11):
    return {
        "bench": "math500",
        "pid": pid,
        "decoding": {"seed": seed, "max_new": max_new},
        "n_tokens": n_tokens,
        "trace": trace,
        "answer_raw": answer_raw,
        "answer_canon": answer_raw,
        "truncated": truncated,
        "correct": correct,
    }


def test_classification_and_grader_disagreement_counts():
    rows = [
        _row(
            "p_correct",
            n_tokens=20,
            max_new=4096,
            trace=r"\boxed{2}",
            answer_raw="2",
            truncated=False,
            correct=True,
        ),
        _row(
            "p_trunc",
            n_tokens=4096,
            max_new=4096,
            trace="still reasoning",
            answer_raw="",
            truncated=True,
            correct=False,
        ),
        _row(
            "p_extract",
            n_tokens=50,
            max_new=4096,
            trace="no marker here",
            answer_raw="",
            truncated=False,
            correct=False,
        ),
        # Old canonicalizer false-negative: structured answer scored wrong.
        _row(
            "p_falseneg",
            n_tokens=1023,
            max_new=4096,
            trace=r"\boxed{(3, \dfrac{\pi}{2})}",
            answer_raw=r"(3, \dfrac{\pi}{2})",
            truncated=False,
            correct=False,
        ),
        _row(
            "p_wrong",
            n_tokens=300,
            max_new=4096,
            trace=r"\boxed{7}",
            answer_raw="7",
            truncated=False,
            correct=False,
        ),
    ]
    gold = {
        "p_correct": "2",
        "p_trunc": "5",
        "p_extract": "5",
        "p_falseneg": r"(3, \frac{\pi}{2})",
        "p_wrong": "5",
    }
    records = [diagnose.classify_row(r, gold[r["pid"]]) for r in rows]
    by_pid = {record["pid"]: record for record in records}

    assert by_pid["p_correct"]["classification"] == "correct"
    assert by_pid["p_trunc"]["classification"] == "token_cap_truncation"
    assert by_pid["p_extract"]["classification"] == "extraction_failure"
    assert by_pid["p_falseneg"]["classification"] == "correct"
    assert by_pid["p_falseneg"]["grader_disagreement"] is True
    assert by_pid["p_wrong"]["classification"] == "genuine_wrong_or_ambiguous"

    summary = diagnose.summarize(records)
    assert summary["grader_false_negatives"] == 1
    assert summary["grader_false_positives"] == 0
    assert summary["stored_accuracy"] == 0.2
    assert summary["regraded_accuracy"] == 0.4

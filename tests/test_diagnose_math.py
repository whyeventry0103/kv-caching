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


def test_load_rows_skips_token_id_sidecars(tmp_path):
    """Sidecars are also *.jsonl and also carry "bench", so the bench filter
    alone lets them through -- and they have no "decoding" block, which crashed
    the audit the first time it was pointed at a compression cell."""
    import json

    load_rows = diagnose.load_rows

    run = tmp_path / "grid-cell"
    (run / "math500").mkdir(parents=True)
    (run / "math500" / "rows.jsonl").write_text(
        json.dumps(
            {
                "bench": "math500",
                "pid": "p1",
                "decoding": {"max_new": 16384},
                "answer_raw": "2",
                "answer_canon": "2",
                "correct": True,
                "truncated": False,
                "finish_reason": "stop",
                "n_tokens": 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = run / "math500" / "token_ids"
    sidecar.mkdir()
    (sidecar / "rows.jsonl").write_text(
        json.dumps({"bench": "math500", "pid": "p1", "prompt_token_ids": [1, 2]})
        + "\n",
        encoding="utf-8",
    )
    rows = load_rows(run)
    assert len(rows) == 1
    assert "decoding" in rows[0]


def test_crosscheck_is_a_different_normalization_from_is_equiv():
    """The whole point of C.1: re-running is_equiv agrees by construction, so
    only a different normalization can speak to under-crediting."""
    from kvcache.grading import crosscheck_equiv, is_equiv

    # A structured form where the two normalizations need not coincide.
    assert crosscheck_equiv("1/2", "0.5") is not None
    # Cannot-compare must be None, never a silent "agrees".
    assert crosscheck_equiv(None, "2") is None
    assert crosscheck_equiv("2", None) is None
    # And is_equiv remains the authority, untouched.
    assert is_equiv("2", "2") is True


def test_summary_separates_replay_determinism_from_grader_validity():
    records = [
        {
            "classification": "correct",
            "grader_disagreement": False,
            "stored_correct": True,
            "ref_correct": True,
            "cross_correct": True,
            "cross_disagreement": False,
        },
        {
            "classification": "genuine_wrong_or_ambiguous",
            "grader_disagreement": False,
            "stored_correct": False,
            "ref_correct": False,
            "cross_correct": True,
            "cross_disagreement": True,
        },
    ]
    got = diagnose.summarize(records)
    assert got["grader_disagreements"] == 0  # replay determinism
    assert got["crosscheck_disagreements"] == 1  # grader validity signal
    assert got["crosscheck_only_correct"] == 1  # the under-crediting direction
    assert "REPLAY DETERMINISM" in got["crosscheck_note"]

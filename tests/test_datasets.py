from kvcache.datasets import _math500_problems, load_benchmark


class FakeDataset(list):
    def select(self, indices):
        return FakeDataset(self[index] for index in indices)


def test_math500_mapping_and_subset():
    dataset = FakeDataset(
        [
            {
                "unique_id": "test/algebra/1.json",
                "problem": "What is one half?",
                "answer": r"\frac{1}{2}",
                "level": 1,
                "subject": "Algebra",
            },
            {
                "unique_id": "test/algebra/2.json",
                "problem": "Unused",
                "answer": "2",
                "level": 2,
                "subject": "Algebra",
            },
        ]
    )
    problems = _math500_problems(dataset, subset=1)
    assert len(problems) == 1
    assert problems[0].pid == "test/algebra/1.json"
    # Gold is kept RAW for is_equiv grading; the canonical form lives in meta.
    assert problems[0].gold_answer == r"\frac{1}{2}"
    assert problems[0].meta["gold_canon"] == "1/2"


def test_math500_level_stratified_subset():
    dataset = FakeDataset(
        {
            "unique_id": f"p{index}",
            "problem": f"problem {index}",
            "answer": str(index),
            "level": level,
            "subject": "Mixed",
        }
        for index, level in enumerate([1, 1, 2, 3, 4, 5])
    )
    problems = _math500_problems(
        dataset,
        subset={"count": 5, "stratify_by": "level"},
    )
    assert [problem.meta["level"] for problem in problems] == [1, 2, 3, 4, 5]


def test_rsp_is_local_and_deterministic():
    problems = load_benchmark("rsp", {"handle": "synthetic"}, subset=2)
    assert len(problems) == 2
    assert problems[0].gold_answer == "14"
    assert "follow refers_to links" in problems[0].question


def test_aime24_problems_raw_gold_and_deterministic_order():
    from kvcache.datasets import _aime_problems

    rows = [
        {"id": 62, "problem": "P62", "answer": "025", "url": "u62", "year": "2024"},
        {"id": 60, "problem": "P60", "answer": "204", "url": "u60", "year": "2024"},
        {"id": 61, "problem": "P61", "answer": "113", "url": "u61", "year": "2024"},
    ]
    problems = _aime_problems(rows, 2, name="aime24")
    # ordered by integer id, subset applied AFTER ordering, gold kept raw
    assert [p.pid for p in problems] == ["aime24-60", "aime24-61"]
    assert problems[0].gold_answer == "204"
    assert problems[0].question == "P60"
    assert problems[0].meta["year"] == "2024"
    # leading-zero answers survive raw (is_equiv handles equivalence at grade time)
    full = _aime_problems(rows, None, name="aime24")
    assert full[2].gold_answer == "025"


def test_gsm8k_gold_parsing_and_refusals():
    import pytest

    from kvcache.datasets import gsm8k_gold

    assert gsm8k_gold("Natalia sold 48/2 = 24 clips.\n#### 72") == "72"
    assert gsm8k_gold("... #### 1,250") == "1250"
    assert gsm8k_gold("#### -3.5") == "-3.5"
    with pytest.raises(ValueError, match="marker"):
        gsm8k_gold("no marker 72")
    with pytest.raises(ValueError, match="not numeric"):
        gsm8k_gold("#### seventy")


def test_gsm8k_problems_seeded_sample_is_frozen_and_raw():
    from kvcache.datasets import _gsm8k_problems

    data = [{"question": f"q{i}", "answer": f"w\n#### {i * 10}"} for i in range(50)]
    a = _gsm8k_problems(data, {"count": 5, "seed": 11})
    b = _gsm8k_problems(data, {"count": 5, "seed": 11})
    assert [p.pid for p in a] == [p.pid for p in b]
    assert [p.pid for p in a] == sorted(
        [p.pid for p in a], key=lambda x: int(x.split("/")[-1])
    )
    i = int(a[0].pid.split("/")[-1])
    assert a[0].question == f"q{i}" and a[0].gold_answer == str(i * 10)
    assert [p.pid for p in _gsm8k_problems(data, 3)] == [
        "gsm8k/test/0",
        "gsm8k/test/1",
        "gsm8k/test/2",
    ]


def test_gsm8k_load_refuses_without_a_pinned_revision(monkeypatch, tmp_path):
    import pytest
    import yaml

    from kvcache.datasets import load_benchmark

    monkeypatch.setenv("REMOTE_DATA_ROOT", str(tmp_path))
    (tmp_path / "datasets").mkdir()
    cfg = yaml.safe_load(open("configs/benchmarks.yaml"))["benchmarks"]["gsm8k"]
    assert "revision" not in cfg  # a human pins it (TODO(human) rule)
    with pytest.raises(RuntimeError, match="pinned"):
        load_benchmark("gsm8k", cfg, subset=3)

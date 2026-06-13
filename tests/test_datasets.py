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

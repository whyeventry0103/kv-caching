from kvcache.generation import finalize_generation
from kvcache.metrics.add import OTHER_ANSWER


def test_generation_at_cap_without_answer_is_retained_failure():
    result = finalize_generation(
        prompt="prompt",
        trace="still reasoning",
        n_tokens=16,
        max_new_tokens=16,
        gold_answer="2",
        wallclock_s=1.0,
    )
    assert result.truncated is True
    assert result.correct is False
    assert result.answer_canon == OTHER_ANSWER


def test_boxed_answer_is_scored():
    result = finalize_generation(
        prompt="prompt",
        trace=r"Therefore \boxed{2}",
        n_tokens=5,
        max_new_tokens=16,
        gold_answer="2",
        wallclock_s=1.0,
    )
    assert result.truncated is False
    assert result.correct is True
    assert result.answer_raw == "2"

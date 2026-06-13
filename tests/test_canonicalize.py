from kvcache.canonicalize import (
    canonicalize,
    canonicalize_ablation,
    extract_answer,
)


def test_boxed_and_option_canonicalization():
    assert canonicalize(r"work \boxed{1/2}") == "1/2"
    assert canonicalize("The answer is (C)") == "C"


def test_second_canonicalizer_is_distinct():
    assert canonicalize_ablation("  Answer   Is C ") == "answer is c"


def test_nested_boxed_answer_is_extracted():
    text = r"Therefore \boxed{\frac{14}{3}}."
    assert extract_answer(text) == r"\frac{14}{3}"


def test_structured_latex_answers_canonicalize_equivalently():
    expected = "(3, pi/2)"
    assert canonicalize(r"(3, \dfrac{\pi}{2})") == expected
    assert canonicalize(r"(3, \frac{\pi}{2})") == expected
    assert canonicalize(r"\boxed{(3, \frac{\pi}{2})}") == expected

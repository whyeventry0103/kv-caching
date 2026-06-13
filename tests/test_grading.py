from kvcache.grading import is_equiv


def test_structured_tuple_and_fraction_forms_match():
    # The smoke-run regression: \dfrac vs \frac inside a tuple must be equivalent.
    assert is_equiv(r"(3, \frac{\pi}{2})", r"(3, \dfrac{\pi}{2})") is True


def test_fraction_decimal_and_dfrac_equivalences():
    assert is_equiv(r"\frac{1}{2}", "0.5") is True
    assert is_equiv(r"\frac{1}{2}", r"\dfrac{1}{2}") is True
    assert is_equiv(r"\sqrt3", r"\sqrt{3}") is True


def test_units_percent_dollar_and_prefix_are_stripped():
    assert is_equiv(r"5\text{ cm}", "5") is True
    assert is_equiv(r"50\%", "50") is True
    assert is_equiv(r"\$5", "5") is True
    assert is_equiv("x=5", "5") is True


def test_genuine_mismatches_and_missing_answers():
    assert is_equiv("2", "3") is False
    assert is_equiv("2", "") is False
    assert is_equiv(None, "2") is False
    # The old sympify canonicalizer mis-scored this structured answer as "2".
    assert is_equiv(r"(3, \frac{\pi}{2})", "2") is False

from kvcache.paper_numbers import (
    Entry,
    classify,
    extract_literals,
    strip_latex_noise,
    summarise,
)


def _entry(key: str, value: float) -> Entry:
    return Entry(key=key, value=value, kind="primary", source="artifact#/x")


def test_displays_covers_prose_rounding_and_bare_magnitude():
    forms = _entry("r", -0.3367).displays()
    assert "-0.337" in forms  # .3f display convention
    assert "0.337" in forms  # quoted without its sign
    assert "-0.34" in forms  # "roughly -0.34"


def test_displays_does_not_generate_percentage_forms():
    """0.015 must not match the '1.5' in 'Qwen-1.5B' (a real false-match class)."""
    assert "1.5" not in _entry("ratio", 0.015).displays()


def test_hyphen_in_a_name_is_not_a_minus_sign():
    lits = extract_literals("accuracy on MATH-500 and Qwen-1.5B", "t", strip=False)
    values = {lit.value for lit in lits}
    assert 500.0 in values and -500.0 not in values
    assert 1.5 in values and -1.5 not in values


def test_real_negative_number_keeps_its_sign():
    lits = extract_literals("a residual of $-0.343$ here", "t", strip=False)
    assert any(lit.value == -0.343 for lit in lits)


def test_double_dash_range_is_not_negative():
    lits = extract_literals("budgets 128--512 inclusive", "t", strip=False)
    assert {lit.value for lit in lits} == {128.0, 512.0}


def test_version_tail_is_skipped():
    lits = extract_literals("transformers 4.56.2 pinned", "t", strip=False)
    assert 4.56 not in {lit.value for lit in lits}


def test_thousands_separator_is_parsed():
    lits = extract_literals("a mean 16{,}384-token cap", "t", strip=False)
    assert 16384.0 in {lit.value for lit in lits}


def test_strip_removes_comments_and_preamble_but_keeps_line_count():
    text = "\\documentclass{article}\n% a comment with 999\n\\begin{document}\n0.5\n"
    cleaned = strip_latex_noise(text)
    assert "999" not in cleaned
    assert "0.5" in cleaned
    assert len(cleaned.splitlines()) == len(text.splitlines())


def test_structural_lengths_are_not_claims():
    lits = extract_literals("\\vspace{6pt} and 0.9\\textwidth", "t", strip=False)
    assert {lit.value for lit in lits} == set()


def test_classification_buckets():
    cat = [_entry("a.residual", -0.3367)]
    lits = extract_literals(
        "resid -0.337, budget 256, year 2023, mystery 0.777", "t", strip=False
    )
    classify(lits, cat, allowed_integers={256})
    got = {lit.raw: lit.classification for lit in lits}
    assert got["-0.337"] == "sourced"
    assert got["256"] == "protocol"
    assert got["2023"] == "year"
    assert got["0.777"] == "unsourced"


def test_summarise_reports_unsourced_rows():
    cat = [_entry("a", 0.5)]
    lits = extract_literals("0.500 and 0.777", "t", strip=False)
    classify(lits, cat)
    report = summarise(lits)
    assert report["total"] == 2
    assert report["by_classification"]["sourced"] == 1
    assert len(report["unsourced"]) == 1
    assert report["unsourced"][0]["value"] == "0.777"


def test_stale_value_is_unsourced_against_the_current_catalogue():
    """The gate's whole purpose: a superseded number must not pass."""
    current = [_entry("diss256.random.residual", -0.3367)]
    lits = extract_literals("residual of $-0.343$", "t", strip=False)
    classify(lits, current)
    assert lits[0].classification == "unsourced"

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
    """The example was "MATH-500 and Qwen-1.5B", asserting 1.5 extracts positive.
    Qwen-1.5B is now dropped entirely as a model-scale suffix -- a stronger
    reason, tested separately -- so the sign rule is exercised on a range
    instead, which is the other case it exists for."""
    lits = extract_literals("MATH-500 at budgets 128--512", "t", strip=False)
    values = {lit.value for lit in lits}
    assert 500.0 in values and -500.0 not in values
    assert 512.0 in values and -512.0 not in values


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


def test_bibliography_label_width_is_not_a_claim():
    """``\\begin{thebibliography}{99}`` is a typesetting hint.

    It was matching a derived ratio rounded to zero places (98.74 -> "99") and
    being counted as *sourced*, which inflates the audit's success metric with
    a number that is not a claim.
    """
    text = "\\begin{document}\n\\begin{thebibliography}{99}\nreal 0.185\n"
    lits = extract_literals(text, "t")
    assert 99.0 not in {lit.value for lit in lits}
    assert 0.185 in {lit.value for lit in lits}
    assert len(strip_latex_noise(text).splitlines()) == len(text.splitlines())


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


def test_protocol_wins_over_a_percentage_that_rounds_onto_it():
    """A catalogued 3.157% must not lend provenance to a bare "3" in the prose.

    displays() legitimately emits the zero-place form of every value, so a
    percentage entry collides with small protocol integers. Matching sourced
    first counted 53 config constants as sourced against unrelated ratios --
    the bibliography-label failure mode one layer up: the gate inflating its own
    success metric.
    """
    cat = [_entry("realized.random-256.ratio_pct", 3.1566)]
    lits = extract_literals("all 3 policies at 3.157", "t", strip=False)
    classify(lits, cat, allowed_integers={3})
    got = {lit.raw: lit.classification for lit in lits}
    assert got["3"] == "protocol"
    assert got["3.157"] == "sourced"


def test_a_non_protocol_integer_still_sources_normally():
    """The ordering must not make every integer unsourceable."""
    cat = [_entry("compute.generations", 7200.0)]
    lits = extract_literals("7200 generations", "t", strip=False)
    classify(lits, cat, allowed_integers={3, 4, 8})
    assert lits[0].classification == "sourced"


def test_displays_omits_the_zero_place_form_of_a_sub_unit_value():
    """0.882 is never written "1". Emitting it matched every bare 0 and 1 in
    the prose against some accuracy or residual -- 418 occurrences, 65% of the
    reported "sourced" count, were protocol constants credited to nothing."""
    forms = _entry("fc", 0.882).displays()
    assert "1" not in forms and "0" not in forms
    assert "0.9" in forms and "0.88" in forms and "0.882" in forms


def test_displays_keeps_the_zero_place_form_of_a_magnitude_above_one():
    """ "230 GPU-hours" and "105%" are legitimate zero-place displays."""
    assert "280" in _entry("gpu_hours", 279.87).displays()
    assert "105" in _entry("rkv_pct", 105.0).displays()


def test_displays_still_covers_an_exact_zero():
    assert "0" in _entry("aime_bonus", 0.0).displays()


def test_bibliography_body_is_not_scanned_for_claims():
    """13 arXiv IDs were being reported as unsourced claims. Nothing in a
    reference list is a claim about our results."""
    tex = (
        "\\begin{document}\nresidual $-0.343$\n"
        "\\begin{thebibliography}{99}\n"
        "\\bibitem{x} Attention Sinks. \\emph{ICLR}, 2024. arXiv:2309.17453.\n"
        "\\end{thebibliography}\n\\end{document}\n"
    )
    values = {lit.value for lit in extract_literals(tex, "t")}
    assert -0.343 in values
    assert 2309.0 not in values and 17453.0 not in values


def test_arxiv_ids_are_stripped_outside_latex_too():
    """PAPER_FINDINGS.md cites them inline, where no LaTeX stripping runs."""
    lits = extract_literals("HOTT (arXiv:2512.12008) reports 0.47", "t", strip=False)
    assert {lit.value for lit in lits} == {0.47}


def test_model_scale_suffixes_are_names_not_measurements():
    """ "1.5B/7B/14B" and "4M tokens" contributed a dozen phantom literals."""
    lits = extract_literals("Qwen-1.5B/7B/14B on 4M tokens", "t", strip=False)
    assert lits == []


def test_a_number_glued_to_a_word_is_an_identifier():
    """PG19 is a corpus name; R1 is a model name."""
    lits = extract_literals("PG19 with R1-Distill at 0.882", "t", strip=False)
    assert {lit.value for lit in lits} == {0.882}


def test_a_space_breaks_the_scale_suffix():
    """A table cell after a number is not a parameter count."""
    lits = extract_literals("residual 0.343 B", "t", strip=False)
    assert {lit.value for lit in lits} == {0.343}


def test_archival_span_is_skipped():
    """PAPER_FINDINGS.md's dated log shows superseded values beside their
    corrections on purpose. The gate stops stale numbers reaching the paper; it
    must not launder the record that documents them."""
    md = (
        "current residual 0.337\n"
        "<!-- AUDIT:ARCHIVAL-BEGIN -->\n"
        "was 0.343 before Amendment K\n"
        "<!-- AUDIT:ARCHIVAL-END -->\n"
        "and 0.268 for R-KV\n"
    )
    values = {lit.value for lit in extract_literals(md, "t", strip=False)}
    assert 0.337 in values and 0.268 in values
    assert 0.343 not in values


def test_archival_span_preserves_line_numbers():
    """A blanked span that ate its newlines would shift every reported line."""
    md = (
        "a 0.1\n<!-- AUDIT:ARCHIVAL-BEGIN -->\n"
        "x 0.2\n<!-- AUDIT:ARCHIVAL-END -->\nb 0.3\n"
    )
    lits = extract_literals(md, "t", strip=False)
    assert [(lit.value, lit.line) for lit in lits] == [(0.1, 1), (0.3, 5)]


def test_section_references_and_dates_are_not_claims():
    lits = extract_literals("see §4.1 and §3 on 2026-07-17 for 0.885", "t", strip=False)
    assert {lit.value for lit in lits} == {0.885}


def test_bare_arxiv_id_in_a_table_column():
    lits = extract_literals(
        "| R-KV | NeurIPS'25 | 2505.24133 | 0.520 |", "t", strip=False
    )
    assert 0.520 in {lit.value for lit in lits}
    assert 2505.0 not in {lit.value for lit in lits}


_GUARDED_DOC = """# Findings

### 4-CURRENT. Post-Amendment-K headline table

residual -0.337 and reference 0.885.

### 4-LOG. Dated read-in log (archival)

<!-- AUDIT:ARCHIVAL-BEGIN -->
superseded residual -0.343
<!-- AUDIT:ARCHIVAL-END -->
"""


def _cat():
    return [_entry("r", -0.3367), _entry("fc", 0.885)]


def test_guarded_section_stops_at_the_next_heading():
    from kvcache.paper_numbers import guarded_section

    section = guarded_section(_GUARDED_DOC)
    assert "-0.337" in section
    assert "-0.343" not in section  # the log is a different section


def test_guarded_section_gate_passes_on_a_clean_document():
    from kvcache.paper_numbers import check_guarded_section

    assert check_guarded_section(_GUARDED_DOC, _cat()) == []


def test_guarded_section_gate_refuses_an_archival_exemption_inside_itself():
    """The archival mechanism is necessary for the log and is also a hole. The
    current-truth section must not be able to use it."""
    from kvcache.paper_numbers import check_guarded_section

    doc = _GUARDED_DOC.replace(
        "residual -0.337 and reference 0.885.",
        "<!-- AUDIT:ARCHIVAL-BEGIN -->\n-0.343\n<!-- AUDIT:ARCHIVAL-END -->",
    )
    problems = check_guarded_section(doc, _cat())
    assert any("may not exempt itself" in p for p in problems)


def test_guarded_section_gate_catches_a_stale_literal():
    from kvcache.paper_numbers import check_guarded_section

    doc = _GUARDED_DOC.replace("residual -0.337", "residual -0.343")
    problems = check_guarded_section(doc, _cat())
    assert any("unsourced literal" in p and "0.343" in p for p in problems)


def test_guarded_section_gate_reports_a_missing_section():
    from kvcache.paper_numbers import check_guarded_section

    problems = check_guarded_section("# Findings\n\nno current section\n", _cat())
    assert any("no 4-CURRENT section" in p for p in problems)

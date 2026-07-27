from __future__ import annotations

import pytest

from kvcache.compute_budget import realized_compression, summarize_wallclock


def test_realized_compression_uses_prompt_plus_generation():
    """The paper's definition. Omitting the prompt term is what made the
    published ratios unreproducible from any artifact for months."""
    got = realized_compression(
        256, mean_prompt_tokens=66.28, mean_generated_tokens=8043.7075
    )
    assert got["mean_trace_tokens"] == pytest.approx(8109.9875)
    assert got["ratio_pct"] == pytest.approx(3.1566, abs=1e-4)
    # The generation-only figure is reported alongside precisely because it
    # differs -- that difference is what the prompt term is worth.
    assert got["ratio_generation_only_pct"] == pytest.approx(3.1826, abs=1e-4)
    assert got["ratio_generation_only_pct"] > got["ratio_pct"]


def test_realized_compression_reproduces_the_papers_pooled_figure():
    """Pooled over the Random/SLM pair at budget 256 the paper quotes 3.8% of a
    6,725-token trace. Both fall out of the prompt-inclusive denominator."""
    random_trace = realized_compression(256, 66.28, 8043.7075)["mean_trace_tokens"]
    slm_trace = realized_compression(256, 66.28, 5273.74)["mean_trace_tokens"]
    pooled = (random_trace + slm_trace) / 2
    assert pooled == pytest.approx(6725.0, abs=0.01)
    assert 100 * 256 / pooled == pytest.approx(3.8, abs=0.01)


def test_realized_compression_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        realized_compression(0, 66.0, 100.0)
    with pytest.raises(ValueError):
        realized_compression(256, 0.0, 0.0)


def test_summarize_wallclock_totals_only_included_directories():
    got = summarize_wallclock(
        {"a": [3600.0, 3600.0], "b": [7200.0], "c": [1800.0]},
        included=["a", "b"],
    )
    assert got["gpu_hours"] == pytest.approx(4.0)
    assert got["generations"] == 3
    assert got["n_directories_included"] == 2
    assert got["n_directories_excluded"] == 1
    assert got["excluded_hours"] == pytest.approx(0.5)


def test_summarize_wallclock_reports_excluded_directories_too():
    """An exclusion a reader cannot see is an exclusion a reader cannot check."""
    got = summarize_wallclock({"a": [3600.0], "superseded": [7200.0]}, included=["a"])
    assert got["by_directory"]["superseded"]["included"] is False
    assert got["by_directory"]["superseded"]["hours"] == pytest.approx(2.0)


def test_summarize_wallclock_refuses_an_unknown_inclusion():
    """A typo'd directory name must fail loudly, not silently total to less."""
    with pytest.raises(ValueError, match="absent from the data"):
        summarize_wallclock({"a": [3600.0]}, included=["a", "typo"])


def test_summarize_wallclock_deduplicates_the_include_list():
    got = summarize_wallclock({"a": [3600.0]}, included=["a", "a"])
    assert got["gpu_hours"] == pytest.approx(1.0)
    assert got["n_directories_included"] == 1

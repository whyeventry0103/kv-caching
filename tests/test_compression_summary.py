"""Realized compression summary: hand-computed mean/median/never-compressed."""

from __future__ import annotations

import pytest

from kvcache.compression_summary import summarize


def test_summary_hand_computed():
    rows = [{"compression_ratio": v} for v in (0.25, 0.5, 1.0, 1.0)]
    s = summarize(rows)
    assert s["mean"] == pytest.approx(0.6875) and s["median"] == pytest.approx(0.75)
    assert s["n_never_compressed"] == 2 and s["never_compressed_share"] == 0.5


@pytest.mark.parametrize("bad", [0.0, 1.5, True, "0.5", float("nan")])
def test_out_of_range_refused(bad):
    with pytest.raises(ValueError):
        summarize([{"compression_ratio": bad}])
    with pytest.raises(ValueError):
        summarize([])

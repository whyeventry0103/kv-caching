"""Realized compression of free-run grid cells (2026-10-15).

Scientific rationale. The framing rules require every absolute budget to be
reported with its REALIZED compression ratio: B / (prompt + generated tokens),
capped at 1, as each grid row records it in ``compression_ratio``. A nominal
budget says little on its own -- at B = 1024 a short trace is never compressed
at all -- so the mean, the median and the share of never-compressed conditions
are summarized per cell from the rows themselves.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from typing import Any


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean, median and never-compressed share of ``compression_ratio``."""
    ratios = []
    for r in rows:
        v = r["compression_ratio"]
        if isinstance(v, bool) or not isinstance(v, int | float) or not 0 < v <= 1:
            raise ValueError(f"compression_ratio out of (0, 1]: {v!r}")
        ratios.append(float(v))
    if not ratios:
        raise ValueError("no rows")
    never = sum(1 for v in ratios if v >= 1.0)
    return {
        "n_rows": len(ratios),
        "mean": statistics.fmean(ratios),
        "median": statistics.median(ratios),
        "n_never_compressed": never,
        "never_compressed_share": never / len(ratios),
        "definition": "B / (prompt + generated tokens), capped at 1, per row",
    }

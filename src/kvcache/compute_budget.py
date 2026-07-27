"""Summed generation wall-clock, for the paper's compute statement.

The compute figure is the one number in the paper a reader cannot check and a
reviewer will not challenge -- which is exactly why it should not be a
remembered constant. It was carried as "~230 GPU-hours over ~7,300
generations" with no artifact behind it.

Every result row already records ``wallclock_s``, so the figure is derivable.
What it needs is an explicit *inclusion rule*, because the raw sum over every
run directory is wrong three ways: the merged full-cache bank double-counts its
own two shards, teacher-forced work (readout, reverse readout, replay) is
excluded by the paper's own wording, and superseded pilots ran on a pipeline
whose data was later invalidated. A wall-clock total is only meaningful
alongside the rule that produced it, so the rule travels in the artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def realized_compression(
    budget: int,
    mean_prompt_tokens: float,
    mean_generated_tokens: float,
) -> dict[str, Any]:
    """Budget as a fraction of the mean full trace (prompt + generation).

    The paper reports absolute budgets *with* their realized ratio, because
    "budget 256" means nothing without the trace length it is 256 out of --
    3.8% is what tells a reader the setting is aggressive. The prompt term is
    small (tens of tokens) but it is not zero, and omitting it was why the
    published ratios could not be reproduced from any artifact: the
    generation-only denominator gives a visibly different number.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    total = float(mean_prompt_tokens) + float(mean_generated_tokens)
    if total <= 0:
        raise ValueError("mean trace length must be positive")
    return {
        "budget": budget,
        "mean_prompt_tokens": float(mean_prompt_tokens),
        "mean_generated_tokens": float(mean_generated_tokens),
        "mean_trace_tokens": total,
        "ratio": budget / total,
        "ratio_pct": 100.0 * budget / total,
        "ratio_generation_only_pct": 100.0 * budget / float(mean_generated_tokens),
    }


def summarize_wallclock(
    per_directory: Mapping[str, Sequence[float]],
    *,
    included: Iterable[str],
) -> dict[str, Any]:
    """Total generation wall-clock over the included run directories.

    ``per_directory`` maps a run directory name to the ``wallclock_s`` values
    of its rows. Directories absent from ``included`` are still reported, with
    their hours, so the exclusions are visible rather than implied -- a reader
    can re-add any of them and see what it does to the total.
    """
    include = list(dict.fromkeys(included))
    missing = [name for name in include if name not in per_directory]
    if missing:
        raise ValueError(f"included directories absent from the data: {missing}")

    def _hours(values: Sequence[float]) -> float:
        return sum(float(v) for v in values) / 3600.0

    breakdown = {
        name: {
            "hours": _hours(values),
            "generations": len(values),
            "included": name in include,
        }
        for name, values in sorted(per_directory.items())
    }
    total_hours = sum(
        entry["hours"] for entry in breakdown.values() if entry["included"]
    )
    total_generations = sum(
        entry["generations"] for entry in breakdown.values() if entry["included"]
    )
    return {
        "gpu_hours": total_hours,
        "generations": total_generations,
        "n_directories_included": len(include),
        "n_directories_excluded": len(breakdown) - len(include),
        "excluded_hours": sum(
            entry["hours"] for entry in breakdown.values() if not entry["included"]
        ),
        "by_directory": breakdown,
    }

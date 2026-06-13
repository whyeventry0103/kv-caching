"""Reusable synthetic interleaved-reference RSP construction."""

from __future__ import annotations


def make_rsp_prompt(values: list[int], references: list[str]) -> str:
    if len(values) != len(references):
        raise ValueError("values and references must have equal length")
    lines = [
        "Start at item 0, follow refers_to links until END, and sum every "
        "encountered value exactly once. Return the sum as a boxed integer."
    ]
    for index, (value, reference) in enumerate(zip(values, references, strict=True)):
        lines.append(f"item {index}: value={value}; refers_to={reference}")
    return "\n".join(lines)

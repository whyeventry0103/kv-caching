"""Generate deterministic interleaved-reference synthetic RSP prompts."""

from __future__ import annotations


def make_rsp_prompt(values: list[int], references: list[str]) -> str:
    if len(values) != len(references):
        raise ValueError("values and references must have equal length")
    lines = ["Resolve the interleaved references and return a boxed integer."]
    for index, (value, reference) in enumerate(zip(values, references, strict=True)):
        lines.append(f"item {index}: value={value}; refers_to={reference}")
    return "\n".join(lines)

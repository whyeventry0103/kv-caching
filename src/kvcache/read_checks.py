"""Value-level row checks shared by the accuracy readers (review of fbf869f, R1).

Scientific rationale. Coverage checks (``kvcache.read_validation``) establish
that the expected conditions exist exactly once; they did not establish that
the correctness values themselves are booleans. Readers then converted with
``bool(...)``, under which the string "False" and NaN are both True, so a
corrupted row could raise an accuracy and still read FINAL. These checks run
BEFORE any conversion and make such rows malformed (INVALID).

A readout row carries two coupled fields: ``readout_defined`` must be a
boolean; a defined readout must carry a boolean ``readout_correct``; an
undefined one must carry none (a value there means the row disagrees with
itself about whether the trace had an answer boundary).

Kept in its own module so the bytes of ``kvcache.read_validation`` -- part of
the dependency fingerprint of every stored read -- do not change.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from kvcache.read_validation import INVALID, validate_population


def bool_problems(row: Mapping[str, Any], fields: Iterable[str]) -> list[str]:
    out = []
    for f in fields:
        v = row.get(f)
        if not isinstance(v, bool):
            out.append(f"{f} is not a boolean: {v!r}")
    return out


def readout_flag_problems(row: Mapping[str, Any]) -> list[str]:
    defined = row.get("readout_defined")
    value = row.get("readout_correct")
    if not isinstance(defined, bool):
        return [f"readout_defined is not a boolean: {defined!r}"]
    if defined and not isinstance(value, bool):
        return [f"defined readout without a boolean readout_correct: {value!r}"]
    if not defined and value is not None:
        return [f"undefined readout carries readout_correct {value!r}"]
    return []


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], tuple],
    expected: set[tuple],
    required: Iterable[str] = (),
    bool_fields: Iterable[str] = (),
    readout: bool = False,
) -> dict[str, Any]:
    """``validate_population`` plus value checks; any problem -> INVALID."""
    block = validate_population(rows, key=key, expected=expected, required=required)
    fields = tuple(bool_fields)
    for i, r in enumerate(rows):
        probs = bool_problems(r, fields)
        if readout:
            probs += readout_flag_problems(r)
        if probs:
            try:
                k = repr(key(r))
            except (KeyError, TypeError, ValueError):
                k = f"<row {i}>"
            block["malformed"][k] = block["malformed"].get(k, []) + probs
            block["status"] = INVALID
    return block

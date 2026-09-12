"""``attn-profile-v1``: columnar on-disk format for Amendment Q.3 (and Q.6).

**Why columnar is mandatory rather than tidier.** At the 8192 generation cap one
trace holds ``28 layers x 4 KV groups x 128 events = 14,336`` records. Written as
a list of per-record JSON objects that is about 1.29 MB per trace and **1.03 GB
across 800 traces**, against roughly 1.6 GB of remaining personal budget on the
remote and a ``runs/`` tree that is 2.5 GB in total. Most of those bytes are the
repeated field names. Four flat arrays plus a header carry exactly the same
information in about 0.41 GB.

An earlier draft of the design spec put this at "roughly 4.6k scalars for an 8k
trace", which was the RECORD count at the MEDIAN trace length (2652 steps), not
the scalar count at the cap -- low by roughly 12x, and low in the direction that
decides whether the cell fits at all. The corrected arithmetic is why the schema
is shaped this way.

**Why a new schema version rather than extending ``tv-profile-v1``.** Amendment
P.4's crossing test could not be answered from the stored profiles because they
held only the TV scalar, so answering it cost a full re-run. The lesson recorded
there is that a profile lacking a field a later question needs is a full pass
wasted. ``attn-profile-v1`` is therefore a separate schema whose field set is
fixed in Amendment Q **before** any cell is queued, and readers must refuse a
version they do not recognise rather than guessing.

Index convention: record ``i`` is ``(layer, group, event)`` flattened in C order,
so ``i = (layer * n_kv_groups + group) * n_events + event``. ``index_of`` is the
single definition of that mapping; nothing recomputes it inline.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "attn-profile-v1"

# The columns, in the order the writer emits them. Adding one is a schema
# version bump, not an in-place edit: a reader that finds an unknown column in a
# v1 file cannot know whether the rest is still meaningful.
FLOAT_COLUMNS = ("mass_loss", "shape_tv", "collateral")
INT_COLUMNS = ("n_evicted",)


def index_of(
    layer: int, group: int, event: int, *, n_kv_groups: int, n_events: int
) -> int:
    """Flat index for one (layer, group, event) record. C order, single source."""
    return (int(layer) * int(n_kv_groups) + int(group)) * int(n_events) + int(event)


def build_attn_profile(
    *,
    n_layers: int,
    n_kv_groups: int,
    n_events: int,
    compression_interval: int,
    event_steps: Sequence[int],
    tau: float,
    columns: dict[str, Sequence[float | None]],
    first_prune_delta_max: float,
) -> dict[str, Any]:
    """Assemble a v1 payload, validating shape before anything reaches disk.

    ``first_prune_delta_max`` is the Q.3 self-control and is stored in the
    payload rather than recomputed later: at the first prune both models have
    consumed identical token IDs, so the shape distortion must be exactly 0.0.
    Storing it means a reader can enforce the gate without the raw rows.
    """
    expected = int(n_layers) * int(n_kv_groups) * int(n_events)
    if len(event_steps) != int(n_events):
        raise ValueError(
            f"event_steps has {len(event_steps)} entries, expected {n_events}"
        )
    missing = set(FLOAT_COLUMNS + INT_COLUMNS) - set(columns)
    if missing:
        raise ValueError(f"attn-profile-v1 is missing columns: {sorted(missing)}")
    extra = set(columns) - set(FLOAT_COLUMNS + INT_COLUMNS)
    if extra:
        raise ValueError(
            f"unknown columns for {SCHEMA_VERSION}: {sorted(extra)}; adding a "
            "field is a schema version bump, not an in-place edit"
        )
    for name, values in columns.items():
        if len(values) != expected:
            raise ValueError(
                f"column {name!r} has {len(values)} entries, expected {expected} "
                f"({n_layers} layers x {n_kv_groups} groups x {n_events} events)"
            )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "n_layers": int(n_layers),
        "n_kv_groups": int(n_kv_groups),
        "n_events": int(n_events),
        "compression_interval": int(compression_interval),
        "collateral_tau": float(tau),
        "index_order": "(layer, group, event) flattened C-order",
        "event_steps": [int(s) for s in event_steps],
        "first_prune_delta_max": float(first_prune_delta_max),
    }
    for name in FLOAT_COLUMNS:
        payload[name] = [None if v is None else float(v) for v in columns[name]]
    for name in INT_COLUMNS:
        payload[name] = [None if v is None else int(v) for v in columns[name]]
    return payload


def write_attn_profile(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a validated payload. Refuses to overwrite, like the TV profiles."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"{p} exists; result rows are append-only, never patched")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return p


def read_attn_profile(path: str | Path) -> dict[str, Any]:
    """Read a v1 profile, refusing any other version rather than guessing."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"expected {SCHEMA_VERSION}, found {version!r}; a reader that guesses "
            "across schema versions is how a silent field mismatch happens"
        )
    return payload


def self_control_passed(payload: dict[str, Any]) -> bool:
    """Amendment Q.3's gate: shape distortion at the first prune must be 0.0.

    Exactly zero, not approximately: under teacher forcing the two models have
    consumed identical token IDs up to the first prune, so the query vectors are
    identical and any non-zero value is harness-introduced divergence.
    """
    return float(payload["first_prune_delta_max"]) == 0.0

"""Schema checks for attn-profile-v1 (Amendment Q.4), including the size claim."""

from __future__ import annotations

import json

import pytest

from kvcache.attn_profile import (
    FLOAT_COLUMNS,
    INT_COLUMNS,
    SCHEMA_VERSION,
    build_attn_profile,
    index_of,
    read_attn_profile,
    self_control_passed,
    write_attn_profile,
)


def _payload(n_layers=2, n_kv_groups=2, n_events=3, delta_max=0.0):
    n = n_layers * n_kv_groups * n_events
    cols = {name: [0.1] * n for name in FLOAT_COLUMNS}
    cols.update({name: [4] * n for name in INT_COLUMNS})
    return build_attn_profile(
        n_layers=n_layers,
        n_kv_groups=n_kv_groups,
        n_events=n_events,
        compression_interval=64,
        event_steps=[64 * (i + 1) for i in range(n_events)],
        tau=0.01,
        columns=cols,
        first_prune_delta_max=delta_max,
    )


def test_index_is_c_order_and_covers_every_record_exactly_once():
    n_layers, n_kv_groups, n_events = 3, 4, 5
    seen = [
        index_of(la, g, e, n_kv_groups=n_kv_groups, n_events=n_events)
        for la in range(n_layers)
        for g in range(n_kv_groups)
        for e in range(n_events)
    ]
    assert sorted(seen) == list(range(n_layers * n_kv_groups * n_events))


def test_build_rejects_a_wrong_length_column():
    cols = {name: [0.1] * 11 for name in FLOAT_COLUMNS}
    cols.update({name: [1] * 11 for name in INT_COLUMNS})
    with pytest.raises(ValueError, match="expected 12"):
        build_attn_profile(
            n_layers=2,
            n_kv_groups=2,
            n_events=3,
            compression_interval=64,
            event_steps=[64, 128, 192],
            tau=0.01,
            columns=cols,
            first_prune_delta_max=0.0,
        )


def test_build_rejects_a_missing_column():
    cols = {name: [0.1] * 12 for name in FLOAT_COLUMNS[:-1]}
    cols.update({name: [1] * 12 for name in INT_COLUMNS})
    with pytest.raises(ValueError, match="missing columns"):
        build_attn_profile(
            n_layers=2,
            n_kv_groups=2,
            n_events=3,
            compression_interval=64,
            event_steps=[64, 128, 192],
            tau=0.01,
            columns=cols,
            first_prune_delta_max=0.0,
        )


def test_build_rejects_an_unknown_column_rather_than_storing_it():
    """Adding a field is a version bump; silently accepting it is the Task 4 bug."""
    cols = {name: [0.1] * 12 for name in FLOAT_COLUMNS}
    cols.update({name: [1] * 12 for name in INT_COLUMNS})
    cols["something_new"] = [0.0] * 12
    with pytest.raises(ValueError, match="schema version bump"):
        build_attn_profile(
            n_layers=2,
            n_kv_groups=2,
            n_events=3,
            compression_interval=64,
            event_steps=[64, 128, 192],
            tau=0.01,
            columns=cols,
            first_prune_delta_max=0.0,
        )


def test_reader_refuses_a_foreign_schema_version(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema_version": "tv-profile-v1"}))
    with pytest.raises(ValueError, match="attn-profile-v1"):
        read_attn_profile(p)


def test_round_trip_and_no_overwrite(tmp_path):
    p = tmp_path / "a" / "prof.json"
    write_attn_profile(p, _payload())
    back = read_attn_profile(p)
    assert back["schema_version"] == SCHEMA_VERSION
    assert back["index_order"].startswith("(layer, group, event)")
    with pytest.raises(FileExistsError, match="append-only"):
        write_attn_profile(p, _payload())


def test_self_control_gate_is_exact_zero_not_approximate():
    """Q.3: identical token IDs at the first prune make 0.0 a guarantee."""
    assert self_control_passed(_payload(delta_max=0.0)) is True
    assert self_control_passed(_payload(delta_max=1e-12)) is False


def test_columnar_encoding_is_materially_smaller_than_per_record_dicts(tmp_path):
    """The size claim that made columnar mandatory, checked rather than asserted.

    Amendment Q.4: 800 traces at the 8192 cap is ~1.03 GB as per-record dicts
    against ~1.6 GB of headroom, and ~0.41 GB columnar.
    """
    n_layers, n_kv_groups, n_events = 4, 4, 8
    n = n_layers * n_kv_groups * n_events
    payload = _payload(n_layers, n_kv_groups, n_events)
    columnar = len(json.dumps(payload))
    per_record = len(
        json.dumps(
            [
                {"mass_loss": 0.1, "shape_tv": 0.1, "collateral": 0.1, "n_evicted": 4}
                for _ in range(n)
            ]
        )
    )
    assert columnar < per_record / 2, (
        f"columnar {columnar}B vs per-record {per_record}B; the whole reason "
        "this schema is columnar is the ~2.5x saving"
    )


def test_cap_record_count_matches_the_amendment():
    """14,336 records / 57,344 scalars at the 8192 cap -- not the 4.6k first stated."""
    n_layers, n_kv_groups, interval, cap = 28, 4, 64, 8192
    n_events = cap // interval
    records = n_layers * n_kv_groups * n_events
    assert n_events == 128
    assert records == 14_336
    assert records * (len(FLOAT_COLUMNS) + len(INT_COLUMNS)) == 57_344

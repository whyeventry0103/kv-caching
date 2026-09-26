"""Experiment identity for Meeting #3 artifacts: what produced a row, and gates.

A row, a basis or a gate marker is only reusable (by ``--resume``, a reader,
or the queue) when it demonstrably came from the same scientific experiment:

* ``implementation_id`` -- sha256 over the GENERATION-relevant source files of
  one runner family (not the whole tree). Editing a reader, a figure script or
  a document does not invalidate expensive generation; editing the code that
  computes a row does.
* ``config_sha256`` -- sha256 of the canonical JSON of the runner's config.
* the model revision, the reference run and a hash of the reference token IDs
  the row was computed from, and (where used) the frozen basis bytes.

Gate markers are written atomically, and writing one outcome retires the other
(an old PASS is renamed, never silently left accepted, and never deleted), so
a failed attempt cannot leave a stale PASS behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATION_SOURCES: dict[str, tuple[str, ...]] = {
    "capture": (
        "src/kvcache/capture.py",
        "src/kvcache/presses/__init__.py",
        "src/kvcache/presses/rkv.py",
        "src/kvcache/replay.py",
        "src/kvcache/models.py",
        "scripts/run_capture.py",
    ),
    "capture_v2": (
        "src/kvcache/capture.py",
        "src/kvcache/capture_v2.py",
        "src/kvcache/presses/__init__.py",
        "src/kvcache/presses/rkv.py",
        "src/kvcache/replay.py",
        "src/kvcache/models.py",
        "scripts/run_capture_v2.py",
    ),
    "basis": (
        "src/kvcache/capture.py",
        "src/kvcache/models.py",
        "scripts/run_basis_fit.py",
    ),
    "exact_lag": (
        "src/kvcache/exact_lag.py",
        "src/kvcache/capture.py",
        "src/kvcache/presses/__init__.py",
        "src/kvcache/metrics/restoration.py",
        "src/kvcache/models.py",
        "scripts/run_exact_lag.py",
    ),
    "restore_v2": (
        "src/kvcache/restore_v2.py",
        "src/kvcache/exact_lag.py",
        "src/kvcache/capture.py",
        "src/kvcache/presses/__init__.py",
        "src/kvcache/metrics/restoration.py",
        "src/kvcache/models.py",
        "scripts/run_restore_v2.py",
    ),
    "fork": (
        "src/kvcache/fork_control.py",
        "src/kvcache/presses/__init__.py",
        "src/kvcache/models.py",
        "scripts/run_fork_control.py",
    ),
    "fork_precision": (
        "src/kvcache/fork_precision.py",
        "src/kvcache/fork_control.py",
        "src/kvcache/presses/__init__.py",
        "src/kvcache/models.py",
        "scripts/run_fork_precision.py",
    ),
    "dacc": (
        "src/kvcache/generation.py",
        "src/kvcache/presses/__init__.py",
        "src/kvcache/models.py",
        "src/kvcache/bank.py",
        "scripts/run_grid.py",
    ),
}


def implementation_id(kind: str, root: Path | None = None) -> str:
    """sha256 over (relative path, bytes) of one family's generation sources."""
    base = Path(root) if root is not None else PROJECT_ROOT
    h = hashlib.sha256()
    for rel in sorted(GENERATION_SOURCES[kind]):
        h.update(rel.encode())
        h.update(b"\0")
        h.update((base / rel).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inputs_sha256(*token_lists: Iterable[int]) -> str:
    """Hash of the exact reference token IDs a row was computed from."""
    return hashlib.sha256(
        json.dumps([list(t) for t in token_lists], separators=(",", ":")).encode()
    ).hexdigest()


PROVENANCE_SCHEMA = "provenance-v1"
INPUT_MANIFEST = "configs/meeting3/input_manifest.json"
REQUIRED_FIELDS = (
    "schema",
    "kind",
    "implementation_id",
    "config_name",
    "config_sha256",
    "model_revision",
    "reference_run",
    "basis_npz_sha256",
    "input_manifest_sha256",
)


def _manifest_path(root: Path | None = None) -> Path:
    # KVCACHE_INPUT_MANIFEST exists for tests with synthetic reference trees;
    # production reads the committed file.
    env = os.environ.get("KVCACHE_INPUT_MANIFEST")
    if env:
        return Path(env)
    return (Path(root) if root is not None else PROJECT_ROOT) / INPUT_MANIFEST


def input_manifest_sha256(root: Path | None = None) -> str:
    return hashlib.sha256(_manifest_path(root).read_bytes()).hexdigest()


def load_input_manifest(root: Path | None = None) -> dict[str, Any]:
    return json.loads(_manifest_path(root).read_text(encoding="utf-8"))


def expected_inputs(manifest: Mapping[str, Any], kind: str, pid: str, seed: int) -> str:
    """The frozen reference-input hash a (problem, seed) row must carry.

    Replay families hash prompt + reference trace; free-running D_acc hashes
    the prompt only (it generates its own trace). A key absent from the
    manifest is outside the frozen design and raises.
    """
    if kind == "dacc":
        return manifest["prompts"][pid]
    return manifest["traces_prompt_plus_trace"][f"{pid}|{int(seed)}"]


def check_inputs(
    manifest: Mapping[str, Any], kind: str, pid: str, seed: int, computed: str
) -> None:
    """Refuse a row whose reference inputs differ from the frozen manifest."""
    try:
        want = expected_inputs(manifest, kind, pid, seed)
    except KeyError:
        raise SystemExit(
            f"REFUSED: ({pid}, {seed}) is not in the frozen input manifest"
        ) from None
    if computed != want:
        raise SystemExit(
            f"REFUSED: reference inputs for ({pid}, {seed}) differ from the frozen "
            "input manifest; the reference bank changed"
        )


def validate_row_provenance(
    prov: Mapping[str, Any] | None,
    *,
    expected: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    kind: str,
    pid: str,
    seed: int,
) -> list[str]:
    """Problems with one row's provenance (empty list = valid).

    Required fields and schema version; equality with the EXPECTED run
    identity derived from committed config/model/reference (not from the row
    itself); and the row's input hash against the frozen manifest.
    """
    if not prov:
        return ["no provenance"]
    out = [f"missing {f}" for f in REQUIRED_FIELDS if f not in prov]
    if prov.get("schema") != PROVENANCE_SCHEMA:
        out.append(f"schema {prov.get('schema')!r} != {PROVENANCE_SCHEMA}")
    if prov.get("kind") != kind:
        out.append(f"kind {prov.get('kind')!r} != {kind!r}")
    if expected is not None:
        out += [
            f"{k}: {prov.get(k)!r} != expected {v!r}"
            for k, v in expected.items()
            if prov.get(k) != v
        ]
    if manifest is not None:
        try:
            want = expected_inputs(manifest, kind, pid, seed)
        except KeyError:
            out.append("condition not in the frozen input manifest")
        else:
            if prov.get("inputs_sha256") != want:
                out.append("inputs_sha256 differs from the frozen input manifest")
    return out


def identity(
    kind: str,
    config: Mapping[str, Any],
    *,
    model_revision: str,
    reference_run: str | None = None,
    basis_npz_sha256: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """The run-level identity every row and gate of one run must share."""
    return {
        "schema": PROVENANCE_SCHEMA,
        "input_manifest_sha256": input_manifest_sha256(root),
        "kind": kind,
        "implementation_id": implementation_id(kind, root),
        "config_name": config.get("name"),
        "config_sha256": config_sha256(config),
        "model_revision": str(model_revision),
        "reference_run": reference_run,
        "basis_npz_sha256": basis_npz_sha256,
    }


def same_identity(a: Mapping[str, Any] | None, b: Mapping[str, Any]) -> bool:
    return bool(a) and all(a.get(k) == v for k, v in b.items())


def write_json_atomic(path: Path, obj: Any) -> Path:
    """Write via a temporary file and rename, so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", "utf-8")
    os.replace(tmp, path)
    return path


def write_gate(
    base: Path, pass_name: str, fail_name: str, report: Mapping[str, Any]
) -> Path:
    """Record a gate outcome atomically and retire the opposite marker.

    The retired file is renamed with a timestamp suffix (evidence is kept, not
    deleted), so the queue -- which accepts a PASS only when it exists, parses
    and says ``passed: true`` -- can never act on a stale PASS after a FAIL.
    """
    passed = bool(report.get("passed"))
    keep, retire = (pass_name, fail_name) if passed else (fail_name, pass_name)
    old = Path(base) / retire
    if old.exists():
        os.replace(old, old.with_name(f"{retire}.superseded-{int(time.time())}"))
    return write_json_atomic(Path(base) / keep, dict(report))


def gate_accepted(path: Path, expected: Mapping[str, Any] | None = None) -> bool:
    """A marker counts only if it parses, says passed, and matches ``expected``."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if d.get("passed") is not True:
        return False
    return expected is None or same_identity(d.get("identity"), expected)

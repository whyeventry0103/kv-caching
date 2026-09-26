"""Typed queue gates for Meeting #3 (R.15): acceptance derived from committed specs.

A gate marker satisfies its dependents only when it proves what it claims. The
EXPECTED identity is computed here from the committed job config, the pinned
model revision in ``configs/models.yaml``, the frozen input manifest and the
current generation code -- never read from the marker itself -- and the marker
must match it field for field.

* ``dacc_smoke``: ``dacc-gate-smoke-PASS.json`` says ``passed: true`` and its
  identity equals the D_acc smoke's expected identity.
* ``basis``: the header says ``passed: true``/``FROZEN``, its npz exists and
  hashes to the recorded value, and the header identity equals the fit's
  expected identity.
* ``capture_smoke``: ``capture-smoke-PASS.json`` says ``passed: true`` and its
  identity equals the capture smoke's expected identity, which includes the
  bytes of the VERIFIED basis.

A generation-code change therefore invalidates a gate (the smoke must rerun),
while reader, figure or documentation edits do not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kvcache.provenance import PROJECT_ROOT, identity

GATE_KINDS = ("dacc_smoke", "basis", "capture_smoke")


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _revision(cfg: dict[str, Any], root: Path) -> str:
    models = _yaml(root / "configs/models.yaml")["models"]
    return str(models[cfg["models"][0]]["revision"])


def _read(path: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_basis(
    runs_root: Path,
    config: str,
    *,
    root: Path = PROJECT_ROOT,
    run_name: str = "q6-basis-fit",
) -> tuple[bool, str, str | None]:
    """(ok, reason, npz sha256) for the frozen basis of ``config``."""
    cfg = _yaml(root / config)
    bdir = Path(runs_root) / run_name / "basis"
    head = _read(bdir / "q6_basis.json")
    if head is None:
        return False, "basis header absent or unreadable", None
    if head.get("passed") is not True or head.get("status") != "FROZEN":
        return False, "basis header not FROZEN/passed", None
    npz = bdir / head.get("npz", "q6_basis.npz")
    if not npz.exists():
        return False, "basis npz missing", None
    sha = _sha(npz)
    if sha != head.get("npz_sha256"):
        return False, "basis npz bytes differ from the header hash", None
    want = identity(
        "basis",
        cfg,
        model_revision=_revision(cfg, root),
        reference_run=cfg["reference_run_name"],
        root=root,
    )
    if head.get("identity") != want:
        return False, "basis identity differs from the expected fit identity", None
    return True, "verified", sha


def gate_status(
    kind: str, runs_root: Path, config: str, *, root: Path = PROJECT_ROOT
) -> tuple[bool, str]:
    """Whether a typed gate is satisfied, with the reason."""
    runs_root = Path(runs_root)
    cfg = _yaml(root / config)
    if kind == "basis":
        ok, why, _ = verify_basis(
            runs_root,
            config,
            root=root,
            run_name=cfg.get("basis_run_name", "q6-basis-fit"),
        )
        return ok, why
    if kind == "dacc_smoke":
        marker = runs_root / "dacc-gate-smoke-PASS.json"
        want = identity(
            "dacc",
            cfg,
            model_revision=_revision(cfg, root),
            reference_run=None,
            root=root,
        )
    elif kind in ("capture_smoke", "capture_v2_smoke"):
        marker = runs_root / (
            "capture-smoke-PASS.json"
            if kind == "capture_smoke"
            else "capture2-smoke-PASS.json"
        )
        basis_cfg = "configs/experiments/basis_fit_q6_r9.yaml"
        ok, why, sha = verify_basis(
            runs_root, basis_cfg, root=root, run_name=cfg["basis_run"]
        )
        if not ok:
            return False, f"capture smoke depends on the basis: {why}"
        want = identity(
            "capture" if kind == "capture_smoke" else "capture_v2",
            cfg,
            model_revision=_revision(cfg, root),
            reference_run=cfg["reference_run_name"],
            basis_npz_sha256=sha,
            root=root,
        )
    elif kind == "fork_precision":
        return _fork_precision_status(runs_root, cfg, root)
    else:
        raise ValueError(f"unknown gate kind {kind!r}")
    d = _read(marker)
    if d is None:
        return False, f"{marker.name} absent, empty or unreadable"
    if d.get("passed") is not True:
        return False, f"{marker.name} does not say passed: true"
    if d.get("identity") != want:
        return False, f"{marker.name} identity differs from the expected identity"
    return True, "verified"


def _fork_precision_status(
    runs_root: Path, cfg: dict[str, Any], root: Path
) -> tuple[bool, str]:
    """R.17.0.5 output: complete, of this identity, with every registered part.

    A parseable file is not enough (R.15): both dtype diagnoses must have
    compared at least one step, and every registered floor trace must carry
    compared steps.
    """
    marker = runs_root / "fork-precision" / "fork_precision.json"
    d = _read(marker)
    if d is None:
        return False, f"{marker.name} absent, empty or unreadable"
    if d.get("passed") is not True:
        return False, f"{marker.name} does not say passed: true"
    want = identity(
        "fork_precision",
        cfg,
        model_revision=_revision(cfg, root),
        reference_run=cfg["reference_run_name"],
        root=root,
    )
    if d.get("provenance") != want:
        return False, f"{marker.name} identity differs from the expected identity"
    diag = d.get("diagnosis") or {}
    for dtype in cfg["dtypes"]:
        if not (diag.get(dtype) or {}).get("n_compared"):
            return False, f"no compared diagnosis steps in {dtype}"
    floor = d.get("floor") or {}
    if len(floor) != int(cfg["floor"]["n_problems"]) or not all(floor.values()):
        return False, "floor comparison missing for a registered trace"
    return True, "verified"


def expected_run_identity(
    kind: str, config: str, runs_root: Path, *, root: Path = PROJECT_ROOT
) -> tuple[dict[str, Any] | None, str]:
    """The identity every row of a run of ``config`` must carry (R.15).

    Derived from committed specs, never from the rows. For capture, the basis
    must verify first (its bytes are part of the identity); ``None`` with the
    reason when it does not.
    """
    cfg = _yaml(root / config)
    rev = _revision(cfg, root)
    if kind in ("capture", "capture_v2"):
        ok, why, sha = verify_basis(
            Path(runs_root),
            "configs/experiments/basis_fit_q6_r9.yaml",
            root=root,
            run_name=cfg["basis_run"],
        )
        if not ok:
            return None, f"basis not verified: {why}"
        return (
            identity(
                kind,
                cfg,
                model_revision=rev,
                reference_run=cfg["reference_run_name"],
                basis_npz_sha256=sha,
                root=root,
            ),
            "ok",
        )
    if kind == "dacc":
        return (
            identity("dacc", cfg, model_revision=rev, reference_run=None, root=root),
            "ok",
        )
    if kind in ("exact_lag", "fork"):
        return (
            identity(
                kind,
                cfg,
                model_revision=rev,
                reference_run=cfg["reference_run_name"],
                root=root,
            ),
            "ok",
        )
    raise ValueError(f"unknown run kind {kind!r}")


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    expected: dict[str, Any] | None,
    key: Any,
) -> dict[tuple, list[str]]:
    """Per-row provenance problems (row key -> problems); empty dict = all valid."""
    from kvcache.provenance import load_input_manifest, validate_row_provenance

    manifest = load_input_manifest()
    bad: dict[tuple, list[str]] = {}
    for r in rows:
        try:
            k = key(r)
            seed = k[1]
            probs = validate_row_provenance(
                r.get("provenance"),
                expected=expected,
                manifest=manifest,
                kind=kind,
                pid=r["pid"],
                seed=seed,
            )
        except (KeyError, TypeError, ValueError) as e:
            k, probs = (repr(r)[:60],), [f"malformed row: {e!r}"]
        if expected is None:
            probs = ["expected identity unavailable", *probs]
        if probs:
            bad[k] = probs
    return bad

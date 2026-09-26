"""Synthetic, provenance-correct run trees for the Meeting #3 acceptance tests.

Every identity is derived exactly as the readers derive it (committed configs,
models.yaml, the real frozen input manifest, the verified basis), so a test
that passes here exercises the production validation rather than bypassing it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kvcache.gates import _revision, _yaml, expected_run_identity  # noqa: E402
from kvcache.provenance import identity, load_input_manifest  # noqa: E402

EXP = "configs/experiments"
PIDS = json.loads((ROOT / "configs/meeting3/q2_subset_pids.json").read_text())["pids"]
SEEDS = (11, 23, 47, 89)
MAN = load_input_manifest()


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def write_basis(runs: Path) -> str:
    cfg = _yaml(ROOT / f"{EXP}/basis_fit_q6_r9.yaml")
    ident = identity(
        "basis",
        cfg,
        model_revision=_revision(cfg, ROOT),
        reference_run=cfg["reference_run_name"],
    )
    d = runs / "q6-basis-fit" / "basis"
    d.mkdir(parents=True, exist_ok=True)
    (d / "q6_basis.npz").write_bytes(b"synthetic-basis")
    sha = hashlib.sha256(b"synthetic-basis").hexdigest()
    (d / "q6_basis.json").write_text(
        json.dumps(
            {
                "passed": True,
                "status": "FROZEN",
                "npz": "q6_basis.npz",
                "npz_sha256": sha,
                "identity": ident,
                "reference_run_name": cfg["reference_run_name"],
                "model_revision": _revision(cfg, ROOT),
            }
        )
    )
    return sha


def capture(runs: Path, budget: int = 256, *, bad_control: bool = False) -> None:
    write_basis(runs)
    # b2048 runs as capture-v2 (R.17.2.1); b256 stays capture-v1
    config, kind = (
        (f"{EXP}/capture_math500_b2048_r17.yaml", "capture_v2")
        if budget == 2048
        else (f"{EXP}/capture_math500_b{budget}_r9.yaml", "capture")
    )
    ident, why = expected_run_identity(kind, config, runs)
    assert ident, why
    rows = []
    for i, pid in enumerate(PIDS):
        for seed in SEEDS:
            for press in ("random", "streamingllm", "h2o", "rkv"):
                rows.append(
                    {
                        "pid": pid,
                        "seed": seed,
                        "press": press,
                        "budget": budget,
                        "provenance": {
                            **ident,
                            "inputs_sha256": MAN["traces_prompt_plus_trace"][
                                f"{pid}|{seed}"
                            ],
                        },
                        "checks": {"n_nonfinite": 0, "layer0_delta_max": 1e-7},
                        "collateral_pooled": {"rkv": 0.1, "h2o": 0.3}.get(press, 0.2)
                        + 0.001 * i,
                        "wanted_any_head_pooled": 0.5,
                        "M_mean": 0.2,
                        "M_event_mean": 0.01,
                        "Delta_mean": 0.1,
                        "slices": {},
                        "latent": {
                            "omega_trace": {"random": 2.0, "streamingllm": 1.0}.get(
                                press, 1.5
                            )
                            + 0.01 * i
                        },
                    }
                )
    if bad_control:
        rows[0]["checks"]["layer0_delta_max"] = 1.0
    _jsonl(runs / f"capture-b{budget}" / "math500" / "m" / "rows.jsonl", rows)


def fc_bank(runs: Path) -> None:
    rows = [
        {
            "pid": p,
            "decoding": {"seed": s},
            "correct": True,
            "press": "fullcache",
            "budget": "full",
        }
        for p in PIDS
        for s in SEEDS
    ]
    _jsonl(runs / "math500-fc-powered-merged" / "math500" / "m" / "fc.jsonl", rows)


def dacc(runs: Path) -> None:
    fc_bank(runs)
    for t0 in (0, 512, 2048):
        ident, why = expected_run_identity(
            "dacc", f"{EXP}/grid_math500_dacc_t{t0}_r3.yaml", runs
        )
        rows, toks = [], []
        for pid in PIDS:
            for seed in SEEDS:
                for press in ("random", "streamingllm"):
                    for budget in (256, 2048):
                        rows.append(
                            {
                                "pid": pid,
                                "press": press,
                                "budget": budget,
                                "decoding": {"seed": seed},
                                "correct": budget == 2048,
                                "truncated": False,
                                "n_tokens": 1000,
                                "provenance": {
                                    **ident,
                                    "inputs_sha256": MAN["prompts"][pid],
                                },
                                "onset": {
                                    "compression_start_step": t0,
                                    "eviction_onset_step": t0 + 65,
                                    "achieved_exposure": 900,
                                    "exposed": True,
                                },
                            }
                        )
                        toks.append(
                            {
                                "pid": pid,
                                "seed": seed,
                                "press": press,
                                "budget": budget,
                                "trace_token_ids": [1, 2, 3, seed],
                            }
                        )
        _jsonl(runs / f"dacc-t{t0}" / "math500" / "m" / "rows.jsonl", rows)
        if t0 == 0:
            _jsonl(runs / "dacc-t0" / "token_ids" / "m" / "t.jsonl", toks)
            for b, run in ((256, "grid-diss256-tok"), (2048, "grid-diss2048-tok")):
                _jsonl(
                    runs / run / "token_ids" / "m" / "t.jsonl",
                    [t for t in toks if t["budget"] == b],
                )


def exact_lag(runs: Path, *, drop_lag: bool = False, bad_gate: bool = False) -> None:
    config = f"{EXP}/exact_lag_math500_r5.yaml"
    ident, _ = expected_run_identity("exact_lag", config, runs)
    gate = {
        "passed": True,
        "checked": "reinserted keys AND values bitwise equal the stash, per head",
    }
    rows = []
    for i, pid in enumerate(PIDS):
        for seed in SEEDS:
            for press in ("random", "streamingllm"):
                lags = {}
                for lag in (1, 4, 16, 64, 256):
                    lags[str(lag)] = {
                        "eligible": True,
                        "requested_lag": lag,
                        "achieved_lag": lag,
                        "applied_step": 192 + lag,
                        "none": 0.3,
                        "matched": 0.2 + 0.0001 * lag + 0.001 * i,
                        "random": 0.29,
                        "relaxed": 0.1,
                        "gates": {
                            a: dict(gate) for a in ("matched", "random", "relaxed")
                        },
                    }
                rows.append(
                    {
                        "pid": pid,
                        "seed": seed,
                        "press": press,
                        "budget": 256,
                        "status": "READ",
                        "cohort_step": 192,
                        "n_trace_tokens": 1000,
                        "n_reinsert": 64,
                        "stash_identity_checks": {"n_entries_checked": 40},
                        "provenance": {
                            **ident,
                            "inputs_sha256": MAN["traces_prompt_plus_trace"][
                                f"{pid}|{seed}"
                            ],
                        },
                        "lags": lags,
                    }
                )
    if drop_lag:
        del rows[0]["lags"]["256"]
    if bad_gate:
        rows[0]["lags"]["1"]["gates"]["matched"]["passed"] = False
    _jsonl(runs / "exact-lag-r5" / "math500" / "m" / "rows.jsonl", rows)


def fork(runs: Path, *, infinite_control: bool = False) -> None:
    for press, short in (("random", "random"), ("streamingllm", "slm")):
        ident, _ = expected_run_identity(
            "fork", f"{EXP}/fork_control_{short}_r11.yaml", runs
        )
        rows = []
        for pid in PIDS:
            for interval in (16, 64, 256):
                rows.append(
                    {
                        "pid": pid,
                        "seed": 11,
                        "press": press,
                        "budget": 256,
                        "interval": interval,
                        "deterministic_control_max_abs_logit": 1e-6,
                        "forks": [
                            {
                                "max_abs_logit": 1e-7,
                                "max_tv": 0.0,
                                "selection_agrees": True,
                            }
                        ],
                        "n_forks": 1,
                        "tv_profile": [0.0] * 40 + [0.1, 0.0, 0.0, 0.0] * 20,
                        "first_removal_step": 40,
                        "provenance": {
                            **ident,
                            "inputs_sha256": MAN["traces_prompt_plus_trace"][
                                f"{pid}|11"
                            ],
                        },
                    }
                )
        if infinite_control and press == "random":
            rows[0]["deterministic_control_max_abs_logit"] = float("inf")
            rows[0]["forks"][0]["max_abs_logit"] = 1.0
        _jsonl(runs / f"fork-control-{short}" / "math500" / "m" / "rows.jsonl", rows)


def t1_tv(runs: Path) -> None:
    fc_bank(runs)
    for b, run in (
        (256, "grid-diss256-tok"),
        (512, "grid-diss512"),
        (1024, "grid-diss1024"),
        (2048, "grid-diss2048-tok"),
    ):
        rows = [
            {
                "pid": p,
                "decoding": {"seed": s},
                "correct": b >= 1024,
                "press": press,
                "budget": b,
            }
            for p in PIDS
            for s in SEEDS
            for press in ("random", "streamingllm")
        ]
        _jsonl(runs / run / "math500" / "m" / "rows.jsonl", rows)
    for b in (512, 1024):
        for press in ("random", "streamingllm"):
            for s in SEEDS:
                d = (
                    runs
                    / f"t1-tv{b}"
                    / "profiles"
                    / "math500"
                    / "m"
                    / f"{press}-{b}-s{s}"
                )
                d.mkdir(parents=True, exist_ok=True)
                for i, p in enumerate(PIDS):
                    (d / f"p{i}.json").write_text(
                        json.dumps(
                            {
                                "schema_version": "tv-profile-summary-v1",
                                "reference_key": [
                                    "math500",
                                    p,
                                    "m",
                                    "fullcache",
                                    "full",
                                    s,
                                ],
                                "condition": {"press": press, "budget": b},
                                "mean_tv": 0.05 + 0.001 * i,
                            }
                        )
                    )

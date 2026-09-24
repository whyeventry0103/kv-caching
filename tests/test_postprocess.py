"""R.14 postprocessing stage end to end on a synthetic, full-size capture grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import postprocess_meeting3 as pp  # noqa: E402

PIDS = json.loads((ROOT / "configs/meeting3/q2_subset_pids.json").read_text())["pids"]


def _grid(tmp_path, *, drop=0, bad=False):
    rows = []
    for i, pid in enumerate(PIDS):
        for seed in (11, 23, 47, 89):
            for press in ("random", "streamingllm", "h2o", "rkv"):
                rows.append(
                    {
                        "pid": pid,
                        "seed": seed,
                        "press": press,
                        "budget": 256,
                        "provenance": {"kind": "capture", "config_sha256": "c"},
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
    rows = rows[drop:]
    if bad:
        rows[0]["checks"]["layer0_delta_max"] = 1.0
    d = tmp_path / "runs" / "capture-b256" / "math500" / "m"
    d.mkdir(parents=True, exist_ok=True)
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path / "runs", tmp_path / "out"


def test_complete_grid_is_final_with_figure_and_terminal_marker(tmp_path):
    runs, out = _grid(tmp_path)
    m = pp.run_stage("capture-b256", runs, out_root=out)
    assert m["status"] == "FINAL" and m["terminal"] and m["passed"]
    assert (out / "figures" / "meeting3_capture_b256.png").exists()
    assert all(v for v in m["outputs"].values())  # every output hashed
    marker = json.loads((runs / "postprocess" / "capture-b256.json").read_text())
    assert marker["terminal"] is True
    d = json.loads((out / "results/diag/meeting3/capture_b256.json").read_text())
    assert d["q3_registered_contrast"]["verdict"].startswith("SUPPORTS")


def test_partial_grid_retries_and_draws_no_figure(tmp_path):
    runs, out = _grid(tmp_path, drop=1)
    m = pp.run_stage("capture-b256", runs, out_root=out)
    assert m["status"] == "PARTIAL" and not m["terminal"]
    assert not (out / "figures").exists()


def test_failed_control_is_terminal_invalid_and_not_passed(tmp_path):
    runs, out = _grid(tmp_path, bad=True)
    m = pp.run_stage("capture-b256", runs, out_root=out)
    assert m["status"] == "INVALID" and m["terminal"] and not m["passed"]

"""R.12 summary-profile AUC read in report_budget_curve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_budget_curve import _summary_auc  # noqa: E402


def _write(root, run, press, pid, seed, mean):
    d = root / run / "profiles" / "math500" / "m" / f"{press}-512-s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text(
        json.dumps(
            {
                "schema_version": "tv-profile-summary-v1",
                "reference_key": ["math500", pid, "m", "fullcache", "full", seed],
                "condition": {"press": press, "budget": 512},
                "mean_tv": mean,
            }
        )
    )


def test_auc_is_seeds_within_problem_then_problems(tmp_path):
    _write(tmp_path, "tv", "random", "a", 11, 0.1)
    _write(tmp_path, "tv", "random", "a", 23, 0.3)
    _write(tmp_path, "tv", "random", "b", 11, 0.5)
    out = _summary_auc(tmp_path, ["512=tv"], 100, 0)
    r = out[("random", 512)]
    assert r["n_clusters"] == 2
    assert r["mean"] == pytest.approx((0.2 + 0.5) / 2)


def test_a_full_profile_is_refused_not_misread(tmp_path):
    _write(tmp_path, "tv", "random", "a", 11, 0.1)
    p = next((tmp_path / "tv").rglob("a.json"))
    p.write_text(json.dumps({"schema_version": "tv-profile-v2", "estimate": [0.1]}))
    with pytest.raises(SystemExit, match="summary"):
        _summary_auc(tmp_path, ["512=tv"], 100, 0)

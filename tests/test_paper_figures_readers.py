"""The figure profile readers must skip sidecars, like every other reader.

`kvcache.replay.is_profile_file` exists because a bare ``*.json`` glob over a
run's profile tree also picks up the ``<pid>.restoration.json`` and
``<pid>.ratio_trace.json`` sidecars written beside each profile. The four
report scripts adopted the predicate when it was written; the two readers in
``scripts/make_paper_figures.py`` were missed and survived only because those
two sidecar payloads happen to carry no ``estimate`` key, so the readers'
``continue`` guard skipped them by accident.

That accident expires as soon as fig4 is pointed at ``replay-ratio-matched-m3-v2``
for the ratio-matched overlay -- a run that writes a sidecar beside every
profile. These tests pin the predicate rather than the accident: the fixture
sidecar deliberately DOES carry an ``estimate``, so a reader that relies on the
missing key fails here.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL = [0.0, 0.0, 0.1, 0.2, 0.3]
SIDECAR = [9.9, 9.9, 9.9, 9.9, 9.9]


def _load_module():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    path = PROJECT_ROOT / "scripts" / "make_paper_figures.py"
    spec = importlib.util.spec_from_file_location("mpf_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profiles_dir(tmp_path: Path) -> Path:
    leaf = tmp_path / "math500" / "model" / "random-256-s11"
    leaf.mkdir(parents=True)
    for name in ("aaaa1111.json", "bbbb2222.json"):
        (leaf / name).write_text(json.dumps({"estimate": REAL, "onset_step": 2}))
    # Carries 'estimate' on purpose: the point is that suffix, not payload
    # shape, is what excludes a sidecar.
    (leaf / "aaaa1111.ratio_trace.json").write_text(
        json.dumps({"estimate": SIDECAR, "onset_step": 2, "matched_ratio": 0.038})
    )
    (leaf / "bbbb2222.restoration.json").write_text(
        json.dumps({"estimate": SIDECAR, "onset_step": 2, "passed": True})
    )
    return tmp_path


def test_onset_aligned_curve_excludes_sidecars(profiles_dir: Path) -> None:
    module = _load_module()
    result = module._onset_aligned_curve(profiles_dir)
    assert result is not None
    _, ys = result
    assert ys == pytest.approx(REAL), "a sidecar leaked into the TV curve"
    assert max(ys) < 1.0


def test_windowed_tv_counts_profiles_only(profiles_dir: Path, capsys) -> None:
    module = _load_module()
    module.print_windowed_tv(profiles_dir)
    out = capsys.readouterr().out
    match = re.search(r"windowed TV over (\d+) profiles", out)
    assert match is not None, out
    # Two profiles and two sidecars on disk; counting 4 is the double-count
    # bug this predicate exists to prevent.
    assert int(match.group(1)) == 2, out

    # The sidecar value (9.9) must not reach any window mean either.
    before = re.search(r"before: per-profile ([\d.]+)", out)
    assert before is not None, out
    assert float(before.group(1)) == pytest.approx(0.0), out

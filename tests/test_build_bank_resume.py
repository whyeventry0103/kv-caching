import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_bank", ROOT / "scripts" / "build_bank.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(pid, seed):
    return json.dumps({"bench": "math500", "pid": pid, "decoding": {"seed": seed}})


def test_resume_done_keys_requires_result_and_sidecar(tmp_path):
    res = tmp_path / "math500" / "m"
    tok = tmp_path / "token_ids" / "math500" / "m"
    res.mkdir(parents=True)
    tok.mkdir(parents=True)
    # p0: result + sidecar -> done.  p1: result only -> NOT done (redo).
    (res / "fullcache-full-s11.jsonl").write_text(
        _row("p0", 11) + "\n" + _row("p1", 11) + "\n"
    )
    (tok / "fullcache-full-s11.jsonl").write_text(_row("p0", 11) + "\n")

    done = MODULE._resume_done_keys(tmp_path, store_token_ids=True)
    assert done == {("math500", "p0", 11)}


def test_resume_heals_partial_trailing_line(tmp_path):
    res = tmp_path / "math500" / "m"
    res.mkdir(parents=True)
    f = res / "fullcache-full-s11.jsonl"
    f.write_text(_row("p0", 11) + "\n" + '{"bench": "math500", "pid": "p1"')  # partial
    done = MODULE._resume_done_keys(tmp_path, store_token_ids=False)
    assert done == {("math500", "p0", 11)}
    # the corrupt trailing line is healed away so append_result won't choke
    assert f.read_text().strip().splitlines() == [_row("p0", 11)]

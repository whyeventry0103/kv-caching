import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_grid", ROOT / "scripts" / "run_grid.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _result(pid, press, budget, seed):
    return json.dumps(
        {
            "bench": "math500",
            "pid": pid,
            "press": press,
            "budget": budget,
            "decoding": {"seed": seed},
        }
    )


def _sidecar(pid, press, budget, seed):
    return json.dumps(
        {"bench": "math500", "pid": pid, "press": press, "budget": budget, "seed": seed}
    )


def test_grid_done_keys_ignore_sidecars_when_not_storing(tmp_path):
    cell = tmp_path / "math500" / "m"
    cell.mkdir(parents=True)
    (cell / "streamingllm-256-s11.jsonl").write_text(
        _result("p0", "streamingllm", 256, 11) + "\n"
    )
    done = MODULE._grid_done_keys(tmp_path, store_token_ids=False)
    assert done == {("math500", "p0", "streamingllm", "256", 11)}


def test_grid_done_keys_require_sidecar_when_storing(tmp_path):
    cell = tmp_path / "math500" / "m"
    tok = tmp_path / "token_ids" / "math500" / "m"
    cell.mkdir(parents=True)
    tok.mkdir(parents=True)
    # p0 has result + sidecar -> done.  p1 has result only -> redo (incomplete).
    rows = _result("p0", "random", 256, 11) + "\n" + _result("p1", "random", 256, 11)
    (cell / "random-256-s11.jsonl").write_text(rows + "\n")
    (tok / "random-256-s11.jsonl").write_text(_sidecar("p0", "random", 256, 11) + "\n")
    done = MODULE._grid_done_keys(tmp_path, store_token_ids=True)
    assert done == {("math500", "p0", "random", "256", 11)}


def test_grid_done_keys_heals_partial_trailing_line(tmp_path):
    cell = tmp_path / "math500" / "m"
    cell.mkdir(parents=True)
    f = cell / "streamingllm-256-s11.jsonl"
    f.write_text(_result("p0", "streamingllm", 256, 11) + "\n" + '{"bench": "math500"')
    done = MODULE._grid_done_keys(tmp_path, store_token_ids=False)
    assert done == {("math500", "p0", "streamingllm", "256", 11)}
    kept = f.read_text().strip().splitlines()
    assert kept == [_result("p0", "streamingllm", 256, 11)]

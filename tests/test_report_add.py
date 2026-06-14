import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_add", ROOT / "scripts" / "report_add.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_add_report_computes_per_cell_add_and_flip(tmp_path):
    full = tmp_path / "full"
    comp = tmp_path / "comp"
    # Two problems, n=2 each. p0: FC all "A" correct; p1: FC all "B" correct.
    full_rows = [
        {"pid": "p0", "decoding": {"seed": 0}, "answer_canon": "A", "correct": True},
        {"pid": "p0", "decoding": {"seed": 1}, "answer_canon": "A", "correct": True},
        {"pid": "p1", "decoding": {"seed": 0}, "answer_canon": "B", "correct": True},
        {"pid": "p1", "decoding": {"seed": 1}, "answer_canon": "B", "correct": True},
    ]
    # Compressed: p0 flips entirely to wrong "C"; p1 unchanged.
    comp_rows = [
        {
            "pid": "p0",
            "decoding": {"seed": 0},
            "press": "streamingllm",
            "budget": 256,
            "answer_canon": "C",
            "correct": False,
        },
        {
            "pid": "p0",
            "decoding": {"seed": 1},
            "press": "streamingllm",
            "budget": 256,
            "answer_canon": "C",
            "correct": False,
        },
        {
            "pid": "p1",
            "decoding": {"seed": 0},
            "press": "streamingllm",
            "budget": 256,
            "answer_canon": "B",
            "correct": True,
        },
        {
            "pid": "p1",
            "decoding": {"seed": 1},
            "press": "streamingllm",
            "budget": 256,
            "answer_canon": "B",
            "correct": True,
        },
    ]
    _write(full / "math500" / "rows.jsonl", full_rows)
    _write(comp / "math500" / "rows.jsonl", comp_rows)

    report = MODULE.build_add_report(fullcache_root=full, compression_root=comp)
    cell = report["cells"][0]
    assert cell["press"] == "streamingllm"
    assert cell["n_problems"] == 2
    # p0: TV(A vs C)=1.0, p1: TV(B vs B)=0.0 -> mean 0.5
    assert cell["mean_add"] == 0.5
    # p0 lost a correct problem (1.0 -> 0.0), p1 unchanged -> net -0.5, lost 0.5
    assert cell["signed_flip_mass"]["net"] == -0.5
    assert cell["signed_flip_mass"]["lost"] == 0.5
    assert cell["signed_flip_mass"]["gained"] == 0.0

#!/usr/bin/env python3
"""R-KV real-generation smoke (GPU): the last gate before R-KV becomes EXECUTABLE.

The synthetic-tensor parity gate (`scripts/rkv_parity.py`) proved the SCORING matches
the authors' code (Jaccard 1.0). This confirms the kvpress `DecodingPress`-wrapped path
(`build_rkv_press`) runs end-to-end on the real model:
  1. it generates non-empty, terminating output without crashing;
  2. every compression keeps the cache within the target budget;
  3. R-KV scores are finite during real decoding (no NaN/inf from the query recompute);
  4. the recent `window` tokens are always retained and exactly `budget` keys survive;
  5. the press's selection during decoding is self-consistent with the parity-verified
     `rkv_retained_indices` on the live model tensors.

Authors-mechanism fidelity stays covered by `rkv_parity.py`; this smoke covers the
INTEGRATION (the recompute + DecodingPress path), not a second authors re-derivation.

PREREGISTRATION Amendment E: R-KV stays PARITY-GATED until BOTH this smoke and the
synthetic parity pass are recorded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--n-problems", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--compression-interval", type=int, default=64)
    parser.add_argument("--i-understand", action="store_true")
    args = parser.parse_args()
    if not args.i_understand:
        raise SystemExit("REFUSED: loads the model on GPU; pass --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: GPU smoke is remote-only (set KVCACHE_REMOTE=1)")

    import kvpress
    import torch

    from kvcache.datasets import load_benchmark
    from kvcache.generation import generate_compressed
    from kvcache.models import load_model_and_tokenizer
    from kvcache.presses import build_rkv_press
    from kvcache.presses.rkv import rkv_retained_indices

    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    model_cfg = models_cfg["r1_distill_qwen_7b"]
    model, tokenizer = load_model_and_tokenizer(model_cfg, allow_download=False)
    problems = load_benchmark(
        "math500",
        benchmarks_cfg["math500"],
        subset={"count": args.n_problems, "stratify_by": "level"},
        allow_download=False,
    )

    compressions: list[dict] = []
    events: list[dict] = []

    class RecordingDecodingPress(kvpress.DecodingPress):
        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            before = int(keys.shape[2])
            base = self.base_press
            budget = int(self.target_size)
            # Verify the live selection on a few layer-0 events that actually prune.
            if int(module.layer_idx) == 0 and len(events) < 5 and before > budget:
                query = base._recompute_window_query(module, hidden_states, kwargs)
                scores = base.score(
                    module, hidden_states, keys, values, attentions, kwargs
                )
                finite = bool(torch.isfinite(scores).all())
                press_kept = scores[0].topk(budget, dim=-1).indices  # (kvh, budget)
                standalone = rkv_retained_indices(query, keys, budget=budget)
                jac = []
                for head in range(scores.shape[1]):
                    a = set(press_kept[head].tolist())
                    b = set(standalone[head].tolist())
                    jac.append(len(a & b) / len(a | b))
                k_len = int(keys.shape[2])
                window = set(range(k_len - int(base.window_size), k_len))
                win_ok = all(
                    window <= set(press_kept[head].tolist())
                    for head in range(scores.shape[1])
                )
                events.append(
                    {
                        "finite": finite,
                        "mean_jaccard": sum(jac) / len(jac),
                        "window_retained": win_ok,
                        "kept": int(press_kept.shape[-1]),
                    }
                )
            compressed_keys, compressed_values = super().compress(
                module, hidden_states, keys, values, attentions, kwargs
            )
            compressions.append(
                {"before": before, "after": int(compressed_keys.shape[2])}
            )
            return compressed_keys, compressed_values

    press = build_rkv_press(
        budget=args.budget,
        compression_interval=args.compression_interval,
        decoding_press_cls=RecordingDecodingPress,
    )

    outputs = []
    failures = []
    for problem in problems:
        gen = generate_compressed(
            model,
            tokenizer,
            problem,
            model_cfg,
            sampling_cfg,
            press=press,
            seed=11,
            max_new_tokens=args.max_new_tokens,
        )
        outputs.append(gen)
        if not gen.trace.strip():
            failures.append(f"{problem.pid}: empty trace")
        if gen.n_tokens <= 0:
            failures.append(f"{problem.pid}: zero tokens generated")

    if not compressions:
        failures.append("no compression events (raise --max-new-tokens)")
    over = [c for c in compressions if c["after"] > args.budget]
    if over:
        failures.append(f"{len(over)} compressions left cache > budget {args.budget}")
    if not events:
        failures.append("no pruning events captured to verify selection")
    if any(not e["finite"] for e in events):
        failures.append("non-finite R-KV scores during decoding")
    if any(not e["window_retained"] for e in events):
        failures.append("recent window not fully retained")
    mean_j = sum(e["mean_jaccard"] for e in events) / len(events) if events else 0.0
    if events and mean_j < 0.99:
        failures.append(f"selection Jaccard {mean_j:.4f} < 0.99 vs parity fn")

    print("=== R-KV real-generation smoke ===")
    print(
        f"problems={len(outputs)} compressions={len(compressions)} "
        f"verified_events={len(events)} budget={args.budget}"
    )
    print(f"selection mean Jaccard vs parity fn: {mean_j:.4f}")
    if outputs:
        print(f"sample trace tail: ...{outputs[0].trace[-160:]!r}")
    if failures:
        print("FAIL:")
        for failure in failures:
            print("  -", failure)
        return 1
    print("PASS — R-KV path sane, within budget, parity-consistent during decoding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

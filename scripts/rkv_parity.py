#!/usr/bin/env python3
"""R-KV port parity gate (CPU, synthetic tensors).

Compares our faithful R-KV selection against the authors' R1KV.update_kv on identical
random (query, key, value) tensors. Mechanism match: the retained token positions per
head must be identical (PREREGISTRATION Amendment E). Runs on CPU; needs the authors'
repo importable via --rkv-repo (its HuggingFace/ dir).

The reference selection here ports the authors' math directly (compute_attention_scores
pooling="max"; cal_similarity with last-similar-link zeroing + softmax). Once this
matches, the same functions back the kvpress R-KV press.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# The parity gate tests the PACKAGE's R-KV scoring (single source of truth).
from kvcache.presses.rkv import rkv_retained_indices  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rkv-repo", required=True, type=Path, help="R-KV HuggingFace/ dir"
    )
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    sys.path.insert(0, str(args.rkv_repo))
    from rkv.compression.r1_kv import R1KV

    torch.manual_seed(args.seed)
    bsz, qh, kvh, d = 1, 28, 4, 128
    query = torch.randn(bsz, qh, args.seq_len, d)
    key = torch.randn(bsz, kvh, args.seq_len, d)
    value = torch.randn(bsz, kvh, args.seq_len, d)

    theirs = R1KV(budget=args.budget, record_kept_token_indices=True)
    theirs.update_kv(key.clone(), query.clone(), value.clone())
    their_idx = theirs.kept_token_indices[-1]  # [kvh, budget]

    our_idx = rkv_retained_indices(query, key, budget=args.budget)

    jaccards = []
    for h in range(kvh):
        a = set(their_idx[h].tolist())
        b = set(our_idx[h].tolist())
        jaccards.append(len(a & b) / len(a | b))
    mean_j = sum(jaccards) / len(jaccards)
    print(f"budget={args.budget} seq_len={args.seq_len} per-head Jaccard={jaccards}")
    print(f"mean Jaccard={mean_j:.4f}  PASS={mean_j > 0.999}")
    return 0 if mean_j > 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())

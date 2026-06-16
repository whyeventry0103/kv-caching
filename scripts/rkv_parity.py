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


def compute_attention_scores(query_states, key_states, pooling="max"):
    import math

    import torch

    b, qh, q_len, d = query_states.shape
    kvh = key_states.shape[1]
    g = qh // kvh
    if g == 1:
        return torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(d)
    q = query_states.view(b, kvh, g, q_len, d)
    k = key_states.unsqueeze(2)
    attn = torch.matmul(q, k.transpose(3, 4)) / math.sqrt(d)
    return attn.mean(dim=2) if pooling == "mean" else attn.max(dim=2).values


def cal_similarity(
    key_states, threshold=0.5, retain_ratio=0.1, retain_direction="last"
):
    import torch

    k = key_states[0]
    num_heads = k.shape[0]
    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    for h in range(num_heads):
        sim[h].fill_diagonal_(0.0)
    mask = sim > threshold
    arange = torch.arange(mask.size(-1), device=mask.device)
    indices = torch.where(mask, arange, torch.zeros_like(mask, dtype=torch.long))
    if retain_direction == "last":
        retain = torch.max(indices, dim=-1)[0]
    else:
        retain = torch.min(indices, dim=-1)[0]
    bidx = torch.arange(num_heads).unsqueeze(1).repeat(1, retain.size(1))
    sidx = torch.arange(retain.size(1)).unsqueeze(0).repeat(num_heads, 1)
    sim[bidx, sidx, retain] = 0
    return sim.mean(dim=1).softmax(dim=-1)


def rkv_retained_indices(query, key, *, budget, window=8, kernel=7, mix_lambda=0.07):
    import torch
    import torch.nn.functional as F

    attn = compute_attention_scores(query, key, pooling="max")
    imp = (
        F.softmax(attn[:, :, -window:, :-window].float(), dim=-1)
        .mean(dim=-2)
        .to(query.dtype)
    )
    imp = F.max_pool1d(imp, kernel_size=kernel, padding=kernel // 2, stride=1)
    red = cal_similarity(key)[:, :-window].unsqueeze(0)
    final = imp * mix_lambda - red * (1 - mix_lambda)
    keep = final.topk(budget - window, dim=-1).indices[0]  # [kvh, budget-window]
    k_len = key.shape[2]
    window_idx = torch.arange(k_len - window, k_len).expand(keep.shape[0], -1)
    return torch.cat([keep, window_idx], dim=-1)  # [kvh, budget]


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

"""R-KV port (Redundancy-aware KV cache compression, arXiv:2505.24133).

The scoring functions below are a **faithful, parity-verified port** of the authors'
HuggingFace implementation: `scripts/rkv_parity.py` shows `rkv_retained_indices`
produces token-for-token identical retained sets to their `R1KV.update_kv` on synthetic
tensors at budgets {128,256,384,512} (mean Jaccard 1.0). They are the single source of
truth for R-KV scoring (the parity harness imports them).

Algorithm: `final_score = importance * mix_lambda - redundancy * (1 - mix_lambda)`,
mix_lambda=0.07, keep top (budget - window_size) non-window tokens + the recent window
(window_size=8). importance = max-pooled (kernel=7) mean attention from the last
`window_size` queries over older keys (`compute_attention_scores`, GQA max-pooled, then
softmax over non-window keys). redundancy = `cal_similarity` (cosine, last-similar-link
zeroing, softmaxed).

PARITY-GATED (PREREGISTRATION Amendment E): `get_press("rkv")` refuses until verified.
The *scoring* is now parity-verified; the remaining gate before EXECUTABLE is a
real-generation smoke confirming the kvpress press's query-recompute path
(`build_rkv_press`) reproduces these retained sets during decoding on the model (GPU).
"""

from __future__ import annotations

from typing import Any

RKV_WINDOW_SIZE = 8
RKV_KERNEL_SIZE = 7
RKV_MIX_LAMBDA = 0.07
RKV_SIM_THRESHOLD = 0.5
RKV_RETAIN_RATIO = 0.1
RKV_RETAIN_DIRECTION = "last"


def rope_inv_freq(
    config: Any,
    head_dim: int,
    device: Any,
    rope_init_functions: Any = None,
) -> tuple[Any, float]:
    """Return ``(inv_freq, attention_scaling)`` for the window-query recompute.

    R-KV needs RoPE-applied queries for the last `window` positions, but kvpress
    supplies ``position_embeddings`` for the CURRENT position only, so the window's
    cos/sin must be rebuilt here. This delegates the frequency schedule to
    transformers' own ``ROPE_INIT_FUNCTIONS`` rather than re-deriving it, because a
    hand-rolled schedule is only correct for unscaled RoPE: the first attempt at the
    R-KV published-configuration cell (R1-Distill-Llama-8B, ``rope_type='llama3'``)
    crashed on every launch against a guard that refused anything scaled, which was
    the guard behaving correctly.

    For an unscaled config this reproduces ``1/theta**(2i/d)`` exactly, so models
    already measured under the previous hand-rolled path are bitwise unaffected --
    that equivalence is locked by a test. Scaling families the installed
    transformers does not implement still refuse rather than silently mis-rotating.
    """
    import torch

    scaling = getattr(config, "rope_scaling", None)
    if not scaling:
        inv_freq = 1.0 / (
            float(config.rope_theta)
            ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                / head_dim
            )
        )
        return inv_freq, 1.0

    rope_type = scaling.get("rope_type") or scaling.get("type")
    if rope_init_functions is None:
        from transformers.modeling_rope_utils import (  # noqa: I001
            ROPE_INIT_FUNCTIONS as rope_init_functions,
        )
    if rope_type not in rope_init_functions:
        raise NotImplementedError(
            "rkv window-query recompute has no frequency schedule for "
            f"rope_type={rope_type!r}; got rope_scaling={scaling!r}"
        )
    inv_freq, attention_scaling = rope_init_functions[rope_type](config, device)
    if inv_freq.shape[-1] * 2 != head_dim:
        raise RuntimeError(
            f"rkv rope inv_freq length {inv_freq.shape[-1]} does not match "
            f"head_dim {head_dim}; the config's rope dim disagrees with the query"
        )
    return inv_freq, float(attention_scaling)


def compute_attention_scores(
    query_states: Any, key_states: Any, pooling: str = "max"
) -> Any:
    """Q@K/sqrt(d) with GQA group pooling. Faithful to R-KV compute_attention_scores."""
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
    key_states: Any,
    threshold: float = RKV_SIM_THRESHOLD,
    retain_ratio: float = RKV_RETAIN_RATIO,
    retain_direction: str = RKV_RETAIN_DIRECTION,
) -> Any:
    """Per-key cosine-redundancy. Faithful to R-KV `cal_similarity` (bsz=1)."""
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


def rkv_final_score(
    query: Any,
    key: Any,
    *,
    window: int = RKV_WINDOW_SIZE,
    kernel: int = RKV_KERNEL_SIZE,
    mix_lambda: float = RKV_MIX_LAMBDA,
) -> Any:
    """R-KV final score over the first (k_len - window) keys: (bsz, kvh, k_len-w)."""
    import torch.nn.functional as F

    attn = compute_attention_scores(query, key, pooling="max")
    imp = (
        F.softmax(attn[:, :, -window:, :-window].float(), dim=-1)
        .mean(dim=-2)
        .to(query.dtype)
    )
    imp = F.max_pool1d(imp, kernel_size=kernel, padding=kernel // 2, stride=1)
    red = cal_similarity(key)[:, :-window].unsqueeze(0)
    return imp * mix_lambda - red * (1 - mix_lambda)


def rkv_retained_indices(
    query: Any,
    key: Any,
    *,
    budget: int,
    window: int = RKV_WINDOW_SIZE,
    kernel: int = RKV_KERNEL_SIZE,
    mix_lambda: float = RKV_MIX_LAMBDA,
) -> Any:
    """Retained token positions per kv-head: top (budget-window) + recent window."""
    import torch

    final = rkv_final_score(
        query, key, window=window, kernel=kernel, mix_lambda=mix_lambda
    )
    keep = final.topk(budget - window, dim=-1).indices[0]
    k_len = key.shape[2]
    # arange defaults to CPU; keep is on the model's device during real decoding, so the
    # window indices must be built on keep.device or torch.cat raises a device mismatch
    # (caught by the GPU smoke; invisible to the CPU-only parity harness).
    window_idx = torch.arange(k_len - window, k_len, device=keep.device).expand(
        keep.shape[0], -1
    )
    return torch.cat([keep, window_idx], dim=-1)


def build_rkv_scorer(kvpress_module: Any) -> Any:
    """Construct the R-KV ScorerPress (kvpress SnapKVPress subclass).

    Uses the parity-verified `rkv_final_score`. The press recomputes the last-window
    queries via kvpress's verified path; a real-generation smoke confirms the retained
    sets match before R-KV is promoted to EXECUTABLE.
    """
    from dataclasses import dataclass

    SnapKVPress = kvpress_module.SnapKVPress

    # MUST be a dataclass: SnapKVPress is a dataclass, so a plain subclass's annotated
    # class attributes do NOT become field defaults -- the generated __init__ would
    # silently set the instance back to SnapKV's window_size=64 / kernel_size=5. That
    # exact shadowing bug shipped once and was caught by the real-generation smoke
    # (selection Jaccard 0.805 vs the parity functions; press-only indices clustered
    # in the last-64 region). Regression-locked in tests/test_presses.py.
    @dataclass
    class RKVScorerPress(SnapKVPress):
        compression_ratio: float = 0.0
        window_size: int = RKV_WINDOW_SIZE
        kernel_size: int = RKV_KERNEL_SIZE
        mix_lambda: float = RKV_MIX_LAMBDA

        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            # Recompute the last-window queries (RoPE-applied), then the parity-verified
            # R-KV final score; pad the window with the max so it is always kept.
            import torch.nn.functional as F

            query = self._recompute_window_query(module, hidden_states, kwargs)
            final = rkv_final_score(
                query,
                keys,
                window=self.window_size,
                kernel=self.kernel_size,
                mix_lambda=self.mix_lambda,
            )
            return F.pad(final, (0, self.window_size), value=final.max().item())

        @staticmethod
        def _recompute_window_query(module, hidden_states, kwargs):
            # R-KV importance uses the last `window` (8) queries. At decode time
            # kvpress supplies position_embeddings for the CURRENT position only, so
            # the window's cos/sin are rebuilt here for positions p-n+1..p and
            # self-checked against the framework's rotation at p -- a silent
            # mis-rotation can never ship a wrong retained set. The frequency
            # schedule comes from transformers' own rope-init (see rope_inv_freq),
            # so scaled families like llama3 are the framework's arithmetic, not
            # ours, and the self-check below certifies it on every event.
            import torch
            from kvpress.presses.snapkv_press import (
                get_prerope_query_states,
                rotate_half,
            )

            n = min(RKV_WINDOW_SIZE, hidden_states.shape[1])
            q = get_prerope_query_states(module, hidden_states[:, -n:])
            config = module.config
            head_dim = q.shape[-1]
            pos_end = int(kwargs["cache_position"][-1].item())
            positions = torch.arange(
                pos_end - n + 1, pos_end + 1, device=q.device, dtype=torch.float32
            )
            inv_freq, attention_scaling = rope_inv_freq(config, head_dim, q.device)
            freqs = positions[:, None] * inv_freq.to(positions.dtype)[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = (emb.cos() * attention_scaling)[None].to(q.dtype)
            sin = (emb.sin() * attention_scaling)[None].to(q.dtype)
            cos_cur, _ = kwargs["position_embeddings"]
            if not torch.allclose(
                cos[:, -1].float(), cos_cur[:, -1].float(), atol=1e-2
            ):
                raise RuntimeError(
                    "rkv analytic RoPE disagrees with model position_embeddings"
                )
            return (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

    return RKVScorerPress()

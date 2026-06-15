"""R-KV port (Redundancy-aware KV cache compression, arXiv:2505.24133).

R-KV ranks tokens by importance AND non-redundancy:

    final_score = importance * mix_lambda - redundancy * (1 - mix_lambda)

with mix_lambda=0.07, then keeps the top (budget - window_size) non-window tokens plus
an always-kept recent window (window_size=8). Source params (HuggingFace impl):
window_size=8, kernel_size=7, mix_lambda=0.07, retain_ratio=0.1,
retain_direction="last".

- **importance** = mean attention from the last `window_size` queries over the older
  keys, max-pooled (kernel_size) -- this is SnapKV's observation-window importance with
  max (not avg) pooling, so we subclass kvpress's verified `SnapKVPress`.
- **redundancy** = a cosine-similarity score between key vectors (R-KV
  `cal_similarity`).

PARITY-GATED (PREREGISTRATION Amendment E): this port is NOT eligible for analysis
until `scripts/rkv_parity.py` shows its retained-token sets match the authors' code
(Jaccard ~= 1) at two budgets on Qwen-7B. The redundancy term in particular
(`_redundancy_score`) reproduces R-KV's intent but its exact `cal_similarity`
resolution (threshold / retain_direction) must be confirmed against the source by the
parity check. `get_press("rkv")` therefore refuses until verification is recorded.
"""

from __future__ import annotations

from typing import Any

RKV_WINDOW_SIZE = 8
RKV_KERNEL_SIZE = 7
RKV_MIX_LAMBDA = 0.07


def build_rkv_scorer(kvpress_module: Any) -> Any:
    """Construct the R-KV ScorerPress (subclass of kvpress SnapKVPress)."""
    import torch
    import torch.nn.functional as F

    SnapKVPress = kvpress_module.SnapKVPress

    class RKVScorerPress(SnapKVPress):
        compression_ratio: float = 0.0
        window_size: int = RKV_WINDOW_SIZE
        kernel_size: int = RKV_KERNEL_SIZE
        mix_lambda: float = RKV_MIX_LAMBDA

        @staticmethod
        def _redundancy_score(keys: torch.Tensor) -> torch.Tensor:
            """Per-key cosine-similarity redundancy, shape (bsz, kv_heads, k_len).

            Higher == more redundant (more similar to another key) == penalized. This
            reproduces R-KV's redundancy intent; the exact resolution scheme
            (threshold, retain_direction) is confirmed by the parity gate.
            """
            k_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
            sim = torch.matmul(k_norm, k_norm.transpose(-1, -2))
            eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool)
            sim = sim.masked_fill(eye, 0.0)
            # redundancy of token i = its strongest similarity to any other key.
            return sim.max(dim=-1).values

        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            bsz, num_key_value_heads, k_len, _ = keys.shape
            groups = module.config.num_attention_heads // num_key_value_heads

            if attentions is not None:
                attn_weights = attentions[..., -self.window_size :, : -self.window_size]
            else:
                attn_weights = self.compute_window_attention(
                    module,
                    hidden_states,
                    keys,
                    self.window_size,
                    kwargs["position_embeddings"],
                )

            importance = attn_weights.mean(dim=-2)
            importance = F.max_pool1d(
                importance,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
            importance = importance.view(
                bsz, num_key_value_heads, groups, k_len - self.window_size
            ).mean(2)

            redundancy = self._redundancy_score(keys)[..., : -self.window_size]

            final = importance * self.mix_lambda - redundancy * (1 - self.mix_lambda)
            # Always keep the recent window: pad it with the max score.
            return F.pad(final, (0, self.window_size), value=final.max().item())

    return RKVScorerPress()

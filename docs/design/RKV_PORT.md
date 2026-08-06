# R-KV port + parity gate

R-KV (arXiv:2505.24133, Zefan-Cai/R-KV, NeurIPS 2025) is the reasoning-specific,
>=100% "denoising" method central to the paper. We **port** it into our kvpress
framework rather than running the authors' code in the grid, so the integrity bar is
**port fidelity** (our reimplementation computes what their code computes on the model
we run), enforced by a hard parity gate before any R-KV row enters analysis
(PREREGISTRATION Amendment E).

## Algorithm (from `HuggingFace/rkv` in the official repo)

Per prune, for a cache of length `L >= budget`:

```
importance  = maxpool_k( mean_over_last_W_queries( softmax(Q_W @ K^T) ) )   # SnapKV-style
redundancy  = cal_similarity(keys)                                          # cosine, R-KV
final_score = importance * mix_lambda - redundancy * (1 - mix_lambda)
keep        = top(budget - W) of final_score   over the first L-W keys
              + the last W keys (recent window, always kept)
```

Source params (HuggingFace impl): `window_size W = 8`, `kernel_size = 7`,
`mix_lambda = 0.07`, `retain_ratio = 0.1`, `retain_direction = "last"`. Note the heavy
weight on **non-redundancy** (0.93) vs importance (0.07).

## Our port

`src/kvcache/presses/rkv.py::build_rkv_scorer` subclasses kvpress `SnapKVPress`
(which already implements R-KV's importance term — observation-window attention with
verified query-recompute + RoPE), overriding `score` to (a) use **max** pooling
(R-KV) instead of avg (SnapKV), `W=8`, `kernel=7`, and (b) subtract the
`mix_lambda`-weighted cosine **redundancy**. The recent window is force-kept by padding
its score with the max. Applied during decoding via `DecodingPress(target_size=budget)`
(`build_rkv_press`). It does **not** require eager attention (it recomputes Q@K itself).

**Open item the parity gate resolves:** R-KV's `cal_similarity` returns a per-key
redundancy via a threshold (0.5) + `retain_direction="last"` resolution scheme; our
`_redundancy_score` uses each key's max cosine similarity to any other key, which
reproduces the intent. The exact form must be confirmed against the source by the
parity check; if the retained sets do not match, `_redundancy_score` is adjusted to
the source resolution until they do.

## Parity gate (run when a GPU is free; blocks all R-KV analysis)

1. **Reference environment:** create a separate venv with the authors' HuggingFace
   requirements; pin and record the R-KV repo commit + package versions as artifacts.
2. **Reference run:** run their `r1_kv.py` (`record_kept_token_indices=True`) on
   `DeepSeek-R1-Distill-Qwen-7B` over a handful of fixed (problem, seed) traces at
   **two budgets** (one moderate, one aggressive), capturing the retained token
   positions after each prune.
3. **Port run:** run our `build_rkv_press` over the same traces/budgets, capturing the
   retained positions (the replay press already records eviction decisions).
4. **Compare:** per (trace, prune, budget), the retained-position sets must match
   (Jaccard ~= 1, with a documented tiny tolerance only if their scoring is
   nondeterministic). Mechanism match, not just final accuracy. Commit the comparison
   + the authors' raw output as artifacts.
5. On pass: move `rkv` from PORT_REQUIRED -> EXECUTABLE and record the gate result.
   On fail: fix `_redundancy_score` (and any GQA/RoPE/index detail) and repeat.

Honest scoping for the paper: R-KV's published >=100% headline is on Llama-8B /
Qwen-14B; we run Qwen-7B and do not reproduce their exact headline. Our R-KV is a
faithful port at our scale; a *no >100% regime at 7B* result (if the port is parity-
certified) is a finding about scale, not a port failure.

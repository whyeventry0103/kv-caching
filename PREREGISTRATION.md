# Preregistration

## Scope and claims discipline

We study the apparent contradiction between severe reasoning degradation under
KV compression and reports of near-lossless compression. The decomposition is:

`total accuracy change = on-reference readout change + residual`.

The residual is reported only as **accuracy change unexplained by on-reference
readout**. It is not a directly measured causal effect of trajectory distortion.

## Hypotheses

- **H1, residual-honest:** on-reference replay explains part, but not necessarily
  all, of compressed free-run accuracy change. The remainder is labeled only as
  the unexplained residual.
- **H2, regime separation:** compression behavior differs by budget, task,
  trace length, and reasoning regime; every absolute budget is accompanied by
  its realized compression ratio.
- **H3, silent flips:** answer-distribution changes can occur without a large
  pass@1 change. Sampled-inference ADD is primary; greedy decoding is analyzed
  only by answer flip rate.

Benjamini-Hochberg correction across problems is mandatory for families of
problem-level tests.

## Models and decoding

Primary: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`.
Generalization: `Qwen/Qwen3-4B`, described as **cross-training-paradigm within
the Qwen family**, never cross-architecture.

Sampling uses temperature 0.6 and top-p 0.95. Seeds are frozen in configuration.

## ADD

Answer Distribution Distortion is total variation between coarsened,
canonicalized full-cache and compressed answer distributions. Excess ADD uses
a per-problem exact or Monte Carlo permutation-test null formed by relabeling
the pooled samples. Split-half subtraction is prohibited. Report entropy shift
with Miller-Madow correction, pass@k with the unbiased Codex estimator, paired
bootstrap intervals, and BH-adjusted problem-level p-values.

MATH-500 uses N=8 as the primary ADD cell. AIME-24/25 uses pass@1 n=16
full-cache and n=8 compressed, ADD for three methods at budgets 256 and 2048
with n=32/16, and a two-cell n=64 asymptote.

## TDP and divergence profile

Per-step divergence is total variation, not KL. Threshold-free AUC and fraction
above tau are primary. TDP localizes onset. Sweep tau over
`{0.05, 0.10, 0.15, 0.20}`. Report onset-aligned, content-token
(numerals/operators/answer-span), and segment-aggregated views.

## Causal ladder

The instrument suite includes evicted-anchor attribution and damage latency;
restoration by reinserting evicted KV at a spike and recomputing d_t; branch
recovery with full-cache-fork and random-fork controls; reverse replay of a
compressed free-run under full cache; and TV-matched noise injected into a
full-cache free-run.

These are interventions and controls. Their interpretation must not upgrade the
decomposition residual into a measured cause.

## Validity and benchmarks

The negative control sends two full-cache banks through the complete suite and
expects excess ADD near zero and TDP percentile near one.

Benchmarks are MATH-500, AIME 2024/2025, GSM8K, FOLIO, StrategyQA, and synthetic
RSP. RSP uses interleaved references, `max_new_tokens=2048`, and treats
truncation as failure. Budgets are 128, 256, 512, 1024, and 2048, plus a
random-eviction floor.

All deviations, exclusions, failed cells, truncations, package versions, and
configuration hashes are retained in machine-readable outputs.

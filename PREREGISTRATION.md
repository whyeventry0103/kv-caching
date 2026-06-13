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

Truncated generations remain in every denominator and are scored incorrect.
They are never discarded as malformed rows. Each cell reports truncation rate.

## Amendment A (2026-06-13): grading authority, claims discipline, off-policy bounds

This dated addendum is append-only; it does not retract anything above.

### Grading authority

Correctness is decided by a vetted pairwise grader, `kvcache.grading.is_equiv`
(canonical Hendrycks MATH normalization; no `eval`/`sympify` on model text), applied
to the **raw** gold and **raw** predicted answer. The `sympify`-based
`canonicalize` is demoted to an ADD bucketing key and a reported canonicalization
sensitivity; it is never the correctness oracle. The exact grader used by HOTT /
the kvpress eval / MATH-500 must be verified and reconciled before the grid; any
deviation is recorded. Each row records `grader_version`.

### ADD is answer-level and a lower bound

ADD is the total variation between answer distributions: a **lower bound** on
behavioral change that only sees changes reaching the answer. It is never
described as "behavioral change." ADD is decomposed (`add_decomposition`) into a
termination-mass shift (probability moving into the `OTHER`/truncation bucket — a
length effect) and the within-answered TV (the reasoning channel), so a truncation
increase is never read as a reasoning change. The identity
`add_tv >= 0.5 * |termination_shift|` is reported.

### Direction discipline (the reference is not assumed better)

Every divergence quantity (ADD, per-step TV, AUC-TV, TDP) is reported alongside its
**signed** outcome consequence — gained-vs-lost correct mass (`flip_mass`,
`aggregate_flip_mass`). Divergence is "behavioral fidelity loss," never "damage";
correctness is the quality arbiter. This guards the >100% ("denoising") regime,
where deviation from the full-cache reference is improvement.

### Off-policy validity of readout/TDP

Teacher-forced readout and TDP are off-policy: they are measured on the full-cache
reference trajectory (good states), not the states the compressed model visits.
The residual `total - on-reference readout` is therefore biased toward the
trajectory share (Ross & Bagnell, DAgger covariate-shift lineage). This is stated
up front, not in rebuttal. The bias is bounded by **reverse replay** (compressed
free-run replayed under full cache; reported symmetrically to forward replay so the
two bracket the on-policy residual — `Decomposition.residual_bracket`) and closed
on-policy by the **TV-matched noise control**, which is mandatory, not optional. TV
is retained over KL/JSD because `1 - acceptance ≈ TV` in speculative decoding
(Leviathan et al.), giving TV a behavioral reading; JSD is an appendix sensitivity.

### Multiple-comparisons pre-registration

The pass@k crossover (the >100% claim) is a confirmatory test pre-registered on
named cells only: R-KV-style methods at high budgets ({1024, 2048}) on math
(MATH-500, AIME), using a **paired bootstrap of the pass@1−pass@k difference**
(the curves are correlated; naive CI overlap is invalid). All other crossovers are
exploratory and BH-corrected. Confirmatory family (decomposition dominance,
moderator, this crossover) is Holm–Bonferroni corrected.

### Negative control as a figure; exchangeability

The two-independent-full-cache-banks negative control is a **figure**, not a
sentence: it must show excess ADD ≈ 0, TDP percentile ≈ 1, and an exact permutation
test rejecting at ≈ α. Exchangeability requires matched batching and kernel settings
across conditions; batch-composition nondeterminism is a real systematic at this
precision and is controlled. Any instrument lighting up on the negative control is a
bug, caught before a reviewer catches it.

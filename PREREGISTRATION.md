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

## Amendment B (2026-06-14): accuracy-scale readout estimand

Per-step TV, mean TV, summed TV, fraction-above-threshold, and TDP remain
distribution-fidelity instruments. They are not accuracy changes and are never
subtracted from an accuracy delta.

The accuracy-scale on-reference readout estimand is measured by a
teacher-force-then-generate intervention. For each full-cache reference trace, define
the reasoning/answer boundary as the end of `</think>` when present, otherwise the
start of the first recognized answer marker. Teacher-force the exact stored reference
token IDs through that boundary into the compressed cache. From the boundary, freely
sample the answer continuation under the same compression method, absolute budget,
temperature 0.6, top-p 0.95, and matched seed. Score the generated answer with the
frozen `is_equiv` grader.

For method-budget cell `m`:

```text
readout_change(m) = readout_accuracy(m) - full_cache_accuracy
total_change(m) = compressed_free_run_accuracy(m) - full_cache_accuracy
unexplained_residual(m) = total_change(m) - readout_change(m)
```

The residual is still named only **accuracy change unexplained by on-reference
readout**. Because the intervention conditions on the good full-cache reasoning
trajectory, it plausibly underestimates readout failure on compressed-model states and
therefore inflates the residual share. Reverse replay and the TV-matched noise control
remain mandatory later controls for this off-policy bias.

Exact prompt and generated token IDs are stored in an additive validated sidecar;
the frozen result-row schema is unchanged. Replay's self-control is run before any
compressed replay. The independent-two-full-cache-bank ADD permutation control remains
with the ADD engine in M5.

## Amendment C (2026-06-14): rendered-prompt tokenization correction

The model chat template already renders its beginning-of-sequence token. Rendered
prompts are therefore tokenized with `add_special_tokens=False`; adding tokenizer
special tokens a second time duplicates BOS and changes the generation protocol.
Exact token-ID sidecar validation exposed this error before M4 replay began.

## Amendment D (2026-06-15): pilot-result discipline and review responses

This addendum records commitments made in response to an external review, before the
powered run, to prevent underpowered pilot directionals from calcifying into facts.

### The M4 pilot decomposition is explicitly UNCONFIRMED

The pilot "double dissociation" (StreamingLLM-256 readout-dominated, Random-256
residual-dominated) is from n=5 problems x 4 seeds. Problem-level bootstrap 95% CIs are
wide (e.g. SLM-256 readout 0.45, CI [0.15, 0.80]; Random-256 readout 0.90, CI
[0.80, 1.00]); the per-method readout-vs-residual split is barely resolved. It is a
**hypothesis-generating observation, not a result**. The powered MATH-500 bank
(250 problems x n=8) is the confirmatory test, and it CAN overturn the dissociation.
**We pre-commit to reporting the powered result whatever it shows**, including a
reversal. No language asserting the dissociation as established may enter any external
artifact until MATH-500 confirms it. Reports must carry the bootstrap CIs, not point
estimates alone.

### Random > StreamingLLM: mundane causes ruled out / controlled

(i) Budget parity is verified: at `target_size=256` both presses prune to
`max_after_prune=256` and `max_before_prune=320` identically — same retained budget,
so the gap is which 256 tokens, not how many. (ii) Stochastic presses now use a
per-(problem, seed) RNG (`press_seed`) so the effect cannot be a single lucky seed.
The mechanistic sentence ("random preserves mid-trace anchors by chance") remains
provisional until the powered bank.

### Readout estimand is math/span-shaped; per-step TV is the general fallback

The accuracy-scale readout (Amendment B) assumes a clean reasoning/answer boundary
(`</think>` or an answer marker) and localizes readout to the answer span. This holds
for math. For logic benchmarks (FOLIO, StrategyQA) where the answer is not a span after
a marker, the **per-step TV profile is the general-purpose readout instrument**, and the
paper states explicitly which instrument carries the decomposition on which benchmark.

### One canonical full-cache reference, enforced in code

`pilot-fullcache-tokenized-v2` (single-BOS, 20/20, with token-ID sidecars) is the sole
canonical pilot FC reference. The double-BOS `pilot-fullcache-v2-tf456` (19/20) is
retired to provenance. `report_m4` asserts every compression and readout row references
this exact bank's `(pid, seed)` set; `total_change` and `readout_change` share one
reference by construction.

### R-KV port is a HARD reproduction gate

R-KV's >=100% regime is central to the paper. Because we port R-KV rather than run the
authors' code, a subtly wrong port that fails to reproduce the regime would masquerade
as a finding. Before ANY R-KV row enters analysis, the port must reproduce a specific
published (model, benchmark, budget) number from the R-KV paper within sampling noise,
committed as a test. If it cannot, we do not have R-KV and no R-KV result is reported.
The same source-sanity gate applies to H2O (ObservedAttention).

### RSP demoted; restoration on natural traces is the primary mechanism instrument

RSP has failed validity twice. The restoration test (re-insert evicted KV at a spike,
recompute d_t) on natural traces is the PRIMARY mechanism instrument; RSP is a bonus.
RSP redesign must not consume restoration-engine effort.

### Bookkeeping

`compression_events` counts layer-level prune calls (x n_layers), not token-prune
steps; it is never divided as a per-step rate. The powered two-full-cache-bank ADD
negative control MUST be run at M8 (split the n=8 bank); the degenerate pilot smoke
does not discharge it.

All M2/M3 pilot rows generated before this correction used duplicate BOS and are
retained only for provenance and harness diagnostics. They are not final paper
evidence. The corrected tokenized full-cache bank is the first canonical M4
reference. Full-cache and compressed free-run rows used in any accuracy change,
signed flip-mass, ADD, or readout/residual decomposition must be regenerated under
the same corrected prompt-tokenization protocol and software stack. In particular,
the earlier M3 compression pilot cannot provide `total_change` for a decomposition
against the corrected M4 reference.

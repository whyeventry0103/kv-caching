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

## Amendment E (2026-06-15): second-review responses (baselines + scaling)

### R-KV port integrity gate (tightened)

R-KV is ported into our kvpress framework, not run from the authors' code, so the
gate certifies **port fidelity** ("does our reimplementation compute what R-KV's code
computes on the model we run"), which is what our paper's integrity depends on — not
headline reproduction (the authors' claim to defend). Before ANY R-KV row enters
analysis:

1. **Mechanism match, not just score.** For a fixed (problem, seed, budget), the set of
   retained token positions after each prune must match the authors' code
   (Jaccard ≈ 1, with a small documented tolerance only if their scoring is
   nondeterministic), on a handful of traces. We already store eviction decisions for
   attribution, so this is cheap and is the test that catches a subtly-wrong port.
2. **Two budgets, not one** — one moderate and one aggressive (the regime where the
   denoising/>100% behavior would appear if it appears at 7B). Match the curve shape.
3. **Committed artifacts:** the authors' `run_math.py` raw output, the exact R-KV repo
   commit, the separate-environment package versions, and seeds — so the comparison is
   reproducible, not asserted.
4. **Honest scoping in the paper:** R-KV's published >=100% headline is on
   R1-Distill-Llama-8B and Qwen-14B; we run Qwen-7B and do **not** reproduce their exact
   headline configuration. Our R-KV is validated as a faithful port at our scale. If the
   faithful port shows **no >100% regime on Qwen-7B, that is a finding about scale, not a
   port failure** — a claim credible *only because* the retained-set mechanism match
   certifies the port. (c) restoring Llama-8B to reproduce their headline is rejected: it
   certifies the wrong thing for us and costs a second GPU.

### H2O is heavy-hitter scoring, labeled precisely

A source read of kvpress `ObservedAttentionPress.score` confirms it is the H2O
heavy-hitter signal (accumulated attention each key receives from all queries,
normalized by causal eligibility) — but it has **no explicit recent-window
reservation** (a component of the H2O paper). It is therefore labeled
"H2O-style attention heavy-hitter eviction (kvpress ObservedAttention, as used by
HOTT)", not bit-identical to the H2O paper. Because it is library code (same library
HOTT used), a behavioral sanity suffices; no published-number gate. It requires eager
attention.

### Eager-attention exchangeability (gates the powered run)

Attention-score presses (H2O) require `attn_implementation="eager"`, whose numerics
differ from the sdpa default. To avoid a stack-mismatch confound, before the powered
run we verify a common numerical baseline: full-cache accuracy under eager vs sdpa on
the pilot problems must agree within sampling noise (no systematic shift). If it does
not, the entire powered suite (all methods + FC reference) runs under eager. This is
the one item that gates the powered run; everything else may run in parallel.

### Readout boundary rule (powered set)

The reasoning/answer boundary is deterministic and logged per row: the LAST `</think>`
(robust to multiple closes and `\boxed{}` emitted mid-reasoning); else an answer-marker
fallback; else **undefined**. Undefined-boundary rows are excluded from readout-accuracy
and counted (`n_readout_undefined`) — never silently dropped or mis-segmented.

### Budget-parity language

StreamingLLM and Random retain the same TOTAL budget (verified: both prune to 256/320
at target_size=256) but a DIFFERENT positional allocation (StreamingLLM = sinks +
recent window; Random = uniform sample). The paper says "same total budget, different
positional allocation," never "same mid-trace coverage" — the differing coverage is
arguably the mechanism, not an artifact to eliminate.

### Powered-run discipline

The powered MATH-500 (250 x n=8) decomposition is where the M4 dissociation survives or
dies; we report it with CIs either way. A reversal or wash-out at n=250 is a publishable
result about the decomposition's resolution limits, not a failure.

## Amendment F (2026-06-19): pre-committed read of the powered readout decomposition

Written *before* the powered readout numbers exist (the run launches tonight). The
powered free-run phenomenon is already settled: at 100 problems x n=4, Random-256 beats
StreamingLLM-256 by +0.185, CI [0.130, 0.242], on 94/100 problems. That is the
PHENOMENON. This amendment pre-registers how we read the decomposition that is supposed
to EXPLAIN it, so the read is not chosen after seeing the data.

### Quantities (budget 256, per method m), all with problem-level bootstrap 95% CIs

Estimator = `scripts/report_m4.py`: one pinned canonical full-cache reference;
problem-clustered bootstrap; undefined reasoning/answer boundary rows excluded from
readout-accuracy and counted (`n_readout_undefined`). Definitions:

- `total_drop(m)   = FC_acc - free_acc(m)`            (free-run loss)
- `readout_drop(m) = FC_acc - readout_acc(m)`         (loss with the FC trajectory held
  fixed, compressed cache active during answer read-out)
- `residual(m)     = readout_acc(m) - free_acc(m) = total_drop - readout_drop`
- `readout_share(m) = readout_drop(m) / total_drop(m)`  (cell-level ratio of means, not
  per-problem ratios, to avoid small-denominator blow-up)

Anchors already known: FC_acc = 0.885; free_acc(SLM) = 0.237 (total_drop 0.648);
free_acc(Random) = 0.422 (total_drop 0.463). Pilot (n=5, unconfirmed) suggested
readout_acc(SLM) ~ 0.45 and readout_acc(Random) ~ 0.90.

### The pre-committed read

**CONFIRMS the thesis (the strong paper)** iff ALL hold:
1. `readout_acc(Random)` 95% CI lower bound >= 0.75 (Random readout is ~lossless), AND
   `residual(Random)` CI excludes 0 (its free-run collapse is trajectory-attributed);
2. `readout_drop(SLM)` CI excludes 0 with `readout_share(SLM)` point estimate >= 0.5
   (StreamingLLM's loss is at least half readout-mediated);
3. `readout_share(SLM)` and `readout_share(Random)` 95% CIs do NOT overlap, in the
   predicted direction SLM > Random.
The headline this licenses: *the same method (Random), at the same budget, is near-
lossless on teacher-forced read-out yet collapses on free generation* — which resolves
the literature contradiction directly.

**COMPLICATES (weaker but honest)** if the split runs in the predicted direction but
either the two `readout_share` CIs overlap (methods not statistically separable) or both
methods are residual/trajectory-dominated. Reported as: "compression damage to reasoning
is predominantly trajectory-mediated, with a method ordering the decomposition cannot
cleanly separate." No clean single-method headline.

**REFUTES** if both methods are readout-dominated (large `readout_drop` for both, small
similar residuals) or the two `readout_share` CIs are indistinguishable AND not in the
predicted direction. Then the +0.185 free-run gap is NOT explained by the readout/
trajectory split; the thesis is dead and we report the phenomenon plus the null
decomposition. Pre-committed to this outcome per Amendment D.

### Reverse-replay bracket result (2026-06-27)

Reverse readout complete (800/800 on grid-diss256-tok traces). Reverse readout acc =
compressed trajectory read out under FULL cache: Random 0.636, StreamingLLM 0.417 (FC
0.882; free 0.422 / 0.237). Clean readout of the compressed trajectory does NOT recover to
FC for either method -> a large component of the loss is genuinely trajectory, not readout
(reverse trajectory damage: Random -0.246, SLM -0.465). Residual bracket (forward..reverse
endpoints): Random [-0.343, -0.214], SLM [-0.347, -0.180]. **VERDICT: the central claim
(large, readout-invisible trajectory residual) SURVIVES the off-policy correction for both
methods -- the residual never approaches 0.** HOWEVER forward and reverse DISAGREE on the
method-level split (forward: Random more trajectory; reverse: SLM more trajectory), and the
brackets overlap, so the method-level readout-vs-trajectory attribution is NOT identifiable
within the off-policy bracket. Report the central residual strongly; report the method
dissociation as bracketed / not separable. Mild confound: differential defined-reverse n
(Random 308 vs SLM 372 -- Random's rambling traces more often lack a clean boundary).

### Causal-claim discipline (off-policy bias)

`residual` is DEFINED as "accuracy change unexplained by on-reference read-out"; it is
not yet "trajectory damage" as a cause. A CONFIRM read licenses only
"consistent with trajectory-dominance." The reverse-replay rung (M7) that brackets the
off-policy bias is REQUIRED before the paper's mechanism section asserts trajectory
causation, and it runs on these same dissociation cells immediately after a CONFIRM,
before any new method/benchmark cells. Differential `n_readout_undefined` between methods
is itself reported as a confound check (a method that more often destroys the boundary
would bias `readout_acc`).

### Outcome (2026-06-23, FINAL forward read at 800/800)

Read honestly against the criteria above. FC=0.882. Random-256: free=0.422,
readout=0.765 [.686,.817], readoutΔ=-0.117, residual=-0.343, share=0.25. StreamingLLM-256:
free=0.237, readout=0.584 [.495,.647], readoutΔ=-0.298, residual=-0.347, share=0.46.
Undefined boundary: 8/cell, balanced (no confound). (The 765/800 preliminary read gave the
same verdict; full power moved every quantity by <=0.012.)

**Verdict: COMPLICATE / qualified confirm.** What CONFIRMED: a large trajectory residual
(~-0.355) exists and is significant for BOTH methods (readout CIs sit far above free-run,
non-overlapping) -> the residual the contradiction needs is real and readout-invisible;
and the methods dissociate significantly on the READOUT channel (SLM 0.595 vs Random 0.777,
non-overlapping CIs), in the predicted direction. Strikingly, the entire +0.185 free-run
gap is in the readout channel (readoutΔ differs 0.182, residual differs 0.003) -- trajectory
damage is essentially METHOD-INVARIANT. What did NOT confirm (the strong pilot headline
dies): Random is NOT readout-lossless (CI lower 0.696 < the 0.75 bar; readoutΔ significantly
<0), and SLM share 0.45 < 0.50. So the clean "SLM=readout-dominated, Random=trajectory-
dominated" framing is FALSE; reality is "both heavily + equally trajectory-damaged, differing
in readout fidelity." REFRAME the paper accordingly. The method-invariant-trajectory claim
rests on the on-reference (off-policy-biased) residual, so reverse replay is now load-bearing,
not optional -- it is the next experiment, on a token-ID-bearing 256 re-run.

## Amendment G (2026-06-27): committed paper spine + external-review responses

External review (2026-06-27) audited the project end-to-end. Committed responses:

### The spine (commit in writing, now)
THE paper is: "compression damage to reasoning is real, large, and largely INVISIBLE to
teacher-forced / readout-style evaluation -- which is why KV-compression method papers
(readout-style) report near-lossless while free-running benchmarks (HOTT) report collapse;
the gap is the trajectory residual." This is fully supported by data in hand (forward
decomposition + the reverse-replay bracket). Method-invariance and any >100% result are
UPSIDE on top of this spine, reported at their true (exploratory / pending) confidence.

### >100% reframe (resolve Gap 2 explicitly)
We have observed NO >100%/denoising regime (no gains on small-budget MATH). So we DROP
">100%" from the framing/title (reframe option b: near-lossless-under-readout vs
collapse-under-free-generation, which the data fully supports). AIME+R-KV at R-KV's reported
>100% budgets is the high-upside swing (Phase 2): reproduce+decompose it (headline) or show
it does not appear at 7B (a clean scale finding, claimable only because the port is
Jaccard-1.0 certified). We do NOT keep ">100%" in the framing without >100% in the results.

### Two-FC negative control -- PASSED (2026-06-27, committed artifact)
`scripts/report_two_fc_control.py` on math500-fc-powered-merged (split seeds {11,23,47,89}
vs {131,197,263,347}): n=250, mean_ADD=0.068, median 0.000, mean perm p=0.976, 0% of
problems p<0.05. The ADD noise floor (~0.07) is far below the compression cells (0.56-0.75),
so ADD does not false-positive. Required figure for submission -- DONE, no GPU.

### Provenance statement (the load-bearing FC bank)
math500-fc-powered-merged is THE canonical full-cache reference (2000 rows = 250 problems x
n=8, accuracy 0.882). Its 4 runmeta commits are orphaned (repo re-init during the move), so
the byte-identical-generation claim is not git-verifiable. Provenance now rests on: identical
pinned package versions (audit_bank hard-check), per-row source_tree_hash, and FC having NO
press logic (the generation path for full-cache is models.py + generation.py only). Appendix
will carry this paragraph. If time permits before freezing results, regenerate the FC bank
from a clean committed state to make it bulletproof (it underlies every number).

### Scope decisions
- RSP is CUT (structurally invalid twice, decoupled from results; mechanism evidence comes
  from natural-trace replay). Late-binding hypothesis -> future-work mention only.
- Generality envelope (Gap 1, venue-deciding) queued for the nightly watcher: budget-2048
  dose-response (R/SLM + H2O), H2O cells 256+2048, R-KV cells (after the gen-smoke gate).
  Target: 4 methods x 2 budgets + onset-localized TV mechanism (re-scoped replay/Fig 4).
- Threats-to-validity section is written as a STRENGTH (the rigor is the paper's edge), not
  an apology.

## Amendment H (2026-07-03): correction of the residual-bracket computation

A computation error is corrected, dated, and disclosed. This amendment does not change any
raw measurement; it changes the derived bracket endpoints and their interpretation.

### The error

`Decomposition.residual_bracket` and `scripts/report_m4.py` computed the reverse bracket
endpoint as `total_change - reverse_readout_change`, with `reverse_readout_change =
acc_rev - acc_fc`. That quantity equals `acc_free - acc_rev`, which is the **on-policy
readout change** (compressed vs clean readout of the same compressed traces), not a
residual estimate. The correct reverse-side residual estimate is `reverse_readout_change`
itself: `acc_rev - acc_fc = total - (acc_free - acc_rev)`, i.e. total minus the readout
change measured on the compressed model's own states. Both bracket endpoints are then
`total - readout` estimates with readout measured on the two state distributions
(reference states forward, compressed states reverse), which is the construction Amendment
A's off-policy section intended. Code, tests, and reports are fixed as of this date.

### Corrected numbers (from the 2026-06-27 reverse-readout run; identities on cell accuracies)

- Random-256: forward residual −0.343, reverse residual = 0.636 − 0.882 = −0.246 →
  bracket **[−0.343, −0.246]** (was, wrongly, [−0.343, −0.214]).
- StreamingLLM-256: forward residual −0.347, reverse residual = 0.417 − 0.882 = −0.465 →
  bracket **[−0.465, −0.347]** (was, wrongly, [−0.347, −0.180]).
  *(Precision note, added 2026-07-03 after re-running report_m4 with the fixed code on the
  full run data: the exact SLM reverse endpoint is −0.466; −0.465 above derives from the
  rounded 0.417 accuracy. Exact brackets: Random [−0.343, −0.246], SLM [−0.466, −0.347].)*

### What changes and what does not

- UNCHANGED (stronger): the central claim. Both brackets exclude zero by a wider margin
  than previously reported; the residual survives the off-policy correction.
- CHANGED: the anticipated bias ordering (forward inflates the residual, reverse deflates
  it) holds for Random but REVERSES for StreamingLLM (reverse −0.465 exceeds forward
  −0.347). Brackets are therefore reported as min/max intervals of the two estimates, not
  directional bias bounds.
- CHANGED: the forward residuals are nearly equal across methods (−0.343 vs −0.347) while
  the reverse estimates separate (−0.246 vs −0.465). "Trajectory damage is
  method-invariant" is therefore a FORWARD-ESTIMAND statement, reported as exploratory;
  the reverse side attributes more trajectory damage to StreamingLLM. Because sampling
  uncertainty is not propagated to bracket endpoints and the point-estimate brackets
  separate only marginally (0.004), the method-level attribution remains reported as
  bracketed and exploratory — the same humility as before, now for the corrected reason.
- The prior text of Amendment F's "Reverse-replay bracket result" is retained unedited
  per append-only discipline; its bracket numbers and its "brackets overlap" reading are
  superseded by this amendment.

## Amendment I (2026-07-05): R-KV promoted to executable — gate history recorded

Amendment E required a real-generation smoke before any R-KV analysis row. Record:

1. **First smoke run FAILED** (selection Jaccard 0.805 vs the parity-verified
   functions). Root cause: kvpress `SnapKVPress` is a dataclass, so the port's plain
   subclass class-attributes (`window_size=8`, `kernel_size=7`) were silently
   shadowed by the dataclass-generated `__init__` defaults (64 / 5) — the live press
   was scoring with SnapKV's window, not R-KV's. The parity harness could not see
   this (it tests the scoring functions, not the press wrapper); the smoke gate
   caught it exactly as designed.
2. **Fixes:** (a) the scorer subclass is now itself a dataclass (regression test
   locks the override); (b) decode-time window queries are now the true last-8
   observation window — DecodingPress buffers 8 hidden states for rkv and the
   cos/sin rotations for positions p-7..p are computed analytically (standard RoPE
   only; scaled-RoPE models are refused) with a per-event self-check against the
   framework-provided rotation at position p, so a convention drift hard-fails
   rather than silently mis-rotating.
3. **Second smoke run PASSED:** 3 problems, 756 compressions within budget,
   selection Jaccard 1.0000 vs the parity functions, coherent trace. The PASS
   marker (`rkv-gen-smoke-PASS.json`, containing the metrics) was written; the
   press registry treats the marker's existence as the promotion condition, so the
   gate remains enforced by code, not convention.

R-KV cells (grid/readout/reverse at 256; AIME-24 at 2048) are queued behind this
marker. Paper labeling: "R-KV (parity-verified port; smoke-verified during
decoding)" — the two-stage gate history above is disclosable as methodology.

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

## Amendment J (2026-07-17): scale-axis extension (R1-Distill-Qwen-1.5B and -14B)

Post-hoc extension, added after all budget-256 Qwen-7B results and the Llama-8B
check were final, in response to an area-chair-style breadth assessment. Two new
subset-scale cell families, run with the frozen protocol (same seeds 11/23/47/89,
same decoding parameters, same 16384 cap, same graders/instruments, 50 problems
x 4 seeds per cell, own pinned FC reference banks):

- Models (pinned from the HF API main revision on 2026-07-17, recorded in
  configs/models.yaml): DeepSeek-R1-Distill-Qwen-1.5B @ad9f0ae0 and
  DeepSeek-R1-Distill-Qwen-14B @1df85071.
- Presses: StreamingLLM (the sharpest observed case) and R-KV (the strongest
  free-runner; the certified port's per-event analytic-RoPE self-check validates
  the port live on the new model dims — a self-check failure aborts the run and
  is a STOP, not a result).
- Pre-committed read (mirrors the Llama-unit branches, decided before any row
  exists): REPLICATES if a cell's forward residual is negative with 95% CI
  excluding zero and its bracket excludes zero; DEGENERATE if free-run accuracy
  floors with truncation > 0.5 (report plainly, decomposition labeled
  uninformative, as for Llama-Random); NON-REPLICATION otherwise (report
  verbatim in threats; no compensating claim elsewhere). All scale-axis results
  are labeled exploratory and subset-scale in the paper regardless of branch.
- Expected-shape gates for the new FC banks (soft bands; outside-band means
  investigate before any dependent cell is read): 1.5B accuracy 0.60-0.85
  (published 0.839 at 32K decode), 14B accuracy 0.82-0.96 (published 0.939),
  both with the ~0.03-0.09 harness gap seen for 7B (0.8825 vs 0.928) and
  Llama-8B (0.805 vs 0.891).

## Amendment K (2026-07-27): correction of the reference-population mismatch in every accuracy-change estimand

A computation error is corrected, dated, and disclosed. Like Amendment H, this
amendment changes no raw measurement: every generated row, grade, and token-ID
sidecar is unchanged. It changes how derived changes are computed against the
full-cache reference, and it changes some reported numbers in both directions.

### Trigger and date of discovery

Found 2026-07-27 during a Phase-0 verification pass that was auditing an
unrelated external-review claim about the 100-vs-250 problem reference bank.
That claim turned out to be immaterial (matched Acc_FC 0.8838 vs 0.8825, a
+0.00125 difference that cancels in the forward residual). The verification
incidentally exposed the real defect below. The sequence is recorded here
because the correction happens to move one pre-registered verdict in our
favour, and the discovery order is therefore part of the evidence.

### The error

`scripts/report_m4.py` computed one global `full_accuracy` over the entire
full-cache bank and subtracted it from quantities measured on strict, non-random
subsets of that bank:

- **Readout accuracy** is defined only on conditions whose reference trace has a
  detected reasoning/answer boundary. A trace has a boundary iff it terminated,
  so the defined subset is systematically easier than the cell population.
- **Reverse-readout accuracy** is defined only on conditions whose *compressed*
  trace has a boundary — a post-treatment selection, differing by policy
  (308 / 372 / 360 / 346 of 400 at MATH-500 budget 256).

`readout_change` and `reverse_readout_change` therefore compared an accuracy on
a selected subset against a reference mean over the unselected population. That
is a population mismatch, not an estimand; no reading of it is defensible.
Measured magnitudes of the mismatch:

- Acc_FC over all 2000 MATH-500 rows = 0.8825; over the 392 readout-defined
  rows = 0.9031; over R-KV-14B's 154 reverse-defined rows = 0.9351 (against
  0.880 for that bank as a whole).

A second, related defect: the point estimates used the unmatched global mean
while their own problem-clustered bootstrap CIs (`_endpoint_cis`) already
restricted the reference to the cell's problems. Point estimate and interval
were computed on different populations.

### The convention adopted (three layers, applied uniformly)

1. **Primary estimand — fixed population.** The population is *all* conditions
   in the cell (400 at MATH-500 budget 256; 200 for the scale-axis cells; 120 on
   AIME-24). The reference is Acc_FC on that same (problem, seed) population.
   Undefined-boundary rows are handled by **imputation bounds in both
   directions** (scored incorrect for one endpoint, correct for the other), so
   every reported change and residual is an interval over a fixed population.
   This is the treatment the MATH-500 reverse rows already received; the
   scale-axis and AIME cells simply never received it.
2. **Secondary — complete-case.** The same quantities restricted to the defined
   subset, with the reference restricted to that identical subset. Reported as a
   sensitivity analysis and labeled complete-case. Because boundary detection is
   post-treatment, complete-case conditions on an outcome the treatment affects
   and compares policies on different populations; that caveat is stated once
   and is why this layer is secondary rather than primary.
3. **Retired — unmatched subset-vs-full-bank.** Removed everywhere. Numbers that
   derived from it are corrected, not re-interpreted.

### Effect on our own claims — in both directions

The correction is not self-serving, and the evidence is that it moves our
numbers both ways:

- **Against us:** the AIME-24 forward residuals shrink, −0.179 / −0.246 / −0.321
  (Random / R-KV / StreamingLLM) becoming −0.118 / −0.194 / −0.280 on matched
  denominators. The AIME "readout exceeds the reference by +0.092" observation
  is deleted outright: on the 93 defined rows, compressed readout agrees with
  full-cache readout on 93/93 conditions (0 gained, 0 lost) for all three
  policies, so the apparent bonus was entirely the denominator.
- **Against us:** the measured pure answer-resampling effect falls from +0.023
  to **+0.003** (full-cache readout of full-cache traces agrees with the bank on
  391/392 rows; 1 gained, 0 lost). This *strengthens* the readout-easiness
  rebuttal but retires a constant used elsewhere as an explanatory scale.
- **For us:** all four MATH-500 reverse endpoints move further from zero
  (Random −0.246 → −0.282, R-KV −0.221 → −0.251, StreamingLLM −0.466 → −0.470,
  H2O-style −0.463 → −0.478).
- **For us:** R-KV-14B's bracket, the only one in the paper that spanned zero,
  no longer does — [−0.178, +0.029] becomes [−0.168, −0.026] complete-case,
  because its reverse endpoint was computed on the 154 conditions where the
  compressed model terminated, a subset on which the reference itself scores
  0.9351 rather than 0.880.

### Decision on the Amendment J scorecard — deliberately NOT restated

Amendment J's pre-registered read gave **3 of 4 scale-axis cells REPLICATE, with
R-KV-14B inconclusive under the bracket**. That outcome stands as the
pre-registered convention's result and is reported as such. The
corrected-denominator outcome for that cell is reported alongside it, with the
discovery timing disclosed, so a reader sees both. We do not bank the upgrade to
4/4 and we do not restate the §5 scope sentence on the strength of a correction
found after the verdict was recorded; a footnote carries the correction. The
value of visibly declining the upgrade exceeds the value of the upgrade.

### Discipline followed in applying it

The fix was written, the code changed, and **every** cell regenerated in one
pass — MATH-500 at 256 and 2048, AIME-24, Llama-8B, 1.5B and 14B — before any
per-cell verdict was inspected. The 14B cell was not fixed first. Uniform
prior application is the control against fitting the convention to the result.

### Post-application record (same day, 2026-07-27): re-run outcome and instrument-class audit

**Verification.** The regenerated cells were cross-checked against derivations
written from the raw rows *before* the code changed (24 of 24 quantities agree to
4 decimal places: every complete-case residual, every reverse endpoint, and the
readout-/reverse-defined reference accuracies).

**The primary estimand reproduces Amendment J independently.** Under layer 1 the
scale-axis scorecard is unchanged at 3 of 4: 1.5B StreamingLLM [-0.4750, -0.2550],
1.5B R-KV [-0.3750, -0.1650] and 14B StreamingLLM [-0.4850, -0.2550] exclude zero;
14B R-KV [-0.1800, +0.0500] spans it. Only the complete-case sensitivity (layer 2)
resolves the fourth cell, at [-0.1675, -0.0260]. The convention change therefore
did not buy the upgrade: the primary estimand returns the pre-registered verdict on
its own, and that is the reason the Amendment J scorecard is not restated -- not
authorial restraint.

**Layer-1 bound width is analytic, not empirical.** Imputing the undefined rows
both ways makes the interval exactly `n_undefined / N` wide. Observed widths match
this identity in every cell (8/400 = 0.020; 27/120 = 0.225; 3/200 = 0.015; 46/200 =
0.230). Consequently a wide layer-1 bound on a high-truncation cell carries no
evidential content beyond the truncation rate, and "spans zero" under layer 1 must
never be read as "no effect." The 14B R-KV cell is the worked example, and its
three statements are consistent rather than contradictory:
forward residual -0.1675 with CI [-0.2733, -0.0850] excluding zero; forward layer-1
bounds [-0.1800, -0.1650] excluding zero (only 3 of 200 readout rows undefined);
bracket spanning zero only because 46 of 200 *compressed* traces never terminated
(truncation 0.230), which is a reverse-side definedness fact. Every reported
layer-1 bound is therefore accompanied by its undefined-row count.

**AIME-24 confidence intervals after the correction.** The AIME residuals move
toward zero (the largest single movement in the re-run), so the CIs were recomputed
before any prose was touched: Random -0.1183, CI [-0.3276, -0.0506]; R-KV -0.1935,
CI [-0.4023, -0.1250]; StreamingLLM -0.2796, CI [-0.4914, -0.1815]. All three still
exclude zero, and all three exclude zero under layer 1 as well ([-0.3000, -0.0750] /
[-0.3667, -0.1417] / [-0.4417, -0.2167]). The second-benchmark replication stands at
3 of 3, with Random's margin reduced (upper bound -0.076 -> -0.051).

**Instrument-class audit -- the defect is confined to accuracy reporting.** Because
this bug survived a pre-registration, a negative control and an exact-replication
check, the sibling instruments were audited directly rather than by inference:

- *ADD.* Computed over every problem in the compressed cell (the loop skips only
  pids absent from the reference bank; zero are). Problem sets on the two sides are
  identical; the seed asymmetry is the disclosed 8-vs-4 and is not a problem-set
  asymmetry. Truncated rows do reach the frozen `__OTHER__` bucket -- verified on
  the rows themselves, 38 of 38 reference and 121 of 121 compressed truncated rows
  carry `answer_canon == "__OTHER__"` -- so the termination-mass channel measures
  what it is claimed to measure. (Mechanism note: they arrive via the canonicalizer,
  not via `coarsen_answer`'s `truncated=` parameter, which this call path never
  sets; the parameter is redundant here, not wrong.)
- *TV profiles.* All three replay cells span exactly the same (problem, seed)
  population as their grid cell, with `auc_tv`, `frac_above` and `onset_step`
  non-null on 400 of 400 rows each, so window means and AUC share one denominator
  and no definedness filter is applied. The H2O mean AUC-TV recomputes to 0.072991,
  matching the archived value exactly.

No population-selection defect was found outside the accuracy-change estimands.

---

## Amendment L (2026-07-28): resolution of held-trace evaluation — a POST-HOC analysis, declared as such

**This analysis was not pre-registered, and it was formed after every cell in the
paper had been read.** It is recorded here before it is written up, with its
design fixed in code (`kvcache.stats.resolution_spread`) rather than described,
so that the choices below cannot be tuned after seeing more of the result than
has already been seen. Every appearance of it in the paper is labelled
exploratory. It does not revise any pre-registered verdict, and no
pre-registered outcome depends on it.

**Provenance of the question.** While correcting Amendment K it became visible
that the three AIME presses share an *identical* readout accuracy (0.462, a
readout change of exactly 0.000 against the matched reference) while free-running
separates them by roughly 2×. The obvious question — whether held-trace
evaluation loses discriminating power precisely in the regime where it certifies
near-losslessness — is answerable from cells that already exist, which is exactly
the situation in which post-hoc analysis is most tempting and least trustworthy.
Hence this amendment.

**Estimand.** For a regime (benchmark × budget), the *spread* of a set of
policies under a protocol is the **mean absolute pairwise difference** of their
per-problem accuracies, paired on the problem. Max-minus-min is reported nowhere:
with three or four presses it is a function of the two extreme cells alone and
moves discontinuously when they swap.

**Primary reported quantity: the DIFFERENCE** (free-run spread − readout spread),
with a problem-clustered percentile CI over 2000 resamples, both spreads driven
by the same bootstrap draw so the difference is paired. **The ratio is reported
as descriptive only, with no interval**: in the saturated regime the readout
spread approaches zero, and a ratio interval on a near-zero denominator is not an
honest interval.

**Secondary:** rank agreement — the fraction of press pairs ordered identically by
the two protocols, with a readout tie against a free-run ordering counted as
disagreement, since that is the failure under study.

**What this analysis may and may not claim.** It compares readout to readout and
free-run to free-run on the same problems, so the off-policy caveat attaching to
the trajectory residual does not apply to it. It may not be stated as "readout
cannot rank policies": that is **false in general**, and our own MATH-500 budget-256
cell is the counterexample. The claim is specific to the saturated regime.

**Result, recorded here as read (artifacts under
`results/diag/amendment-k/classb/resolution_*.json`, seed 0, 2000 resamples):**

| regime | free-run spread | readout spread | difference (95% CI) | ratio | rank agreement |
|---|---|---|---|---|---|
| MATH-500 @256 | 0.1704 | 0.1328 | +0.0376 [−0.0117, +0.0837] | 0.779 | 6/6 |
| MATH-500 @2048 | 0.0283 | 0.0033 | +0.0250 [+0.0067, +0.0467] | 0.118 | 1/3 |
| AIME-24 @2048 | 0.1012 | 0.0000 | +0.1012 [+0.0357, +0.1786] | 0.000 | 0/3 |

At budget 256 the difference **spans zero and rank agreement is perfect**: readout
retains essentially full resolution in the regime where it reports large damage.
At the two near-lossless cells the difference **excludes zero** and rank agreement
falls to 1/3 and then 0/3. The monotone ordering of the ratio (0.78 → 0.12 → 0.00)
is descriptive; the two collapse cells are what carry statistical weight.

**Confirmatory follow-up, pre-registered now.** The budget 512 and 1024 cells
described in the queue have not been run. Before they are, we pre-commit the
prediction this analysis generates: **the readout-to-free-run spread ratio at 512
and 1024 will fall strictly between the 256 value (0.78) and the 2048 value
(0.12), and the spread difference will move from spanning zero toward excluding
it as budget rises.** If the ratio at 512 or 1024 falls outside [0.12, 0.78], or
the ordering is non-monotone in budget, the gradual-collapse reading is refuted
and we will report it as such rather than reinterpreting. This converts the
present post-hoc observation into a prediction testable on data that does not yet
exist, which is worth more than the cells themselves.

---

## Amendment M (2026-07-28): pre-registered reads for the built-but-unlaunched experiments

Four experiments are specified and instrumented but **not run**. Their reads are
committed here first, so that if they are run later the analysis is fixed before
the data exists. Any of them may never run; that is recorded as a limitation
rather than resolved by quietly dropping the entry.

### M.1 TV-matched full-cache noise control (`noise_control`)

**Status: instrument gate built and tested; injection engine NOT built; run NOT
launched.** `docs/project/context.md` §1.3 designates this control mandatory, and the paper
does not currently contain it. That is stated as a limitation in §6 rather than
left implicit.

**Design.** During full-cache free-running generation, inject calibrated
per-step perturbations `(1 - eps) * p + eps * q` whose induced per-step TV
matches the profile measured for method *M* under replay. Cells:
StreamingLLM-256 and Random-256, 100 problems × 4 seeds each, matched seeds.

**Instrument gate, which runs first and can veto the experiment.**
`kvcache.metrics.noise_control.match_gate` requires *both* (i) first-50 and
later window means within 10% relative of the target profile, and (ii) the
later/first-50 growth ratio within 25% relative. Both are required because
either alone is satisfiable by an obviously wrong profile — a flat curve with
the right average passes (i) and fails (ii); a uniformly doubled curve passes
(ii) and fails (i). **If the gate does not pass, accuracy from this control is
not read at all.** A failed calibration means the experiment did not run; a null
result under a failed match is uninterpretable, and a positive one worse.

**Pre-committed read, conditional on the gate passing.** Let ΔM be method *M*'s
free-running accuracy change and ΔN the change under TV-matched generic noise,
both against the matched reference, with problem-clustered CIs.

- **GENERIC** — if ΔN's CI contains ΔM for both cells: the accuracy loss is
  explained by divergence *magnitude*, and the trajectory residual is generic
  drift rather than structured damage. **This outcome is reported as the primary
  result of the control and it substantially weakens the paper's mechanism
  reading.** It is not a reason to reframe around something else; the correct
  response is to say so, in the abstract.
- **STRUCTURED** — if ΔN is significantly smaller in magnitude than ΔM (CIs
  non-overlapping) for both cells: matched-magnitude noise does not reproduce
  the damage, so what compression does is structured. This is the outcome the
  mechanism reading predicts, and it is the strongest rung of the causal ladder.
- **MIXED** — if the two cells disagree: reported as mixed, per policy, with no
  pooled claim.

Signed flip-mass accompanies both arms, so a *reproduced improvement* is never
mislabelled as reproduced damage.

### M.2 The missing factorial cell (full-cache trace, compression at the boundary)

**Status: specified, not built, not launched.** Force the full-cache reference
trace through a **full** cache and activate compression only at the
reasoning/answer boundary. 100 × 4, one budget (256), StreamingLLM and Random.

This separates readout damage localized to the answer span from damage caused by
a cache state degraded across the trace — the decomposition currently cannot
tell those apart, which is the reviewers' central structural objection. Read:
accuracy near 0.885–0.906 ⇒ answer-time compression alone explains little and
essentially all readout damage is trace-accumulated; near 0.765 (Random's forward
readout) ⇒ answer-time compression alone explains it. Intermediate values are
reported as a proportion with a CI and not rounded to either pole.

The inverse cell (compressed trace, compressed answer) is deliberately **not**
specified: it is a replay of the free-running condition and carries no extra
information.

### M.3 Ratio-matched TV series

**Status: specified, not built, not launched.** Budget is fixed at 256 while
context grows, so the observed TV staircase may be a compression-*ratio* curve
rather than a horizon curve. Re-measure per-step TV holding the realized ratio
fixed as the trace grows. Read: if the staircase survives at fixed ratio, the
horizon reading stands and the word "mechanism" is licensed for it; if it
flattens, the profile is a ratio effect and §4.4 is restated in those terms.

### M.4 Content-token TV variant

**Status: specified, not built, not launched.** Restrict the TV profile to
numerals, operators, and the answer span. Read: if the content-restricted
profile tracks the full profile, the "your TV is paraphrase noise" objection is
answered; if it is materially flatter, the profile is substantially lexical and
we say so.

### M.5 Budgets 512 and 1024 — the confirmatory resolution test

**Status: configs written, not launched.** The read is already pre-committed in
**Amendment L**: the readout-to-free-run spread ratio must fall strictly between
the 256 value (0.78) and the 2048 value (0.12), monotone in budget. A ratio
outside [0.12, 0.78] at either budget, or a non-monotone ordering, refutes the
gradual-collapse reading, and we report that rather than reinterpreting.

### M.6 One HOTT-configuration cell, and R-KV in its published configuration

**Status: specified, not launched.** We claim to resolve HOTT's contradiction
while failing to reproduce its baseline within a factor of two (0.882 vs 0.47),
and we characterize R-KV's published claim from an adjacent configuration rather
than its own (R1-Distill-Llama-8B / AIME-24). Both are honest gaps in the current
paper and are stated as such in §6. Reads: for the HOTT cell, whether our
full-cache baseline moves toward 0.47 under their decode-length and sampling
configuration — if it does, the digit gap is configuration and the channel
argument is unaffected; if it does not, we cannot explain the gap and say so.
For the R-KV cell, whether the >100% regime appears in its own configuration; if
it does, our "no >100% regime at 7B" finding is confirmed as scale-specific
exactly as Amendment E anticipated, and if it does not, the published claim does
not replicate under a parity-certified port and that is a finding about the
claim.

---

## Amendment K, addendum (2026-07-28): the fix was incomplete, and the class is now swept and enforced

**Disclosure first: Amendment K was applied to the nine decomposition cells and
stopped there.** For roughly a day the retired mismatched-population comparison
remained inside `fc-readout-control` — the control whose only job is to calibrate
the readout channel against the "readout is easy by construction" objection. It
differenced `Acc_ro(FC) = 0.9056`, defined on 392 rows, against the 2000-row bank
at `0.8825`, giving `+0.023`. On the matched population the answer-resampling
effect is **`+0.003`**. The correction happens to *help* the paper — the residuals
are two orders of magnitude above the control rather than one — so per the
standing rule the timing is disclosed here rather than quietly banked, and the
old value is retained in the artifact under `resampling_effect_unmatched_RETIRED`
so it stays recognisable.

That a systemic fix could be applied to nine artifacts and miss a tenth is the
generalisable lesson, not the instance. The class has now been enumerated,
classified, and made enforceable.

### The sweep

Every code path that computes a quantity against a reference, classified. "K-irrelevant"
carries a reason; assertion alone was not accepted.

| Path | Class | Reason |
|---|---|---|
| `scripts/report_m4.py` | **K-compliant** | Emits `full_accuracy` = the cell's own reference population (line ~379) as the comparator for every change; the 2000-row bank figure is emitted separately as `full_accuracy_all_rows` and at `#/full_cache/accuracy` for provenance only. Declares `reference_convention`. Layer-1 `primary_bounds` fix the population at all cell conditions. |
| `scripts/harvest_results.py` (`readout_acc`) | **was K-violating, now compliant** | The finding above. Now restricts the bank to the readout-defined conditions and emits `fc_accuracy_matched`, `resampling_effect_matched`, and `reference_convention`; the retired form is kept under an explicit `_RETIRED` suffix. |
| `scripts/report_add.py` | **K-irrelevant, now declared** | Signed flip-mass *is* reference-relative, but it is formed per problem: mean over pids of (compressed accuracy on that pid − FC accuracy on that pid), with pids absent from the reference bank skipped. Both sides are the same population by construction, so no reference-mean question arises. ADD itself is a per-problem distribution TV, likewise paired. Now declares the convention and the basis so this is machine-checkable rather than re-derivable only by reading the loop. |
| `scripts/report_two_fc_control.py` | **K-irrelevant** | Compares two full-cache banks to each other, paired on pid. There is no compressed arm and no defined-subset restriction. |
| `scripts/report_self_replay.py` | **K-irrelevant** | Emits TV quantities (`global_mean_tv` and per-profile summaries). No accuracy, no reference mean. |
| `scripts/report_tv_windows.py` | **K-irrelevant** | Windowed per-step TV over replay profiles. Distribution-fidelity scale; never differenced against an accuracy. |
| `scripts/report_policy_contrast.py` | **K-irrelevant** | Random vs StreamingLLM, paired per problem. The full-cache reference does not enter the statistic at all. |
| `scripts/report_residual_contrast.py` | **K-irrelevant, and provably so** | The per-problem residual is `free − readout`; `Acc_FC` cancels exactly out of `(free − Acc_FC) − (readout − Acc_FC)`. This contrast cannot be affected by the reference-population choice. |
| `scripts/report_resolution.py` | **K-irrelevant** | Compares free-run spread against readout spread. Both are within-protocol spreads; no reference term. |
| `scripts/report_compute.py`, `scripts/report_prompt_lengths.py` | **K-irrelevant** | Wall-clock and token counts. No accuracy. |
| `scripts/diagnose_math_failures.py` | **K-irrelevant** | Per-row regrade and classification. Compares a row to its own gold, never to a population mean. |
| `src/kvcache/decomposition.py` | **K-compliant** | Defines the estimands; the reverse endpoint's collapse to `acc_rev − acc_fc` is documented in the module. Population selection happens in the caller. |
| `src/kvcache/stats.py` | **K-irrelevant** | `paired_policy_contrast`, `paired_residual_contrast`, `resolution_spread` and `percentile_ci_mc_error` are all paired or within-protocol; none takes a reference mean. |
| `src/kvcache/metrics/add.py`, `tdp.py`, `noise_control.py` | **K-irrelevant** | Distribution-scale instruments. `noise_control` compares TV profiles to TV profiles. |
| `scripts/make_paper_figures.py` | **K-compliant** | Since 2026-07-27 it loads every constant from `paper_numbers.json`, so it inherits the convention rather than holding a copy. It names `FC_BANK` and `FC` separately so the two references cannot be confused. |

**Retired-pattern grep.** Searching the tree for a defined-subset accuracy
differenced against a full-bank mean returned three hits, all accounted for:
`harvest_results.py` (the deliberately retained `_RETIRED` value),
`report_m4.py:236` (the bank figure, emitted as provenance and never used as a
comparator — verified by tracing all four of its uses), and
`report_self_replay.py:58` (a TV mean, not an accuracy). **No K-violating path
remains.**

### Enforcement

`kvcache.paper_numbers.check_reference_conventions` enumerates the fourteen
artifacts that carry a reference-relative quantity and fails the audit gate if
any lacks `reference_convention`, declares a value other than
`amendment-k-cell-population`, or carries a key whose name marks it pre-K
without marking it retired. A missing artifact is a failure rather than a skip:
silently passing because a file was absent is how a class-wide check stops
checking anything. This runs in `make audit`, so the class cannot regress
unnoticed the way it did here.

Building the check immediately caught the four ADD artifacts, which were
compliant but undeclared — exactly the state in which a future reader has to
re-derive compliance by reading the code, which is what allowed the original
omission.

---

## Amendment N (2026-07-29): pre-registered reads for the three newly queued cells

Written before any of these runs. Each read states what every outcome means,
including the outcomes that cost us something.

### N.1 The HOTT-configuration cell (`grid_math500_hottcap`)

**What is matched, and what is not.** HOTT's setup, verified against the paper's
own text: *"Each model is allowed to generate a maximum of 2048 new tokens via
greedy decoding"*, over budgets {128, 256, 384, 512}. This cell changes **only
the generation cap**, to 2048. Sampling stays at T=0.6 / top-p 0.95, because the
generation path hardcodes `do_sample=True` and adding greedy decoding is a change
to result-bearing code that deserves its own unit rather than a late edit.
**Greedy therefore remains an unmatched variable and every statement of this
result must say so.**

**Prior, stated before the run:** our Random/StreamingLLM traces average 6,725
tokens, so a 2048 cap truncates the majority of them, and truncated generations
score incorrect in every denominator here. We expect a large drop.

**Pre-committed read.** Let `Acc_FC@2048cap` be full-cache MATH-500 accuracy
under the cap, against our 0.882 at 16384 and HOTT's 0.47.

- **EXPLAINED** — `Acc_FC@2048cap` lands within roughly 0.05 of 0.47: the
  digit-level gap is a decode-length artifact, our channel argument is untouched,
  and §2 stops needing to hedge about "different decode-length and sampling"
  because we will have measured it. Report the truncation rate alongside, since
  it is the mechanism.
- **PARTLY EXPLAINED** — it moves substantially toward 0.47 but stops short
  (say 0.55–0.70): the cap is the dominant term and something else (greedy
  decoding, prompt template, grader) carries the rest. Report the residual gap
  as unexplained rather than attributing it to the unmatched variable by
  assumption.
- **NOT EXPLAINED** — it stays near 0.882: our stated explanation for the gap is
  wrong. This is the outcome that costs us: §2's parenthetical becomes a claim we
  tested and failed to support, and it must be rewritten to say we cannot account
  for the difference. **Report it that way.**

The StreamingLLM-128 arm is read the same way against HOTT's 0.03.

### N.2 R-KV in its published configuration (`grid_aime24_rkv_published`)

R1-Distill-Llama-8B on AIME-24 at budget 2048 (~16% of the mean trace at this
scale), against its own new reference bank (`bank_aime24_llama8b`, N.3). This is
the model and benchmark of the 105%-at-16% headline; §5 currently says we tested
the claim "in an adjacent configuration".

**Pre-committed read.** Let `r = Acc_rkv / Acc_FC` on this cell, with a
problem-clustered CI on the difference.

- **REPLICATES (r ≥ 1.0, CI on the difference includes or exceeds zero)** — the
  >100% regime is real in its own configuration, our "no >100% regime at 7B on
  MATH-500" finding is confirmed as **scale- and benchmark-specific exactly as
  Amendment E anticipated**, and §5's wording strengthens in *their* favour. We
  report that plainly; it does not weaken the trajectory residual, which is a
  separate quantity.
- **DOES NOT REPLICATE (r < 1.0 with the difference CI excluding zero)** — a
  parity-certified port does not reproduce the published headline in the
  published configuration. That is a finding about the claim, and it must be
  stated with maximal care: our port matched their code at Jaccard 1.0 on
  retained sets, so the honest phrasing is that we could not reproduce it, not
  that the claim is false. Report our accuracy, theirs, the CI, and the port
  certification together.
- **INCONCLUSIVE (CI spans, n=30 problems)** — say so. AIME-24 is 30 problems and
  this cell may simply lack power; that is a limitation, not a result.

**We do not re-cut this in our favour under any branch.** If it replicates, the
paper says the regime is real at their scale.

### N.3 `bank_aime24_llama8b`

A reference bank, not a test. Gate: accuracy in [0.30, 0.70] (R-KV's own paper
and the DeepSeek report put R1-Distill-Llama-8B AIME-24 pass@1 near 0.50). Out of
band ⇒ investigate as an instrument problem before any cell is read against it,
per the standing rule that a value outside its expected shape is a bug until the
instrument is re-validated.

---

## Amendment L, result (2026-07-31): the confirmatory test at budget 512 CONFIRMS

`grid-diss512` and `readout-diss512` completed 800/800 each. The prediction
Amendment L pre-committed — that the readout-to-free-run spread ratio at budgets
512 and 1024 would fall strictly between the budget-256 value and the budget-2048
value, monotone in budget — **holds at 512**. Budget 1024 is still running and
its arm of the prediction is untested.

**Result as pre-registered** (artifact
`results/diag/amendment-k/classb/resolution_math500_512.json`, seed 0, 2000
resamples): free-run spread 0.2025, readout spread 0.0608, **ratio 0.3004**,
spread difference +0.1417 with 95% CI [+0.0700, +0.2075], excluding zero. The
pre-registered interval was (0.12, 0.78); 0.300 is strictly inside it.

**A comparability check, and the disclosure that goes with it.** The budget-256
and budget-2048 resolution values were computed over four and three presses
respectively, while budget 512 has two (Random and StreamingLLM). Because the
spread is a mean absolute pairwise difference, the press *set* affects it, so the
budget comparison was not strictly like-for-like. **After seeing the 512 result**
we recomputed 256 and 2048 restricted to the same two presses. This is a
sensitivity analysis of a pre-registered quantity rather than a new estimand, but
it was run after the fact and is labelled accordingly.

| budget | free-run spread | readout spread | ratio | difference CI |
|---|---|---|---|---|
| 256 | 0.1850 | 0.1792 | 0.9685 | [−0.0783, +0.0883] spans zero |
| 512 | 0.2025 | 0.0608 | 0.3004 | [+0.0700, +0.2075] excludes zero |
| 2048 | 0.0425 | 0.0025 | 0.0588 | [+0.0075, +0.0750] excludes zero |

The ordering is monotone on the matched pair as well, and 0.300 lies strictly
between the matched-pair endpoints (0.0588, 0.9685). **The verdict is the same
under both readings**, which is the useful property: the confirmation does not
depend on which press set the endpoints were computed over.

Two things the matched-pair view shows more sharply than the mixed-press one.
At budget 256 the readout spread is 0.1792 against a free-run spread of 0.1850 —
readout retains **essentially all** of the policy separation, not merely most of
it. And at 2048 rank agreement on this pair is 0, so the two protocols order the
two policies oppositely where readout certifies near-losslessness.

**Not yet claimed.** Budget 1024 is unrun. Until it lands, the confirmatory test
is passed at one of its two points, and the paper must say exactly that.

---

## Amendment L, final result (2026-08-02): the confirmatory prediction is REFUTED

Budget 1024 completed (grid and readout, 800/800 each). **The prediction
Amendment L pre-committed does not hold, and we report it as a refutation rather
than reinterpreting it.**

**The pre-committed criterion, quoted from Amendment L:** *"the readout-to-free-run
spread ratio at 512 and 1024 will fall strictly between the 256 value (0.78) and
the 2048 value (0.12), and the spread difference will move from spanning zero
toward excluding it as budget rises. If the ratio at 512 or 1024 falls outside
[0.12, 0.78], or the ordering is non-monotone in budget, the gradual-collapse
reading is refuted and we will report it as such rather than reinterpreting."*

**Full series, matched press pair (Random, StreamingLLM) at all four budgets:**

| budget | free-run spread | readout spread | ratio | spread difference (95% CI) | rank agr. |
|---|---|---|---|---|---|
| 256 | 0.1850 | 0.1792 | 0.9685 | +0.0058 [−0.0783, +0.0883] | 1/1 |
| 512 | 0.2025 | 0.0608 | 0.3004 | +0.1417 [+0.0700, +0.2075] | 1/1 |
| 1024 | 0.1125 | 0.0050 | **0.0444** | +0.1075 [+0.0600, +0.1600] | 1/1 |
| 2048 | 0.0425 | 0.0025 | 0.0588 | +0.0400 [+0.0075, +0.0750] | 0/1 |

**Both clauses fail.** The 1024 ratio of 0.0444 is **outside** the pre-committed
interval (0.12, 0.78), and the ordering is **non-monotone**: the ratio falls
0.9685 → 0.3004 → 0.0444 and then *rises* to 0.0588 at 2048. Either alone is a
refutation under the stated criterion; both fail.

**Diagnosis, offered as description and not as rescue.** The readout spread is
monotone across all four budgets (0.1792 → 0.0608 → 0.0050 → 0.0025). The
free-run spread is not (0.1850 → 0.2025 → 0.1125 → 0.0425): it *rises* from 256
to 512 and collapses by 2048, because at the near-lossless pole the two policies
converge in free-running accuracy and there is little separation left to lose.
The ratio therefore inverts through its denominator, not because readout recovers
resolution. **We note this because it is true, and we do not substitute the
monotone numerator for the ratio we pre-registered.** A prediction about a ratio
is refuted by that ratio.

**What survives, stated separately so it is not mistaken for a rescue.** The
spread *difference* excludes zero at 512, 1024 and 2048, and spans zero at 256.
Readout retains essentially all policy separation at budget 256 (0.1792 against
0.1850) and almost none at 1024 and 2048 (0.0050 and 0.0025). The endpoint
observation — that held-trace evaluation discriminates where it reports damage
and not where it certifies — is measured and stands. **The graded, monotone-in-
budget reading of it does not.**

**Consequences, applied.** Per the branch written before the data landed
(`docs/operations/INTEGRATION_CHECKLIST.md` §1, Branch B): §4.1 is demoted from a primary
contribution and retitled to state the endpoints without the graded claim; the
proposed reframe title is withdrawn, since both candidates were built on the
graded reading; the contribution reorder that placed resolution collapse first is
withdrawn. The post-hoc status of the original observation is unchanged and
remains disclosed.

---

## Amendment O (2026-08-03): the N.3 bank gate FAILED, and the generation cap is raised to the published one

### O.1 The gate failure, stated first

`bank_aime24_llama8b` completed 120/120 (30 problems x seeds 11/23/47/89) and
**failed its N.3 gate**. Pre-registered band: accuracy in [0.30, 0.70]. Measured:
**0.2750**. Per N.3 the bank is an instrument problem until re-validated, and no
cell may be read against it. `grid_aime24_rkv_published` (N.2) is therefore
**not readable** and stays unread.

This is recorded before any remedy, and it is not deleted by what follows.

### O.2 Diagnosis: the failure is the generation cap, not the pipeline

- **Truncation rate 0.2583** (31/120 rows hit the 16384-token cap). Per standing
  protocol truncations stay in the denominator scored incorrect, so they enter
  the accuracy directly.
- **Accuracy among non-truncated rows: 0.3708**, inside the gate band.
- The already-accepted Qwen-7B AIME bank (`fc-aime24`, n=240) shows the *same
  behaviour* at the same cap: truncation 0.2208, accuracy 0.3708, non-truncated
  0.4759. The pipeline is doing at 8B what it does at 7B.
- Both banks land far below their published references (DeepSeek report
  R1-Distill-Qwen-7B ~0.555 and R1-Distill-Llama-8B ~0.504 on AIME-24), by
  similar proportions (0.67x and 0.55x).

The instrument is not broken. The 16384 cap costs both models roughly a quarter
of their generations, and at 8B that is enough to push the point estimate below
the floor.

### O.3 The cap change, and the honest account of its timing

**The change.** `bank_aime24_llama8b` and `grid_aime24_rkv_published` move from
`max_new_tokens: 16384` to **32768** for AIME-24, and are re-run from scratch.

**The justification that does not depend on the result.** N.2 exists to test
R-KV *in its published configuration*. The published configuration includes the
generation length: DeepSeek's own evaluation of the R1-Distill models and R-KV's
AIME-24 headline both use a 32768-token budget. Capping at 16384 was a design
error in our config, discoverable from the published setup before any data
existed. Running a "published configuration" cell at half the published
generation length does not test the published configuration.

**The timing, disclosed because it matters.** We did not notice this before
running. The error was found while investigating an out-of-band gate result, and
a change to a protocol parameter made after seeing a disappointing number is
exactly the move this preregistration exists to constrain. We therefore bind
ourselves as follows:

1. **The 16384 result is permanent.** Accuracy 0.2750 at truncation 0.2583, gate
   failed, is reported in the paper's record alongside whatever 32768 produces.
   It is not superseded and not deleted.
2. **Both caps are reported wherever this cell is stated**, with the truncation
   rate at each. A reader sees the cap's effect, not just the final number.
3. **The gate band is unchanged**: [0.30, 0.70]. It is not widened, and 0.2750
   is not reclassified as a pass.
4. **One attempt only.** If the bank fails the gate again at 32768, N.2 and N.3
   are **cut**, §2 keeps its "adjacent configuration" hedge, and no third cap is
   tried. This clause exists to stop an unbounded search for a cap that passes.

### O.4 Pre-registered read for the re-run, written before it runs

- **Bank in band [0.30, 0.70]** ⇒ the instrument is re-validated, N.2 becomes
  readable, and N.2's pre-registered branches (REPLICATES / DOES NOT REPLICATE /
  INCONCLUSIVE, Amendment N.2, unchanged) apply against the 32768 bank.
- **Bank still out of band** ⇒ per O.3(4), N.2/N.3 are cut. We report that we
  could not build a reference bank for R1-Distill-Llama-8B on AIME-24 that met a
  pre-registered sanity band under our generation path, and that we therefore did
  not test R-KV's published configuration. That is a limitation, stated as one.
- Either way, the **truncation-rate finding is reported**: a quarter of AIME-24
  traces from these models do not terminate within 16384 tokens, which is a
  measurement about evaluating reasoning models under a token budget and is
  relevant to any AIME result quoted at a cap.

### O.5 Scaled-RoPE support in the R-KV window-query recompute

`grid_aime24_rkv_published` had additionally crashed on every launch since
2026-08-01 with `NotImplementedError: rkv window-query recompute supports
standard RoPE only` — R1-Distill-Llama-8B uses llama3-scaled RoPE and the
recompute hand-rolled the unscaled frequency schedule. The guard was correct:
it refused rather than scoring with wrong rotations.

**Fix.** The frequency schedule now comes from transformers' own
`ROPE_INIT_FUNCTIONS` for whatever `rope_type` the config declares, rather than
being re-derived locally. This is a deletion of hand-rolled arithmetic, not an
addition. The guard is narrowed, not removed: a scaling family the installed
transformers has no schedule for still refuses.

**Certification.** (i) A test locks the unscaled branch to **bitwise-identical**
frequencies against the previous hand-rolled formula, so every number already
measured under the old path is unchanged; (ii) the per-event self-check against
the model's own `position_embeddings` is retained and fires on every compression;
(iii) the real-generation smoke was run on R1-Distill-Llama-8B / AIME-24 and
**passed at selection Jaccard 1.0000** over 448 compressions and 5 verified
events. The R-KV PASS marker was deliberately **not** written by that run: the
marker is the promotion mechanism for R-KV as a whole and is defined over
R1-Distill-Qwen-7B, so a smoke on another model reports its result and promotes
nothing.

---

## Amendment N.1, result (2026-08-03): the HOTT baseline gap is a generation-cap artifact

`bank_hottcap` (400 rows) and `grid_hottcap` (400 rows) completed. Read as
pre-registered in N.1, branch **EXPLAINED**.

**Result.** Re-running our own full-cache condition under HOTT's 2048-token
generation cap, changing nothing else, drops accuracy from **0.882** to
**0.515**, with **0.430** of generations truncated at the cap and scored
incorrect (230/400 cap hits). HOTT report 0.47. The cap alone therefore accounts
for nearly all of the baseline gap, leaving **0.045**.

The StreamingLLM-128 arm, read the same way against HOTT's 0.03, lands at
**0.100** with **0.475** truncation.

**Mandatory caveat, as pre-written.** This cell matches HOTT's 2048-token
generation cap and **not** their greedy decoding, which our generation path does
not support; sampling remains $T=0.6$, top-$p$ 0.95. It therefore measures how
much of the baseline gap the cap alone accounts for, and any residual gap is not
attributable to the cap.

**Consequences, applied per `docs/operations/INTEGRATION_CHECKLIST.md` §2 (EXPLAINED).**
§2's parenthetical no longer asserts a decode-length explanation; it reports the
measurement. §5's "we do not reproduce HOTT's baseline" item is removed from the
not-measured list, because it is now measured; what remains untested there is
greedy decoding, and that is what §5 now says.

**A gate limitation found while landing this, recorded rather than hidden.** The
0.045 residual is sourced, but the number audit attributes it to
`llama.random.readout_ci1` — an unrelated Llama CI bound that happens to equal
0.045 exactly. A derived entry `hottcap.residual_gap` was added and does **not**
change the attribution, because exact-match precedence prefers the entry whose
stored float is exactly 0.045. The number and its arithmetic are checked; the
audit's provenance trail for that one literal points at the wrong quantity.
Value-based attribution cannot separate two distinct quantities that share a
value without context-aware matching, which is not built.

---

## Amendment O, result (2026-08-04): the re-run bank PASSES the N.3 gate, exactly at the floor

`fc-aime24-llama8b-cap32768` completed 120/120 (30 problems x seeds 11/23/47/89)
at the published 32768-token cap.

**Result, stated precisely because the margin is zero.** Accuracy
**36/120 = 3/10 = 0.300000 exactly**. The pre-registered N.3 band is
[0.30, 0.70], inclusive. The value sits **exactly on the floor**, so this is a
**PASS under the criterion as written** — and it is a pass with no margin
whatsoever, which we state rather than round away. We do not re-cut the
criterion in either direction: the band was fixed in advance, 0.300000 is inside
it, and a boundary pass is a pass.

**The cap diagnosis was correct.** Truncation fell from **0.2583** at 16384 to
**0.0750** at 32768 (14 cap hits, down from 42), and mean generated length rose
to 14799 tokens. The rows that were being discarded as truncated are now
completing.

**What did not happen: accuracy did not rise to the published reference.**
Overall accuracy moved only 0.2750 -> 0.3000, and accuracy among non-truncated
rows *fell*, 0.3708 -> 0.3243. Recovering the truncated generations mostly
recovered *hard* problems the model then answers incorrectly. So the cap
explained the gate failure without closing the distance to the ~0.504 that
DeepSeek report for this model; our bank sits well below its published
reference, and that remains true and is reported.

**Consequences, per O.4.** The bank is re-validated as an instrument, so
**N.2 becomes readable** and `grid_aime24_rkv_published` runs against this bank
at the matched 32768 cap. N.2's pre-registered branches (REPLICATES / DOES NOT
REPLICATE / INCONCLUSIVE, Amendment N.2) are unchanged and apply as written.
Both caps remain reported per O.3(2): 16384 gave 0.2750 at 0.2583 truncation and
failed the gate; 32768 gives 0.3000 at 0.0750 truncation and passes it at the
floor.

**O.5 confirmed in production.** The R-KV cell launched and is generating on both
GPUs against the llama3-scaled RoPE path. Before the fix it raised
`NotImplementedError` within seconds of every launch.

---

## Amendment N.2, result (2026-08-04): REPLICATES, on a one-generation margin

`grid_aime24_rkv_published` completed 120/120 at the 32768 cap against the
re-validated bank (Amendment O.4). Read as pre-registered in N.2. Artifact:
`results/diag/amendment-k/classb/rkv_published_cap32768.json`.

**The branch was selected by code, not by judgment.** N.2's rule is implemented
in `kvcache.stats.published_config_branch` and applied by
`scripts/report_published_config.py`, which emits the branch as a field of the
artifact. This exists so the read cannot be re-cut after the numbers are seen.

**Result.** R1-Distill-Llama-8B, AIME-24, budget 2048, mean realized cache
0.1886, 30 problems × 4 seeds, matched (problem, seed) population under the
Amendment K convention:

| | full cache | R-KV 2048 |
|---|---|---|
| accuracy | 0.3000 (36/120) | 0.3083 (37/120) |
| truncation @32768 | 0.0750 | 0.2083 |
| accuracy, non-truncated | 0.3243 (n=111) | 0.3895 (n=95) |
| mean tokens | 14799.3 | 18869.7 |

r = **1.0278**. Paired difference (R-KV − full cache) **+0.0083**, problem-
clustered 95% CI **[−0.0750, +0.0917]**, 7 wins / 17 ties / 6 losses over 30
problems. Branch: **REPLICATES** (r ≥ 1.0 and the difference CI includes zero,
N.2 as written).

**What this does and does not license, stated together.** The pre-registered
branch strengthens R-KV's claim and the paper says so plainly. It is also a
one-generation margin — 37 correct against 36 of 120 — on an interval that spans
zero, over 30 problems, against a bank that cleared its N.3 band at exactly the
floor. The defensible statement is that our certified port is **not worse than
full cache at ~19% of it** in the published configuration; the 105% headline is
neither reproduced at that magnitude nor contradicted. We do not upgrade this to
a confirmed >100% regime, and we do not downgrade it to INCONCLUSIVE because we
would prefer that — the rule was written first and it selects REPLICATES.

**Two facts in R-KV's favour, recorded because they are.** R-KV's traces run
longer (18870 vs 14799 mean tokens) and truncate nearly three times as often
(0.2083 vs 0.0750), every truncation scored incorrect; it leads anyway. Among
non-truncated generations it leads 0.3895 to 0.3243.

**An unplanned replication, reported rather than discarded.** The queue relaunched
this cell under its own run name while the manually launched run was in flight,
because the manual launch used different run names (`rkvpub-g0/g1`) and the queue's
progress counter reads the run directory. That duplicate produced 47 rows before
its window closed. They are **not** merged into the cell — that would double-count
problems. On all 47 overlapping (problem, seed) conditions the two independent
processes agree **exactly**: same trace, same canonical answer, same token count,
same correctness. Sampled decoding at T=0.6 reproduced bitwise across processes,
runs and GPU assignments. The wasted GPU time is the cost of the run-name
mismatch and is recorded as an operational error, not a protocol one.

**Consequences.** §2's "adjacent configuration" hedge is replaced by the direct
test; §5's no->100%-regime paragraph reports the cell with its precision; §5's
not-measured item changes from "not tested in its own configuration" to "tested
and under-powered". `tab:audit` is unchanged — what R-KV claims is unchanged.

---

## Amendment P (2026-08-10): Meeting #2 instrumentation — pre-registered reads

Five reads, all written before their computations run. Convention 2 (fixed by the
supervisor at Meeting #2) is binding for every new quantity below:

```
Delta_tot       = Acc_free - Acc_fc
readout_error   = Acc_ro   - Acc_fc          (measured)
residual_fwd    = Delta_tot - readout_error  = Acc_free - Acc_ro
reverse_readout = Acc_free - Acc_rev
residual_rev    = Delta_tot - reverse_readout = Acc_rev - Acc_fc
bracket         = [min(residual_fwd, residual_rev), max(residual_fwd, residual_rev)]
```

**Naming note, recorded rather than acted on.** This repo has never used
Convention 1: there is no `compression_error` anywhere, and no `trajectory_error`
computed as `Acc_free - Acc_rev` (verified by search over all tracked files). The
existing artifact keys are the repo's own names and map one-to-one onto
Convention 2: `readout_change` = `readout_error`, `unexplained_residual` =
`residual_fwd`, `reverse_readout_change` = `residual_rev`, `residual_bracket` =
`bracket`. Renaming the stored keys would change every artifact, invalidate the
29/29 re-derivation baseline and every `paper_numbers.json` source pointer, for
zero semantic gain. **New** artifacts use Convention 2 names; existing keys keep
theirs, with the mapping recorded here and in the docs.

### P.1 The additivity (interaction) test

`readout_error` and `residual_rev` are measured independently of each other and
of `Delta_tot`, so their sum against `Delta_tot` is a real test, not an identity.

**The population rule, which is the whole difficulty.** Amendment K fixed each
term's reference to that term's own population: `readout_error` is measured on
the readout-defined conditions, `residual_rev` on the reverse-defined ones, and
`Delta_tot` on all 400. Those are three different denominators. Summing them and
calling the shortfall an interaction conflates a real interaction with a
population mismatch — the exact defect class Amendment K exists for.

**Therefore the primary estimand is computed on the COMMON population**: the
conditions where the free-run row, the readout row, the reverse-readout row and
the full-cache reference are all defined. All four accuracies are recomputed
there; the interaction is

```
interaction = Delta_tot - (readout_error + residual_rev)
            = Acc_free - Acc_ro - Acc_rev + Acc_fc      (on the common set)
```

Secondary, reported alongside for disclosure: the same quantity under each
term's own Amendment-K population (what the stored artifacts give directly).

**Uncertainty.** Problem-clustered bootstrap, resampling problems jointly across
all four protocols (possible because they share the same problem set), 2000
resamples, seed 0, percentile interval — the repo's standing convention,
unchanged.

**Pre-registered branches.** Only the SIGN is claimed, and only where the CI
excludes zero.

- **SPLIT CONFIRMED** — the interaction is negative for Random and R-KV and
  positive for StreamingLLM and H2O-style, with all four CIs excluding zero.
  Then a third structurally unrelated quantity recovers the
  recoverable-vs-concentrated partition, and it is reported as convergent
  evidence with the other two.
- **PARTIAL** — the sign pattern holds but not every CI excludes zero. Report
  per cell which signs are supported; the partition claim is weakened to the
  cells that carry it.
- **NO STRUCTURE** — most CIs include zero. Then additivity is not detectably
  violated at this sample size, the convergent-evidence claim is **withdrawn**,
  and that is the reported result. This is a real outcome, not a failure to
  measure.
- **CONTRADICTS** — signs exclude zero but do not follow the predicted
  partition. Reported plainly, and the two prior derivations of the partition
  are re-examined rather than this one being discarded.

### P.2 Per-step profile extensions (Task 1)

Computed from the 1200 already-stored profiles; no new generation. Per policy
rather than pooled, over the full horizon (the stored profiles already contain
it; only the rendered figure truncates at +/-400):

- **running integral** of per-step TV, which distinguishes plateau (recovery)
  from continued climb (compounding); per-step TV is memoryless and cannot.
- **eviction-event alignment** — whether spikes coincide with the 64-step prune
  interval and whether they partially decay between prunes.
- **profile-shape classification** — monotone-drift vs spike-and-recover, per
  trace, with the per-policy proportions reported.

**Read.** Descriptive only. These are TV-scale quantities: per Definition 6 they
may not license an accuracy-scale claim by adjacency. Any connection to accuracy
must go through P.3's cross-cell prediction or a matched-magnitude control.

**Confound, stated in advance.** Budget is fixed while context grows, so the
realised compression ratio worsens monotonically within a trace. A rising
profile is therefore consistent with a compression-ratio curve as well as a
horizon curve, and **no mechanistic reading is licensed until the ratio-matched
control (M.3) has run.** The shape statistics are reported before M.3; their
interpretation is not.

**Property to preserve and state.** These profiles are teacher-forced with the
trace held fixed, so they isolate cache-state divergence with no feedback
through sampled tokens. Free-running adds a second divergence source, making
the teacher-forced profile an a fortiori lower bound on free-running divergence.

### P.3 AUC-TV versus the residual (Task 3)

A cross-scale prediction: a TV-scale quantity predicting an accuracy-scale one.
Legitimate as a prediction that can fail; never as an assertion that divergence
explains accuracy.

- **Primary statistic: Spearman rank correlation** between per-cell AUC-TV and
  the per-cell residual. "Proportional" is a strictly stronger claim than
  "correlated"; linearity through the origin is tested only after, and a rank
  correlation is never described as proportionality.
- **Reported pooled AND within-budget.** AUC-TV and the residual both scale with
  trace length and budget, so a pooled cross-cell correlation may be a budget
  effect wearing a mechanism's clothes. If it holds only pooled, it is reported
  as a budget effect.
- **Problem-clustered bootstrap on the correlation**, 2000 resamples, seed 0.
- **n is stated with every coefficient.** The current 3-point version is not a
  correlation and is not reported as one.

**Pre-registered branches.**

- **PREDICTS** — Spearman rho > 0 in magnitude-consistent direction with a CI
  excluding zero, pooled AND within at least one budget.
- **BUDGET EFFECT ONLY** — pooled CI excludes zero, every within-budget CI
  includes it. Reported as a budget effect, not a divergence-to-accuracy link.
- **REFUTED** — CI includes zero pooled and within-budget. **This is a
  substantive finding and is reported as one:** trace-divergence *magnitude*
  would then not be what drives accuracy loss, which is more surprising and more
  useful than the positive result. The mechanism section would be rewritten to
  say that divergence localises *when* damage begins without predicting *how
  much*.

**Prior caution.** At budget 256 AUC-TV spans 0.073-0.086 across three policies
(a 17% spread, no intervals) while residuals span 0.268-0.344. If AUC-TV barely
varies where the residual does, any pooled correlation is carried by the budget
axis. This is expected and is why within-budget is pre-committed.

### P.4 Extended per-step measures (Task 4)

The stored profiles retain only the TV scalar per step (`schema_version:
tv-profile-v1`, fields `estimate/lower/upper` plus `reference_token_ids`) — no
distributions — so JSD, top-1 agreement and rank displacement require a replay
re-run. Verified by inspection, not assumed.

Measures, all on the **softmax next-token distribution only**. Raw logits are
excluded: uncalibrated across positions, arbitrary scale, and a temperature
difference alone would make identical policies look different. Latent-state
divergence is out of scope.

- **TV** — retained for continuity with the published result.
- **JSD** — bounded and symmetric, and unlike KL does not diverge exactly when
  the compressed model commits confidently to a token the full model gave
  near-zero mass. If KL is computed at all, both directions are computed and
  both reported.
- **top-1 agreement** — did the argmax change.
- **rank displacement** — where the full-cache argmax lands under compression.

**Read.** Whether JSD or top-1 localises onset earlier or more sharply than TV.
Descriptive; the same Definition 6 constraint applies.

### P.5 The restoration test (Task 2)

At a fixed step in a compressed teacher-forced replay, re-insert the most
recently evicted K/V entries and measure whether per-step divergence against the
full cache falls.

**Instrument gate, before any divergence is read.** Verify that the reinserted
entries are actually present in the cache after reinsertion and bitwise
identical to what was evicted — a retained-set check in the spirit of the R-KV
Jaccard parity gate. A failed gate means the experiment did not run.

**Three arms.**

| Arm | Design | Reads as |
|---|---|---|
| budget-relaxed | reinsert; cache grows | upper bound, confounded with budget |
| **budget-matched** | reinsert AND evict an equal number of others under the same policy | **the causal test** — does *which* tokens are present matter at fixed budget |
| random-reinsert | reinsert an equal number of randomly chosen previously-evicted entries, budget-matched | separates "these tokens were load-bearing" from "any perturbation reduces divergence" |

The budget-matched arm is the result. The relaxed arm is a reference point and
must not be reported as the result.

**Four arms, not three (added 2026-08-11 per the Meeting #2 reply).** A
`none` arm — the same traces with no intervention at all — is required, because
divergence at the trigger step has a trajectory of its own and some of it would
decay anyway. Every effect is a *paired difference against `none` on the same
traces*, never a before/after within one arm. Without it, "divergence dropped
after reinsertion" is unfalsifiable. `random` is the null for *which* tokens;
`none` is the null for *time*.

**Accuracy under restoration, if the cost permits.** Divergence recovery cannot
license an accuracy claim, and an accuracy-scale restoration result would need
no bridging argument. If it is one extra generation pass it is worth running,
and it is reported as the primary read if it exists.

**Trigger.** Fixed step at 2x the median onset (step 130), pre-committed.
Divergence-triggered selection is *not* used as primary: reinserting where
divergence happened to spike invites regression to the mean, so some decay would
be expected with no causal effect. If a divergence-triggered variant is run it
carries a matched sham-trigger control.

**Fraction reinserted:** one quarter of the budget, stated in advance.
**Policies:** Random and StreamingLLM at minimum — they sit on opposite sides of
the recoverable/concentrated split and should behave differently if the
mechanism is real.

**Pre-registered branches**, on the budget-matched arm, over the N=200 steps
after the trigger:

- **RESTORES** — mean per-step divergence after the trigger falls relative to
  the same trace without reinsertion, CI on the paired difference excluding
  zero, AND the random-reinsert control shows a significantly smaller drop.
- **NON-SPECIFIC** — divergence falls but the random-reinsert control falls as
  much. Then any cache perturbation helps and nothing is shown about *which*
  tokens matter. Reported as such.
- **NO RESTORATION** — no significant drop in the budget-matched arm. Evicted
  information is not recoverable by reinsertion at this budget, which bounds how
  the residual can be described.

---

## Amendment P.1, result (2026-08-10): the interaction partition does NOT survive a common population

Run as pre-registered in P.1 over raw rows, all four budget-256 MATH-500 cells.
Artifacts: `results/diag/amendment-k/classb/additivity/{random,streamingllm,h2o,rkv}-256.json`.

**Primary estimand — common population, problem-clustered CI (2000 resamples, seed 0):**

| cell | n cond | n prob | Delta_tot | readout_error | residual_rev | sum | **interaction** | 95% CI | excludes 0 |
|---|---|---|---|---|---|---|---|---|---|
| Random | 304 | 94 | −0.378 | −0.128 | −0.289 | −0.418 | **+0.039** | [−0.010, +0.088] | **no** |
| StreamingLLM | 364 | 100 | −0.648 | −0.302 | −0.484 | −0.786 | **+0.137** | [+0.060, +0.218] | yes |
| H2O-style | 352 | 100 | −0.636 | −0.287 | −0.489 | −0.776 | **+0.139** | [+0.063, +0.219] | yes |
| R-KV | 340 | 98 | −0.318 | −0.076 | −0.256 | −0.332 | **+0.015** | [−0.029, +0.060] | **no** |

**Branch: PARTIAL, and the predicted partition is not recovered.** Two of four
interactions are distinguishable from zero, both positive, both concentrated
policies — that part matches. But Random and R-KV do **not** show the predicted
negative interaction: their point estimates are *positive* and their intervals
include zero. The claim that the interaction sign "splits exactly along the
recoverable-vs-concentrated line" is **not supported**. What survives is
narrower: *the concentrated policies over-explain; the other two show no
detectable interaction.*

**Why the earlier reading looked different, which is the substantive point.**
The proposed computation summed three terms measured on three different
denominators — `Delta_tot` over all 400 conditions, `readout_error` over the
readout-defined 392, `residual_rev` over the reverse-defined 308–372 — and read
the shortfall as an interaction. That shortfall is partly population mismatch,
the defect class Amendment K exists for. Three computations side by side:

| cell | mixed-reference (as proposed) | per-term Amendment-K populations | **common population (estimand)** |
|---|---|---|---|
| Random | −0.076 | −0.042 | **+0.039** |
| StreamingLLM | +0.140 | +0.142 | **+0.137** |
| H2O-style | +0.135 | +0.146 | **+0.139** |
| R-KV | −0.037 | −0.009 | **+0.015** |

The two concentrated cells are stable across all three. **Random and R-KV flip
sign**, and they are exactly the two cells with the smallest reverse-readout
population (308 and 346 of 400, against 372 and 360 for StreamingLLM and
H2O-style) — because reverse readout is defined only where the *compressed*
trace reaches an answer boundary, and those two cells truncate most (0.233 and
0.135, against 0.070 and 0.105). The apparent partition tracked how much
population each cell lost, not a property of the channels.

**Scope, which must be attached wherever these numbers appear.** The estimand
is *the interaction among conditions where all four protocols terminated* — not
"the interaction". The common population drops every condition where any
protocol is undefined, disproportionately non-terminating traces, so it is an
easier subset: `Delta_tot` for Random falls from −0.463 to −0.378, a 0.085 shift
larger than several effects the paper reports.

**Selection bounded both ways, not absorbed (added 2026-08-11).** The dropped
conditions are scored both ways, Amendment-K layer-1 style, to bound what the
selection could be worth:

| cell | interaction | sampling CI | dropped | **both-ways bound** |
|---|---|---|---|---|
| Random | +0.039 | [−0.010, +0.088] | 96 | [−0.270, +0.210] |
| StreamingLLM | +0.137 | [+0.060, +0.218] | 36 | **[+0.008, +0.188]** |
| H2O-style | +0.139 | [+0.063, +0.219] | 48 | [−0.040, +0.200] |
| R-KV | +0.015 | [−0.029, +0.060] | 60 | [−0.193, +0.108] |

**This narrows the surviving claim again.** Only **StreamingLLM** excludes zero
under *both* sampling uncertainty and the worst-case imputation. **H2O-style's
interaction is robust to sampling but not to selection** — its both-ways bound
spans zero. So the defensible statement is not "the concentrated policies
over-explain" but: *StreamingLLM's two damage channels demonstrably destroy
overlapping problems; H2O-style points the same way without surviving the
selection bound; Random and R-KV show nothing detectable.*

The bound is deterministic and carries no sampling uncertainty; it is a range
over imputations, never compared for width against a CI.

**Consequence.** The additivity test does not currently provide convergent
evidence for the recoverable-vs-concentrated partition. The two prior
derivations (reverse-side endpoints, ADD channels) are unaffected — this
amendment removes a third line of support that was never validly established,
rather than contradicting them.

---

## Amendment P.2, result (2026-08-10): eviction-locked for Random only; and a statistic of mine that was noise

Computed over the 1200 stored profiles, no new generation. Artifacts:
`results/diag/amendment-k/classb/shape/{diss256,h2o256}.json`.

**A statistic was withdrawn before it was reported.** The first implementation
classified a trace as "recovering" when its per-trace phase profile fell by at
least 10% from peak to trough, and reported 85–89% of traces as
"drift+recover" with a median within-cycle decay of 0.75. That statistic is
noise. With 64 buckets of noisy per-step values, peak-minus-trough is dominated
by sampling variation: a **sham control** — the same steps assigned to shuffled
phase labels, destroying all real phase structure — reproduced it almost
exactly (observed median 0.752 against sham 0.737 over 200 Random traces). The
per-trace recovery axis is not measurable at this noise level and has been
removed from the module. Eviction alignment is now tested only on the aggregate
phase profile, always against its sham.

**Aggregate phase profile (mean divergence by position in the 64-step prune cycle):**

| policy | amplitude | sham amplitude | ratio | peak at phase |
|---|---|---|---|---|
| Random | 0.179 | 0.073 | 2.5× | **0 — the prune step** |
| StreamingLLM | 0.106 | 0.065 | 1.6× | 13 |
| H2O-style | 0.101 | 0.063 | 1.6× | 21 |

**Read: eviction events are the proximate trigger for Random, and are not for
the concentrated policies.** Random peaks exactly on the prune step with an
amplitude 2.5× its sham and decays across the cycle. StreamingLLM and H2O-style
peak mid-cycle with barely more amplitude than shuffled labels produce. The
pooled artifact could not show this — it averaged Random and StreamingLLM
together.

**Intervals added 2026-08-11** (trace-clustered bootstrap on the EXCESS over
sham, 2000 resamples, seed 0; observed and sham recomputed together on each
resample so their correlation is preserved):

| policy | amplitude | sham | **excess** | 95% CI | excludes zero | peak phase |
|---|---|---|---|---|---|---|
| Random | 0.179 | 0.073 | **+0.099** | [+0.061, +0.127] | **yes** | **0 — the prune step** |
| StreamingLLM | 0.106 | 0.065 | **+0.038** | [+0.001, +0.056] | yes, barely | 13 |
| H2O-style | 0.101 | 0.063 | **+0.026** | [−0.011, +0.044] | **no** | 21 |

**Revised read, more conservative than the ratios suggested.** Only Random has
eviction-locked divergence that is clearly established: its phase peak sits
exactly on the prune step and the excess is far from zero. StreamingLLM has weak
phase structure whose interval only just clears zero, and its peak is mid-cycle,
so it is not "locked to eviction" in the same sense. H2O-style shows **no
detectable phase structure at all**. The earlier "2.5x vs 1.6x" framing
overstated the concentrated pair; ratios of an amplitude to its sham are not a
test, and the intervals are.

**Scope limit that must travel with this.** R-KV — the other policy on the
recoverable side of the taxonomy — has **no replay profiles at all** (the replay
runs cover Random, StreamingLLM and H2O-style only). So the proposed
correspondence "recoverable policies are eviction-locked, concentrated policies
diverge on a lag" currently rests on one clear case per side plus one marginal
case. It is a hypothesis this data is consistent with, not a demonstrated
partition. An R-KV replay run would be the direct test and is not queued.

**Trace-level growth (robust; quarter-means over hundreds of steps):**
monotone-drift in 86–90% of traces, median last-quarter/first-quarter ratio
3.94 (Random), 2.77 (StreamingLLM), 3.74 (H2O-style).

**Running integral** of per-step TV shows no plateau — the accumulation *rate*
climbs monotonically from 0.0013/step at onset+50 to 0.079/step at onset+1600.
**Survivorship caveat:** only 269 of 400 traces reach +1600, and longer traces
diverge more, so the late points are computed on a longer-trace subpopulation.
The climb is therefore not cleanly separable from selection at this stage.

**Per-policy AUC-TV** (needed for P.3): Random 0.0857, StreamingLLM 0.0829,
H2O-style 0.0730 — consistent with the three previously published values.

**Interpretation remains withheld.** Per P.2 these are TV-scale quantities and
the ratio-matched control (M.3) has not run, so no mechanistic reading of the
rising profile is licensed: it is still consistent with a compression-ratio
curve rather than a horizon curve.

---

## Amendment P.2, replication (2026-08-12): Random holds, StreamingLLM does not, and R-KV enters

The extended replay pass (`replay-ext-diss256`, 800 v2 profiles: 25 problems x 4
seeds x 4 presses x budgets {256, 2048}) is an **independent sample** from the
one the 2026-08-10 read used (100 problems x 4 seeds, budget 256 only). It also
closes that read's stated scope limit: **R-KV now has replay profiles.**

**Budgets must not be pooled.** The first pass of this report pooled 256 and
2048 and was wrong to: budget 2048 barely compresses these traces (AUC-TV 0.013
vs 0.084-0.108 at budget 256; 86% of traces classified flat, against 87-96% drift
at 256), so pooling dilutes any eviction-locked signal toward zero. The pooled
Random excess read +0.105 and StreamingLLM +0.020; the split below is the
readable one. `scripts/report_profile_shape.py` gained `--budget` for this, and
its output now records the filter so a pooled number cannot be mistaken for a
single-budget one.

**Phase-amplitude excess over sham, budget 256** (n = 100 profiles per press;
prior single-budget values from `results/diag/amendment-k/classb/shape/`):

| press | excess, this sample | 95% CI | peak phase | prior excess | prior CI | prior peak |
|---|---|---|---|---|---|---|
| Random | +0.1068 | [+0.0395, +0.1315] | 0 | +0.0991 | [+0.0614, +0.1271] | 0 |
| StreamingLLM | +0.0195 | [-0.0212, +0.0499] | 9 | +0.0384 | [+0.0013, +0.0560] | 13 |
| H2O-style | +0.0293 | [-0.0306, +0.0480] | 24 | +0.0257 | [-0.0106, +0.0436] | 21 |
| R-KV | +0.0620 | [+0.0016, +0.0775] | 9 | not measured | -- | -- |

**Budget 2048: the phenomenon is absent for all four presses.** No excess
excludes zero, and three of four point estimates are negative (Random -0.0538,
StreamingLLM -0.0644, R-KV -0.0350, H2O-style +0.0127).

**Reads.**

1. **Random replicates cleanly** on an independent sample, same peak at phase 0
   (the prune step). This is the one eviction-locked result the record carries.
2. **StreamingLLM does not replicate.** Its prior interval excluded zero by
   0.0013 -- a borderline result -- and the new sample's does not. Per the
   standing rule that a pre-committed outcome is never re-cut in our favour, the
   2026-08-10 StreamingLLM claim is **withdrawn**, not averaged with this one.
   The honest statement is that StreamingLLM's phase locking is not established.
3. **H2O-style is consistently null**, both samples, as before.
4. **R-KV excludes zero but marginally** (lower bound +0.0016, the same knife
   edge that StreamingLLM's prior result sat on and then fell off). With eight
   press x budget comparisons in this family, one marginal exclusion is the
   expected yield under no effect. **R-KV is recorded as suggestive, not
   established**, and specifically must not be used to claim the recoverable /
   concentrated partition: its peak phase is 9, not 0.
5. The partition hypothesis from 2026-08-10 is therefore **weaker**, not
   stronger, after adding the policy it most needed. One clear case (Random),
   one withdrawn (StreamingLLM), one null (H2O-style), one marginal with the
   wrong peak (R-KV).

**Unchanged:** interpretation stays withheld pending M.3, and all of this is
TV-scale (Definition 6).

---

## Amendment P.3, result (2026-08-12): BUDGET EFFECT ONLY

`scripts/report_auc_residual.py` over the 800 v2 profiles, Spearman rank
correlation with a problem-clustered bootstrap (2000 resamples, seed 0), against
the catalogued per-cell residuals.

* **Pooled across budgets:** rho = **-0.857**, CI [-0.893, -0.643], 7 cells,
  excludes zero. (rkv-2048 skipped: no catalogued residual.)
* **Within budget 256:** rho = **-0.200**, CI [-0.400, +0.800], 4 cells. Does
  not exclude zero.
* **Within budget 2048:** rho = -1.0, CI [-1.0, +1.0], 3 cells. Three points is
  the script's floor and the interval is the whole range; this is not evidence.

This is the pre-registered **BUDGET EFFECT ONLY** branch. The pooled correlation
is carried by the contrast between budgets -- budget-256 cells have both large
AUC and large residuals, budget-2048 cells have both small -- and does not
survive holding budget fixed. The within-budget tests are also underpowered
(4 and 3 cells), so the correct statement is *not established either way at this
n*, not *refuted*.

**Answer to the work order's Task 3 as asked:** AUC-TV is **not** shown to be
proportional to the subtraction residual. Two separate reasons, both of which
must travel with any restatement: (a) rank correlation is not proportionality --
that is a strictly stronger claim (linearity through the origin) which was not
tested; (b) what pooled correlation there is, is a budget effect.

---

## Amendment P.4, result (2026-08-12): the crossing test is degenerate by construction

`scripts/report_onset_localization.py` over the 800 v2 profiles.

**The pre-registered comparison cannot distinguish TV from JSD, and the reason
is structural rather than empirical.** In teacher-forced replay the compressed
and full-cache distributions are *identical* until the first prune, so pre-onset
TV is exactly 0.0 and its spread is exactly zero. The crossing rule's zero-spread
fallback then reduces "first crossing" to "first non-zero step". TV and JSD are
non-zero on exactly the same steps -- verified directly on a profile: identical
zero/non-zero pattern at all 10,769 steps -- so their crossings coincide
necessarily. The medians agree to the step (192 at budget 256, 1920 at 2048,
both exact multiples of the 64-step compression interval) and the means differ
in the third decimal (138.26 vs 138.25) only through trace-length rounding.

**No conclusion about JSD's onset sensitivity may be drawn from this test**, in
either direction. Recording it as "JSD localises onset the same as TV" would be
reporting an identity as a measurement.

**Rank displacement does differ, and in the opposite direction to the one
predicted.** Median first crossing 280-451 steps after onset versus 192 for
TV/JSD, i.e. it is **coarser** at onset, not sharper. `divergence.py`'s docstring
asserted the opposite ("Sharper than top-1 for localising onset"); that claim was
written from the mechanism, not from data, and is corrected. The mechanism is
sound but its sign was wrong: an integer rank does not move at all until a
perturbation is large enough to reorder tokens, so it is *insensitive* early.

**What would answer the actual question**, if it is worth answering: a
scale-free formulation -- for each trace and measure, the first step reaching a
fixed fraction of that trace's own late-trace level -- which compares the
*shape* of the rise rather than its support. That is a post-hoc design, chosen
after seeing that the pre-registered one is degenerate, and it is therefore
**exploratory and must be labelled so** if it is ever run. It is not run here,
and the P.4 verdict stands as written above.

---

## Amendment P.5, instrument gate (2026-08-12): PASSED, and two design faults it caught

`restoration-smoke` (3 problems x 1 seed x 2 presses x 4 arms = 24 conditions),
run 2026-08-12 12:43-13:26 remote. Artifact
`results/diag/amendment-p/restoration_smoke_diag.json`.

**The gate passes: 24/24 reports, zero failures.** Every treated arm reinserted
entries **bitwise identical** to what was evicted (`max_abs_difference` 0.0 in
all cases). `matched` and `random` hold the cache at exactly 256; `relaxed`
grows to exactly 320 = 256 + 64 reinserted, which is what makes it the
budget-confounded reference arm. The powered cell is therefore unblocked.

**Live-path validation, over and above the gate.** On a checked trace the arms
are bit-identical up to step 256 and first differ at **step 257** -- exactly one
step after the intervention, which is where a cache edit can first reach the
distribution -- then differ materially per step (0.0567 / 0.0990 / 0.0817 for
none / matched / relaxed at the same step). Pre-intervention identity is what
makes the paired design valid, and it is now observed rather than assumed.

**Fault 1: the pre-registered trigger step is not the step the intervention
lands on.** `TRIGGER_STEP = 130` was pre-committed as 2x the median onset. The
press can only intervene during a compression event that already holds a full
stash, so the observed `applied_at_step` is **192 or 256 depending on the
trace** -- it tracks prompt length, not the nominal trigger. Anchoring the read
window at 130 would place a *trace-dependent* stretch of pre-intervention steps
inside the window. Those steps contribute exactly zero to the paired difference
(the arms are identical there), so the effect would be diluted by a factor that
differs per trace: a heterogeneous estimand, not merely a noisy one.
`restoration_read` now takes `applied_steps` and anchors per trace, records
`anchored_on` and the applied-step range, and `report_restoration.py` drops any
trace whose treated arms disagree on the step rather than pooling them onto one
anchor. In the smoke the arms agree within every trace, so no trace is dropped.
**This is a correction to how P.5 is READ, not to what it does**; the
intervention itself is unchanged and `TRIGGER_STEP` retains its meaning as the
earliest step at which intervention is permitted.

**Fault 2 (mine, in the runner's guard).** The post-prune budget check was given
a relaxed-arm allowance but the between-prune check was not, so the first launch
died on `between-prune cache exceeded expected bound`. The relaxed arm inflates
both peaks. Fixed and the allowance applied to both.

**Not a result.** Per this amendment, divergence from the smoke is not read. The
reader was nonetheless exercised end-to-end on it to prove it runs, and its n=3
output is recorded as a diagnostic only: `NO RESTORATION` in both cells, with
all effects within +/-0.001 of zero. That is three traces and carries no
evidential weight; it is noted only so that a near-zero powered result is not
mistaken for a late surprise. The powered cell `restoration-diss256` (800
conditions, both GPUs) launched 13:29 remote the same day.

---

## Amendment P.5, correction (2026-08-13): the powered cell is INVALIDATED by an index-space bug in the arm design

`restoration-diss256` completed 800/800 on 2026-08-13 and was read. **The read is
withdrawn and the 800 profiles are invalidated for the arm comparison.** They are
kept as permanent record under their original run name, and the corrected cell
runs under a new one; nothing is resumed into, renamed, or deleted.

**The numbers that were produced** (archived, explicitly named invalid, at
`results/diag/amendment-p/restoration_diss256_INVALID_indexbug.json`; effects are
`none` minus arm, so negative = divergence INCREASED):

| cell | matched | random | relaxed | verdict as read |
|---|---|---|---|---|
| random-256 | -0.0012 [-0.0018, -0.0007] | **-0.2484** [-0.3328, -0.1747] | -0.0014 [-0.0021, -0.0008] | NO RESTORATION |
| streamingllm-256 | -0.0014 [-0.0021, -0.0009] | **-0.2443** [-0.3279, -0.1710] | -0.0014 [-0.0021, -0.0008] | NO RESTORATION |

**What triggered the investigation:** the `random` control was ~200x the size of
`matched`. A control arm that dwarfs the treatment it controls for is a claim
about the instrument, not about the mechanism, so it was checked before being
reported.

**Root cause.** `plan_reinsertion` chose the eviction set with

```python
pool = [p for p in retained if p not in set(reinsert)]
```

where `reinsert` indexes the **stash** of previously evicted entries and
`retained` indexes the **live cache** -- two independent 0-based index spaces in
which a shared number means nothing. The intent ("do not evict something you are
reinserting") is vacuous: a stashed entry was evicted earlier and is not in the
live cache, so it can never be selected. The line's actual effect was to delete
from the eviction pool whichever live positions numerically coincided with the
reinserted stash indices.

**Measured consequence.** `matched` reinserts the highest stash indices (the most
recent evictions), so the exclusion barred it from evicting any cache position
above 191: **0 of its 64 evictions fell in the most recent quarter of the cache,
against 18 of 64 for `random`** (~16 expected under uniform sampling; verified
directly, stash = cache = 256, n = 64). Cache index tracks recency, so the arms
differed in two ways at once -- which entries returned, and which were dropped to
make room. `random`'s -0.248 is therefore substantially a *we evicted recent
entries* effect and cannot be read as *these tokens were not load-bearing*, which
is the only thing that arm exists to test. `matched`'s eviction set was likewise
a biased subset, not "an equal number of others under the same policy" as this
amendment specifies, so its number is affected too. The whole cell goes.

**Fix, and a design strengthening it forced.**

1. `pool = list(retained)` -- no cross-space exclusion; every live position stays
   eligible.
2. The eviction set is now drawn from an **independent RNG stream** (`evict_rng`),
   seeded identically in every arm. Previously both draws came from one stream,
   so an arm that had already consumed draws choosing what to reinsert landed on
   a different eviction set. With the fix, `matched` and `random` evict the
   **same** entries and differ only in which entries return -- which is what
   makes `random` a control at all. This is a strengthening of the pre-registered
   design, not a change of estimand: the arm definitions in this amendment are
   unchanged, the implementation now matches them.

Regression tests: `tests/test_restoration.py` asserts that no position is
structurally excluded from the eviction pool and that arms sharing a seed evict
identical sets. An older test that asserted the buggy invariant (built on
deliberately overlapping index spaces) is replaced, with its premise recorded.

**What survives.** The instrument gate is unaffected: `retained_set_report`
checks that reinserted entries are bitwise identical to what was stashed, which
is independent of the selection policy. Its 24/24 pass and the live-path
validation (arms bit-identical to step 256, first differing at 257) stand. The
smoke's n=3 diagnostic is withdrawn along with the powered read.

**Pre-registered read for the re-run is unchanged** and is restated here so it
cannot be adjusted after the fact: RESTORES iff `matched` is positive and
excludes zero and exceeds `random`; NON-SPECIFIC iff `random` reproduces it;
NO RESTORATION otherwise. The corrected cell is `restoration-diss256-v2`.

---

## Amendment M.3, correction (2026-08-14): the ratio-matched control was a constant-32 cell; INVALIDATED

`replay-ratio-matched-m3` completed 200/200 on 2026-08-14. **The run is
invalidated: it did not implement the control it names.** Evidence archived at
`results/diag/amendment-p/m3_INVALID_floor_collapse.json`.

**What it actually ran.** Across **all 200 traces and all 377,104 compression
events, the chosen `target_size` was 32 -- the `min_budget` floor -- 100% of the
time.** The largest target ever chosen was 32; the largest cache ever observed
was 408 entries. The requested ratio was 0.038; the realized median ratio was
0.333. So the cell held a **constant, very small absolute budget**, which is
precisely the property M.3 exists to remove, and it is more aggressive than the
budget-256 cells it was meant to be compared against.

**Root cause: a feedback loop.** The press computed

```python
context = int(keys.shape[2])
base.target_size = max(min_budget, round(ratio * context))
```

`keys.shape[2]` is the CURRENT CACHE SIZE -- itself the output of the previous
compression -- not the context the model would have had. So: compress to 32,
accrue one 64-step interval to 96, `round(0.038 * 96) = 4` -> floored to 32,
compress to 32, forever. The cache can never grow, so the ratio can never be
realized. The intended quantity is the logical context (prompt + everything
generated so far).

**Why no test caught it.** The unit test's fake base returned its input
unchanged, so the cache never shrank and the loop could not form. A test whose
fake cannot exhibit the failure mode is not a test of that failure mode. It has
been replaced with a fake that actually evicts to `target_size`, and the new
test reproduces the collapse on the old code (target 250 -> 32 on the first
event).

**Fix.** Everything ever evicted was once in the cache, so
`logical = cumulative_evicted + current_cache` reconstructs the full context
without the press needing to know the prompt length. Eviction is accumulated
**per layer**, since `compress` fires once per layer per event and a single
counter would multiply the total by the layer count. The trace now records
`logical_context` and an explicit `at_floor` flag per event, so a future floor
collapse is visible in the artifact itself rather than requiring this analysis.

**Consequence for the reading.** M.3's pre-registered read is UNCHANGED and is
restated so it cannot be adjusted later: if the divergence staircase survives at
fixed ratio, the horizon reading stands and a mechanistic reading is licensed;
if it flattens, the profile is a compression-ratio effect and section 4.4 is
restated in those terms. **That read has still not been performed.** Until the
corrected cell runs, no mechanistic reading of any divergence profile is
licensed -- including the Amendment P.2 finding that Random is eviction-locked,
whose phase-0 peak is exactly what a compression-ratio curve would produce.

The corrected cell is `replay-ratio-matched-m3-v2`. The invalid 200 profiles are
permanent record under their original run name: not resumed into, not renamed,
not deleted.

### Amendment M.3-v2, read condition (2026-08-15, written before the cell completes)

The corrected press works -- on the first v2 trace `logical_context` rises
116 -> 1012 and `target_size` varies (32-38) instead of pinning at the floor.
But a floor of 32 at ratio 0.038 only stops binding once the logical context
reaches 32/0.038 ~= 842 tokens. **Below that the cell is a fixed-32 budget whose
realized ratio WORSENS as context grows -- the exact confound M.3 exists to
remove.** On that first (short) trace, 336 of 420 events sat at the floor and
the realized median ratio was 0.0567, not 0.038.

`min_budget` is NOT being changed: it is pre-registered at 32, and altering a
protocol parameter after seeing data is the thing this document exists to
prevent. Instead the read is conditioned, and the condition is fixed here before
the cell finishes:

**The M.3 comparison uses only steps where the press was above the floor**
(`at_floor == False` in the ratio trace). Steps at the floor are reported as a
coverage figure, never silently pooled. Measured on the budget-256 profiles of
the extended pass, the threshold is ~step 692 for a typical ~150-token prompt;
**24 of 25 traces exceed it and 83.9% of all steps qualify**, so the restriction
is a coverage caveat rather than a power problem.

If the above-floor coverage of the completed v2 cell falls materially below that
83.9% estimate, the read is reported as underpowered rather than forced.

**The verdict itself is unchanged:** staircase survives at fixed ratio -> the
horizon reading stands and a mechanistic reading is licensed; staircase flattens
-> the profile is a compression-ratio effect and section 4.4 is restated in
those terms.

---

## Amendment P.5, result (2026-08-15): NO RESTORATION, on the corrected arms

`restoration-diss256-v2`, 800/800 conditions, 200 per seed. **Instrument gate:
800/800 pass, zero failures** -- every treated arm bitwise identical to what was
stashed (`max_abs_difference` 0.0), `matched`/`random` holding the cache at 256,
`relaxed` at 320. Zero anchor disagreements: within every trace the treated arms
landed on the same step, so one anchor per trace is valid. Artifact
`results/diag/amendment-p/restoration_diss256_v2.json`.

Effects are `none` minus arm over a 200-step window anchored per trace on the
applied step (192 or 256), so **positive = divergence reduced**:

| cell | matched (the causal test) | random (specificity control) | relaxed (confounded reference) | verdict |
|---|---|---|---|---|
| Random-256 | **-0.0013** [-0.0019, -0.0007] | -0.2485 [-0.3330, -0.1748] | -0.0014 [-0.0021, -0.0008] | **NO RESTORATION** |
| StreamingLLM-256 | **-0.0015** [-0.0022, -0.0009] | -0.2442 [-0.3278, -0.1709] | -0.0014 [-0.0021, -0.0008] | **NO RESTORATION** |

Baseline (no-intervention) mean divergence over the same window: 0.0383
(Random), 0.0445 (StreamingLLM).

**The pre-registered answer to Meeting #2 Task 2 is NO.** Reinserting the most
recently evicted K/V entries does not reduce divergence. The `matched` effect is
*negative* -- divergence very slightly INCREASED -- and excludes zero, though at
0.0013-0.0015 against a baseline of 0.038-0.045 it is ~3% of the divergence
being explained and is better read as "indistinguishable from no help" than as
a real harm. The `relaxed` arm, which is allowed to grow the cache, does no
better, so the null is not an artifact of holding the budget fixed.

**The specificity control carries the interesting part.** Reinserting the same
*number* of entries chosen randomly from the evicted pool -- with the SAME
entries dropped to make room, which is what the corrected implementation
guarantees -- increases divergence by 0.245-0.249, roughly six times the
baseline divergence. So the cache is acutely sensitive to *which* entries occupy
it, while restoring the specific entries a policy just evicted is nearly
neutral. Eviction damage, at this budget and horizon, is not undone by putting
the tokens back.

**A prior hypothesis of mine was wrong, and the record should show it.** When
the v1 cell was invalidated (2026-08-13) I attributed `random`'s large effect to
the index-space bug, which had barred `matched` from evicting recent entries (0
of 64 in the top quarter, against 18 for `random`). The corrected run reproduces
the same aggregate to four decimals (v1 `random` -0.2484 vs v2 -0.2485; v1
`matched` -0.0012 vs v2 -0.0013) while **every single per-trace profile differs**
(0 of 25 identical per cell checked). So the fix did take effect and did change
the trajectories; the eviction confound simply was not what produced the effect.
The invalidation was still correct -- with arms differing in two ways the
contrast was uninterpretable regardless of how the numbers landed -- but the
mechanism I proposed for it was not.

**Scope.** One model, MATH-500, budget 256, two policies, a single intervention
at one trigger with `reinsert_fraction` 0.25 and a 200-step read window. A null
here is a null for that intervention, not for reinsertion in general: a larger
fraction, a different trigger, or repeated reinsertion are untested.

---

## Amendment M.3, result (2026-08-15): the staircase is a COMPRESSION-RATIO effect

`replay-ratio-matched-m3-v2`, 200/200 conditions on the corrected press.
Artifacts: `results/diag/amendment-p/{m3_verdict,m3v2_coverage,
shape_m3v2_UNRESTRICTED}.json`.

**The pre-committed read condition is satisfied.** Above-floor coverage is
**83.5%** of compression events against the 83.9% predicted on 2026-08-15 before
the cell finished; per-trace median coverage 74.2%; realized ratio on above-floor
events exactly **0.038**, and the largest target chosen was 626, so the budget
genuinely tracked context this time.

**Verdict, on the restricted and paired window** (each trace trimmed to where
its own press left the floor; the fixed-budget counterpart cut at the SAME
absolute step; drift = last-quarter mean / first-quarter mean; problem-clustered
bootstrap, 2000 resamples, seed 0; 24 problems and 92 observations per press;
median 75.9% of each trace retained; excluded: 8 profiles never left the floor,
8 windows shorter than 200 steps):

| press | drift, FIXED budget 256 | drift, RATIO-matched 0.038 | branch |
|---|---|---|---|
| Random | **1.201** [1.061, 1.339] | **0.662** [0.614, 0.709] | FLATTENS -> ratio effect |
| StreamingLLM | **1.299** [1.100, 1.520] | **0.739** [0.680, 0.805] | FLATTENS -> ratio effect |

**This is the pre-registered "flattens" branch for both policies.** Section 4.4
is to be restated in compression-ratio terms: the rising divergence profile is
what a monotonically worsening ratio produces, not evidence of a horizon
mechanism. **No mechanistic reading of the rising staircase is licensed.**

Three things must travel with this result.

1. **What was actually manipulated.** Holding the ratio fixed *is* letting the
   absolute budget grow with context (32 -> 626 here). "Fixed ratio" and
   "growing absolute cache" are the same manipulation by construction, so the
   honest statement is: the staircase disappears when the budget grows in
   proportion to the context. It is not a demonstration that horizon length is
   irrelevant holding cache size fixed -- that is a different experiment.
2. **The restriction shrinks the staircase it is testing.** Over the whole
   trace the fixed cell drifts 2.93 (Random) and 2.26 (StreamingLLM); over the
   comparable above-floor window it drifts 1.20 and 1.30. Trimming removes the
   early trace where most of the rise happens. The staircase is still present
   and its interval still excludes 1, so the comparison stands, but it is a
   milder staircase than the headline figure.
3. **The matched cell does not merely flatten -- it reverses** (drift 0.66-0.74,
   below 1). Divergence *falls* across the trace when the budget grows with
   context. That is consistent with the ratio being the driver, and is the
   direction a growing absolute cache would produce.

**An aggregation bug in the verdict script was caught before this was reported.**
The first run keyed observations by problem id alone, so each of the four seeds
overwrote the previous one and three quarters of the data was silently dropped.
On that reduced sample Random read `drift_fixed = 1.085 [0.923, 1.247]` -- an
interval spanning 1 -- and the script correctly refused it as "no staircase in
the fixed cell over this window -> not readable". With all 92 observations the
same cell reads 1.201 [1.061, 1.339]. The verdict above uses the full data; the
refusal branch worked as designed on the partial data.

**Consequence for Amendment P.2.** The eviction-locking finding (Random's phase
excess peaking at the prune step) was flagged on 2026-08-12 as possibly being
what a ratio-driven curve would produce. That question is now testable and is
NOT yet answered here: the phase statistics computed on this cell
(`shape_m3v2_UNRESTRICTED.json`: Random excess +0.326 [0.263, 0.339] peak 0;
StreamingLLM +0.102 [0.065, 0.128] peak 6, both excluding zero) were computed
over the WHOLE trace including the floor-bound period, which is the harshest
compression in the run and would inflate phase amplitude. **They are recorded as
preliminary and must not be cited as evidence that eviction-locking survives
ratio matching** until the phase analysis is repeated on the same restricted
window as the drift comparison above.

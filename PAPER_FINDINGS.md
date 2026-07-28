# Paper findings — consolidated evidence base (for drafting)

This is the authoritative, self-contained record of all results, framing rules, and honest
limitations for the paper. All numbers are final unless marked *(in progress)*.

## 1. The problem we resolve
Two literatures contradict each other on KV-cache compression for reasoning models:
- **KV-compression method papers** report *near-lossless* accuracy, but on **readout-shaped
  evidence** (see the audit table below).
- The **HOTT benchmark** (arXiv:2512.12008, *"Hold Onto That Thought"*) evaluates the same
  methods **free-running on reasoning** and reports **method-specific** severe degradation.

**IMPORTANT framing correction (verified against the papers 2026-07):** do NOT say "HOTT
reports collapse" broadly. HOTT (MATH-500, DeepSeek-R1-Distill-Qwen-7B, Table 2, budgets
128/256/384/512) finds StreamingLLM collapses (**0.03** at budget 128, from full-cache **0.47**)
while heavy-hitter/prefill-aware methods survive (H2O 0.14, **SnapKV 0.38** at 128); its verbatim
conclusion is *"heavy-hitter methods H2O and SnapKV-D win over all budgets and datasets for
reasoning models."* The **sharpest, cleanest contradiction** is StreamingLLM: certified
near-lossless on its own **perplexity** evaluation, yet **0.03** free-running on HOTT — that
single method IS the readout-vs-trajectory gap in the wild, and our decomposition explains the
mechanism. Reconcile the **channel structure**, NOT the digits: HOTT full-cache MATH-500 = 0.47
vs our 0.882 (different decode-length/sampling), so no digit-for-digit merge. HOTT also
independently notes eviction makes models *"more talkative, generating longer sequences"* —
corroborates our Random termination-mass / truncation finding. **HOTT does NOT test random
eviction** (its methods are Full/ShadowKV/H2O/KNorm/R-KV/SnapKV/StreamingLLM), so our
Random-beats-recency (+0.185) result is our OWN contribution — do NOT imply HOTT corroborates it.
HOTT does evaluate R-KV free-running (relevant to the audit table / related work).

**Evaluation-protocol audit (verified):**

| Method | Venue | arXiv | Primary near-lossless evidence | Trace fixed? | Free-run reasoning acc? |
|---|---|---|---|---|---|
| StreamingLLM | ICLR'24 | 2309.17453 | LM perplexity (PG19, streaming to 4M tok) | Yes | No |
| H2O | NeurIPS'23 | 2306.14048 | few-shot task acc (lm-eval suite), summ (XSUM/CNN), throughput | Mostly | No |
| SnapKV | NeurIPS'24 | 2404.14469 | LongBench, Needle-in-a-Haystack retrieval | Yes | No |
| R-KV | NeurIPS'25 | 2505.24133 | **MATH-500, AIME-24 free-running**; 105% of full-KV @ ~16% cache | No | **Yes** |

R-KV is the **counterexample** to "methods evaluate readout-style": it evaluates free-running
reasoning and claims **gains** (105% of full-KV @ ~16% cache, R1-Distill-Llama-8B, AIME-24).
That claimed >100% regime is exactly what our in-progress R-KV/AIME cells test — NOT something we
claim to have explained. Do not lump R-KV with the readout-evaluated methods.

Our contribution is a **measurement/analysis paper** (not a new compression method) that
explains *why*, by decomposing the accuracy change under compression into a **readout** part
and a **trajectory** part.

## 1b. Reproducibility facts (verified from repo/runs 2026-07)
- **Bootstrap:** problem-clustered (resamples problems, not rows), **2000 resamples**, percentile
  CI (2.5/97.5 indices). ADD permutation null = **10,000** resamples, Benjamini-Hochberg across
  problems.
- **Truncation rate @256:** Random **0.233** (93/400), StreamingLLM **0.070** (28/400) — Random
  truncates 3.3x more, corroborating its larger ADD termination-mass shift.
- **Exact decomposition split (post-Amendment-K):** readout-gap **0.181**, residual-gap
  **0.008**. SLM readout share = **0.492 -> 49.2%**. The residual gap is the number that
  moved most in relative terms (it was 0.004 against the mismatched reference), so the
  "the whole +0.185 lives in readout" sentence is now "0.181 of the 0.185".
- **Compute:** **280 GPU-hours** of generation (summed wall-clock over **7,200** generations)
  under the inclusion rule recorded in `results/diag/amendment-k/classb/compute.json`:
  free-running and reference-bank generation only, excluding teacher-forced work (readout,
  reverse readout, replay), superseded pilots, and the two shards subsumed by the merged
  bank. Every directory is listed there with its hours and an inclusion flag.
- **Repo:** github.com/whyeventry0103/kv-caching (ANONYMIZE for submission).
- **EXACT REPLICATION (verified 2026-07-04):** the original budget-256 grid
  (grid-math500-dissociation-v1) and the token-ID re-run (grid-diss256-tok) agree on ALL
  800 rows — identical per-row correctness, truncation, AND n_tokens → generation is
  deterministic under our seeding; quote as "an independent re-run reproduced all 800
  rows exactly." (In the paper: Threats/instrument-validity.)

## 2. Thesis (the spine — commit to this framing)
> **Compression damage to reasoning is real, large, and largely *invisible to teacher-forced /
> readout-style evaluation* — which is why method papers (that evaluate on held-fixed traces)
> report near-lossless, while free-running benchmarks (HOTT) report collapse. The gap is the
> trajectory residual, and it arises from a divergence that is ~zero at the moment of eviction
> and *compounds* over the generated trajectory.**

Do NOT frame around ">100% / denoising": we did not observe any accuracy gains on small-budget
MATH, so that regime is explicitly out of scope for the headline (see §6).

## 3. Setup
- **Model:** DeepSeek-R1-Distill-Qwen-7B. **Benchmark:** MATH-500. Sampling T=0.6, top-p=0.95.
- **Full-cache reference:** 250 problems × n=8 = 2000 generations, **FC accuracy 0.882**.
- **Compression cells:** 100 problems × n=4, budget **256** (the collapse pole); budget **2048**
  *(in progress)*.
- **Methods (eviction presses):** StreamingLLM, Random (per-(problem,seed) RNG), H2O
  (ObservedAttentionPress; eager attention) *(in progress)*, R-KV (parity-verified port;
  cells pending) *(in progress)*.
- **Stack:** transformers 4.56.2, kvpress 0.5.3; decoding-time `DecodingPress`,
  `compression_interval=64`, `target_size=budget`.
- **Estimands:**
  - *total change* = free-run accuracy − FC accuracy.
  - *on-reference readout* = teacher-force the FC reasoning trace to the reasoning/answer
    boundary, then generate the answer span under compression. *readout change* = readout
    accuracy − FC.
  - *residual* = total − readout (labeled "accuracy change unexplained by on-reference
    readout"; **a definition, not a measured cause** — see §5).
  - *reverse readout* = the *compressed* free-run trace read out under the *full* cache; brackets
    the on-policy residual together with forward readout.
  - *ADD* = total-variation between coarsened FC vs compressed answer distributions
    (termination-mass vs within-answered split; permutation null; signed flip-mass).
  - *TV profile / TDP* = per-step total variation between full and compressed teacher-forced
    replay; onset = first prune step.
- **Grader:** Hendrycks `is_equiv` (no eval/sympify on model text). Exact token-ID sidecars for
  bitwise-faithful replay; single-BOS tokenization verified by decode↔encode round-trip.

## 4. Results (final unless marked)

### 4-CURRENT. Post-Amendment-K headline table — READ THIS ONE

Everything below §4-CURRENT is a **dated read-in log**, not current truth. Each entry
is correct as of its date and several exist specifically to show a superseded value
beside its correction. Those are the honesty trail; they are never rewritten. When you
want *the current number*, use this table or `paper_numbers.json`, which is generated
from the artifacts and gates the `.tex`.

Reference discipline (Amendment K): the bank is 0.882 over 2000 rows and is quoted in
the setup section only. **Every Δ, readout change, and reverse endpoint below is
measured against 0.885** — the same bank restricted to the cell's 400 matched
conditions. The third value (100×8, 800 rows) is provenance only, never a comparator.

| method @256 | free | readout (95% CI) | total Δ | readout Δ | residual (95% CI) | primary bound | reverse ep | readout share | trunc |
|---|---|---|---|---|---|---|---|---|---|
| Random | 0.422 | 0.765 [0.686, 0.817] | −0.463 | −0.138 | **−0.337** [−0.404, −0.262] | [−0.348, −0.328] | −0.282 | 29.8% | 0.233 |
| StreamingLLM | 0.237 | 0.584 [0.495, 0.647] | −0.647 | −0.319 | **−0.344** [−0.405, −0.270] | [−0.355, −0.335] | −0.470 | 49.2% | 0.070 |
| H2O-style | 0.247 | 0.597 [0.502, 0.667] | −0.637 | −0.306 | **−0.344** [−0.403, −0.270] | [−0.357, −0.337] | −0.478 | 48.0% | 0.105 |
| R-KV | 0.520 | 0.798 [0.710, 0.850] | −0.365 | −0.105 | **−0.268** [−0.335, −0.195] | [−0.282, −0.262] | −0.251 | 28.7% | 0.135 |

`primary bound` is the Amendment K layer-1 interval: the population is fixed at all 400
conditions and rows with no answer boundary are scored both ways. **Its width is
exactly the cell's undefined fraction**, so always read it next to `trunc` — a wide
bound means many undefined rows, not a weak effect.

**A primary bound is NOT a confidence interval and must never be compared for width
against one.** It is deterministic and contains no sampling uncertainty; the CI beside
it is a variance statement. At low truncation the bound is necessarily narrower, which
is arithmetic, not evidence. A bound excluding zero means the point estimate is negative
under every imputation — it is not significance. Every inferential claim in the paper
rests on the complete-case CI.

Other current values that moved and are easy to re-introduce stale:
- Pure answer-resampling effect (FC readout of FC traces, matched): **+0.003**.
  Residuals are 132× (Random) / 105× (R-KV) that — *two* orders of magnitude.
  (The superseded unmatched figure is in the archival log, not here.)
- Dose-response factor 256 vs 2048: **4.4–10.2×**.
- Budget-2048 residuals: **−0.033 / −0.079 / −0.059**, CIs [−0.065, −0.010] /
  [−0.125, −0.042] / [−0.102, −0.020], all excluding zero.
- AIME residuals: **−0.118 / −0.194 / −0.280** (Random / R-KV / SLM). AIME readout Δ is
  **exactly 0.000** for all three presses.
- Policy contrast CI: **[0.130, 0.240]** at the pinned seed. The published upper bound
  came from a seed that was never recorded; the artifact now pins seed, resample count,
  and a difference histogram sufficient to recompute the interval.
- Compute: **280 GPU-hours over 7,200 generations** under the artifact's inclusion rule.

Superseded values are listed at the top of the archival log below.

### 4-LOG. Dated read-in log (archival — not current truth)

<!-- AUDIT:ARCHIVAL-BEGIN -->
<!-- Everything from here to AUDIT:ARCHIVAL-END is the dated read-in log: a
     historical record whose numbers are correct as of their dates. The number
     audit skips this span deliberately. Do NOT "update" these entries -- several
     exist to show a wrong value beside its correction, which is the point. New
     results append a new dated subsection; corrections go in §4-CURRENT and in a
     PREREGISTRATION amendment. The heading above must stay ABOVE this marker:
     the 4-CURRENT gate ends the guarded span at the next heading, so a marker
     placed first would fall inside the section it is meant to sit outside. -->

**Superseded values, listed so they are recognised if they resurface — do not reuse.**
Dose factor 3.6–6.8×. Resampling effect +0.023. Residuals −0.343 / −0.347 / −0.349 and
R-KV −0.278. Reverse endpoints −0.246 / −0.466 / −0.463 / −0.221. Readout shares
25/46/45/23%. Residual spread 0.006, readout-change spread 0.169. Policy-contrast upper
bound 0.242 (200-seed range [0.2375, 0.2450]). Compute ~224–230 GPU-h / ~7,307
generations. Budget-2048 residuals −0.051 / −0.096 / −0.075. AIME residuals −0.179 /
−0.246 / −0.321. Truncation-adjusted −0.110 / −0.277.

### 4a. The powered phenomenon (budget 256)
FC 0.882 → **StreamingLLM 0.237**, **Random 0.422**. Random beats the "principled" recency
policy by **+0.185, 95% CI [0.130, 0.242]**; per-problem difference **≥ 0 on 94/100 problems
(44 strict wins, 50 ties, 6 losses — verified from rows 2026-07-04; say "non-negative on
94/100", NEVER "beats on 94/100", because half the 94 are ties and Fig 1's histogram makes
that visible)**. (n=100 problems × 4 seeds, problem-clustered bootstrap.)

### 4b. Forward decomposition (readout vs trajectory), 800/800 conditions
| method @256 | free | readout (95% CI) | total Δ | readout Δ | residual | readout share |
|---|---|---|---|---|---|---|
| Random | 0.422 | 0.765 [0.686, 0.817] | −0.460 | −0.117 | **−0.343** | 25% |
| StreamingLLM | 0.237 | 0.584 [0.495, 0.647] | −0.645 | −0.298 | **−0.347** | 46% |

Key facts: (i) a large trajectory residual (~−0.34) exists for **both** methods; (ii) the
entire +0.185 Random-vs-SLM gap lives in the **readout** channel (readout Δ differs by 0.181;
residual differs by 0.004) → **trajectory damage is method-invariant** (report as exploratory).
Undefined reasoning/answer boundary: 8/cell, balanced.

### 4c. Reverse-replay bracket (the off-policy defense), 800/800
**CORRECTED 2026-07-03 (bracket-computation bug; PREREGISTRATION Amendment H).** The reverse
endpoint is the reverse change ITSELF (acc_rev − acc_FC = total − on-policy-readout-change);
the old code used total − (acc_rev − acc_FC), which equals the on-policy READOUT change, not a
residual estimate. Reverse readout (compressed trajectory read out under FULL cache): **Random
0.636, SLM 0.417** (vs FC 0.882) — clean readout does NOT recover to FC, so a large part of the
loss is genuinely the trajectory. Corrected **residual bracket: Random [−0.343, −0.246], SLM
[−0.466, −0.347]** — even farther from 0 than the old (wrong) [−0.34,−0.21]/[−0.35,−0.18], so
the residual **survives the off-policy correction with room to spare**. Nuance changed by the
correction: the forward residuals are nearly equal (−0.343/−0.347) while the reverse estimates
separate (−0.246 vs −0.466 — SLM more trajectory-damaged on the reverse side; the expected
"forward inflates the residual" bias ordering holds for Random but REVERSES for SLM). The
point-estimate brackets barely separate (gap 0.004) and no sampling uncertainty is propagated to
the endpoints, so the *method-level* split is still reported as **bracketed and exploratory**,
and "method-invariant trajectory damage" is a **forward-estimand** statement. Mild confound:
differential defined-reverse counts (Random 308 vs SLM 372 — Random's rambling traces more often
lack a clean answer boundary). (Exact decimals CONFIRMED 2026-07-03 by re-running report_m4 with
the fixed code on the remote: SLM reverse endpoint is −0.466 — the earlier −0.465 came from the
rounded 0.417 accuracy. Fig 2 is generated from this report in DATA mode.)

**ENDPOINT CIs (2026-07-04, closes the "no sampling uncertainty on the bracket" gap):**
problem-clustered bootstrap (2000 resamples) on each bracket endpoint — forward
**[−0.402, −0.258]** (Random) / **[−0.403, −0.268]** (SLM); reverse **[−0.383, −0.213]** /
**[−0.553, −0.410]**. ALL four exclude zero. Forward endpoints statistically
indistinguishable (method-invariance stands); reverse endpoints separate with
NON-OVERLAPPING CIs (SLM significantly more trajectory-damaged on the reverse estimand)
→ attribution is estimand-dependent; still reported as bracketed/exploratory. Archived:
`results/diag/m4-powered-diss256-summary.json`; fields `unexplained_residual_ci95`,
`reverse_residual_ci95` in report_m4.

### 4d. Figure 4 — the mechanism (TV profiles), 800/800
**Estimator (defined 2026-07-04, script-reproducible via `make_paper_figures.py
--print-windows`):** table entries are means of PER-TRACE window means (each trace counts
once; matches the problem-clustered convention). The step-pooled later mean is LARGER
(0.1182) because longer traces diverge more — quote 0.0884 as the conservative per-trace
estimator, never mix the two.
Onset-aligned per-step TV (800 profiles, median first-eviction at step 65 = compression interval):
| window relative to first eviction | mean per-step TV |
|---|---|
| before eviction | **0.0000** |
| first 50 steps after | 0.0018 |
| later steps (>50 after) | **0.0884** |
Divergence is **zero before eviction, near-zero immediately after, then compounds** to ~0.09.
This is the mechanism: trajectory damage is a *slow-compounding drift*, not an instantaneous
readout error — which is exactly why single-step/readout evaluation misses it. mean AUC-TV 0.086
(Random) vs 0.083 (SLM) — matched across methods, corroborating method-invariant trajectory
damage. (The zero pre-eviction TV also validates the replay instrument.)

### 4e. ADD — silent answer-distribution distortion (budget 256)
| method | mean ADD | within-answered TV | termination shift | signed flip (net) |
|---|---|---|---|---|
| StreamingLLM | 0.751 | 0.749 | +0.081 | −0.646 |
| Random | 0.559 | 0.504 | +0.224 | −0.461 |
Both far above the noise floor. SLM's divergence is almost entirely *within-answered* (it keeps
answering, but a different, wrong answer); Random's has more *termination* mass (it rambles /
fails to terminate — consistent with its higher truncation). So the two methods damage the answer
distribution differently even where accuracy is similar.
**H2O-style row (read 2026-07-06): ADD 0.715, within-answered 0.713, termination +0.101,
flip net −0.636 (gained 0.000)** — patterns with StreamingLLM (concentrated pair). In paper
Table 3 + §4.5 + Fig 3.
**Dose response (add-2048, read 2026-07-06): Random ADD 0.085 (≈ the 0.068 floor),
SLM 0.146 (net flip −0.076)** → distortion tracks the dose; the pre-registered
silent-flip-at-flat-accuracy regime did NOT materialize at 2048 — reported honestly in §4.5.
Archived: runs harvest/add-2048 + harvest/add-256-h2o (remote) and stdout JSONs.

### 4f. Two-full-cache negative control (instrument validity)
Splitting the FC bank into two disjoint seed halves and running ADD between them: **mean ADD
0.068, median 0.000, mean permutation p 0.976, 0% of problems significant.** The ADD noise floor
(~0.07) is far below the compression cells (0.56–0.75) → ADD does not false-positive.

### 4g. R-KV port integrity — GATE COMPLETE (2026-07-05)
The R-KV port reproduces the authors' retained-token sets exactly (Jaccard 1.0 at budgets
{128,256,384,512}, synthetic tensors). The real-generation smoke (Amendment E's last gate)
initially **FAILED at Jaccard 0.805** — the press wrapper was silently running SnapKV's
window/kernel (64/5) instead of R-KV's (8/7) due to dataclass field shadowing; the parity
harness could not see it, the smoke caught it (the gate worked). Fixed (dataclass subclass +
regression test + true 8-query window with analytic RoPE + per-event self-check), re-run:
**PASS, Jaccard 1.0000**, marker written -> rkv executable; cells queued (256 grid/readout/
reverse + AIME-2048). Sell the two-stage catch as methodology, never hide it (Amendment I).

### 4h. H2O-style FORWARD DECOMPOSITION IN — method-invariance extends to 3 policies (2026-07-05)
readout-diss256-h2o 400/400 (report archived: results/diag/m4-h2o256-forward-summary.json):
**free 0.2475 [0.180, 0.318], readout 0.597 [0.503, 0.668], totalΔ −0.635, readoutΔ −0.286,
residual −0.349 [−0.403, −0.270], share 45%, trunc 0.105, flip net −0.6375 (gained 0.000),
n_readout_undefined 8 (same 8 FC rows as the other cells).** THE result: forward residuals
across three mechanistically unrelated policies = −0.343 / −0.347 / −0.349 (spread 0.006,
CIs mutually overlapping, all excluding zero) while readout changes span 0.169 → trajectory
damage method-invariant on the forward estimand; readout channel carries ALL method
differences. In the paper: Table 2 third row (dagger footnote: ObservedAttention ≠ H2O,
8192 cap), abstract, §4.1/§4.2, threats. **REVERSE BRACKET IN (2026-07-06): rev acc 0.4194 (151/360 defined; 40/400 undefined),
rev change −0.463, bracket [−0.463, −0.349], rev CI [−0.566, −0.420]** (archived:
results/diag/m4-h2o256-full-summary.json). Structure: H2O-style twins StreamingLLM on the
reverse side (−0.463 vs −0.466, CIs overlap) and both separate from Random (−0.246) with
non-overlapping CIs → reverse side splits UNIFORM vs CONCENTRATED, not method vs method.
Bias ordering reverses for both concentrated policies. Imputation bounds: endpoints
−0.505 (all-incorrect) … −0.405 (all-correct) — zero excluded under any imputation.
Figures: Fig 1 H2O bar + Fig 2 third panel DONE (3-hue palette validated, #CC79A7).

### (superseded preliminary note, kept for history) — free-run @256 (2026-07-04)
grid-diss256-h2o 400/400: **free-run 0.2475, truncation 0.105, mean 4,359 tokens**
(100 problems × 4 seeds; audited from rows). Third eviction policy collapses — worse than
Random (0.422), at StreamingLLM's level (0.237). **LABELING LANDMINE (do not step on):**
our press is kvpress ObservedAttentionPress — the accumulated-attention heavy-hitter
signal WITHOUT the recent-window reservation of the actual H2O paper (and of HOTT's H2O,
which SURVIVES free-running). So this does NOT contradict HOTT and must never be written
as "H2O collapses" — always "H2O-style / ObservedAttentionPress (no recency reservation)".
Positive reading (hypothesis-level): pure attention-score selection collapses like pure
recency selection while uniform Random does best of the three → *concentrated* allocations
(positional OR score-based) underperform uniform coverage at this budget; HOTT's surviving
H2O combines both components. Decomposition (readout-diss256-h2o) runs tonight — only then
does the method-invariance test on a third method read out. Tempered claim if it lands
clean: we predicted ObservedAttentionPress behavior, not H2O's.

### 4i. Budget-2048 free-run (the near-lossless pole) — READ 2026-07-04
grid-diss2048-tok 800/800: **Random 0.850 (trunc 0.030, mean 3,454 tok), StreamingLLM
0.8075 → displays 0.807 per the §5 display convention (trunc 0.028, mean 3,650 tok)**
vs FC 0.882. Free-running at 8× budget is within
0.03–0.07 of the reference → (a) the collapse is a BUDGET effect, (b) doubles as a harness
control (free-running per se isn't broken), (c) the Random>SLM ordering PERSISTS at 2048
(+0.042). In the paper: §4.1 + Fig 1 dose bars. readout-2048 (decomposition) queued.

### 4j. m4-2048 — dose-response DECOMPOSITION IN (read 2026-07-07)
readout-diss2048 800/800 joined grid-diss2048-tok 800/800; FC 0.8825. **Random@2048:
free 0.8500 [0.7825, 0.9125], readout 0.9005 [0.82, 0.9425], residual −0.0505
CI [−0.065, −0.01]**, trunc 0.030, 8 undefined readout boundaries, flip net −0.035.
**SLM@2048: free 0.8075 [0.735, 0.87], readout 0.9031 [0.8225, 0.9425], residual
−0.0956 CI [−0.1275, −0.0425]**, trunc 0.028, 8 undefined, flip net −0.0775.
Readout at 2048 is statistically indistinguishable from (nominally above) the
reference — the remaining free-run loss is effectively ENTIRELY trajectory-channel.
Residuals shrink by ×3.6 (SLM) to ×6.8 (Random) vs the −0.343/−0.347 values at 256,
with CIs still excluding zero. Exact residuals derivable from counts: Random
−0.0325 − (353/392 − 0.8825) = −0.05051; SLM −0.075 − (354/392 − 0.8825) = −0.09556.
Branch taken: A (both residuals in [−0.10, 0]) with one template correction — the
playbook's "order of magnitude below" was replaced by "factor 3.6–6.8" (SLM's
−0.096 vs −0.343 is 3.6×, not 10×). In the paper: new §4.2 dose-response paragraph;
Setup 2048 status; §4 preamble; §4.1 clause; threats-generality; conclusion
extensions list. Abstract unchanged (256-led, per standing decision).
Archived: results/diag/m4-2048-summary.json (fetch in flight — link down at read
time; digest values above are report-exact from harvest_results.py output).

### 4k. m4-2048-h2o — H2O-style dose point IN (read 2026-07-10)
grid-diss2048-h2o 400/400 + readout-diss2048-h2o 400/400; FC 0.8825. **H2O@2048:
free 0.8225 [0.7525, 0.8875] (displays 0.823), readout 0.8980 [0.82, 0.9375]
(displays 0.898; 352/392, 8 undefined), residual −0.0755 CI [−0.1025, −0.02]
(displays −0.075, [−0.102, −0.020])**, trunc 0.085, flip net −0.0625. Free ≥ 0.75 →
playbook Branch "behaves alike": near-lossless free-run, readout ≥ reference,
residual sits BETWEEN Random (−0.051) and SLM (−0.096), factor 4.6 below its −0.349
at 256 (inside the paper's 3.6–6.8 range — no text change needed). Dose response now
holds for ALL THREE policies. In the paper: one clause in the §4.2 dose paragraph;
§4 preamble, threats-generality, conclusion extensions updated (H2O fully done).
Archived: results/diag/m4-2048-h2o-summary.json

### 4l. replay-256-h2o — TV profiles for the 3rd policy IN (read 2026-07-10)
replay-diss256-h2o 400/400. **mean AUC-TV 0.0730 (displays 0.073), median onset 65**
— |0.073 − 0.085| = 0.012 ≤ 0.02 → playbook Branch A: §4.4 sentence extended to
"matched across all three policies (0.086 / 0.083 / 0.073)". STOP-rule check PASSED:
before-window 0.0000 per-profile AND pooled (26,000 pre-eviction steps) — instrument
valid on the third policy. Windows (H2O only, NOT merged into Table 3 which stays
R+SLM 800): first50 0.0008, later per-profile 0.0765 / pooled 0.1023 (estimators
kept separate per §5 discipline). Fig 4 unchanged (R+SLM data).
Archived: results/diag/replay-256-h2o-auc.json, replay-256-h2o-windows.txt

### 4m. m4-rkv-256 — R-KV full decomposition IN, Branch C (read 2026-07-12)
grid/readout/reverse-diss256-rkv all 400/400; FC 0.8825. **R-KV@256: free 0.5200
[0.435, 0.61] (displays 0.520 [0.435, 0.610]) — BEST free-runner of the four;
readout 0.7985 = 313/392 (displays 0.798 [0.710, 0.850]); residual −0.2785 CI
[−0.335, −0.195] (displays −0.278); reverse 0.6618 = 229/346 (displays 0.662);
bracket [−0.278, −0.221]; revCI [−0.3554, −0.2025] (displays [−0.355, −0.203]);
trunc 0.135; 8 fwd-undefined; 54 rev-undefined; flip gained 0.0025 / net −0.365.**
Imputation (derived from counts): all-incorrect → 229/400 − FC = −0.310;
all-correct → 283/400 − FC = −0.175 — bracket excludes zero under ANY imputation.
Branch taken: **C** (free 0.520 in the 0.30–0.55 intermediate band — 0.03 short of
Branch B's 0.55) — row + bracket reported plainly, NO framing overhaul, figures
unchanged per playbook. Structure found: (a) forward residual nominally smaller
than the −0.34 trio, CIs OVERLAP → we claim neither equality nor separation;
method-invariance stays scoped to the original trio; (b) reverse side: R-KV lands
with Random on the RECOVERABLE side (revCIs overlap each other, both separated
from the concentrated pair with non-overlapping CIs) → §4.3 split renamed
"recoverable vs concentrated"; (c) no gains (gained 0.0025) → >100% threat
paragraph gets the datum; AIME remains the direct test of the published claim.
In the paper: Table 2 fourth row + ‡ footnote (certified-port-at-7B labeling);
§4.2 new R-KV paragraph; §4.3 all lists extended (eight endpoints); abstract one
clause + "every endpoint"; contributions bullet 4; summary-of-findings clause;
Setup fourth press; §5 gate tense (both gates passed); threats generality
(four policies, gate cleared) + >100% datum; related work tense; conclusion
opening clause + extensions list. Fig 1/Fig 2 UNCHANGED (Branch C rule).
Archived: results/diag/m4-rkv-256-summary.json

### 4n. add-256-rkv + figure refresh for the 4-policy grid (read 2026-07-12)
report_add over grid-diss256-rkv (400 rows) vs the FC bank: **R-KV ADD 0.45875
(displays 0.459) — lowest of the four; within-answered 0.4499 (0.450);
termination shift +0.14875 (+0.149); flip gained 0.00375 (0.004) / net −0.36375
(−0.364).** Ordering SLM 0.751 > H2O 0.715 > Random 0.559 > R-KV 0.459 — less
distribution damage in step with higher accuracy; no denoising anywhere.
In the paper: Table 3 fourth row; §4.5 R-KV sentence; §4.5 + Fig-3 caption split
renamed "recoverable-vs-concentrated" (consistency with §4.3).
FIGURES refreshed for the 4-policy grid (user-requested; supersedes the Branch C
figures-unchanged default): Fig 1 = 8 bars (R-KV sky #56B4E9 + H2O-2048 tint
#E5BCD3; budget-group gap; palette validated: worst adjacent CVD ΔE 15.7–20.7,
≥12 floor; contrast WARN relieved by direct labels), DATA-mode histogram regen
on remote; Fig 2 = 2×2 ladders, top row recoverable (Random, R-KV), bottom row
concentrated (SLM, H2O) — layout encodes the reverse split; DATA-mode from the
three archived 256 summaries; Fig 3 = 4 groups. All three eyeballed. NOTE for
successors: MC/_cells_from_summary now know "rkv"→"R-KV"; C["R-KV"] carries the
full constant set.
Archived: results/diag/add-256-rkv-summary.json

### 4o. m4-llama-256 — MIXED, fits no branch, reported plainly (read 2026-07-15)
All four Llama runs complete; own bank FC 0.805 (161/200, gate-passed 07-14).
**SLM@256-Llama REPLICATES the central result: free 0.2350 [0.15, 0.32] (Qwen:
0.2375!), readout 0.5077 (displays 0.508), residual −0.2727 (−0.273) CI [−0.345,
−0.175], bracket [−0.426, −0.273], revCI [−0.523, −0.327] — zero excluded
throughout; n_rev 198; trunc 0.010.** **Random@256-Llama DEGENERATES: free 0.0000
(0/400), trunc 0.805, readout 0.0205 (0.021), residual −0.021 with bracket
[−0.021, +0.028] SPANNING ZERO; only 6/400 defined reverse rows; flip lost
0.805.** INSTRUMENT CHECK before interpretation (out-of-shape rule): eyeballed
Random-Llama traces = fluency collapse into '.\\n\\n' filler loops (tens of
thousands of chars), NOT rambling reasoning and NOT garbage-token breakage; SLM
cell on the identical harness/bank is sane → real behavioral finding. Mechanism
hypothesis (labeled as such in the paper, cite xiao2023streamingllm): uniform
eviction removes attention-sink tokens the Llama family depends on; the
sinks-preserving policy is protected. CONSEQUENCES, reported plainly (fits
neither playbook branch — per the no-forcing rule): (a) central claim (large
readout-invisible residual) replicates cross-architecture on the sharpest case;
(b) random-beats-recency is ARCHITECTURE-DEPENDENT — reverses on Llama (0.000 vs
0.235); (c) a near-zero residual at floor accuracy is uninformative (both
channels floored — no accuracy left to attribute); never cite Llama-Random as
"small residual". In the paper: new threats Cross-architecture-check paragraph;
scope sentence → "one primary model (subset-scale Llama-base check below)";
§4-preamble/conclusion status flips. Abstract/§4 untouched (subset scale stays
in threats).
Archived: results/diag/m4-llama-256-summary.json

### 4p. fc-aime24-acc — AIME-24 reference bank (read 2026-07-15)
240/240 rows (30 problems × 8 seeds): **accuracy 0.3708 (displays 0.371), trunc
0.2208 (0.221), mean 10,415 tokens; ratio 2048/mean = 0.197.** Slightly below
the playbook's soft 0.4–0.7 sanity band; mechanism visible in the truncation
number: non-truncated subset ≈ 0.371/0.779 ≈ 0.476, and published 0.555 is at
32K decode vs our 16K cap over much longer AIME traces. Same-direction harness
gap as MATH-500 banks (Qwen 0.8825 vs 0.928; Llama 0.805 vs 0.891) → accepted
as reference, NOT a STOP. FC_AIME slot for the m4-aime-2048 read = 0.371; note
the 0.221 in-denominator truncation when interpreting AIME free-run cells.
Archived: results/diag/fc-aime24-bank.json

### 4q. fc-readout-control — resampling control IN, Branch A (read 2026-07-17)
readout-fc-control 400/400 (full-cache readout of the FC bank; the control queued
by the 07-08 submission audit): **readout_accuracy 0.9056 = 355/392 (displays
0.906), 8 undefined boundaries (same 8 rows as every readout run on this bank
subset — consistent).** Acc_ro(FC) − Acc_FC = **+0.0231 (displays +0.023)** —
the pure answer-resampling effect of the readout protocol, measured directly with
zero compression. Inside the playbook band [0.88, 0.92] → Branch A. Ratios:
14.8× below the trio residuals (0.343), 12.0× below R-KV's (0.278) — "an order
of magnitude below" is exactly true. Also confirms the mechanism of the
above-reference 2048 readout values (0.898–0.903). In the paper: the §4.2
calibration sentence upgraded from the 2048-derived bound to the direct
measurement. The "readout conflates compression fidelity with resampling
variance" reviewer attack now has a measured answer.
Archived: results/diag/fc-readout-control.json


### 4r. m4-aime-2048 — Branch A + two bonus findings (read 2026-07-18)
grid-aime24-2048 360/360 + readout-aime24-2048 360/360 vs fc-aime24 (FC 0.3708,
trunc 0.221, mean 10,415 tok; ratio 19.7%). 30 problems × 4 seeds = 120/cell.
**(1) NO >100% REGIME (the headline question): best free Random 0.2833 (displays
0.283) [0.15, 0.425]; R-KV 0.2167 (0.217) [0.10, 0.358]; SLM 0.1417 (0.142)
[0.042, 0.258] — all ≪ FC 0.371; gained flip ≤ 0.050.** Per Amendment E: a
scale/protocol finding, not a port failure. Trunc: R 0.308 / RKV 0.450 / SLM
0.233 (in-denominator as always).
**(2) RESIDUAL REPLICATES ON A 2ND BENCHMARK: −0.1790 (−0.179) CI [−0.349,
−0.076] / −0.2457 (−0.246) [−0.419, −0.139] / −0.3207 (−0.321) [−0.509, −0.202]
(R/RKV/SLM) — all CIs exclude zero.**
**(3) READOUT IS LITERALLY PRESS-INDEPENDENT AT THIS DOSE: all three presses
readout 0.4624 (0.462) [0.327, 0.661], IDENTICAL correctness vectors on all 120
keys.** Investigated BEFORE reading (invariant-violation rule): NOT a bug —
generations differ (only 30/120 share first-40 tokens), but 85/93 defined rows
converge to the same extracted answer (e.g. three different texts all reaching
\\boxed{204}), and on the 8 answer-divergent rows every press is wrong. The 27
undefined keys are shared by construction (boundary is a reference-trace
property). Readout 0.462 > FC 0.371 — same above-reference resampling pattern
as MATH-2048, consistent with §4q's +0.023 calibration.
Branch taken: A (playbook) + the two extra findings reported plainly per the
no-forcing rule. In the paper: threats >100% paragraph rewritten with the
result; related-work tense + audit-table caption; §4 preamble + conclusion
extensions now name only the scale axis. Abstract unchanged (MATH-500-led per
playbook either-branch rule; n=30 — no per-problem claims, clustered CIs only).
Archived: results/diag/m4-aime-2048-summary.json
ALSO: fc-q15b bank GATE PASSED same morning — 200/200, acc 0.800 (band
[0.60, 0.85], published 0.839), trunc 0.04.

### 4s. m4-q15b-256 — scale axis, 1.5B cell IN, both presses REPLICATE (read 2026-07-20)
grid/readout/reverse-readout-diss256-q15b complete (400/400 each) on
DeepSeek-R1-Distill-Qwen-1.5B against its own pinned bank (fc-q15b, acc
0.800, gate passed 2026-07-18). Amendment J pre-committed branch check, both
presses:
- **StreamingLLM-256: free 0.220 [0.130, 0.315], readout 0.495 [0.378,
  0.617], residual −0.275 CI [−0.362, −0.193], bracket [−0.418, −0.275],
  reverse CI [−0.550, −0.358], trunc 0.150 → REPLICATES.**
- **R-KV-256: free 0.355 [0.240, 0.465] (again the stronger free-runner, as
  at 7B), readout 0.688 [0.566, 0.806], residual −0.333 CI [−0.439, −0.230],
  bracket [−0.333, −0.262], reverse CI [−0.443, −0.235], trunc 0.210 →
  REPLICATES.**
Both CIs and both brackets exclude zero; truncation for both cells is far
below the 0.5 degenerate threshold. This is 1 of 2 scale-axis cells
(Amendment J); m4-q14b-256 (14B) is still running (0/400 all three stages as
of this read) — do NOT apply the ≥3/4-replicate scope-upgrade sentence
("one model family across three scales…") until the 14B cell lands; that
edit is pre-drafted in RESULTS_PLAYBOOK.md and fires only then.
In the paper: new threats paragraph "Scale-axis check (subset scale,
Amendment J, in progress)" added after the Cross-architecture paragraph
(`kv_readout_vs_trajectory_paper.tex` §threats); Generality paragraph's
stale "AIME-24 is queued" corrected to "AIME-24 is in" (it was already read
2026-07-18, §4r — the sentence had gone stale) and a scale-axis landing
clause added. No figure change (per FIGURES_PLAYBOOK.md — subset-scale
units are threats-only unless ≥3/4 replicate and the story earns a
small-multiples panel; decide once the 14B cell is in).
Archived: results/diag/m4-q15b-256-summary.json (harvest digest; copy after
this session's `harvest_results.py` run).
ALSO (infra note): overnight of 2026-07-19→20 was the first full window
under the `shard_seeds(total_gpus=...)` fair-share fix — both GPUs ran in
lockstep at 34–48% util from 01:45 through past 09:00 (verified in
`logs/nightly_autorun.log`), unlike the single-GPU-idle night the fix was
written for. Fix confirmed working under real load, not just the unit test.

### 4t. m4-q14b-256 — scale axis, 14B cell IN, Amendment J COMPLETE; 3/4 REPLICATE (read 2026-07-24)
grid/readout/reverse-readout-diss256-q14b complete (400/400 each; last job in
the 31-job nightly queue — the queue is now fully drained) on
DeepSeek-R1-Distill-Qwen-14B against its own pinned bank (fc-q14b, acc 0.880,
band [0.82, 0.96] published 0.939 → gate PASSED). `fails: 0` on every stage;
R-KV self-check (RuntimeError on RoPE-convention drift, `src/kvcache/presses/
rkv.py:209`) did NOT fire — verified by grepping the grid log for
error/traceback/abort (zero hits) and confirming full 200/200 rows landed —
so this is a genuine result, not an instrument abort.
- **StreamingLLM-256: free 0.240 [0.150, 0.335], readout 0.503 [0.395,
  0.600], residual −0.263 CI [−0.335, −0.170], bracket [−0.485, −0.263],
  reverse CI [−0.590, −0.385], trunc 0.000 → REPLICATES cleanly.**
- **R-KV-256: free 0.670 [0.535, 0.795] (again the stronger free-runner —
  persists across all three scales tested), readout 0.848 [0.750, 0.922],
  residual −0.178 CI [−0.273, −0.085] (excludes zero) — but bracket
  [−0.178, 0.029] SPANS ZERO, the first non-degenerate cell in the paper
  where it does. reverse CI [−0.131, 0.029]. trunc 0.230.**
  Root cause of the span: reverse-readout accuracy is 0.909, *above* the
  0.880 reference (reverse-readout change +0.029). This is the same
  magnitude as the +0.023 pure answer-resampling effect measured directly
  in §4j/§4a-dose (full-cache readout of full-cache traces, zero
  compression) — i.e. the bracket's positive edge is consistent with known
  resampling noise on a readout channel that is already near-lossless at
  this scale (0.848 vs 0.880 reference, only −0.032 readout loss) rather
  than evidence the trajectory effect reverses sign. Not floored (free
  0.670, trunc 0.230 ≪ 0.5), so it does not fit DEGENERATE either.
  **Per Amendment J's binding wording ("CI AND bracket exclude zero" for
  REPLICATES), this cell does not qualify — it falls under "anything else":
  reported verbatim, no compensation, no forcing into either branch.**
Scorecard across all four scale-axis cells (§4s + this unit): **3 of 4
REPLICATE outright** (SLM-1.5B, R-KV-1.5B, SLM-14B); **1 of 4 inconclusive
under the bracket** (R-KV-14B) — not contradictory, not degenerate, just
statistically unresolved at n=50×4 with a near-lossless readout channel.
Per Amendment J: ≥3/4 replicate → **scope-upgrade sentence now applies**:
"the readout-invisible trajectory residual holds across one model family
spanning three scales (1.5B/7B/14B), plus an independent Llama-base check."
Applied in the paper.
In the paper: Scale-axis threats paragraph rewritten in full (both models,
all four cells, R-KV-14B exception spelled out with the resampling
comparison) — `in progress` → `complete` in its own label. Generality
paragraph: scope-upgrade sentence added; stale "subset-scale ... checks
below" parenthetical removed (redundant with the explicit enumeration that
follows); "scale-axis check is landing, one of two cells in" → "complete on
both". §4 preamble (line ~196): "in progress" → "All experimental cells
reported in this paper ... are final." Conclusion "Near-term extensions in
progress" paragraph rewritten past tense: full grid is in, 3/4 scale cells
replicate, 4th reported as inconclusive-not-contradictory. Abstract/title
unchanged (headline was already scoped to 7B/budget-256; scale axis lives
only in threats+conclusion per the playbook's original design — this was
never meant to enter the headline).
No figure change: this is the second of the two subset-scale checks and
still doesn't clear FIGURES_PLAYBOOK.md's bar (visual finding worth its own
panel) — the 3/4-replicate story is a sentence, not a chart; the one
partial exception (R-KV-14B) is better served by the resampling-comparison
prose than by a bar that would need its own caveat anyway.
Archived: results/diag/m4-q14b-256-summary.json.
**ORIGINAL CAMPAIGN + AMENDMENT J: 100% READ. Nightly queue fully drained —
no more scheduled runs.** Remaining work is the pre-planned endgame only
(anonymization scrub, restore double-blind author block, arXiv upload,
ICLR conversion) per `docs/strategy/SUBMISSION_STRATEGY.md` Phase 11.

<!-- AUDIT:ARCHIVAL-END -->

## 5. Framing discipline (non-negotiable — apply throughout)
- **Divergence ≠ damage.** ADD / TV / residual are "behavioral-fidelity loss," never "damage."
  Correctness is the only quality arbiter. Carry the **signed flip-mass** (gained vs lost correct
  answers) with every divergence quantity.
- **Residual is a definition, not a cause.** "Accuracy change unexplained by on-reference
  readout." The reverse-replay *bracket* is what licenses the (bounded) causal reading; state the
  off-policy/covariate-shift caveat up front, not in rebuttal.
- **Truncated generations stay in every denominator, scored incorrect** (never dropped). Report
  truncation rate per cell.
- **Method-invariance and any generality claims are exploratory** at current scope (one model,
  one benchmark, budget 256).
- Report the pre-registered outcome honestly: the decomposition read was pre-committed
  (confirm/complicate/refute) and came out **COMPLICATE** — the central residual is robust; the
  strong pilot "SLM=readout / Random=trajectory" dissociation did **not** hold cleanly.
- **Terminology update (2026-07-17, AC-plan reframe): "method-invariant" is retired.**
  Say "a residual of similar magnitude exists for every policy tested" (existential,
  exploratory-labeled) — the invariance wording invited an estimand-dependence attack
  the reverse split makes partially valid. The suite of estimands is now NAMED: "the
  trajectory audit" (title, abstract, contribution bullet 1, §6 ¶2) — it is the
  paper's stated deliverable, released as the pinned harness.
- **Display convention (pinned 2026-07-07): every 3-decimal number is `f"{v:.3f}"` of the
  report-artifact float** — the string numpy/R reproduce from raw rows. Do NOT hand-round
  exact ties up. This rule is now enforced structurally rather than by discipline:
  `make_paper_figures.py` loads every constant from `paper_numbers.json` and formats it
  at render time, so a hand-rounded tie has nowhere left to enter.
<!-- AUDIT:ARCHIVAL-BEGIN -->
  Historical instances caught under this rule, kept because they are how the rule was
  learned: H2O-style free 0.248 → 0.247 (99/400 = 0.2475) and SLM@2048 free 0.808 →
  0.807 (323/400 = 0.8075), both found by eyeballing Fig 1's hand-rounded constant
  against Fig 2's DATA-mode value for the same quantity. Four more were found in Unit 3
  (2026-07-27): −0.498/−0.428/−0.313/−0.178 hand-rounded where `.3f` gives
  −0.497/−0.427/−0.312/−0.177.
<!-- AUDIT:ARCHIVAL-END -->

## 6. Limitations / threats to validity (write this as a STRENGTH section)
- **Generality (primary):** results are one model / one benchmark / budget 256 with two eviction
  methods. Budget-2048 (dose-response), H2O, R-KV, and AIME are *in progress* to widen this.
- **No >100% regime observed** → the paper is framed as "near-lossless-under-readout vs
  collapse-under-free-generation," not around denoising/gains.
- **Off-policy bias** on the residual is *bounded* by reverse replay, not eliminated; the
  method-level split is not identifiable within the bracket.
- **Kernel confound:** H2O runs on eager attention (others on sdpa); full-cache eager-vs-sdpa
  exchangeability was checked (19/20 vs 20/20, no systematic shift).
- **FC-bank provenance:** the reference bank's build commits were orphaned in a repo move;
  provenance now rests on pinned package versions + per-row source-tree hashes + FC having no
  compression logic.

## 7. Status
- **DONE (writable core):** phenomenon, forward decomposition, reverse-replay bracket
  (+ endpoint CIs, + imputation bounds), Figure 4 TV mechanism (estimator defined), ADD,
  two-FC negative control, R-KV parity.
- **DONE (added 2026-07-07):** budget-2048 decomposition (§4j) — dose response closed.
- **DONE (added 2026-07-10):** h2o-2048 dose point (§4k) + H2O TV profiles (§4l) —
  H2O-style is now COMPLETE on every instrument (forward, reverse, ADD, dose, TV).
- **DONE (added 2026-07-12):** R-KV full decomposition (§4m, Branch C) — the paper
  now has FOUR policies with brackets excluding zero on all eight endpoints.
- **DONE (added 2026-07-15):** Llama cross-architecture check (§4o — SLM
  replicates, Random degenerates, both reported plainly); AIME-24 FC bank (§4p).
- **DONE (added 2026-07-17):** fc-readout-control (§4q — resampling effect
  +0.023, measured; Branch A).
- **DONE (added 2026-07-18):** m4-aime-2048 (§4r) — ORIGINAL CAMPAIGN 100%
  READ. No >100% regime; residual replicates on 2nd benchmark; readout
  is press-INDEPENDENT at that dose (restated prompt-inclusive as 19.5% in
  Unit 3). Drop-dead 2026-08-01 met with two weeks to spare.
- **DONE (added 2026-07-20 + 2026-07-24):** m4-q15b-256 (§4s) + m4-q14b-256
  (§4t) — Amendment J COMPLETE, all four scale-axis cells read. 3/4
  REPLICATE (both 1.5B cells, SLM-14B); R-KV-14B inconclusive under the
  bracket (span-zero, tied to the +0.023 resampling floor), reported
  plainly. Scope-upgrade sentence applied: "one model family spanning
  three scales (1.5B/7B/14B), plus a Llama-base check."
- **ALL EXPERIMENTAL WORK COMPLETE 2026-07-24 — nightly queue fully
  drained, no runs remain.** Endgame only from here: anonymization scrub,
  restore double-blind author block, arXiv, ICLR conversion
  (`docs/strategy/SUBMISSION_STRATEGY.md` Phase 11). Drop-dead 2026-08-01
  met with 8 days to spare.

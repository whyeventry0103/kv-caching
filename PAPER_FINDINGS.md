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
- **Exact decomposition split:** readout-gap 0.1811, residual-gap 0.0039 (use 0.181 / 0.004;
  the prereg's 0.182/0.003 is a mis-rounding). SLM readout share = 0.4625 -> 0.46.
- **Compute:** ~224 GPU-hours of generation (summed wall-clock over 7,307 generations; excludes
  model-load/teacher-forcing overhead — quote as "~230 GPU-hours").
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
0.808 (trunc 0.028, mean 3,650 tok)** vs FC 0.882. Free-running at 8× budget is within
0.03–0.07 of the reference → (a) the collapse is a BUDGET effect, (b) doubles as a harness
control (free-running per se isn't broken), (c) the Random>SLM ordering PERSISTS at 2048
(+0.042). In the paper: §4.1 + Fig 1 dose bars. readout-2048 (decomposition) queued.

## 7. Status
- **DONE (writable core):** phenomenon, forward decomposition, reverse-replay bracket
  (+ endpoint CIs, + imputation bounds), Figure 4 TV mechanism (estimator defined), ADD,
  two-FC negative control, R-KV parity.
- **IN PROGRESS (generality/breadth):** budget-2048 grid DONE 800/800 (read pending);
  readout-2048 queued; H2O-style: grid DONE (§4h), readout tonight; R-KV smoke + cells;
  AIME (drop-dead 2026-08-01).

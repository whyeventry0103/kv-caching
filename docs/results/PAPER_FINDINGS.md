# Paper findings — consolidated evidence base (for drafting)

This is the authoritative, self-contained record of all results, framing rules, and honest
limitations for the paper. **All experimental work is complete as of 2026-08-27;
every number here is final.** The *(in progress)* convention is retained only for
reading historical entries in the dated log below.

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
That claimed >100% regime is exactly what our R-KV/AIME cells test (both complete and read — §4m, §4r) — NOT something we
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
- **Compute:** **375 GPU-hours** of generation (summed wall-clock over **10,007** generations)
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

>
> **MECHANISM CLAUSE REQUIRES RESTATEMENT (flagged 2026-08-23, unresolved).**
> The final clause — "*compounds* over the generated trajectory" — reads as a
> HORIZON mechanism. Amendment M.3 (§4v) shows the rising divergence profile is
> what a monotonically worsening compression ratio produces, and states that no
> mechanistic reading of it is licensed. The clause is not false as a
> *description* of the fixed-budget regime everyone deploys — divergence does
> accumulate there, and §4v gives the drift figures for both windows; what is
> unlicensed is attributing that accumulation to horizon
> rather than to the worsening ratio. The first two clauses — invisibility to
> readout-style evaluation, and the residual as the gap — are untouched by M.3
> and stand. **Do not cite the mechanism clause as-is; the restatement wording
> is an open decision, not yet made.**

Do NOT frame around ">100% / denoising": we did not observe any accuracy gains on small-budget
MATH, so that regime is explicitly out of scope for the headline (see §6).

## 3. Setup
- **Model:** DeepSeek-R1-Distill-Qwen-7B. **Benchmark:** MATH-500. Sampling T=0.6, top-p=0.95.
- **Full-cache reference:** 250 problems × n=8 = 2000 generations, **FC accuracy 0.882**.
- **Compression cells:** 100 problems × n=4, budget **256** (the collapse pole) and budget
  **2048** (dose-response) — both **complete and read** (§4-CURRENT).
- **Methods (eviction presses):** StreamingLLM, Random (per-(problem,seed) RNG), H2O
  (ObservedAttentionPress; eager attention), R-KV (parity-verified port) — all four
  **complete and read** (§4-CURRENT).
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
- Compute: **375 GPU-hours over 10,007 generations** under the artifact's inclusion rule.
  Corrected 2026-08-05 — the previous figure is in the superseded list below. Cause:
  `INCLUDED` in `report_compute.py` is an allowlist and had drifted nine result-bearing
  run directories behind. Found by `scripts/rederive_artifacts.py`, not by reading.

Superseded values are listed at the top of the archival log below.

### 4-LOG. Dated read-in log (archival — not current truth)

<!-- AUDIT:ARCHIVAL-BEGIN -->
<!-- Everything from here to AUDIT:ARCHIVAL-END is the dated read-in log: a
     historical record whose numbers are correct as of their dates. The number
     audit skips this span deliberately. Do NOT "update" these entries -- several
     exist to show a wrong value beside its correction, which is the point. New
     results append a new dated subsection; corrections go in §4-CURRENT and in a
     PREREGISTRATION amendment.

     PATH NOTE (added 2026-08-29), so an archival citation stays mechanically
     resolvable. Entries below cite nine artifacts as `results/diag/m4-*-
     summary.json`. Those files were re-derived on 2026-07-27/29 under Amendment
     K's population discipline -- each change measured against its OWN
     population rather than the unrestricted 0.8825 bank -- and the corrected
     versions live at `results/diag/amendment-k/<same name>.json`. That
     amendment-k copy is CANONICAL: build_paper_numbers.py and
     rederive_artifacts.py both read it, nothing reads the originals, and the
     paper quotes only its values. The originals are retained, renamed
     `<name>_PRE-AMENDMENT-K.json`. To resolve a citation below, append that
     suffix; to get the number the paper uses, read the amendment-k copy.

     The re-derivation was not cosmetic. For H2O at budget 2048 the same named
     quantity changed SIGN: `readout_change` is +0.0155 in the pre-Amendment-K
     file and -0.0051 in the canonical one. A bare filename resolving to two
     files with opposite signs is exactly the ambiguity the rename removes; an
     invariant test now keeps identity-bearing basenames unique under
     results/diag/ (tests/test_paper_numbers_properties.py).

     The heading above must stay ABOVE this marker:
     the 4-CURRENT gate ends the guarded span at the next heading, so a marker
     placed first would fall inside the section it is meant to sit outside. -->

#### 2026-08-03 — the N.3 AIME bank FAILED its gate; cap raised to the published 32768 (Amendment O)

`bank_aime24_llama8b` completed 120/120 and **failed** its pre-registered N.3
gate: accuracy **0.2750** against a band of [0.30, 0.70]. Per N.3 the bank is an
instrument problem until re-validated, so `grid_aime24_rkv_published` (N.2) is
**not readable** and stays unread. §2's "adjacent configuration" hedge for R-KV
therefore still stands, unchanged.

Diagnosis (Amendment O.2): truncation **0.2583** (31/120 rows hit the 16384
cap), accuracy among non-truncated rows **0.3708** (in band). The accepted
Qwen-7B AIME bank behaves the same way at the same cap (truncation 0.2208,
accuracy 0.3708, non-truncated 0.4759). The pipeline is not broken; the cap
costs both models about a quarter of their generations, and at 8B that is enough
to cross the floor.

Remedy and its honest account (Amendment O.3): the cap moves to **32768**, the
published generation length used by DeepSeek's own R1-Distill evaluation and by
R-KV's AIME headline — a cell whose purpose is "the published configuration"
must use the published generation length, which was a config design error
independent of any result. But it was found while investigating a disappointing
number, so: the 16384 result is permanent and reported beside whatever 32768
gives; both caps and both truncation rates are reported wherever the cell is
stated; the band is **not** widened and 0.2750 is **not** reclassified as a pass;
and there is **one attempt** — a second failure cuts N.2/N.3 outright.
Re-runs use new run names so the 16384 artifacts are never touched.

Separately (Amendment O.5), the N.2 cell had been crashing on every launch since
2026-08-01: the R-KV window-query recompute hand-rolled unscaled RoPE and
R1-Distill-Llama-8B uses llama3 scaling, so the guard refused — correctly. The
schedule now comes from transformers' own `ROPE_INIT_FUNCTIONS`; the unscaled
branch is locked bitwise-identical to the old formula (so no existing number
moves), the per-event self-check is retained, and the real-generation smoke on
Llama-8B/AIME-24 passed at **Jaccard 1.0000** over 448 compressions.

Reportable regardless of what happens next: **about a quarter of AIME-24 traces
from these models do not terminate within 16384 tokens.** That is a measurement
about evaluating reasoning models under a token budget, and it applies to any
AIME number quoted at a cap.

#### 2026-08-04 — Part F verification: the core cell re-derived end to end, and a gate weakness quantified

**Re-derivation.** `report_m4` was re-run on the remote over the raw rows
(`math500-fc-powered-merged`, `grid-diss256-tok`, `replay-diss256`,
`readout-diss256-g0/g1`, `reverse-readout-diss256`) and its output compared
field by field against the committed `m4-powered-diss256-summary.json`:
**every field identical**, including the full-cache line (0.8825, 1765/2000, 38
truncated). This re-derives the abstract's budget-256 numbers — readout 0.765 /
0.584, free-run 0.422 / 0.237, reverse 0.636, residuals −0.337 / −0.344 — from
rows rather than from the summary they were read out of. Free-run accuracies and
the H2O cell were additionally re-counted straight from the JSONL rows
(0.4225 / 0.2375 / 0.2475) and agree to the displayed digit.

**A gate weakness, quantified rather than left implicit.** The number audit
answers "does every literal have a source", not "is that the right source".
Restricted to result-precision (3+ decimal) literals, **351 occurrences match
more than one catalogue entry**, so for those the green tick does not pin the
number to one quantity. The concrete instance: the paper's 0.045 (0.515 − 0.47)
attributes to `llama.random.readout_ci1`, an unrelated Llama CI bound, and the
derived `hottcap.residual_gap` entry does not displace it because exact-match
precedence prefers the entry whose stored float is exactly 0.045. The audit now
**reports this count on every run** instead of printing an unqualified pass.
Most of the exposure is small-magnitude values where coincidence is inherent; a
real fix needs context-aware matching, which is not built.

**Superseded values, listed so they are recognised if they resurface — do not reuse.**
Dose factor 3.6–6.8×. Resampling effect +0.023. Residuals −0.343 / −0.347 / −0.349 and
R-KV −0.278. Reverse endpoints −0.246 / −0.466 / −0.463 / −0.221. Readout shares
25/46/45/23%. Residual spread 0.006, readout-change spread 0.169. Policy-contrast upper
bound 0.242 (200-seed range [0.2375, 0.2450]). Compute ~224–230 GPU-h / ~7,307
generations. Budget-2048 residuals −0.051 / −0.096 / −0.075. AIME residuals −0.179 /
−0.246 / −0.321. Truncation-adjusted −0.110 / −0.277. Compute 280 GPU-hours / 7,200 generations
(superseded 2026-08-05: the inclusion allowlist had drifted nine result-bearing run
directories behind; the corrected figure is 375 / 10,007 in §4-CURRENT).

#### 2026-08-04 — R-KV in its published configuration: pre-registered branch REPLICATES

`grid_aime24_rkv_published` finished 120/120 at the 32768 cap against the
re-validated bank. Read per Amendment N.2, branch selected **by code**
(`kvcache.stats.published_config_branch`, emitted as a field of the artifact) so
it could not be re-cut after the numbers were seen. Artifact:
`results/diag/amendment-k/classb/rkv_published_cap32768.json`.

**Result.** R1-Distill-Llama-8B, AIME-24, budget 2048 (mean realized cache
**0.1886**), 30 problems x 4 seeds, matched population. Full cache **0.3000**
(36/120), R-KV **0.3083** (37/120), **r = 1.0278**. Paired difference
**+0.0083**, problem-clustered 95% CI **[-0.0750, +0.0917]**; 7 wins / 17 ties /
6 losses. Branch **REPLICATES**.

**Say it with its precision, always.** The margin is one generation (37 vs 36 of
120), the CI spans zero, n=30 problems, and the reference bank cleared its N.3
band at exactly the floor. Defensible: our certified port is **not worse than
full cache at ~19% of it** in the published configuration. NOT defensible:
"we reproduced 105%". Also not defensible in the other direction: calling this
INCONCLUSIVE — r >= 1.0 with a CI including zero is REPLICATES under the rule as
pre-registered, and the rule was written before the data.

**Reported because it favours R-KV:** its traces run longer (18869.7 vs 14799.3
mean tokens) and truncate at **0.2083** vs **0.0750**, every truncation scored
incorrect, and it still leads; among non-truncated rows it leads **0.3895** to
**0.3243**.

**Free determinism check.** The queue relaunched the same cell under its own run
name while the manual run (`rkvpub-g0/g1`) was in flight — the queue's progress
counter reads the run directory, so different run names made the work invisible
to it. The duplicate reached 47 rows. They are **not merged** (that would
double-count problems). On all 47 overlapping conditions the two independent
processes agree exactly: same trace, same canonical answer, same token count,
same correctness. Sampled decoding at T=0.6 reproduced bitwise across processes
and GPU assignments.

**Operational note, so it is not repeated:** launching outside the queue must use
the queue's run name, or the queue will relaunch the same job.

#### 2026-08-05 — full verification sweep: 29/29 artifacts re-derived from rows; one stale number found and fixed

**The gap this closed.** `audit_paper_numbers.py` proves every literal matches
*some* catalogue entry, and the catalogue is built *from* `results/diag/`.
Nothing checked that the artifacts still follow from the rows. An artifact
written by a since-fixed pipeline, or edited by hand, would have passed every
gate in the repo.

**Artifact re-derivation (`scripts/rederive_artifacts.py`, new).** Runs on the
remote, reconstructs each producing command — from the artifact's own recorded
`runs` block where it has one, else from a manifest — re-runs it against the raw
rows, and diffs field by field. Result: **29 of 29 IDENTICAL**, including the
core cell (`m4-powered-diss256`), all four grader audits, all four ADD cells,
all seven resolution regimes, and the R-KV published-configuration cell.
Archived: `results/diag/rederive-2026-08-05.json`.

**One real defect, found by it.** `compute.json` differed: `INCLUDED` in
`report_compute.py` is a hardcoded allowlist and had drifted **nine
result-bearing run directories** behind (grid-diss512, grid-diss1024,
bank-hottcap, grid-hottcap, both AIME-Llama banks, both rkvpub shards, and the
duplicate relaunch). Those cells *are* reported in the paper, so the compute
statement had silently stopped covering the work it describes.
**280 GPU-hours / 7,200 generations → 375 GPU-hours / 10,007 generations.**
Corrected in the paper, in §1b, and in §4-CURRENT; the old figure is in the
superseded list. Nothing else in the paper depends on it.

**Intent gate (`scripts/audit_paper_claims.py` + `paper_claims.json`, new).**
Value-matching cannot tell two quantities apart when they round alike — the
known 360-occurrence ambiguity. `paper_claims.json` pins a number to the
*quantity it asserts*: **79 load-bearing numbers** (abstract, contributions,
`tab:decomp` in full, the R-KV published cell, the HOTT-cap cell, the compute
and truncation statements) each verified to display the value of their named
catalogue key. A regression test moves `rkvpub.ratio` and asserts the gate
fails, so the gate is proven to bite. Wired into `make audit`.

**Coverage, stated rather than implied.** **155** result-precision literal
occurrences in the `.tex` remain both ambiguous and unpinned. They are not known
to be wrong — every one is sourced and every backing artifact re-derives — but
they rest on value-matching, so they are *not* verified to the standard the 79
are. Pinning more is cheap; the number is printed by `make audit` so it cannot
quietly grow.

**Also checked.** PDF: 0 errors, 0 undefined references, 0 undefined citations,
0 multiply-defined labels, 1 pre-existing Overfull (line 293, 1.4pt), 26 pages.
Local/remote artifact parity: all 75 files match by sha256 (the 6 previously
local-only artifacts were copied up). `make parity` still reports a commit
mismatch — structural, because `sync_source.sh` never transfers `.git`.

#### 2026-08-10 — additivity test (Amendment P.1): the interaction partition is a population artifact

**What was proposed.** That `readout_error + residual_rev` under-explains
`Delta_tot` for Random and R-KV and over-explains it for StreamingLLM and
H2O-style, by 10–22% in every cell — a third, structurally unrelated recovery of
the recoverable-vs-concentrated partition.

**What the rows say.** Computed on the COMMON population (conditions where all
four protocols are defined), with problem-clustered CIs:

| cell | interaction | 95% CI | excludes zero |
|---|---|---|---|
| Random | +0.039 | [−0.010, +0.088] | no |
| StreamingLLM | +0.137 | [+0.060, +0.218] | yes |
| H2O-style | +0.139 | [+0.063, +0.219] | yes |
| R-KV | +0.015 | [−0.029, +0.060] | no |

**The partition is not recovered.** Random and R-KV do not show the predicted
negative interaction — their point estimates are positive and their intervals
include zero. Only the two concentrated policies have a detectable interaction.

**Root cause, and it is the familiar one.** The proposed figures summed three
terms measured on three different denominators (400 / 392 / 308–372) and read
the shortfall as an interaction; part of it was population mismatch. Random and
R-KV flip sign between conventions and are exactly the two cells with the
smallest reverse-readout population, because reverse readout is defined only
where the compressed trace terminates and those two truncate most (0.233,
0.135). The "partition" tracked truncation rate.

**Standing rule this reinforces:** never sum accuracy terms measured on
different populations. Amendment K fixed each term against its own reference;
that makes each term correct in isolation and makes their SUM invalid without a
common-population recomputation.

**Selection bounded both ways (2026-08-11).** Scoring the dropped conditions
both ways narrows it further: only **StreamingLLM** excludes zero under both the
sampling CI and the worst-case imputation ([+0.008, +0.188]). H2O-style's
both-ways bound spans zero ([−0.040, +0.200]) despite its sampling CI excluding
it. Random [−0.270, +0.210] and R-KV [−0.193, +0.108] are wide open.

**The surviving claim, stated at its true strength:** StreamingLLM's two damage
channels demonstrably destroy overlapping problems; H2O-style points the same
way without surviving the selection bound; Random and R-KV show nothing. The
estimand is "interaction among conditions where all protocols terminated", and
that scope travels with the number.

**Do not reuse:** interaction −0.076 / +0.140 / +0.135 / −0.037, and the claim
that additivity fails by 10–22% in every cell (it fails by 4.6% for R-KV and is
not significant).

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
edit is pre-drafted in docs/results/RESULTS_PLAYBOOK.md and fires only then.
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

### 4u. Meeting #2 Tasks 3 and 4, and the P.2 replication (2026-08-12)

Source run `replay-ext-diss256`: 800 `tv-profile-v2` profiles (25 problems x 4
seeds x 4 presses x budgets {256, 2048}), the first replay data that includes
**R-KV**. Artifacts: `results/diag/amendment-p/{shape_ext_b256,shape_ext_b2048,
auc_resid_tv,onset_ext}.json`. Full reads in PREREGISTRATION Amendments P.2
(replication), P.3 and P.4, all dated 2026-08-12.

**Task 3 (AUC-TV vs the subtraction residual): BUDGET EFFECT ONLY.** Pooled
Spearman rho = -0.857, CI [-0.893, -0.643] over 7 cells; within budget 256
rho = -0.200, CI [-0.400, +0.800] over 4 cells; budget 2048 has only 3 cells and
an interval spanning the whole range. The pooled association does not survive
holding budget fixed. **Write it as "not established at this n", never as
"proportional"** -- rank correlation is not proportionality, and that stronger
claim (linearity through the origin) was not tested.

**Task 4 (JSD / top-1 / rank displacement): the crossing test is degenerate.**
TV and JSD are non-zero on exactly the same steps in teacher-forced replay
(pre-onset both exactly 0.0; identical zero/non-zero pattern across all 10,769
steps of the profile checked), so their first crossings coincide *necessarily*.
Medians agree to the step (192 at budget 256, 1920 at 2048). **No claim about
JSD's onset sensitivity may be made from this**, in either direction. Rank
displacement does differ and is **coarser** at onset (median 280-451 vs 192),
the opposite of what `divergence.py` predicted; that docstring is corrected.

**Task 1 / P.2 replication: one result holds, one is withdrawn.** Budgets must
be split -- budget 2048 barely compresses (AUC-TV 0.013 vs 0.084-0.108; 86% of
traces flat) and pooling dilutes the signal. At budget 256, phase-amplitude
excess over sham: **Random +0.1068 [+0.0395, +0.1315], peak phase 0** (replicates
+0.0991 [+0.0614, +0.1271], peak 0, on an independent sample);
**StreamingLLM +0.0195 [-0.0212, +0.0499]** against a prior +0.0384 [+0.0013,
+0.0560] -- **the prior StreamingLLM claim is withdrawn**, its interval had
excluded zero by 0.0013; H2O-style +0.0293 [-0.0306, +0.0480], null in both
samples; **R-KV +0.0620 [+0.0016, +0.0775], peak 9** -- marginal, on the same
knife edge StreamingLLM just fell off, and with 8 comparisons in the family it
is the expected yield under no effect. At budget 2048 nothing excludes zero and
three of four estimates are negative.

**Framing consequence.** The "recoverable policies are eviction-locked" partition
is **weaker** after adding the policy it most needed: one clear case (Random),
one withdrawn (StreamingLLM), one null (H2O-style), one marginal whose peak is
at phase 9 rather than 0 (R-KV). Random's eviction-locking is the only phase
result the record carries. Interpretation of the rising profile stays withheld
pending M.3.

### 4v. Meeting #2 Task 2 (restoration) and the M.3 ratio-matched control (2026-08-15)

Two powered cells landed and were read. Both required a re-run: the first
attempt at each was invalidated by a defect in our own instrument, and both
invalidations are recorded in PREREGISTRATION (P.5 correction 2026-08-13, M.3
correction 2026-08-14). Artifacts:
`results/diag/amendment-p/{restoration_diss256_v2,m3_verdict,m3v2_coverage}.json`.

**Task 2 — reinsertion does NOT reduce divergence.** `restoration-diss256-v2`,
800/800 conditions, 200 per seed. Instrument gate **800/800 pass, zero
failures, zero missing, zero anchor disagreements**: every treated arm bitwise
identical to what was stashed (`max_abs_difference` 0.0), `matched`/`random`
holding the cache at 256, `relaxed` at 320. Effects are `none` minus arm over a
200-step window anchored per trace on the applied step (192 or 256), so
**positive would mean divergence reduced**:

| cell | matched (the causal test) | random (specificity control) | relaxed (confounded reference) | baseline divergence |
|---|---|---|---|---|
| Random-256 | **-0.0013** [-0.0019, -0.0007] | -0.2485 [-0.3330, -0.1748] | -0.0014 [-0.0021, -0.0008] | 0.0383 |
| StreamingLLM-256 | **-0.0015** [-0.0022, -0.0009] | -0.2442 [-0.3278, -0.1709] | -0.0014 [-0.0021, -0.0008] | 0.0445 |

Verdict **NO RESTORATION** in both cells. The `matched` effect is negative and
excludes zero, but at 0.0013-0.0015 against a baseline of 0.038-0.044 it is ~3%
of the divergence and is to be written as **indistinguishable from no help**,
never as a measured harm. The `relaxed` arm, which is allowed to grow the cache,
does no better, so the null is not an artifact of holding the budget fixed.

**The specificity control carries the interesting part.** Reinserting the same
*number* of entries chosen randomly from the evicted pool -- with the SAME
entries dropped to make room, which the corrected implementation guarantees via
a shared eviction stream -- increases divergence by 0.244-0.249, roughly six
times the baseline. So the cache is acutely sensitive to *which* entries occupy
it, while restoring the specific entries a policy just evicted is nearly
neutral. **Eviction damage, at this budget and horizon, is not undone by putting
the tokens back.**

Scope, and it is narrow: one model, MATH-500, budget 256, two policies, a single
intervention at one trigger with `reinsert_fraction` 0.25 and a 200-step read
window. A null here is a null for that intervention, not for reinsertion in
general.

**M.3 — the divergence staircase is a COMPRESSION-RATIO effect.**
`replay-ratio-matched-m3-v2`, 200/200. The read condition was fixed in writing
before the cell completed (PREREGISTRATION, M.3-v2 read condition) and is met:
above-floor coverage **83.5%** of 377,104 compression events against the 83.9%
predicted, per-trace median coverage 74.2%, realized ratio on above-floor events
exactly **0.038**, largest target chosen 626.

Drift = last-quarter mean / first-quarter mean over the restricted, paired
window (each trace trimmed to where its own press left the floor; the
fixed-budget counterpart cut at the SAME absolute step; problem-clustered
bootstrap, 2000 resamples, seed 0; 24 problems / 92 observations per press;
median 75.9% of each trace retained; excluded: 8 profiles never left the floor,
8 windows under 200 steps):

| press | drift, FIXED budget 256 | drift, RATIO-matched 0.038 | matched - fixed |
|---|---|---|---|
| Random | **1.201** [1.061, 1.339] | **0.662** [0.614, 0.709] | -0.539 [-0.655, -0.417] |
| StreamingLLM | **1.299** [1.100, 1.520] | **0.739** [0.680, 0.805] | -0.559 [-0.760, -0.366] |

Both take the pre-registered **FLATTENS** branch. **Section 4.4 is to be
restated in compression-ratio terms, and no mechanistic reading of the rising
divergence profile is licensed.** The rise is what a monotonically worsening
ratio produces.

Three qualifiers that must travel with it:

1. **What was manipulated.** Holding the ratio fixed *is* letting the absolute
   budget grow with context (32 -> 626 here); they are the same manipulation by
   construction. The honest claim is that the staircase disappears when the
   budget grows in proportion to context -- NOT that horizon length is
   irrelevant at fixed cache size, which is a different experiment.
2. **The restriction shrinks the staircase being tested.** Over the whole trace
   the fixed cell drifts 2.93 (Random) and 2.26 (StreamingLLM); over the
   comparable above-floor window it drifts 1.20 and 1.30. Trimming removes the
   early trace where most of the rise sits. The staircase is still present and
   its interval still excludes 1, so the comparison stands, but the milder
   figure is the one licensed for this comparison.
3. **The matched cell does not merely flatten -- it reverses** (drift 0.66-0.74,
   below 1): divergence *falls* across the trace when the budget grows with
   context.

**RESOLVED 2026-08-27 -- see §4w. Eviction-locking SURVIVES ratio-matching
(branch A): Random +0.252 [0.184, 0.271], peak phase 0, on the same restricted
window as the drift comparison. The preliminary whole-trace numbers below are
superseded and must not be cited.**

**Consequence for §4u's P.2 result, as recorded at the time (now resolved).** The eviction-locking
finding (Random's phase excess peaking at the prune step) was flagged as
possibly being what a ratio-driven curve would produce. Phase statistics
computed on this cell (`shape_m3v2_UNRESTRICTED.json`: Random excess +0.326
[0.263, 0.339] peak 0; StreamingLLM +0.102 [0.065, 0.128] peak 6, both excluding
zero) were computed over the WHOLE trace including the floor-bound period -- the
harshest compression in the run, which would inflate phase amplitude. **They are
preliminary and must not be cited as evidence that eviction-locking survives
ratio matching** until the phase analysis is repeated on the same restricted
window as the drift comparison above. That analysis is not yet written.

### 4w. Amendment P.2-v2: eviction-locking survives ratio-matching (2026-08-27)

The last open Meeting-#2 item, and it closes **branch A**. CPU-only re-analysis
of the `replay-ratio-matched-m3-v2` profiles on the same restricted window as
the M.3 drift comparison. Read condition pre-registered 2026-08-24, before any
number existed. Artifact
`results/diag/amendment-p/shape_m3v2_RESTRICTED.json`; full detail and the
correction disclosure in PREREGISTRATION "Amendment P.2-v2, result".

Coverage matches the drift comparison exactly -- 24 problems / 92 observations
per press, median 75.9% of each trace retained; excluded and counted: 8 profiles
that never left the floor, 8 windows under 200 steps.

| press | whole trace | restricted (above-floor) | peak phase |
|---|---|---|---|
| Random | +0.326 [0.263, 0.339] | **+0.252 [0.184, 0.271]** | **0** (prune step) |
| StreamingLLM | +0.102 [0.065, 0.128] | +0.076 [0.031, 0.099] | 1 |

**Random's eviction-locking is real and survives ratio-matching.** Both
restricted excesses are smaller than their whole-trace values, as pre-committed,
because the floor-bound period is the harshest compression in the run.

**State this together with M.3, never alone.** Within each prune cycle,
divergence is locked to the eviction event -- that is a mechanism and is now
licensed. Across cycles, the rising envelope is a compression-ratio effect and
no mechanistic reading of it is licensed (M.3, §4v). The restricted window shows
both halves at once: phase excess excludes zero while drift is 0.65-0.71 and
97.8% of traces classify flat. **Do not let the restored phase mechanism leak
into the staircase claim.**

**StreamingLLM stays WITHDRAWN.** Its restricted excess excludes zero, but a
fixed-budget claim that failed to replicate on 2026-08-12 is not rehabilitated
by a different cell, and its peak is phase 1 rather than 0. Descriptive only.

**A one-bucket windowing bug was found and corrected mid-analysis, and it moved
the verdict from branch B to branch A** -- i.e. in our favour, so the timing is
disclosed and both readings are published side by side in the amendment. The
re-based onset was not preserving phase alignment, rotating every phase index by
a measured, constant 63 buckets across all 96 traces. The excesses barely moved
(+0.248 to +0.252); what moved was which bucket counts as the prune step, which
is exactly what the branch turns on. Amendment M.3's drift verdict does not use
the onset re-basing and reproduces bit-identically.

### 4x. The mechanism clause is decided and propagated: candidate A, sharpened (2026-09-08)

No new computation. This entry records a **wording decision** that M.3 (§4v) and
P.2-v2 (§4w) made necessary, and its propagation into the paper. Rationale and
the rejected candidates: `docs/project/MECHANISM_CLAUSE_CANDIDATES.md`.

**Chosen: candidate A (regime-anchored), anchored on the measured `0.0018`
rather than the constructed `0.0000`.** The thesis previously said the residual
"arises from a divergence that is approximately zero at the moment of eviction
and *compounds* over the generated trajectory." Both halves were problems: the
compounding is the reading M.3 delicensed, and the zero is an **instrument
property**, not a finding — under teacher forcing with identical token IDs the
two models see identical input until the first prune, so pre-eviction TV *must*
be `0.0000`, and §4.4 already spends that fact as the replay harness's
self-control. The thesis now reads:

> The gap is the trajectory residual: under the fixed cache budgets deployments
> actually use, the divergence behind it is small in the steps immediately
> following an eviction and far larger later in generation, so held-fixed-trace
> protocols sample it exactly where it is smallest.

`sec:mechanism` retitled to **"Where the divergence is, and where readout
evaluation looks"**. The clause carries no numeral by design — landmine 11 (a
prose quantity is invisible to `audit --strict`) means a numeric thesis claim
would be unchecked; the numbers stay in §4.4 and `tab:tv`, where the audit sees
them.

**Candidate C was ruled out by a blocked dependency, not by argument**, and this
is reversible. C is the better fit *if* `fig4_tvprofile` gets the ratio-matched
overlay; the overlay could not be plotted because neither `replay-ext-diss256`
nor `replay-ratio-matched-m3-v2` exists locally and the remote was refusing
connections. **If that figure is ever re-plotted, revisit this decision** — the
site-4 comment in the `.tex` carries the instructions and the blocker.

**The `TODO(M.3-RESTATEMENT)` marker set was incomplete, and this is the finding
worth carrying forward.** Propagating the seven markers and then sweeping for
the delicensed vocabulary found three further sites, one serious:

- the **abstract** claimed the profiles "show it grows with horizon" and that
  short-horizon evaluation "measures the pre-compounding regime" — the strongest
  unlicensed claim in the paper, in its most-read paragraph, and unmarked;
- **Summary of findings** said "near-zero immediate effect and strong compounding";
- the **trajectory-audit protocol** told future users to check the divergence
  profile "for compounding"; it now tells them to hold the intervention's dose
  fixed before reading growth as a horizon effect, which folds our own M.3
  correction into the reusable protocol.

A find-and-replace over the seven markers would have shipped a paper whose
abstract still asserted what its own §4.4 had withdrawn. **A marker set is a
starting point for a sweep, not a substitute for one.**

**Site 6 was visited and deliberately left unchanged** — it already pairs both
findings and attributes the rise to nothing, which is what candidate A asks of a
mechanism site. Recorded because "unchanged" here is a decision, not an
oversight.

**Verification.** PDF rebuilt ×2: 0 LaTeX errors, 1 Overfull, 9 Underfull, **29
pages** — identical to a baseline build of `HEAD`'s `.tex` in a scratch
directory, so the edits introduced no new typesetting problems and the surviving
Overfull (lines 309–310) predates them. Gates bare: lint clean, 313 passed / 14
skipped, dryrun clean, `audit --strict` clean on both the numbers gate and the
claims gate. No artifact, no protocol, and no number changed — prose only.

### 4y. The M.3 figure, and a caption that described a figure the paper never printed (2026-09-08)

Same day as §4x, after the remote came back. No new computation on the GPUs; the
raw rows are untouched. Two defects found and fixed, one new figure added.

**1. `fig4_tvprofile`'s caption described a curve the figure does not show.** The
committed figure is the three-window SUMMARY bar chart (before / first-50 /
later; 0.0000 / 0.0018 / 0.0884), because `make_paper_figures.py` only renders
the onset-aligned curve when `--profiles-dir` is given and the profiles live on
the remote. The caption nonetheless described "a compounding staircase whose
steps align with eviction events" and stated "the curve is shown to $+400$ steps
of onset." Verified by reading the PNG and by extracting text from the embedded
PDF (axis label: "Window rel. to first eviction (onset step 65.0)"). The
mismatch predates the M.3 restatement. The caption now describes the three
windows it plots, and labels the 0.0000 bar as the self-control it is.

This also retires a premise carried in `MECHANISM_CLAUSE_CANDIDATES.md`: the
"figure asserts the horizon reading through its x-axis" problem applied to a
DATA-mode rendering nobody had looked at, never to the paper as printed.

**2. The planned ratio-matched overlay was rejected on measurement, not taste.**
The two arms are perfectly paired — 200 profiles each, all 200
`(press, seed, pid)` keys shared — but they sit ~2 orders of magnitude apart in
level: first-50 window mean TV `0.0036` fixed vs `0.1972` ratio-matched; at
onset offset 0, `0.0008` vs `0.3289`. The ratio-matched press spends the early
trace pinned at `min_budget` 32 against the fixed arm's 256. M.3's verdict is a
drift statement *within* each arm and never a level comparison *between* them, so
a shared-axis overlay would have invited precisely the unlicensed reading the
figure exists to prevent.

**3. New figure `fig:ratiomatched`** (`figures/fig7_ratiomatched.pdf`): the
paired above-floor window the verdict uses (92 paired traces per policy; 8 never
left the floor, 8 windows under 200 steps, excluded and counted), with each
trace divided by its own first-quarter mean before averaging. 1.0 is "no drift,"
so the plotted quantity is the verdict itself: fixed rises
(`1.201` / `1.299`), ratio-matched reverses (`0.662` / `0.739`). Levels are
normalized away, with the reason stated in the caption.

Artifact: `results/diag/amendment-p/m3_drift_curves.json`, emitted by
`report_m3_verdict.py --curves-out`. **Additive by construction** — the flag adds
a second file and `m3_verdict.json` is byte-identical with or without it,
confirmed by sha256 against the committed artifact on the remote. That match is
also a free re-derivation check: the committed verdict reproduces from the raw
rows.

**4. Thirteen numbers moved from uncited to pinned.** The drift figures had never
appeared as literals in the paper, so they were not in `paper_numbers.json`;
printing them in the caption made `audit --strict` fail with 11 unsourced
literals, which is the gate working. Both drift means, all eight CI bounds, the
paired-trace count and `max_target` 626 are now catalogued from their artifacts
and **pinned** in `paper_claims.json` (102 → 115 claims), so each asserts its
specific quantity rather than merely matching some value.

**The candidate-C reopening condition recorded in §4x was evaluated and is now
closed, not pending.** C's case rested on a figure showing *both* levels on
shared axes; `fig:ratiomatched` shows one — the across-cycle envelope and its
control. The within-cycle phase result is still unplotted. Candidate A stands.

**Verification.** PDF ×2: 0 errors, 1 Overfull, 9 Underfull, 29 pages (the new
figure did not add one). Gates bare: lint clean, 318 passed / 14 skipped, dryrun
clean, `audit --strict` clean on both gates. `make parity` **OK** — commit,
source tree, and all 98 result artifacts agree after sync.
**Addendum (same day): `rederive_artifacts.py` ran for the first time this
session and caught a stale artifact.** 28/29 identical on the first pass; the
miss was `amendment-k/classb/compute.json`, whose per-directory ledger predates
the Meeting-#2 runs. Nine directories were absent from it -- `replay-ext-diss256`,
`replay-ext-smoke`, `replay-ratio-matched-m3`, `replay-ratio-matched-m3-v2`,
`restoration-diss256`, `restoration-diss256-v2`, `restoration-smoke`, `rext-g0`,
`rext-g1` -- so `excluded_hours` read `130.00` against a true `219.42` and
`n_directories_excluded` `22` against `31`.

**The paper's cited compute numbers were not affected and did not move:**
`gpu_hours` `374.958` (printed as $375$) and `generations` `10007` are
byte-identical before and after, because all nine missing directories are
teacher-forced replay/restoration work that the artifact's own inclusion rule
excludes (verified: `included=False` on all nine, and `n_directories_included`
unchanged at 23). Rebuilding `paper_numbers.json` produced no diff at all,
which is the check that proves it.

What was actually broken was the artifact's stated promise -- "every directory
is listed with its hours and an 'included' flag, so any exclusion can be re-added
and its effect seen" -- which cannot hold while nine directories are missing from
the ledger. Regenerated in place from the rows (a derived report over valid rows,
the Amendment H pattern, not a data patch). Re-run after the fix: **29/29
identical, exit 0.**

This is the second time this artifact's exclusion list has gone stale as new runs
landed, so the pattern is worth naming: `compute.json` does not self-update, and
any cell that adds a run directory leaves it stale until it is regenerated.

### 4z. Meeting #3 pre-registration, the ICLR conversion, and a scoop-watch verdict that changed (2026-09-12/13)

No new measurements. Three things that bear on how results are read.

**1. Amendment Q pre-registers six Meeting-#3 cells, and widens P.4's latent
scope.** Every read is written before any cell was queued. Q.0 replaces
P.4's "Latent-state divergence is out of scope" with a dated, limited admission:
structural claims only, exploratory by construction, and the basis fitted on the
full-cache condition only and frozen, with circularity a STOP condition rather
than a caveat. Task C's estimand is **fixed in the amendment rather than after
the fact**: four policies is four points and a rank correlation over them is not
a test, so the pre-committed quantity is the pairwise contrast
`ω(Random) − ω(StreamingLLM)` with a problem-clustered CI, with the four-policy
ordering reported descriptively and no correlation coefficient.

**2. The Q.1 sawtooth control was redesigned before it was queued, and it can go
against us.** The proposed control held mean cache at 288 while varying
amplitude. Mean cache is `B + I/2`, so that forces `B = 288 − I/2` and locks
amplitude to floor — the small-amplitude arm also never falls below 284, and a
collapse there would be equally consistent with the hypothesis and with "the
cache never fell far enough to do damage." The parameterisation is genuinely
two-dimensional, so the identity is broken the other way: **fixed floor 256, the
published cell's own floor, with the interval swept.** Mean cache then varies,
and the direction closes the confound rather than opening it — interval 16 has
the lowest mean cache and is the harshest arm, so a collapse there cannot be
attributed to that arm being gentler.

**This cell is aimed at P.2 branch A, which is already written into the paper.**
If interval 16's phase excess does not exclude zero while interval 256's does,
branch A is not refuted but must be **qualified**: "locked to eviction events at
the deployed amplitude (64), and not detectably at amplitude 16," with the
threshold and the untested region stated in §4.4 and the threats section. That
wording exists in Amendment Q before the read, which is the point.

**3. The scoop-watch verdict changed, and the novelty claim narrows.**
arXiv:2608.01631 (Ai, He, Guo) runs a "controlled fixed-trace replay protocol" —
structurally our forward readout — and its Table 2 (AIME26, Qwen3-8B, budget
256) reports end-to-end **and** fixed-trace accuracy for the same methods: AdaKV
`0.233` against `0.400`, SnapKV `0.267` against `0.400`. **Those differences,
`−0.167` and `−0.133`, are the quantity this project is about, and the paper
never names or computes them** — its estimand is the answer–evidence gap and the
two accuracies sit in different tables serving a different claim.

The standing claim that "the replay-decomposition instrument remains unclaimed"
is **retracted**. What remains unclaimed is narrower and should be stated that
way from here: the residual as a *defined estimand* with problem-clustered
intervals, the reverse-replay bracket that corrects its off-policy bias, and the
use of the decomposition to reconcile two literatures. Related Work now states
the overlap **before** the difference and quotes their numbers as containing our
residual, and also reads the overlap as corroboration — an independent group, on
a different model and benchmark, measured a large free-running-minus-fixed-trace
gap for eviction methods.

**Framing rule added.** Two design-spec errors were corrected before they could
propagate: an attention-profile storage estimate low by ~12x (4,592 records at
the median trace length restated as 14,336 records / 57,344 scalars at the 8192
cap, which is what made the schema columnar), and a single "roughly six times
baseline harm" standing in for two different ratios. The specificity control is
**6.49× baseline (Random)** and **5.49× baseline (StreamingLLM)**, always both,
per landmine 11.

**Venue conversion.** The paper now builds under `iclr2027_conference` at **26
pages, ~24 of it main text** against a 9-page limit — the previous "29 against
9" was measured under `article` at 11pt and was never comparable. Appendices are
unlimited after the bibliography, so the remaining work is relocation, proposed
in `docs/strategy/ICLR2027_PAGE_BUDGET.md`. Anonymity is structural via
`\iclrfinalcopy` and was verified in the built PDF rather than assumed. The
`0.882` in SUBMISSION_STRATEGY's Phase-6 intro draft was **not** used: it is the
full-cache bank accuracy, which Amendment K designates not the cell comparator,
so the applied text uses the matched references `0.903` and `0.885`.

### 5a. Amendment Q.1 result: eviction-locking is AMPLITUDE-DEPENDENT, and branch A is qualified (2026-09-15)

The first Meeting-#3 read, and it constrains a result already in the paper. Read
applied exactly as pre-registered on 2026-09-12, before any of these profiles
existed. Artifacts `results/diag/amendment-q/shape_sawtooth_i{16,64,256}.json`.

All arms at `target_size` 256 — the published cell's own floor — with the
compression interval, and therefore the per-event eviction amplitude, swept.
100 profiles per press per arm.

| arm | Random excess | excl. 0 | peak | StreamingLLM excess | excl. 0 | peak |
|---|---|---|---|---|---|---|
| I=16 | +0.0110 [−0.0252, +0.0300] | **no** | 1 | +0.0032 [−0.0236, +0.0184] | no | 9 |
| I=64 | +0.1193 [+0.0446, +0.1410] | **yes** | **0** | +0.0431 [−0.0142, +0.0617] | no | 9 |
| I=256 | +0.3183 [+0.1881, +0.3361] | **yes** | **0** | +0.0309 [−0.0357, +0.0860] | no | 9 |

**Random → AMPLITUDE-DEPENDENT.** T1 does not exclude zero, T3 does. Per the
branch as written, **P.2 branch A is not refuted but is qualified**: locking is
established at the deployed amplitude (64) and is not detectable at amplitude
16, with amplitudes below 64 otherwise untested. The paper's threats section now
says exactly that.

**StreamingLLM → INDETERMINATE.** No arm excludes zero and every peak sits at
phase 9, away from the prune step. Consistent with its P.2 withdrawal, adds
nothing to it, and no phase claim is made for it in any arm.

**Binding framing rule, pre-committed and not adopted after the fact: do NOT
compare excess magnitudes across arms.** Raw phase amplitude scales with the
number of phase buckets, and the arms have 16, 64 and 256 buckets. The apparent
monotone growth +0.011 → +0.119 → +0.318 is exactly the artifact the
pre-registration forbade reading, and it is not evidence that the effect grows
with amplitude. The licensed statement is per-arm exclusion of zero, nothing
more.

**Write "not detectably locked", never "not locked".** T1's interval
[−0.0252, +0.0300] excludes an effect of the size seen at I=64 but cannot
exclude a small one. The distinction is the difference between an honest null
and an overclaim.

**The deployed arm reproduces the published shape**, which is a consistency
check on the instrument rather than new evidence: I=64 is the configuration
every other cell uses, and Random there excludes zero peaking at phase 0.

**Instrument note — the reads were stuck for two days on a path defect, not on
data.** `run_shape_analysis.py` resolved the SOURCE profiles against the
launcher-scoped `REMOTE_RUNS_ROOT` rather than `KVCACHE_BASE_RUNS_ROOT`, giving
`runs/shape-sawtooth-i16/replay-sawtooth-i16/profiles` with the run name
doubled, so all three cells failed and were marked stuck. **Amendment P.2-v2
masked the same defect**: its artifact had been produced out of band, so the
cell read "complete" by marker while its own launch log recorded rc=2. Fixed,
with regression tests that simulate the launcher's environment — the earlier
tests set only `REMOTE_RUNS_ROOT` and so could not see it.

### 5b. Amendment Q.2 result: MIXED, and the mechanism clause stays as decided (2026-09-21)

The second Meeting-#3 read. Applied exactly as pre-registered on 2026-09-12,
before any onset profile existed. Artifact
`results/diag/amendment-q/onset_surface_q2.json`.

**Verdict: MIXED.** At budget 256 the slope of divergence on exposure excludes
zero at all three read positions but **changes sign** — `+4.15e-06` at position
2304, `−8.88e-07` at 3328, `−1.35e-06` at 4352. At budget 2048 no cell excludes
zero. Not RATIO-ONLY (cells exclude zero), not HORIZON-PRESENT (sign is not
consistent). Per the branch: report the interaction, claim neither main effect.

**The headline consequence is a non-event, and that is worth stating plainly:
the mechanism clause is NOT reopened.** Amendment Q.2 pre-committed that a
HORIZON-PRESENT outcome would partially relicense what M.3 delicensed and reopen
the candidate-A decision of 2026-09-08. MIXED does not, so candidate A stands
unchanged and for its original reasons. This is the outcome that required no
paper edit, and no paper edit was made.

**The obvious confound was checked and did not explain it.** Deeper read
positions retain only longer traces, so pids fall 19 → 14 → 10 and 192 of 400
traces never reach the deepest position — meaning the full panel compares
positions on *different problem populations*. A balanced panel (the 116 traces
readable at every position; 10 problems throughout) was computed as a
diagnostic, and the **sign change survives on the fixed population**
(`+4.47e-06` → `−9.78e-07` → `−1.35e-06` at budget 256, all excluding zero). So
MIXED is a real interaction, not trace-length attrition.

The balanced panel is a **diagnostic, never a re-cut**: the verdict is taken on
the full panel as pre-registered. It exists only to say whether the variation
driving that verdict is a population artifact. It is not.

**Binding scope.** MIXED claims no main effect, so nothing here supports either
"exposure contributes nothing at fixed dose" or "exposure is separable from
dose." Slopes are per-step TV per decode step (~1e-6 at budget 256), not
accuracy-scale; Definition 6 applies. The manipulated quantity is **exposure to
the compression process**, never "horizon" — in a staggered arm the early
entries were never subject to eviction, so the retained set is drawn from a
different history. Scope: one model, MATH-500, two budgets, three read
positions, 10–19 problems per cell.

### 5c. Meeting-#3 Task 1: the damage-vs-cache-size curve, built from rows already on disk (2026-09-23)

**The graph the original Task 1 asked for had never been drawn.** Its ingredients
existed — the budget-512 and budget-1024 grid and readout cells ran in July — but
their committed artifacts held only resolution spreads, not per-press accuracy.
`report_m4.py` was re-run over those existing rows (a derived report over valid
rows, not new measurement) and archived as
`results/diag/amendment-k/m4-512-summary.json` and `m4-1024-summary.json`. Both
re-derive **IDENTICAL** from the raw rows under `rederive_artifacts.py`, and both
use the same Amendment-K cell reference as every other unit (FC 0.885, readout
reference 0.903). Figure: `figures/fig8_budget_curve.{pdf,png}` (not in the paper).

| budget | press | free-run | total damage | readout loss | trajectory residual [95% CI] |
|---|---|---|---|---|---|
| 256 | Random | 0.422 | 0.463 | 0.138 | −0.337 [−0.404, −0.262] |
| 512 | Random | 0.680 | 0.205 | 0.020 | −0.189 [−0.252, −0.129] |
| 1024 | Random | 0.800 | 0.085 | 0.000 | −0.087 [−0.133, −0.048] |
| 2048 | Random | 0.850 | 0.035 | 0.003 | −0.033 [−0.065, −0.010] |
| 256 | StreamingLLM | 0.237 | 0.647 | 0.319 | −0.344 [−0.405, −0.270] |
| 512 | StreamingLLM | 0.477 | 0.408 | 0.082 | −0.334 [−0.394, −0.263] |
| 1024 | StreamingLLM | 0.688 | 0.198 | 0.005 | −0.196 [−0.263, −0.135] |
| 2048 | StreamingLLM | 0.807 | 0.078 | 0.000 | −0.079 [−0.125, −0.042] |

**1. The requested negative correlation holds, monotonically, for both policies.**
Damage falls at every step from 256 to 2048. The full cache is zero **by
construction** (full cache against itself), drawn as a hollow anchor and not
claimed as a measurement.

**2. The curve hides the finding, and the decomposition exposes it: readout
damage closes by 512–1024 while trajectory damage persists.** Readout loss is
already ≤0.02 for Random at 512 and ≤0.005 for both at 1024 — held-trace
evaluation reports essentially lossless compression from budget 1024 up. The
residual is still −0.087 (Random) and −0.196 (StreamingLLM) at 1024, with CIs
excluding zero. **StreamingLLM's residual barely moves from 256 to 512
(−0.344 → −0.334) while its readout loss collapses (0.319 → 0.082).** So in the
256–1024 range the readout-style evaluation the method papers use has already
declared near-losslessness while the free-running loss is still large — the
paper's thesis, now visible as a function of budget rather than at one budget.

**3. Divergence is not on this curve, and cannot be yet.** Per-step TV exists at
budgets 256 and 2048 only (AUC-TV Random 0.108 → 0.013, StreamingLLM 0.103 →
0.019); the 512 and 1024 cells were generation-only and have no replay profiles.
Panel (c) of the figure shows the two measured points and draws no curve
between them. A divergence-vs-budget curve needs teacher-forced replay at 512
and 1024 — cheap (no generation, traces exist) but a new measurement, so it
needs a pre-registered read first.

**Scope and what this is not.** Two policies, MATH-500, one model, four budgets.
Budget and compression ratio move together here (`r = B/t`), so this is a
dose curve: it cannot by itself separate "more cache" from "less compression
over the trace", which is exactly why Amendments Q.1/Q.2 exist. The residual is
"accuracy change unexplained by on-reference readout", never a measured causal
effect; these are forward-readout residuals without the reverse bracket, which
exists at 256 only.


<!-- AUDIT:ARCHIVAL-END -->

### 5d. R.17 corrections to the paper (2026-09-27): reinsertion retracted and rewritten; locking reworded per R.2

**0.1 Reinsertion (§4.8, "Follow-up probes").**

- **Retracted.** The P.5 cell stashed the tail of a score-ordered tensor, not
  the evicted entries (PREREGISTRATION R.5). Its specificity control and its
  matched-arm null no longer carry any claim. The figures 6.49× (Random) and
  5.49× (StreamingLLM) (`p5.*.random_over_baseline`, from
  `results/diag/amendment-p/restoration_diss256_v2.json`) and the sentence "the
  cache is acutely sensitive to which entries occupy it" now appear only inside
  the retraction.
- **Old (removed):**
  > "The specificity control is where the information is. Re-inserting the same
  > number of entries drawn at random from the evicted pool … moves divergence by
  > −0.249 (Random) and −0.244 (StreamingLLM), which is 6.49× and 5.49× their
  > respective baselines. The cache is acutely sensitive to which entries occupy
  > it, while restoring the specific entries a policy just evicted is close to
  > neutral."

  Also removed: the matched-arm null −0.0013 [−0.0019, −0.0007] and
  −0.0015 [−0.0022, −0.0009], and "The instrument gate passed on all 800
  profiles". That gate checked re-inserted against stashed entries, not
  stashed against evicted ones.
- **New.**
  - Identity-repaired exact-lag read (`results/diag/meeting3/exact_lag.json`,
    R.10). At no lag and for neither policy does the matched arm meet the
    pre-registered restoration criterion.
  - Random, lag 1: matched **+0.0031 [+0.0004, +0.0061]**; the random draw
    gives +0.0044 [+0.0010, +0.0085] (NON-SPECIFIC).
  - Random, lags 4–256: every matched CI includes 0.
  - StreamingLLM: every matched CI includes 0.
  - Scope, from `results/diag/meeting3/r17/exact_lag_eligibility.json`: 8 of 25
    problems (32 of 100 traces) are eligible; budget 256 is a 3.4% realized
    cache on these traces. Full-cache accuracy is **0.594 [0.250, 0.875]**
    eligible against **0.971 [0.926, 1.000]** ineligible, so the eligible
    problems are long-prompt and low-accuracy. Exploratory.
  - The text states that no numeric precision floor has been measured yet
    (R.17.0.5), and that the redesigned probe (R.17.2.3) has not run.
- **Also qualified:** the "A diagnosis of ours was wrong" paragraph now notes
  that the corrected P.5 cell itself stashed the wrong entries, so it records a
  diagnosis, not a result.
- **Pinned:** 11 new claims in `paper_claims.json` (keys `p5.*` and
  `m3.el.*`). The trigger step 130 is listed as a protocol constant.

**0.2 Eviction-locking wording (§ threats, "A windowing defect…").**

- **Old:**
  > "\textbf{The locking is amplitude-dependent, and we qualified it because a
  > pre-registered control said to.} … the outcome is the branch that
  > constrains us: **locking is established at the deployed amplitude and is not
  > detectable at amplitude 16, and amplitudes below 64 are otherwise
  > untested.**"

  It also said: "Within a prune cycle the divergence is locked to the eviction
  event at the compression interval this paper deploys." The `**…**` was
  Markdown inside LaTeX and printed literal asterisks; that is fixed too.
- **New:**
  > "\textbf{Locking is detected at two amplitudes and is not yet validated.} …
  > \textbf{phase locking is detected for Random at compression intervals 64 and
  > 256 and not detected at 16; amplitudes below 64 are otherwise untested.} The
  > phase statistic is invariant to circular rotation, so it cannot separate
  > physical removal from the selection that precedes it (Amendment R.2). The
  > direct control that does separate them … has not yet validated this
  > amplitude-dependent reading, so we report the locking as detected, not
  > established."

  The earlier sentence now reads "the Random divergence is detectably
  phase-locked to the eviction event …", and StreamingLLM is called
  indeterminate.
- **Numbers unchanged** (`results/diag/amendment-q/shape_sawtooth_i{16,64,256}.json`):
  - Random I=16: +0.0110 [−0.0252, +0.0300];
  - Random I=64: +0.1193 [+0.0446, +0.1410];
  - Random I=256: +0.3183 [+0.1881, +0.3361].

**PDF.** Rebuilt twice: no `^!` errors and no undefined references. One
sub-point overfull box in §3 predates these edits (it is identical in the HEAD
build). The page count is unchanged.

### 5e. R.17 reads on existing data, and the mechanism-section draft (2026-09-27)

Target venue: proposed ICML 2027, pending supervisor confirmation. All reads
were registered in PREREGISTRATION R.17 (`ac78eab`) before they ran.
Artifacts are under `results/diag/meeting3/r17/`.

- **R.17.1.2 D_acc onset, restricted to reference > 2304 tokens (not blind;
  pre-committed reading applied): NOT ESTABLISHED for both policies.**
  - Random: +0.110 [+0.030, +0.200] unrestricted → **−0.048 [−0.184, +0.075]**
    restricted.
  - StreamingLLM: +0.190 [+0.080, +0.310] → **+0.026 [−0.066, +0.118]**.
  - 60/100 pairs retained, full-cache accuracy 0.798 [0.623, 0.939], 0
    attrition. Source: `dacc_restricted.json`.
- **R.17.1.5 (b256, exploratory).**
  - **H2 NEITHER:** spread(M) 0.3147 against half-width 0.0312; spread(Δ)
    0.0984 against 0.0223.
  - **H3 HOLDS:** M(H2O-style) − M(Random) = −0.315 [−0.330, −0.298], while
    H2O-style's free-run total change is −0.637 against −0.463.
  - Confirmatory at b2048 is pending. Source: `mdelta.json`, which also holds
    the first-prune control (layer-0 Δ exactly 0.0) and the per-layer profiles.
- **R.17.1.7 ω.**
  - Per-layer profiles are in `omega_layers.json`.
  - Addendum D: log(‖δ⊥‖/‖δ∥‖) Random − StreamingLLM = +0.080 [+0.064, +0.092],
    so the contrast is carried by proportionally larger off-subspace
    displacement.
  - k-sensitivity is not computable on b256 (norms at k = 64 only).
- **R.17.1.8 bridge: pre-commitment NOT CONFIRMED.**
  - AUROC 0.478–0.659; smallest Benjamini–Hochberg q 0.165. ω ≈ 0.5.
  - Δ's positive AUROC does not survive BH.
  - Source: `bridge.json`.
- **R.17.1.3 AUC-TV CIs.**
  - b256: Random 0.1075 [0.0906, 0.1240], StreamingLLM 0.1028
    [0.0831, 0.1235].
  - b2048: Random 0.0128 [0.0063, 0.0201], StreamingLLM 0.0186
    [0.0091, 0.0291].
  - 512/1024 are pending (t1-tv). Source: `auc_tv.json`.
- **Collateral.** The instrument is validated by a positive control (1.0). The
  b256 zero is empirical. Whether real attention at 256 can produce collateral
  is not established at this n, and the rank diagnostic is recorded as
  missing.

**Mechanism-section draft (P.1).**
- It sits in `kv_readout_vs_trajectory_paper.tex` inside a marked `\iffalse`
  block, so it is not compiled. Its 32 literals are pinned in
  `paper_claims.json`. It states only reads that have been read, and names
  every pending cell as pending.
- **Discrepancy recorded, not resolved silently:** the instruction said the
  draft replaces "the current §4.4". In the compiled PDF §4.4 is the
  reverse-replay bracket, a core estimand result. The draft therefore targets
  §4.5 "Where the divergence is…" (`sec:mechanism`) and §4.8, and leaves §4.4
  untouched. Adopting it is a pending decision.
- Page plan: `docs/strategy/ICML2027_PAGE_BUDGET.md`.

## 5. Framing discipline (non-negotiable — apply throughout)
- **Divergence ≠ damage.** ADD / TV / residual are "behavioral-fidelity loss," never "damage."
  Correctness is the only quality arbiter. Carry the **signed flip-mass** (gained vs lost correct
  answers) with every divergence quantity.
- **Q.1 phase excess: per-arm exclusion of zero only** (pinned 2026-09-15). Raw phase amplitude
  scales with the number of phase buckets, so excess magnitudes from arms with different
  compression intervals are NOT comparable. Never write that the excess grows with amplitude.
  Locking is "established at the deployed amplitude (64), not detectable at 16, untested below
  64" — and "not detectably locked", never "not locked", because the I=16 interval excludes an
  I=64-sized effect but not a small one.
- **The divergence profile is a localization, not a mechanism for the rise** (pinned 2026-09-08,
  §4x). Say "small in the steps immediately following an eviction and far larger later," or
  "readout-shaped protocols sample it where it is smallest." **Never** "compounds," "slow drift,"
  "grows with horizon," or "pre-compounding regime" — M.3 showed the rise does not survive
  ratio-matching, so every one of those attributes the growth to a horizon the evidence cannot
  carry. Two further rules travel with this: (a) the exactly-zero pre-eviction TV is an
  **instrument property** (true by construction under teacher forcing) and is spent as the
  replay harness's self-control — never promote it to a finding or into the thesis; (b) the
  within-cycle eviction-locking (§4w) and the ratio-effect envelope (§4v) never travel alone.
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

  **Scale exception (added 2026-08-30, do NOT "correct" it back).** The rule
  fixes how a number is rounded, not that every quantity must be shown at three
  decimals regardless of magnitude. The Amendment P.5 matched- and relaxed-arm
  effects and their CI bounds are shown at **four** decimals in the paper,
  because at three each point estimate lands on one of its own displayed
  interval bounds (Random `-0.001` against an upper bound of `-0.001`;
  StreamingLLM `-0.002` against a lower bound of `-0.002`), which makes a
  correct null read as a malformed interval. The values are still `f"{v:.4f}"`
  of the artifact float, so the anti-hand-rounding guarantee is unchanged; only
  the width differs. This is the single quantity family where that applies.
<!-- AUDIT:ARCHIVAL-BEGIN -->
  Historical instances caught under this rule, kept because they are how the rule was
  learned: H2O-style free 0.248 → 0.247 (99/400 = 0.2475) and SLM@2048 free 0.808 →
  0.807 (323/400 = 0.8075), both found by eyeballing Fig 1's hand-rounded constant
  against Fig 2's DATA-mode value for the same quantity. Four more were found in Unit 3
  (2026-07-27): −0.498/−0.428/−0.313/−0.178 hand-rounded where `.3f` gives
  −0.497/−0.427/−0.312/−0.177.
<!-- AUDIT:ARCHIVAL-END -->

## 6. Limitations / threats to validity (write this as a STRENGTH section)
- **Generality (primary):** the primary decomposition is one model / one benchmark / budget 256
  with two eviction methods. Budget-2048 (dose-response), H2O, R-KV and AIME are **complete** and
  widen it (all read in §4-CURRENT); the scope sentence is "one model family spanning three scales
  (1.5B/7B/14B), plus a Llama-base check." Method-invariance and generality claims remain
  exploratory at this scope.
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
  +0.023 unmatched, measured; Branch A). **Superseded for citation:** the
  current matched figure is **+0.003** (§4-CURRENT). The +0.023 values in this
  changelog are historical records of what was measured on those dates.
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
- **DONE (added 2026-08-15):** Meeting #2, all four tasks (§4u, §4v). Task 1 —
  P.2 replicated for Random, **StreamingLLM WITHDRAWN**, R-KV marginal. Task 2 —
  **NO RESTORATION** (gate 800/800). Task 3 — **BUDGET EFFECT ONLY**. Task 4 —
  crossing test **degenerate by construction**. Plus M.3 — the staircase is a
  **compression-ratio effect**, which delicenses the mechanism reading of the
  divergence profile (see the restatement flag in §2). Both powered cells were re-runs after
  defects in our own instruments; both v1 cells invalidated wholesale, never
  patched. **Unresolved at the time; RESOLVED 2026-08-27** — Amendment P.2-v2
  closed it on branch A (eviction-locking survives ratio-matching); the figures
  are in §4w. **All Meeting-#2 items are now closed.**
- **ORIGINAL CAMPAIGN + AMENDMENT J COMPLETE 2026-07-24 — that queue fully
  drained.** (Superseded as a statement about the project as a whole: the
  Meeting-#2 cells ran afterwards. See the 2026-08-15 entry below.) Endgame only from here: anonymization scrub,
  restore double-blind author block, arXiv, ICLR conversion
  (`docs/strategy/SUBMISSION_STRATEGY.md` Phase 11). Drop-dead 2026-08-01
  met with 8 days to spare.

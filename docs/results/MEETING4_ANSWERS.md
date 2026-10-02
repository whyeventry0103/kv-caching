# Meeting #4 questions — answers with evidence

First written 2026-10-02; corrected the same day after the independent
review of `f69dce3` (wording narrowed, paired uncertainty added, matched-key
comparisons added, the Llama arithmetic fixed, the retention simulation
committed with its device). Every number was recomputed on the GPU host from
the raw rows by `scripts/audit_meeting4.py`; the outputs are the JSON files in
`docs/results/meeting4_evidence/`. Nothing is quoted from an earlier write-up.

## Question 1 — Does the contradiction exist, or is it a bug?

**The numbers reproduce, and nothing we checked is a bug.** That is a bounded
claim: it covers the stored rows, the scoring path and the generation path as
described below, not every possible defect.

### 1a. What was checked, and what it establishes

| Cell (budget 256, 100 problems × 4 seeds) | Rows | Duplicates | Stored accuracy | Re-scored accuracy | Mismatches | Truncation |
|---|---|---|---|---|---|---|
| Random | 400 | 0 | 0.4225 | 0.4225 | 0 | 0.233 |
| StreamingLLM | 400 | 0 | 0.2375 | 0.2375 | 0 | 0.070 |
| H2O-style | 400 | 0 | 0.2475 | 0.2475 | 0 | 0.105 |
| R-KV | 400 | 0 | 0.5200 | 0.5200 | 0 | 0.135 |
| Full cache on the same 400 keys | 400 | 0 | 0.8850 | — | — | — |

(`accuracies.json`; the 512/1024/2048 cells are in the same file.)

- *Re-scoring.* Each row's raw trace went through answer-marker detection,
  `extract_answer` and the Hendrycks `is_equiv` grader against the dataset's
  gold answer, loaded fresh, with the truncation rule. **3,200 rows over
  eight 400-row cells, 0 mismatches.** This establishes that the stored
  `correct` flags are consistent with the project's grader. It is not an
  independent check of answer semantics (a different grader could disagree
  on some normalisations), and the full-cache rows were not re-scored.
- *Readouts.* Forward readout (0.765 / 0.584 / 0.597 / 0.798 pooled over
  the 392 defined conditions) and reverse readout re-score with 0 mismatches
  (`readouts.json`).
- *Same code then and now.* The September D_acc cell regenerated Random and
  StreamingLLM at budgets 256 and 2048 with the current runner and was
  token-identical to the June rows on all 200 shared generations (R.8.3,
  `results/diag/meeting3/dacc.json`).
- *Compression was active.* Teacher-forced replays of these cells show
  divergence exactly 0 before the first eviction and above 0 after it, for
  every policy (`replay-diss256*`).
- *The ordering replicates.* Random > StreamingLLM holds at budgets 512,
  1024 and 2048 and on AIME-24 (0.283 vs 0.142); R-KV's lead holds at 1.5B
  and 14B. It reverses on Llama-8B, where Random collapses into filler loops
  (PAPER_FINDINGS §4o): model-dependent, not a code artefact.

### 1b. The policy gap, with its uncertainty

The residual is a descriptive decomposition, free-run minus on-reference
readout on the conditions whose reference trace has an answer boundary. It is
not a measured causal "trajectory damage". Paired by (problem, seed), with
equal-problem means and problem-cluster bootstrap CIs (`paired.json`):

| Contrast (a − b), budget 256 | Free-run (400 pairs) | Readout (392 pairs) | Residual (392 pairs) |
|---|---|---|---|
| Random − StreamingLLM | +0.185 [+0.130, +0.240] | +0.179 [+0.118, +0.243] | +0.006 [−0.078, +0.088] |
| Random − H2O-style | +0.175 [+0.120, +0.233] | +0.167 [+0.103, +0.233] | +0.006 [−0.081, +0.093] |
| R-KV − Random | +0.098 [+0.045, +0.155] | +0.031 [−0.028, +0.085] | +0.069 [−0.006, +0.144] |
| H2O-style − StreamingLLM | +0.010 [−0.038, +0.063] | +0.013 [−0.053, +0.078] | 0.000 [−0.078, +0.078] |

So: Random's advantage over StreamingLLM and over H2O-style is carried by
the readout channel (CI excludes 0) and the residual difference is not
distinguishable from 0 at this n (CI [−0.08, +0.09]: that is "not detected",
not "equal"). R-KV's advantage over Random sits partly in the residual, with
a CI that touches 0.

**On the same conditions the capture metrics were measured on.** M, Δ and ω
come from 25 problems (and at budget 2048 from the 21-problem / 69-trace set
with a measured eviction event). The accuracy rankings on exactly those keys
(`matched_keys.json`, equal-problem means, budget-256 cells):

| Keys | Policy | Free-run | Readout | Residual |
|---|---|---|---|---|
| capture 25 problems (100 conditions) | Random | 0.37 [0.22, 0.54] | 0.727 [0.58, 0.86] | −0.37 [−0.54, −0.19] |
| | StreamingLLM | 0.25 [0.12, 0.40] | 0.54 [0.39, 0.69] | −0.30 [−0.42, −0.18] |
| | H2O-style | 0.18 [0.08, 0.30] | 0.57 [0.41, 0.73] | −0.39 [−0.52, −0.26] |
| | R-KV | 0.37 [0.21, 0.56] | 0.73 [0.56, 0.87] | −0.36 [−0.49, −0.23] |
| 2048-eligible keys (69 conditions) | Random | 0.31 [0.14, 0.50] | 0.65 [0.46, 0.82] | −0.35 [−0.58, −0.10] |
| | StreamingLLM | 0.11 [0.02, 0.23] | 0.36 [0.21, 0.52] | −0.26 [−0.44, −0.07] |

The ordering Random > StreamingLLM holds on the capture keys too, with
overlapping CIs at n = 100 and n = 69. The relationship between the capture
metrics and accuracy is therefore being discussed on the same population,
but with the uncertainty those n carry.

### 1c. What the metrics measure, and what "Random" actually keeps

M, Δ and ω are per-step perturbation measures taken under teacher forcing:
they say how much each eviction disturbs the computation at that step. They
agree with the trajectory side of the picture: Random has the highest
truncation rate (0.233 vs 0.070) and the longest generations (mean 8,044 vs
5,274 tokens). They do not measure which information survives to the end.

What each press retains is a property of its code, and for Random it is not
what the name suggests. `kvpress.RandomPress` reseeds its generator at every
prune, so for a given cache size each cache *slot* has a fixed score, and
`ScorerPress` gathers the top-k in score order. Eviction is a fixed pattern
over slots, not a fresh uniform draw. Because the scores depend only on
shape, dtype, device and seed, the retention process can be simulated
exactly with the press's own `torch.rand` call and no model
(`kvcache.metrics.retention`; `retention.json` holds the run on the GPU
host's CUDA device, torch 2.7.1+cu126, bf16, prompt 94, budget 256, interval
64, 4 KV heads, 4,096 steps; the CPU generator gives a different score
vector with the same structure, also recorded):

| After 4,096 steps (CUDA run) | Random | StreamingLLM |
|---|---|---|
| Of the last 64 tokens, kept per KV head | 48–55 | 64 |
| Survival of the 64-token cohort entering at step 1,024, after 64 / 256 / 512 / 1,024 steps | 0.62–0.70 / 0.27–0.33 / 0.12–0.16 / 0.02–0.05 | 1 / 0 / 0 / 0 |
| Median / 90th-percentile age of a cached token | 198 / 757 | 128 / 230 |
| Tokens older than 1,024 steps, per head | 6–21 | 4 (the sinks) |
| Distinct positions held across the 4 KV heads | 566 (2.2× the budget) | 256 (1×) |
| Prompt tokens kept per head (of 94) | 1–3 | 4 |

Random is a soft-recency sieve with a long sparse tail that differs per KV
head: it keeps most of the recent window, a geometrically decaying sample of
older tokens, and across heads covers about twice as many positions.
StreamingLLM is a hard window: four sinks plus the last 252 tokens, identical
in every head. Both lose the prompt within about a thousand steps, and an
evicted entry stays absent unless it is explicitly restored.

The real rows agree with the simulation where they can be compared: in
restore-v3 at step 1,024, Random's stash holds a median 63 of a median 64
prompt entries per head (`results/diag/amendment-s/origin.conditions.tsv`).

### 1d. Hypothesis, and what the direct tests found so far

The hypothesis that follows from 1b–1c: at each prune Random evicts some of
the most recent, most attended tokens (high M and Δ at that step), but what
it holds between prunes is a longer, head-diverse sample of the context,
while StreamingLLM holds only the last 252 tokens; the readout channel, which
needs the earlier derivation and the question, is where that difference
shows. Forward readout accuracy by reference-trace length is consistent with
it (`by_length.json`): StreamingLLM 0.80 / 0.48 / 0.31 against Random
0.92 / 0.70 / 0.68 for traces of 1–2k / 2–4k / 4–8k tokens. This is a
hypothesis about accuracy, with two direct tests registered in Amendment S:

- **S.2 restoration by origin: read, FINAL** (MEETING3_RESULTS §3e). For
  Random, returning the evicted prompt entries recovers 37% [31%, 42%] of
  what returning every evicted entry recovers, against 22% [16%, 28%] for an
  equal-sized random draw of trace entries: prompt entries carry more per
  entry (PROMPT-SPECIFIC), and they do not carry most of it (PARTIAL). For
  StreamingLLM the prompt entries recover no more than an equal number of
  trace entries (NON-SPECIFIC). Divergence, not accuracy; and the Random
  prompt arm carried a material pad of trace entries in some heads (up to
  25), reported in §3e.
- **S.1 prompt-protected decomposition: running.** Free-run at budget 256 is
  complete; the forward readout is pending, so the read is PARTIAL and
  carries no label. The accuracy question is answered there, not here.

**What to say to the supervisor.** The numbers are right. M, Δ and ω rank the
policies by per-step disturbance; accuracy ranks them by what survives, and
the two rankings part in the readout channel (paired difference CI excludes
0) while the residual difference is not detected. Whether the surviving
*prompt* is the reason is being tested directly; the divergence-level test
says it is part of the answer for Random and not for StreamingLLM.

**Paper wording to change.** Our "Random" is kvpress's `RandomPress` with
per-prune reseeding: a fixed per-slot pattern that behaves as a soft-recency
sieve. The paper should say so rather than imply memoryless uniform eviction.

## Question 2 — How ω is computed

ω asks: when the compressed model's hidden state differs from the full
model's at the same token, does it differ along directions the full model
itself uses, or off them?

1. **Basis fit (`scripts/run_basis_fit.py`, Amendment R.6).** Teacher-force
   each full-cache reference trace of the frozen 25-problem × 4-seed subset
   once. At up to 256 decode positions fixed from the trace length alone,
   record the output of every decoder block (28 layers × 3,584 dims) at the
   prediction position. Weight problems equally, seeds equally within a
   problem, positions equally within a trace. Per layer, in float64:
   standardise each coordinate by its mean and standard deviation; take the
   exact eigendecomposition of the standardised covariance
   (`numpy.linalg.eigh`); keep the top k = 64 eigenvectors as an orthonormal
   basis U. The basis is frozen (sha256 in the run header) before any
   compressed trace is projected; the compressed condition is never seen by
   the fit.
2. **Per-step split (`capture.split_norms`).** Replay a trace under a policy.
   At each decode forward and layer, δ = h_comp − h_full. Standardise:
   z = δ/σ. Project: c = Uᵀz; δ∥ = Uc; δ⊥ = z − δ∥. Record ‖δ∥‖ = ‖c‖ and
   ‖δ⊥‖.
3. **ω per step and layer** = ‖δ⊥‖ / ‖δ∥‖. Steps with δ = 0 (before the first
   eviction) are undefined and excluded: 2.3M of 48.6M step-layers at budget
   256, 20.1M at 2048 (where 31 traces per policy are never compressed).
   Unbounded (‖δ∥‖ = 0, ‖δ⊥‖ > 0) never occurred.
4. **ω per trace** (`capture.omega_trace_value`): mean over positions within
   each layer, then an unweighted mean over the 28 layers.
5. **The registered contrast:** ω(Random) − ω(StreamingLLM), paired by
   (problem, seed), equal-problem mean, problem-cluster bootstrap CI.
   +0.200 [+0.159, +0.234] at 256; +0.225 [+0.178, +0.273] at 2048; the
   ordering holds at every nested k from 8 to 256.

Caveats to state: ω is a ratio, and at 256 the contrast comes from larger
‖δ⊥‖, not smaller ‖δ∥‖; the basis explains 21–41% of variance per layer, so
"parallel" means "inside the dominant full-cache subspace". ω is geometry,
not a calibrated measure of harm. By registration it carries no accuracy
claim. The bridge tests (per-trace ω against free-run truncation, AUROC 0.54
and 0.48, none of six tests surviving correction) mean a predictive
association is **not established**, not that one has been ruled out.

## Question 3 — What collateral = 0 means (and does not mean)

**It does not mean the query heads agree.** The registered definition (R.4,
`capture.collateral_stats`): an evicted position is *collateral* iff (a) at
least one query head of the KV group gave it more than τ = 0.01 of its
full-model attention, **and** (b) the *group-mean* full-model attention
ranks it below the budget cut among the pre-prune candidates. Condition (a)
alone, `wanted_any_head`, is far from zero:

| Policy | Evicted (position, group, layer, event) instances checked | Wanted by some head (> 1%) |
|---|---|---|
| Random | 46,451,328 | 3,731,830 |
| R-KV | 46,451,328 | 1,294,665 |
| StreamingLLM | 46,451,328 | 739,823 |
| H2O-style | 46,451,328 | 54,932 |

Heads wanted millions of the evicted positions. The zero is condition (b).
A position one head attends at > 0.01 has a group mean of at least
0.01/7 ≈ 0.0014 under a 7-to-1 grouping. To rank outside the top 256 of
about 320 candidates, 256 other candidates would each need a group mean
above that, i.e. more than 36% of the group's attention spread flat over
256 positions. Attention is concentrated, so at budget 256 this never
happened on this model; the positive control (a synthetic event built to
violate it) returns 1.0, so the instrument can fire. At budget 2048 the zero
is forced outright (fewer than 700 positions can exceed 1/700). Heads also
disagree measurably: the per-head M within a group spreads by 0.145
(Random) to 0.075 (H2O-style) at 2048 (`r17/mdelta_capture-b2048.json`).

**Training or grouping?** Neither forces agreement. GQA trains 7 query
heads on one shared K/V, so they attend over the same keys, but their
attention *patterns* differ (the dispersion above). What makes collateral
zero is the metric's threshold against the group-mean cut at an 80% keep
fraction, plus the concentration of attention. A per-head top-k criterion
would be more sensitive; it was not registered and the per-position ranks
were not stored in capture-v1, so it needs a re-run.

**Llama-8B (4 query heads per KV head).** The same arithmetic is *tighter*,
not looser, than on Qwen: a position one head attends at > 0.01 has a group
mean of at least 0.01/4 = 0.0025, and 256 competitors above that would need
more than 64% of the group's attention spread flat. Under this bound
collateral is harder to realise on Llama than on Qwen. The Llama capture
(400/400, not yet read) reports collateral without reading it, as
registered.

## A note on the collapse test

The pooled ΔR² of budget after log r (0.029 and 0.033, CIs above 0) is an
in-sample gain from adding a term to a nested model, which is non-negative by
construction. It is descriptive, as the Meeting 3 write-up and the mechanism
draft already say ("suggested, not established"); it is not evidence of
explanatory significance without a held-out or null comparison.

## Reproduce

```bash
ssh kvcache-gpu 'cd ~/kv-caching && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  HF_HOME=~/kvcache-storage/cache/huggingface REMOTE_DATA_ROOT=~/kvcache-storage/data && \
  R=~/kvcache-storage/runs && for r in accuracies readouts by-length paired matched-keys; do \
  .venv/bin/python scripts/audit_meeting4.py --runs-root $R $r; done && \
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/audit_meeting4.py --device cuda retention'
```

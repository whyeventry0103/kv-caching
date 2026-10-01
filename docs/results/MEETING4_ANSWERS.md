# Meeting #4 questions — answers with evidence (2026-10-02)

Three questions from the supervisor meeting. Every number here was recomputed
on the GPU host from the raw rows on 2026-10-02 by the scripts in
`docs/results/meeting4_evidence/` (their output is archived there). Nothing is
quoted from an earlier write-up.

## Question 1 — Does the contradiction exist, or is it a bug?

**It exists, and it is not a bug.** The early accuracies reproduce exactly from
the raw rows, independently of every report script.

### 1a. The audit

| Cell (budget 256, 100 problems × 4 seeds) | Rows | Duplicates | Stored accuracy | Re-scored accuracy | Mismatches | Truncation |
|---|---|---|---|---|---|---|
| Random | 400 | 0 | 0.4225 | 0.4225 | 0 | 0.233 |
| StreamingLLM | 400 | 0 | 0.2375 | 0.2375 | 0 | 0.070 |
| H2O-style | 400 | 0 | 0.2475 | 0.2475 | 0 | 0.105 |
| R-KV | 400 | 0 | 0.5200 | 0.5200 | 0 | 0.135 |
| Full cache on the same 400 keys | 400 | 0 | 0.8850 | — | — | — |

"Re-scored" means: take each row's raw trace, re-run answer-marker detection,
`extract_answer`, and the Hendrycks `is_equiv` grader against the dataset's
gold answer (loaded fresh), and apply the truncation rule. 2,800 rows across
the 256/1024/2048 cells: 0 mismatches. Forward readout (0.765 / 0.584 / 0.597 /
0.798) and reverse readout re-score with 0 mismatches too, and the residuals
recompute to −0.337 / −0.344 / −0.344 / −0.268, the headline table's values.

Checks on whether the right thing ran:
- *Press mapping.* `build_press` maps `random` → `RandomPress`, `streamingllm`
  → `StreamingLLMPress`, `h2o` → `ObservedAttentionPress`, `rkv` → the
  certified port; the grid writes the config's press name into each row.
- *The June code path equals the current one.* The September D_acc cell at
  onset 0 regenerated Random and StreamingLLM at budgets 256 and 2048 with the
  current, audited runner and was **token-identical to the June rows on all
  200 shared generations** (R.8.3 determinism control,
  `results/diag/meeting3/dacc.json`). Whatever ran in June is what runs now.
- *Compression was active.* Teacher-forced replay of the same cells shows
  divergence exactly 0 before the first eviction and > 0 after it, for every
  policy (`replay-diss256*`; the pre-eviction zero is the harness
  self-control).
- *The ordering replicates.* Random > StreamingLLM holds at budgets 512, 1024
  and 2048 and on AIME-24 (0.283 vs 0.142); R-KV's lead holds at 1.5B and
  14B. It reverses on Llama-8B, where Random collapses into filler loops
  (PAPER_FINDINGS §4o): model-dependent, not a code artifact.

### 1b. Why Random does better despite the worst M, Δ and ω

The contradiction dissolves once two facts are on the table.

**Fact 1: the policy gap is in the readout channel, not the trajectory
channel.** Residuals (the trajectory damage) are the same for the three
policies, −0.337 / −0.344 / −0.344; what differs is readout accuracy, 0.765 /
0.584 / 0.597. M, Δ and ω are per-step perturbation measures taken during
teacher forcing; they say how much each eviction disturbs the computation at
that step. They do agree with the trajectory side: Random has the highest
truncation rate (0.233 vs 0.070) and the longest generations (mean 8,044 vs
5,274 tokens), which is the non-termination the ω hypothesis predicted at the
policy level. They say nothing about which *information* survives to the end.

**Fact 2: what "Random" actually keeps.** `kvpress.RandomPress` reseeds its
generator at every prune, so the score of a cache slot is a fixed number for a
given cache size, and after the first prune the slots are in score order. The
eviction is therefore a fixed pattern over slots, not a fresh uniform draw.
An exact simulation of that code (same shapes, same reseeding) at budget 256,
interval 64, prompt 94, gives after 4,096 steps:

| | Random | StreamingLLM |
|---|---|---|
| Of the last 64 tokens, kept per KV head | 46–55 | 64 |
| Survival of a 64-token cohort after 64 / 256 / 512 / 1024 steps | 0.6 / 0.35 / 0.15 / ~0 | 1 / 0 / 0 / 0 |
| Median age of a cached token | 210 | 128 |
| 90th-percentile age | 633 | 230 |
| Tokens older than 1,024 steps, per head | 4–14 | 4 (the sinks) |
| Distinct positions held by the union of the 4 KV heads | 558 (2.2× the budget) | 256 (1×) |
| Prompt tokens kept per head (of 94) | 0–4 | 4 |

So Random is a *soft-recency sieve with a long sparse tail that differs per
KV head*: it keeps most of the recent window, a geometrically decaying sample
of older context, and across heads it covers twice as many positions.
StreamingLLM is a hard window: four sinks plus the last 252 tokens, identical
in every head; everything older is gone from every head. H2O-style keeps old
heavy hitters and drops low-attention recent tokens, so it loses recency
instead.

**Why that makes M, Δ and ω high and accuracy good at the same time.** At
each prune Random evicts about a fifth of the newest 64 tokens, which the full
model attends to heavily; that is high attended mass (M) and a large attention
reshaping (Δ) *at that step*, and the hidden state moves off the full-cache
subspace (ω). But reasoning text restates what it needs, and the other heads
still hold copies, so the step-level damage is survivable, while the long tail
keeps earlier derivations reachable. StreamingLLM loses almost nothing the
model attends to at the moment of eviction (old tokens, low attention: low M,
low Δ), but after 252 tokens the question and every earlier step are gone from
every head, which is what a readout needs. The data say exactly this. Forward
readout accuracy by reference-trace length at budget 256:

| Reference length (tokens) | n | Full cache | Random | StreamingLLM | H2O-style | R-KV |
|---|---|---|---|---|---|---|
| < 1,000 | 33 | 0.939 | 0.909 | 0.879 | 0.909 | 0.939 |
| 1,000–2,000 | 132 | 0.992 | 0.924 | 0.803 | 0.712 | 0.985 |
| 2,000–4,000 | 137 | 0.891 | 0.701 | 0.482 | 0.562 | 0.788 |
| 4,000–8,000 | 65 | 0.862 | 0.677 | 0.308 | 0.400 | 0.538 |
| ≥ 8,000 | 25 | 0.560 | 0.320 | 0.320 | 0.280 | 0.360 |

StreamingLLM's deficit grows with the length of the context it has thrown
away; Random's holds until everything fails. This is the H3 result in another
form: the quantity an attention-score policy optimises (keep what is attended
now) is not the quantity that governs correctness (keep what will be needed).
Concurrent work (arXiv:2609.03430, "Random Attention") reports the same
ordering on four models and explains it the same way, with one addition we
are now testing directly: that the prompt is the fragile part (the
prompt-protected cells of Amendment S.1, and the restoration-by-origin cell
S.2, both running).

**What to say to the supervisor.** M, Δ and ω rank the policies by how much
they disturb each step. Accuracy ranks them by how much usable context
survives to the end. Random is worst on the first and second-best on the
second because it damages every step a little and nothing permanently;
StreamingLLM damages every step less and the past completely. Both rankings
are correct measurements of different things, and the decomposition already
shows where they part: in the readout channel.

**One wording change for the paper.** Our "Random" is kvpress's `RandomPress`
with per-prune reseeding, a fixed per-slot pattern. The paper should say so
("random as implemented by kvpress: a fixed per-slot score pattern reseeded at
each prune, which behaves as a soft-recency sieve with a long tail"), not
imply memoryless uniform eviction.

## Question 2 — How ω is computed

ω asks: when the compressed model's hidden state differs from the full
model's at the same token, does it differ *along directions the full model
itself uses*, or *off them*?

1. **Basis fit (`scripts/run_basis_fit.py`, Amendment R.6).** Teacher-force
   each full-cache reference trace of the frozen 25-problem × 4-seed subset
   through the model once (full cache, never pruned). At up to 256 decode
   positions fixed from the trace length alone, record the output of every
   decoder block (28 layers × 3,584 dims) at the prediction position. Weight
   problems equally, seeds equally within a problem, positions equally within
   a trace. Per layer, in float64: mean μ and per-coordinate standard
   deviation σ; standardise each coordinate ((x − μ)/σ, not covariance
   whitening); take the exact eigendecomposition of the standardised
   covariance (`numpy.linalg.eigh`); keep the top k = 64 eigenvectors as an
   orthonormal basis U (3,584 × 64). The basis is frozen (sha256 in the run
   header) before any compressed trace is projected; the compressed condition
   is never seen by the fit (circularity is a stop condition).
2. **Per-step split (`capture.split_norms`).** Replay a trace under a policy.
   At each decode forward and each layer, δ = h_comp − h_full (block outputs
   at the same token; the means cancel). Standardise: z = δ/σ. Project:
   c = Uᵀz (64 coefficients); δ∥ = U c; δ⊥ = z − δ∥. Record ‖δ∥‖ = ‖c‖ and
   ‖δ⊥‖ = ‖z − Uc‖.
3. **ω per step and layer** = ‖δ⊥‖ / ‖δ∥‖ (`latent_split.omega_with_reason`).
   A step with ‖δ∥‖ = 0 is counted separately (undefined if both are 0,
   unbounded if only ‖δ⊥‖ > 0) and excluded from the mean. Undefined steps
   are common and expected: they are the steps before the first eviction,
   where δ = 0 exactly (2.3M of 48.6M step-layers at budget 256, 20.1M of
   48.6M at 2048, where 31 traces per policy are never compressed).
   Unbounded never occurred.
4. **ω per trace** (`capture.omega_trace_value`): mean of finite ω over
   positions within each layer, then an unweighted mean over the 28 layers.
5. **The registered contrast**: ω(Random) − ω(StreamingLLM), paired by
   (problem, seed), equal-problem mean, problem-cluster bootstrap CI (2,000
   resamples, seed 0). PREDICTED / NULL / REVERSED was fixed before the data.
   Result: +0.200 [+0.159, +0.234] at 256; +0.225 [+0.178, +0.273] at 2048;
   ordering holds at every nested k from 8 to 256.

Two things to be clear about with the supervisor: (i) ω is a *ratio*, so the
contrast is "proportionally more off-subspace", and at 256 the log-ratio
decomposition shows it comes from larger ‖δ⊥‖, not smaller ‖δ∥‖; (ii) the
basis explains 21–41% of variance per layer at k = 64, so "parallel" means
"inside the dominant full-cache subspace", not "inside everything the full
model does". The registration forbids using ω for any accuracy claim or
calling it the mechanism; the bridge test confirmed per-trace ω does not
predict truncation (AUROC 0.54 / 0.48).

## Question 3 — What collateral = 0 means (and does not mean)

**It does not mean the query heads agree.** The registered definition
(R.4, `capture.collateral_stats`): an evicted position is *collateral* iff
(a) at least one query head of the KV group gave it more than τ = 0.01 of its
full-model attention, **and** (b) the *group-mean* full-model attention ranks
it below the budget cut among the pre-prune candidates. It measures one
specific failure of grouping: a position one head needed, which the group's
averaged view would also have discarded. Condition (a) alone,
`wanted_any_head`, is far from zero:

| Policy | Evicted (position, group, layer, event) instances checked | Wanted by some head (> 1%) |
|---|---|---|
| Random | 46,451,328 | 3,731,830 |
| R-KV | 46,451,328 | 1,294,665 |
| StreamingLLM | 46,451,328 | 739,823 |
| H2O-style | 46,451,328 | 54,932 |

So heads *do* want millions of the positions that were evicted. The zero is
condition (b): none of those positions ranked outside the group-mean top-256.
That is close to arithmetic. A position one head attends at > 0.01 has a
group mean of at least 0.01/7 ≈ 0.0014. To fall outside the top 256 of about
320 candidates it would need 256 other candidates each with group mean above
0.0014, i.e. more than 36% of the group's attention spread flat over 256
positions. Attention is concentrated (sinks and a few tokens carry most of
it), so this never happens on this model at this grouping and threshold. The
positive control (a synthetic event built to violate it) returns 1.0, so the
instrument can fire; the real attention never produces the configuration. At
budget 2048 the zero is forced outright (fewer than 700 positions can exceed
1/700). Heads also disagree measurably: the per-head M within a group spreads
by 0.145 (Random) to 0.075 (H2O-style) at 2048 (`r17/mdelta_capture-b2048.json`).

**Is it training, or grouping?** Neither forces agreement. GQA trains 7 query
heads against one shared K/V, so they attend over the same keys, but their
attention *patterns* differ (the dispersion above). What makes collateral
zero is the metric's threshold against the group-mean cut at an 80% keep
fraction, plus the concentration of attention. A more sensitive instrument
would use a per-head top-k criterion instead of a τ threshold against the
group mean; that was not registered and the per-position ranks were not
stored in capture-v1, so it cannot be computed from existing rows without a
re-run. The Llama-8B capture now running has a 4-to-1 grouping, where the
same arithmetic is looser (1/400 instead of 1/700); its collateral is
reported, not read.

## Reproduce

The scripts are archived as `.py.txt` (evidence copies, outside the lint
gate, like the tick scripts). Copy them to the host and run:

```bash
for f in audit_accuracies audit_readouts audit_by_length; do
  scp docs/results/meeting4_evidence/$f.py.txt kvcache-gpu:/tmp/$f.py; done
ssh kvcache-gpu 'cd ~/kv-caching && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  HF_HOME=~/kvcache-storage/cache/huggingface REMOTE_DATA_ROOT=~/kvcache-storage/data && \
  for f in audit_accuracies audit_readouts audit_by_length; do .venv/bin/python /tmp/$f.py; done'
```

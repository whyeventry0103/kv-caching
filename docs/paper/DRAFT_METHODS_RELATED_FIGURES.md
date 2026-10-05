# Draft: methods, related work, table shells, figure specifications (WP7 preparation)

Drafted 2026-10-05. This is not manuscript text yet.
- Every number refers to a claim-ledger ID (`docs/paper/CLAIM_LEDGER.md`), not to a value typed here.
- The abstract and headline are written only after the evidence freeze (plan §12.3).
- Wording follows the ledger's §E.

---

## 1. Methods (draft prose)

### 1.1 Question and operational design

We compare two ways of evaluating a KV-cache eviction policy C against the full cache F on the same problems and sampling seeds:
- **held-trace readout:** a reference reasoning trace is supplied;
- **free-running generation:** the model writes its own trace.

Crossing the source of the trace with the cache used to read it out gives four arms:

| | read out with F | read out with C |
|---|---|---|
| **trace written with F** | A_FF (full-cache bank) | A_FC (forward readout) |
| **trace written with C** | A_CF (reverse readout) | A_CC (compressed free run) |

The loss D = A_CC − A_FF splits two ways:
- readout first: (A_FC − A_FF) + (A_CC − A_FC);
- trajectory first: (A_CF − A_FF) + (A_CC − A_CF).

A_CC − A_FC and A_CF − A_FF are the same trajectory contrast, measured under the compressed and the full-cache readout. Their difference, I, is the 2×2 interaction.

We treat all of these as **operational comparisons**. They are not bounds on a causal "trajectory share". Such a reading would need assumptions about how conditioning on a reference trajectory biases each direction, which our data do not support.

### 1.2 Readout mechanics

**Forward readout.** The exact full-cache reasoning token IDs are teacher-forced under C up to the reasoning/answer boundary (the last `</think>`). The answer span is then generated under C.

**Reverse readout.** The same procedure in the other direction: the compressed free-run trace is teacher-forced under F, and the answer span is generated under F.

Teacher forcing rebuilds the cache states under the reading policy. It does not reuse the states of the original generation.

**Defined readouts.** A readout is defined only when its source trace has an answer boundary. That property is post-treatment for the reverse arm, because it depends on whether the compressed run terminated.

### 1.3 Prompt protection and cache accounting

**Budget.** B is the total number of resident KV entries per layer and KV head after each prune. It includes the prompt.

**Prompt-protected policies** never evict a prompt position:
- *Positions.* Positions are tracked absolutely, per layer and head, and checked at every prune.
- *Trace slots.* The policy's own scores choose among trace positions for the remaining B − P slots.
- *Short budgets.* When P > B − 64, the target is relaxed to P + 64. Those problems are flagged, and a sensitivity analysis excludes them.
- *Prune schedule.* Every 64 decoding steps.

**Policies:**
- *StreamingLLM* keeps 4 sinks plus the recent window; protected, it keeps the whole prompt plus the recent window.
- *Random* (legacy kvpress) reseeds its generator at every prune, so it evicts a fixed slot pattern rather than drawing fresh uniform samples. We call it "legacy Random" wherever the distinction matters.
- *H2O-style* is kvpress `ObservedAttentionPress`: observed current attention, no recency window. It is not canonical accumulated H2O.

### 1.4 Populations, estimators, uncertainty

Free-run accuracy is reported on every scheduled condition. Truncations and missing answers count as incorrect.

Each readout contrast is reported:
- on its own defined support, against the full-cache bank restricted to that same support;
- on the common support where all four arms are defined;
- as worst-case bounds over the whole population, with every undefined readout imputed anywhere in [0, 1].

**Estimator.** Every estimate is an equal-problem mean: seeds averaged within problem, then problems averaged. Intervals are 95% percentile intervals from a problem-cluster bootstrap (2,000 resamples, seed 0). Quantities from one analysis share their bootstrap draws, so differences and ratios stay paired. Families of pairwise policy contrasts are Bonferroni-adjusted.

**Practical relevance.** Tolerances of 1, 3 and 5 accuracy points were proposed after the B=256 result was seen, so they are reported as sensitivities. A "false reassurance" finding needs both:
- the readout loss's upper one-sided bound at or below the margin;
- the free-run loss's lower one-sided bound above it.

Equivalence needs the 90% interval to fall inside ±margin.

### 1.5 Grading

**Primary grader.** The frozen Hendrycks `is_equiv` on the last boxed answer.

**Secondary sensitivity.** A format-aware score, specified after inspecting bank errors (`docs/protocol/QWEN_SCORING_SENSITIVITY.md`):
- the answer is taken only from the post-reasoning region;
- a fixed list of rules applies, each with an ID;
- ambiguous rows keep their primary score and are reported as bounds.

Any cross-model comparison uses the same grader on both sides.

### 1.6 Instrument validity

Divergence (TV) measurements carry a bf16/fp32 precision floor: mean per-step TV 0.006. The registered fork-precision gate failed at a ratio of 2.006 against 2.0, so divergence values near the floor are reported as not distinguished from numerics (claim ledger §D). Accuracy-scale results do not use this instrument.

---

## 2. Related work (draft prose)

**Prompt retention and selector comparisons.**
- Random Attention [2609.03430] shows that once every method keeps the prompt, most of the free-running accuracy gap between eviction selectors disappears. Uniform random retention with the prompt protected matches the strongest selector at about 4× compression.
- The Pitfalls of KV Cache Compression [ACL 2026] shows that compressing a multi-instruction query drops some instructions silently and leaks system prompts. It proposes fairer eviction and whitelisting.

Our results agree that prompt retention matters. What we add is how protection changes the gap between *evaluation protocols*, which neither paper measures, since both evaluate free generation only.

**Free-running reasoning benchmarks.** HOTT [2512.12008] free-runs eviction families on reasoning tasks and reports severe, method-specific degradation. It does not hold traces fixed or include a random baseline.

**Fixed-trace replay.** Ai et al. [2608.01631] replay a full-cache reasoning trace under each compressed cache to study answer–evidence faithfulness. They also report end-to-end accuracy, but at a larger budget (2048 against 256 for the fixed-trace runs). End-to-end accuracy is lower even with the larger cache. They form no matched-budget difference and no reverse replay.

**Exposure bias and off-policy evaluation.** Conditioning on reference rather than self-generated trajectories is a known source of bias [Ranzato et al. 2016; Bengio et al. 2015; Ross et al. 2011]. We do not rename it. We measure its consequence for one intervention under matched conditions, and report that the two replay directions can disagree.

**Not claimed:**
- that random eviction is competitive;
- that the prompt is fragile;
- that free-running reasoning degrades under eviction;
- that fixed-trace replay is a new instrument.

---

## 3. Table shells

**T1. Protection × protocol at B=256 and B=1024.** Values come from ledger rows A1–A9 (B=256) and are pending for B=1024. One row per policy × protection:
- free-run accuracy (all conditions);
- forward readout accuracy (defined support);
- forward residual with CI;
- reverse trajectory contrast with CI;
- interaction with CI;
- full-population bounds;
- n defined forward / reverse / common.

**T2. Paired protection effect by direction** (ledger A3, A5): shrinkage ratio with CI and absolute change with CI, forward and reverse, per policy.

**T3. Matched-key diagnostics** (ledger B1–B3): accuracy contrast, M contrast, Δ contrast, the family-adjusted joint label, n = 100 / 25 problems.

**T4 (appendix). Grader sensitivity** (ledger C1–C2): primary vs secondary, both banks, transitions by rule.

---

## 4. Figure specifications (vector, data-driven, built from the named artifacts)

| Fig | Construction | Data source | Reader takeaway | Status |
|---|---|---|---|---|
| 1. Protocol and cache | (a) The 2×2 above as four boxes with arrows: trace source on rows, reading cache on columns, each arrow labelled with its contrast. (b) Cache strips at one prune, protected vs unprotected at the same B: prompt block, retained trace, evicted trace, plus a dashed P+64 relaxation marker | Schematic; the strip proportions use the mean prompt (93.7) and B=256 | Why holding the trace fixed changes the experiment; that protection spends the same budget | Spec only |
| 2. Main protected comparison | Two panels (forward residual, reverse trajectory contrast). x: policy × protection; points with 95% CIs; budgets as two facets (1024 pending, drawn as an empty facet with an "awaiting" label, never as zeros) | `results/diag/paper_audit/reverse_b256.json` (common support) and `protection_b256.json` | Protection shrinks the forward gap a lot and the reverse gap by about half | Data present for 256 |
| 3. Fair-selector evidence | Per pair, two aligned dot-and-whisker columns (accuracy contrast, M contrast), on separate axes, family-adjusted whiskers | `matched_b256.json` (unprotected); WP5 adds protected | Whether more mass lost ranks policies by accuracy | Unprotected only |
| 4. Practical margin / replication | Forest plot by model × task × budget; shaded ±0.03 band; n per row | `decision_b256.json`; WP6 | Practical size and heterogeneity | Pending WP6 |
| 5. Support and missingness | Stacked bars per cell: common / forward-only / reverse-only / neither, with bound intervals overlaid on the point estimates | `reverse_b256.json` `population`, `bounds_full_population` | How undefined boundaries affect each direction (unprotected Random: 92 undefined reverse rows) | Data present |
| A. Budget/retention audit | Resident entries vs decoding step for the shortest- and longest-prompt smoke traces, protected vs unprotected | prompt_protect smoke records | Equal nominal budget ≠ equal memory | Spec only |
| B. Grader transitions | Sankey-style bars: primary → secondary, by rule, both banks | `results/diag/qwen_scoring/*` | Which scores changed and why | Data present |

**Rules for every figure:**
- sign conventions match the text;
- the population is named in the caption;
- unavailable values are marked, never drawn as zero;
- colours are colour-blind-safe, with fixed policy colours across figures;
- the PNG is viewed before the figure is called done.

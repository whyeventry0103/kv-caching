# Meeting 5 protocol draft — matched protection, finite information channel, interventions

**Status: DRAFT v0.1, dated 2026-10-14, written before any Meeting 5 outcome
exists.** No M5 row has been generated: every M5 config is `held: true`, and
no M5 campaign is approved. Until the first M5 read, this draft may change only
by a dated entry in §9. After any M5 outcome has been read, a change needs a
lettered amendment in `PREREGISTRATION.md`. Source of the design:
`docs/paper/MEETING5_REVIEW_AND_PLAN.md` (§§6–11, 15). Where this draft makes a
choice the review left open, it says so and gives the reason.

## 1. Variables (review §6.1)

| Symbol | Meaning in this repo |
|---|---|
| X | the history: prompt plus generated trace (M5-A/B), or protected instruction plus decode-path fact body (M5-C) |
| C_F | full cache at the stated point |
| C_π | complete persistent cache under policy π: retained absolute positions per layer/KV head, plus any scorer state |
| Y | task target: final-answer correctness (M5-A/B); the queried fact (M5-C) |
| S, N | M5-C only: target and nuisance facts. Each is independent and uniform on 4 values, and all 16 combinations are enumerated per context block |
| U | M5-C context block: aliases, layout, filler, query. U never contains S or N |
| R | eviction randomness, keyed separately from generation sampling (`press_seed`, `draw_seed`) |
| Z | finite readout output: 4 single-token answer symbols plus OTHER (all remaining vocabulary mass) |

A subset argument C_π ⊂ C_F is used only at a common-history crop: M5-C
readouts and forward readout. It is never applied across free-running
trajectories.

## 2. Hypotheses and what would weaken them (review §7)

The directions below are frozen. The "weakens" column is a declared outcome,
not a failure to be explained away.

| ID | Question | Prediction | Weakened or rejected by |
|---|---|---|---|
| H1 | Is the accuracy/proxy mismatch a prompt-loss effect? | Under identical protection and capacity, the Random − H2O-style and Random − StreamingLLM accuracy contrasts move toward 0 | The protected contrasts stay as large as the unprotected ones. If instead they vanish (§4.3), protection is the reported explanation, with no deeper information mechanism invented |
| H2 | Does useful retained content explain a residual? | M5-C target-query channel MI, measured before the answer exists, predicts later accuracy beyond M/Δ/ω, occupancy and exposure | No held-out gain, or it only tracks an answer already produced |
| H3 | Does target information matter causally? | At equal budget, protecting the target span beats protecting an equal nuisance span on the target query; the sign follows the task when the query switches to N | Nuisance protection helps equally, or the effect is explained by cache size, timing or numerics |
| H4 | Is the limit accessibility? | Optional probe; enters only if native readout fails (review §11.3) | Not frozen here; needs its own dated entry before use |
| H5 | Is historical Random representative? | Fresh-uniform agrees with the fixed-slot Random within the §4.3 margin | Only the fixed-slot algorithm shows the advantage: then the claim is narrowed to that algorithm, by name |
| H6 | Does it generalize? | The protected contrast survives on GSM8K / BBH | It reverses or vanishes: report a boundary, and make no universal Random claim |

## 3. Cohorts and capacity (review §8.2)

- **Discovery cohort:** the frozen 100 MATH-500 problems
  (`configs/meeting3/grid100_pids.json`), seeds 11/23/47/89, Qwen-7B at the
  pinned revision, B = 256. This cohort is discovery, not confirmation.
- **Strict-fit cohort (primary fixed-B comparison):** the 97 problems with
  P + 64 ≤ 256. It was computed from the full-cache bank's prompt token IDs
  before any M5 outcome (frozen as `configs/meeting5/strict_fit_b256.json`; derivation in
  `results/diag/meeting5/inventory.json`).
- **Relaxed problems:** test/algebra/1349 (P = 344, B_eff = 408),
  test/prealgebra/1646 (B_eff = 417) and test/precalculus/1199 (B_eff = 280).
  They enter the whole-cohort comparison only after the unprotected arms have
  also run at the same B_eff (manifest item M5A-CAPMATCH, ≤ 48 rows). Until
  then, the whole-cohort protection contrast is labelled capacity-unmatched
  for these three problems.
- **Unprotected** means prompt tokens are eligible for eviction, not that they
  are forcibly evicted.
- **Per-arm reporting:** effective budget, K/V bytes, peak occupancy between
  prunes, prune count, interval, first eviction, prompt retention,
  completion/truncation and token count. Truncated runs stay in every
  denominator, scored incorrect.

## 4. M5-A endpoints and inference (review §8.4)

### 4.1 Primary

The primary endpoint is the paired free-running accuracy difference: average
over seeds within a problem, then average over problems (equal-problem). The
two scientific pairings are **Random − H2O-style** and
**Random − StreamingLLM**. Random − R-KV is a declared third, descriptive
comparison.

The protection-by-policy interaction for pairing (R, j) is:

    I_j = (A_{R,+P} − A_{j,+P}) − (A_{R,−P} − A_{j,−P})

It is computed per problem on matched keys.

### 4.2 Inference

Inference is the same as the registered B = 1024 interaction
(`problem_level_t_tests`, PREREGISTRATION 2026-10-14):
- a two-sided problem-level paired t test of I_j against 0;
- Holm over the two scientific pairings at α = 0.05;
- a problem-cluster bootstrap 95% CI (2000 resamples, seed 0) for every
  contrast, labelled descriptive.

Separate CIs are not compared as a significance test, and a non-significant
difference is not called equivalence.

### 4.3 Practical equivalence margin

The margin is ±0.05 accuracy, frozen here. "The difference vanishes" means the
two one-sided tests (α = 0.05 each, i.e. a 90% CI inside ±0.05) both reject. A
CI that straddles the margin is reported as **inconclusive**, not as vanished.

The review left the value open. 0.05 was chosen because it is the review's own
example of a practically meaningful effect (§9.4, "a five-point superiority
claim"), and because N = 97–100 with a paired SD near 0.3 gives a half-width
of about 0.06. The margin is therefore at the edge of what this cohort can
resolve. This is stated now so that an inconclusive result is not reread as
equivalence.

### 4.4 Protected proxies

Protected M/Δ/ω are read only from new protected captures on the same reference
traces, keys, layers and checkpoints. Protected accuracy is never combined with
old unprotected metrics.

## 5. M5-B replication (review §9)

The planning N is 200 problems × 2 seeds, from a hashed selection of untouched
problems. Calibration and confirmation manifests are disjoint, by problem ID.
Caps: GSM8K 8,192 and BBH 4,096 tokens; MATH keeps its historical cap. A cap
changes only by a pre-outcome amendment, and only for all arms at once.

Graders are frozen before outcomes:
- **GSM8K:** gold parsing (`gsm8k_gold`) is tested separately from final-answer
  extraction.
- **BBH:** exactly one valid final option under the frozen format; a
  multi-answer or ambiguous output is incorrect, with its reason recorded.
  There is no LLM judge.

The BBH revision is a human pin. The proposal is `9ee07bd…` (review §9.1), and
it is not used until a person pins it.

Results are reported per dataset with heterogeneity, never pooled into one
p-value.

## 6. M5-C finite information channel (review §10)

### 6.1 Generator

The generator is `src/kvcache/information_tasks.py`:
- two 4-valued variables per block, with all 16 combinations;
- facts `" <alias> is <value>."` whose token length is equal for all values
  and aliases (`check_encodings` against the pinned tokenizer, run on
  calibration before freezing);
- the target/nuisance order alternates by block (counterbalanced ages);
- filler is screened for value tokens and alias token sequences, with the
  source hash retained;
- the calibration/confirmation split is by whole block.

### 6.2 Primary regime

The primary regime is a 1,024-token decode-path body at B = 256. The early fact
is at body offset 64 and the late fact at offset 768, so the ages at readout
are about 950 and about 250 tokens. An age sweep is secondary.

### 6.3 Readout

The readout is an independent clone of the cache at the checkpoint. It cannot
evict again, it receives no target-bearing state, and its query tokens are
always retained. Query S and query N read the same cache.

### 6.4 Channel

The channel is computed by `src/kvcache/task_information.py`:
- the native log-softmax over the vocabulary, with the 4 symbol probabilities
  plus OTHER and no renormalization;
- exact I(S; Z | u) with N marginalized uniformly;
- log loss in bits from log-probabilities;
- expected correct probability, OTHER mass and native greedy accuracy;
- a rounding tolerance of SUM_TOL = 1e-6, declared before outcomes.

### 6.5 Labels

Every number is labelled **"output-channel MI (bits) through the declared
answer channel — not total cache MI."** The score 2 − cross-entropy is a
declared-reader score, can be negative, and is not "bits in the cache."
At a fixed u and eviction seed R, S → C_π → Z, so by data processing the
channel MI is a **lower bound** on I(S; C_π | u, R), never an estimate of it.

### 6.6 Reference cases

The reference cases are implemented and tested:

| Case | MI (bits) | Correct probability |
|---|---|---|
| identity | 2 | 1 |
| uniform | 0 | 0.25 |
| wrong permutation | 2 | 0 |
| always OTHER | 0 | 0 |
| copy-nuisance | 0 about S, 2 about N | — |

### 6.7 Primary M5-C endpoints

- Per policy: the paired per-context contrast of target-query MI against full
  cache.
- The Random − H2O-style contrast of target-query MI.
- Uncertainty resamples contexts.
- The confirmation-precision target is a 0.1-bit half-width. The final N is
  frozen after calibration, before any confirmation outcome.

### 6.8 Calibration

Calibration is 4 blocks × 16 × 5 caches = 320 prefix executions. It is a
separate request with a proposed 4 GPU-hour cap, and calibration records are
never confirmatory.

The calibration passes only if:
- full-cache readout works;
- end-to-end and cloned readouts agree;
- the task is not at floor or ceiling in every arm;
- eviction actually removes fact spans in some arms;
- OTHER is understood.

## 7. M5-D equal-budget interventions (review §11.1)

### 7.1 Arms

Within each M5-C context, at the same B_eff, there are four arms:
1. **base policy;**
2. **target-span protection:** the target fact's tokens are added to the
   mandatory set, and the base policy fills the remaining slots;
3. **nuisance-span protection:** the equal-length nuisance fact is protected
   instead;
4. **placebo-span protection:** a filler span matched in length and age,
   predeclared by offset.

### 7.2 What stays equal

- K/V bytes;
- protected instruction tokens;
- the intrinsic recent window (R-KV: 8);
- the pruning schedule;
- query/readout;
- sampling.

Prevention of eviction is used, not after-the-fact restoration, which avoids
the prompt-plus-padding problem. Each arm logs whether the intervention changed
retention; an already-retained span is a real zero-dose case.

### 7.3 Primary contrast

The primary contrast is target protection − nuisance protection, on target-query
log score (bits) and on greedy accuracy, paired by context. The task-switch
check repeats it with the N query, where the sign should reverse.

These are effects of the specified cache interventions. They do not show that
the MI scalar is the sole mediator.

### 7.4 Implementation

Implementation is deferred. It extends the protected-press mandatory set with
predeclared body spans, and it is not built until the M5-C calibration passes
(gate G2).

## 8. Decision gates (review §15)

| Gate | Passes when | Otherwise |
|---|---|---|
| G1 | The four-policy protected accuracy read is FINAL | If the mismatch vanishes (§4.3), lead with protection; no information explanation is forced |
| G2 | Calibration validity, the reference tests, full/full identity and no leakage all hold | One redesign on calibration data, then abandon the instrument |
| G3 | The earlier channel adds held-out value beyond simpler predictors, and at least one intervention supports it | Descriptive diagnostics only; no mechanistic headline |
| G4 | New datasets give an explicit scope | A failure narrows the claim; the condition is not removed |

## 9. Dated changes

- 2026-10-14: v0.1 drafted. No M5 outcome exists.

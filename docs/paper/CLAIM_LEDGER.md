# Claim ledger (execution plan WP0)

Created 2026-10-05 at baseline `1b644dc`. One row per claim the paper makes or
may make. A claim enters the manuscript only in the wording of its row, with
its scope and status. Numbers are copied from the named artifact. They are not
recomputed by hand: rerun the artifact's `reproduce` command to check one.

Status values:
- **CONFIRMED-SCOPED.** Registered analysis, FINAL, within the stated scope.
- **DESCRIPTIVE.** Valid, but not a registered test.
- **SENSITIVITY.** A separately named, post-hoc analysis.
- **EXPLORATORY.** Post-hoc, or the CI is not adjusted for its family.
- **PENDING.** The data are not read yet.
- **RETIRED.** Must not appear.

Estimators:
- **EP.** Equal-problem mean: seeds within problem, then problems; 2000-resample problem-cluster percentile bootstrap, seed 0.
- **M4.** `report_m4` complete-case per-condition mean (Amendment K layer 2).

The two estimators are never mixed under one label.

## A. Central claims (accuracy scale; no dependence on the fork-precision instrument)

| ID | Sentence (exact wording allowed) | Source artifact | Denominator | Estimator, CI | Scope | Status | What would invalidate it |
|---|---|---|---|---|---|---|---|
| A1 | "At budget 256, with policies that may evict the prompt, free-running accuracy falls far below held-trace readout accuracy: forward residual −0.332 (Random) and −0.338 (StreamingLLM)." | `results/diag/amendment-s/protection_b256.json` `per_press.*.primary.residual_base` | 392 conditions with a defined forward readout (of 400) | EP; Random [−0.404, −0.262], SLM [−0.405, −0.270] | Qwen-7B (R1-Distill), MATH-500, 100 problems × 4 seeds, B=256 | CONFIRMED-SCOPED (descriptive decomposition, not causal) | A data or grading correction |
| A2 | "Protecting the prompt at the same total budget shrinks the forward residual to −0.056 (Random) and −0.078 (StreamingLLM); a residual persists." | same, `residual_pp`, `residual_label` = RESIDUAL PERSISTS | 392 | EP; [−0.098, −0.016] / [−0.125, −0.034]; sensitivity without the 3 budget-relaxed problems −0.058 / −0.081, also PERSISTS | same, B=256 only | CONFIRMED-SCOPED (S.1) | B=1024 is separate (PENDING). Not "most of the residual was prompt loss" (RETIRED wording). |
| A3 | "The paired change in the forward residual is +0.276 (Random) and +0.259 (StreamingLLM)." | same, `delta_residual`, label SHRINKS | 392 pairs | EP paired; [+0.204, +0.349] / [+0.182, +0.339] | same | CONFIRMED-SCOPED (S.1) | — |
| A4 | "Forward residual magnitude shrinks by about 83% (Random) and 77% (StreamingLLM)." | `results/diag/paper_audit/decision_b256.json` `per_press.*.shrinkage` | 392 | Joint EP bootstrap of `1 − abs(R_pp)/abs(R_base)`: 0.832 [0.709, 0.949] / 0.768 [0.624, 0.898]; base near zero in 0% of replicates | same | SENSITIVITY (ratio of point estimates, not a causal share) | Must be paired with A5 wherever it appears |
| A5 | "On the four-arm support, protection improves the forward residual by +0.239 (Random) and +0.271 (StreamingLLM), and the reverse trajectory contrast by +0.161 and +0.278. The forward improvement is larger for Random (difference +0.078 [+0.007, +0.149]); for StreamingLLM a difference was not detected (−0.007 [−0.081, +0.069]). As ratios of each direction's own unprotected value: 0.82/0.78 forward, 0.51/0.57 reverse." | `results/diag/paper_audit/reverse_b256.json` `protection_by_direction` (incl. `difference_of_improvements`) | 298 conditions / 92 problems (Random), 335 / 98 (SLM): all four arms defined in both cells | Joint EP; ratios Random 0.817 [0.655, 0.968] / 0.507 [0.309, 0.687], SLM 0.778 [0.622, 0.910] / 0.570 [0.447, 0.691] | same | SENSITIVITY (new 2×2 analysis, 2026-10-05; difference added after the review of fbf869f) | Percentages use different denominators: never read "mainly forward" off them. The four-arm support is selected (undefined reverse rows are post-treatment) |
| A6 | "After protection, the two replay directions disagree: the trajectory contrast is −0.049 under compressed readout but −0.179 under full-cache readout (Random; interaction +0.130)." | `reverse_b256.json` `per_policy.random_pp.common_support` | 376 common-support conditions (Random-pp), 355 (SLM-pp) | Joint EP; interaction Random-pp +0.130 [+0.078, +0.185], SLM-pp +0.127 [+0.065, +0.197]; full-population worst-case bounds [+0.067, +0.127] / [+0.022, +0.140] | same | SENSITIVITY | Unprotected Random shows no detected interaction (+0.029 [−0.026, +0.084]); unprotected SLM +0.146 [+0.068, +0.227] |
| A7 | "At budget 256 the held-trace readout is itself already lossy: no 'false reassurance' (readout within m of full cache while free-run is not) at m = 1, 3 or 5 points." | `decision_b256.json` `practical_*` | 392 | Bonferroni pair of one-sided 97.5% bounds; R_loss protected 0.169 [0.114, 0.227] (Random), 0.199 [0.138, 0.274] (SLM) | same | SENSITIVITY (margins proposed after the B=256 result) | The "near-lossless certification" story needs B=1024/2048 evidence; at B=256 it is underestimation, not false certification |
| A8 | "Free-running accuracy rises with protection from 0.422 to 0.662 (Random) and from 0.237 to 0.610 (StreamingLLM), still well below the full cache." | `protection_b256.json` `free_accuracy_*`, `delta_free` | 400 | EP; Δfree +0.240 [+0.180, +0.307] / +0.372 [+0.297, +0.450] | same | CONFIRMED-SCOPED | — |
| A9 | "Protection does not detectably change Random's forward readout (Δ −0.033 [−0.094, +0.024]); it raises StreamingLLM's (+0.116 [+0.063, +0.177])." | `protection_b256.json` `delta_readout` | 392 | EP paired | same | CONFIRMED-SCOPED | Not "Random's readout did not need the prompt" (RETIRED) |
| A10 | Unprotected four-policy decomposition and reverse envelope at 256 (paper Table 2, §4.4) | `results/diag/amendment-k/m4-*-summary.json` | per cell | M4 | Qwen-7B, B=256 | CONFIRMED-SCOPED as *operational comparisons*; the envelope is not a causal bound | Wording fixed 2026-10-05 (§2.3 of the plan) |

## B. Diagnostics (supporting; scoped)

| ID | Sentence | Source | Denominator | Estimator, CI | Status | Limit |
|---|---|---|---|---|---|---|
| B1 | "On the 25 capture problems, Random is more accurate than H2O-style (+0.19) while losing more attention mass M (+0.315): a policy-level mismatch." | `results/diag/paper_audit/matched_b256.json` `contrasts_M["random - h2o"]` | 100 conditions / 25 problems, unprotected B=256 | Joint EP; accuracy +0.190 [+0.070, +0.310], Bonferroni-3 [+0.040, +0.340]; M +0.315 [+0.298, +0.329] | EXPLORATORY (descriptive) | Four policy means cannot establish a per-trace predictor; unprotected policies only; teacher-forced M |
| B2 | "Random vs StreamingLLM: the accuracy advantage (+0.12) is not established after adjustment across the three pairs." | same, `random - streamingllm` | same | +0.120 [+0.020, +0.220], Bonferroni-3 [0.000, +0.240] → NOT ESTABLISHED | EXPLORATORY | — |
| B3 | "Random and R-KV have equal accuracy (+0.000) while Random loses more mass (+0.269)." | same, `random - rkv` | same | acc [−0.120, +0.110] | EXPLORATORY | "Equal" is not equivalence: no TOST was run |
| B4 | "ω(Random) − ω(StreamingLLM) reverses on Llama-8B (−0.355 [−0.453, −0.275])." | `results/diag/meeting3/capture_llama8b_b256.json` | R.17.3 design | registered contrast | CONFIRMED-SCOPED (REVERSED) | Does not isolate architecture as the cause; "ω proves the mechanism" is RETIRED |
| B5 | S.2: "Random's prompt-plus-padding restoration arm recovers ρ 0.367 [0.310, 0.419], more than the trace arm (+0.147 [+0.083, +0.200]); for StreamingLLM a prompt-versus-trace contrast was not detected (−0.030 [−0.090, +0.025])." | `results/diag/amendment-s/origin.json` | S.2 eligible set | registered | CONFIRMED-SCOPED, **but TV-based: carries the precision floor (section D)** | All 91 Random rows padded with 1–25 entries: not a pure per-prompt-entry effect |
| B6 | Collateral (GQA) = 0 on the capture rows | capture reads | — | — | DESCRIPTIVE | — |

## C. Qwen3 bank (WP2)

| ID | Sentence | Source | Status |
|---|---|---|---|
| C1 | "Under the frozen grader the Qwen3-4B full-cache bank scores 169/200 = 0.845, just below the registered band [0.85, 0.99]; the gate FAILS." | `results/diag/qwen_scoring/bank_audit_qwen3_4b.json` `primary` (0 recompute mismatches) | CONFIRMED (gate FAIL stands) |
| C2 | "A post-inspection, format-aware secondary score (v2) gives 0.905, with an AMBIGUITY RANGE of 0.900–0.960 over 12 ambiguous rows (not a confidence interval); the same scorer moves the Qwen-7B anchor from 0.885 to 0.920 (range 0.920–0.930)." | same, plus `bank_audit_r1_qwen7b.json` | SENSITIVITY (never a gate input) |
| C3 | Any Qwen3 decomposition result | — | PENDING. Cells incomplete and held behind B=1024. Disposition (a) chosen 2026-10-06: exploratory only, gate FAIL shown beside every result, secondary score unusable until the 42-row human review is done |

## D. Fork-precision gate: scope of the FAIL (WP0 item 6)

Recorded 2026-10-05.

**The gate.** R.17.0.5 compares the fork statistic with the bf16/fp32 precision floor. It FAILED on one arm out of six:
- Failing arm: Random I=256, fork p95 0.1159 against 2× floor 0.0578. The ratio is 2.006 against the 2.0 limit.
- Floor: mean per-step TV 0.0062, p95 0.0273.
- fp32 diagnosis: IMPLEMENTATION, by threshold.

The FAIL is kept. It is not renamed and not dismissed for being small.

**What the instrument underwrites.** Next-token total-variation (TV) divergence between two cache conditions:
- the per-step TV profiles,
- fork and exact-lag effects,
- the restoration ρ ratios (S.2, restore-v2),
- eviction locking.

**Claims that depend on it.** Each carries the floor in its sentence, or is not made:
- Paper abstract: "≈0.002 per step immediately after" the first eviction. This is **below** the floor mean of 0.006. The sentence now says so (edited 2026-10-05).
- The "0.088 later in the trajectory" figure is about 14× the floor mean. It stands, with the floor stated.
- B=2048 divergence (0.0128 / 0.0186) and exact-lag effects (~0.003) are *not established above numerics*.
- Eviction locking stays "detected, not established" (R.2).
- S.2 ρ ratios (B5) are TV ratios. Their numerators and denominators sit far above the floor at B=256. They are still flagged as instrument-dependent.

**Claims that do not depend on it.** Every accuracy-scale quantity:
- A1–A10, C1–C2,
- the capture mass metrics M, Δ, ω (teacher-forced attention mass, not fork TV).

No software gate substitutes for this instrument gate.

## E. Retired wording (plan §2.3)

| Retired | Replacement in use |
|---|---|
| "Most of the residual was prompt loss." | "Prompt protection reduces the measured forward residual magnitude by about 83%/77% in this setting" — always with A5 and A6 beside it |
| "Protection works mainly in the forward direction" (said in chat, 2026-10-05) | A5's paired difference: larger forward improvement for Random only; not detected for StreamingLLM. Keep the significant post-protection interaction (A6) a separate statement |
| "Random's readout did not need the prompt." | "A readout effect was not detected for Random; its CI includes both negative and positive changes." (A9) |
| "The prompt is not special for StreamingLLM." | "The prompt-versus-trace restoration contrast was not detected." (B5) |
| "ω proves the mechanism." | "ω describes a model-dependent geometric difference; the accuracy connection is not established." (B4) |
| "The reverse bracket proves a causal trajectory effect" / "forward and reverse bracket the on-policy residual" | "Forward and reverse readout are complementary operational comparisons; a causal bound requires additional assumptions." (paper edited at 13 places, 2026-10-05) |
| UNCHANGED meaning equal | "UNCHANGED (registered label): the CI includes zero; equivalence was not tested." |
| "All gates pass." | Name each gate. Software: lint/test/dryrun/audit pass. Provenance: legacy summaries pass. Bank: Qwen3 FAIL. Instrument: fork-precision FAIL. |
| Title "Near-Lossless Under Readout, Collapse Under Free Generation" | Working title "Auditing KV-Cache Eviction: Prompt Retention, Readout, and Free-Running Reasoning". At B=256 readout loses 0.136–0.315 (A7). |

## F. Open questions this ledger cannot close yet

1. **B=1024 (protected free run + forward readout).** Does a practically meaningful protected residual persist? This decides the budget scope of A2. Status: PENDING (grid 796/800 on 2026-10-14). The forward/reverse interaction at 1024 has a registered t-test + Holm procedure, frozen 2026-10-14 before the data.
2. **B=1024/2048 near-lossless readout.** Is there a cell where readout is within 3 points of the full cache while free running is not? This is the only form of the "false certification" story the plan allows. Status: not measured under protection.
3. **H2O-style under protection, and a fresh-uniform random baseline.** Do B1–B3 survive equal prompt protection? This needs WP5, which is held for authorization.
4. **Qwen3 gate disposition.** This is the owner's and supervisor's decision; see `docs/protocol/QWEN_SCORING_SENSITIVITY.md`.

## G. Review of fbf869f (2026-10-05): what changed in this ledger

- A5 rewritten: absolute improvements and their paired difference replace a
  reading based on percentages with different denominators.
- C2: "bounds" renamed ambiguity range; secondary scorer is v2 (0 rows changed
  vs v1 in either bank).
- Reader validation repaired (R1); every affected read re-read on the host:
  0 numeric changes in protection_b256, origin, reverse_b256, matched_b256,
  decision_b256. matched joint labels reworded by the sign-table fix (R4), same
  classification.

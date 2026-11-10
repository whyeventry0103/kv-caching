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
| A2 | "Protecting the prompt at the same total budget shrinks the forward residual to −0.056 (Random) and −0.078 (StreamingLLM); a residual persists." | same, `residual_pp`, `residual_label` = RESIDUAL PERSISTS | 392 | EP; [−0.098, −0.016] / [−0.125, −0.034]; sensitivity without the 3 budget-relaxed problems −0.058 / −0.081, also PERSISTS | same, B=256 only | CONFIRMED-SCOPED (S.1) | B=1024 is A11–A12. Not "most of the residual was prompt loss" (RETIRED wording). |
| A3 | "The paired change in the forward residual is +0.276 (Random) and +0.259 (StreamingLLM)." | same, `delta_residual`, label SHRINKS | 392 pairs | EP paired; [+0.204, +0.349] / [+0.182, +0.339] | same | CONFIRMED-SCOPED (S.1) | — |
| A4 | "Forward residual magnitude shrinks by about 83% (Random) and 77% (StreamingLLM)." | `results/diag/paper_audit/decision_b256.json` `per_press.*.shrinkage` | 392 | Joint EP bootstrap of `1 − abs(R_pp)/abs(R_base)`: 0.832 [0.709, 0.949] / 0.768 [0.624, 0.898]; base near zero in 0% of replicates | same | SENSITIVITY (ratio of point estimates, not a causal share) | Must be paired with A5 wherever it appears |
| A5 | "On the four-arm support, protection improves the forward residual by +0.239 (Random) and +0.271 (StreamingLLM), and the reverse trajectory contrast by +0.161 and +0.278. The forward improvement is larger for Random (difference +0.078 [+0.007, +0.149]); for StreamingLLM a difference was not detected (−0.007 [−0.081, +0.069]). As ratios of each direction's own unprotected value: 0.82/0.78 forward, 0.51/0.57 reverse." | `results/diag/paper_audit/reverse_b256.json` `protection_by_direction` (incl. `difference_of_improvements`) | 298 conditions / 92 problems (Random), 335 / 98 (SLM): all four arms defined in both cells | Joint EP; ratios Random 0.817 [0.655, 0.968] / 0.507 [0.309, 0.687], SLM 0.778 [0.622, 0.910] / 0.570 [0.447, 0.691] | same | SENSITIVITY (new 2×2 analysis, 2026-10-05; difference added after the review of fbf869f) | Percentages use different denominators: never read "mainly forward" off them. The four-arm support is selected (undefined reverse rows are post-treatment) |
| A6 | "After protection, the two replay directions disagree: the trajectory contrast is −0.049 under compressed readout but −0.179 under full-cache readout (Random; interaction +0.130)." Abstract form: "the two replay directions disagree at the tighter budget of 256 entries (median nominal budget ratios 0.097 for Random and 0.098 for StreamingLLM)". | `reverse_b256.json` `per_policy.random_pp.common_support`; ratios `results/diag/amendment-s/realized_compression.json` `cells.b256.{random_pp,streamingllm_pp}.median` (100 problems × 4 seeds) | 376 common-support conditions (Random-pp), 355 (SLM-pp) | Joint EP; interaction Random-pp +0.130 [+0.078, +0.185], SLM-pp +0.127 [+0.065, +0.197]; full-population worst-case bounds [+0.067, +0.127] / [+0.022, +0.140] | same | SENSITIVITY | Unprotected Random shows no detected interaction (+0.029 [−0.026, +0.084]); unprotected SLM +0.146 [+0.068, +0.227] |
| A7 | "At budget 256 the held-trace readout is itself already lossy: no 'false reassurance' (readout within m of full cache while free-run is not) at m = 1, 3 or 5 points." | `decision_b256.json` `practical_*` | 392 | Bonferroni pair of one-sided 97.5% bounds; R_loss protected 0.169 [0.114, 0.227] (Random), 0.199 [0.138, 0.274] (SLM) | same | SENSITIVITY (margins proposed after the B=256 result) | The "near-lossless certification" story needs B=1024/2048 evidence; at B=256 it is underestimation, not false certification |
| A8 | "Free-running accuracy rises with protection from 0.422 to 0.662 (Random) and from 0.237 to 0.610 (StreamingLLM), still well below the full cache." | `protection_b256.json` `free_accuracy_*`, `delta_free` | 400 | EP; Δfree +0.240 [+0.180, +0.307] / +0.372 [+0.297, +0.450] | same | CONFIRMED-SCOPED | — |
| A9 | "Protection does not detectably change Random's forward readout (Δ −0.033 [−0.094, +0.024]); it raises StreamingLLM's (+0.116 [+0.063, +0.177])." | `protection_b256.json` `delta_readout` | 392 | EP paired | same | CONFIRMED-SCOPED | Not "Random's readout did not need the prompt" (RETIRED) |
| A10 | Unprotected four-policy decomposition and reverse envelope at 256 (paper Table 2, §4.4) | `results/diag/amendment-k/m4-*-summary.json` | per cell | M4 | Qwen-7B, B=256 | CONFIRMED-SCOPED as *operational comparisons*; the envelope is not a causal bound | Wording fixed 2026-10-05 (§2.3 of the plan) |
| A11 | "With the prompt protected at B=1024 (median nominal budget ratios 0.434 Random, 0.487 StreamingLLM), the forward residual persists and shrinks: −0.047 (Random) and −0.040 (StreamingLLM)." | `results/diag/amendment-s/protection_b1024.json` `per_press.*.primary.residual_pp`; ratio in `realized_compression.json` | 392 of 400 conditions per policy (readout defined) | EP; [−0.086, −0.015] / [−0.070, −0.013]; Δresidual Random lower bound +0.001 | Qwen-7B, MATH-500, 100 problems × 4 seeds, B=1024 | CONFIRMED-SCOPED (S.1; consequence A) | Random's SHRINKS rests on a +0.001 bound: never "robust shrinkage". The ratio is nominal (budget/sequence length), not hardware memory |
| A12 | "At protected B=1024 the trajectory contrast is negative under either readout direction (compressed: −0.037 / −0.039; full cache: −0.040 / −0.034), and no forward/reverse interaction was detected (Holm p 0.64 for both policies)." | `results/diag/paper_audit/reverse_b1024.json` `registered_inference`, `per_policy.*.common_support` | common support 382 / 98 problems (Random), 383 / 97 (SLM) | registered problem-level t test + Holm; EP bootstrap CIs descriptive | same, B=1024 | CONFIRMED-SCOPED (registered 2026-10-14, before the data) | "No interaction detected" ≠ "no interaction" (no equivalence test was registered). Operational comparisons, not a causal share. Undefined readouts are excluded; SLM's worst-case bound lies below 0 |
| A13 | "Prompt protection changes the Random contrast for both scientific pairings: interaction −0.186 (Random − H2O-style) and −0.137 (Random − StreamingLLM), both rejecting after Holm." | `results/diag/meeting5/m5a/m5a_read_strict.json` `registered_inference` (+ `.conditions.tsv`) | 388 conditions / 97 problems per arm (frozen strict cohort) | registered problem-level t test + Holm (t −5.45, −4.02; df 96); EP bootstrap [−0.250, −0.113] / [−0.201, −0.070] | Qwen-7B, MATH-500, B=256, strict cohort | CONFIRMED-SCOPED (registered 2026-10-14, before any M5 outcome) | The whole-cohort read is held (capacity comparators absent) |
| A14 | "Under protection, a Random − H2O-style accuracy difference was not detected (−0.005); it lies within ±0.05 (sensitivity margin) but was not shown within the ±0.03 primary margin." | same, `contrasts["random-h2o"].protected`, `equivalence_protected_contrasts` | same | EP [−0.036, +0.031]; TOST (1 − 2α) [−0.034, +0.023]; Holm TOST p 0.151 (±0.03), 0.010 (±0.05) | same | CONFIRMED-SCOPED (owner margins frozen 2026-10-14) | Never "equal" or "equivalent" without naming the ±0.05 margin |
| A15 | "Under protection Random's advantage over StreamingLLM shrinks from +0.191 to +0.054 (descriptive paired 95% interval [+0.023, +0.090]); it is not shown within either equivalence margin." | same, `contrasts["random-streamingllm"]` | same | EP | same | DESCRIPTIVE (the protected contrast is not a registered superiority test; the registered tests are the interactions, A13) | Never "Random beats StreamingLLM under protection" as a confirmed claim |
| A17 | "On the strict-fit cohort, the median nominal budget ratio is 0.074 unprotected and 0.103 protected for Random; 0.068 and 0.111 for H2O-style; 0.098 and 0.104 for StreamingLLM; 0.045 and 0.058 for R-KV." (§4.5 only) | `results/diag/meeting5/m5a/strict_realized_compression.json` `arms.*.median` | 388 conditions / 97 problems per arm; 0 budget-relaxed, 0 at ratio one | per-row min(1, B/(prompt + generated)), median | Qwen-7B, MATH-500, B=256, strict cohort | DESCRIPTIVE | Never mixed with the 100-problem S.1 ratios (A6, A11): different populations. Nominal, not occupancy |
| A16 | "Protection raises free-running accuracy by +0.247 (Random), +0.433 (H2O-style), +0.384 (StreamingLLM) and +0.108 (R-KV)." | same, `accuracy` | same | EP means (protected − unprotected) | same | DESCRIPTIVE | R-KV is our certified port; H2O-style is current-attention, not canonical H2O |

## B. Diagnostics (supporting; scoped)

| ID | Sentence | Source | Denominator | Estimator, CI | Status | Limit |
|---|---|---|---|---|---|---|
| B1 | "On the 25 capture problems, Random is more accurate than H2O-style (+0.19) while losing more attention mass M (+0.315): a policy-level mismatch." | `results/diag/paper_audit/matched_b256.json` `contrasts_M["random - h2o"]` | 100 conditions / 25 problems, unprotected B=256 | Joint EP; accuracy +0.190 [+0.070, +0.310], Bonferroni-3 [+0.040, +0.340]; M +0.315 [+0.298, +0.329] | EXPLORATORY (descriptive) | Four policy means cannot establish a per-trace predictor; unprotected policies only; teacher-forced M. **2026-10-17:** under protection the accuracy half is not detected on the strict cohort (A14), and no protected M exists. Never combine protected accuracy with unprotected M to claim that the mismatch persists |
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
| C3 | "On Qwen3-4B (EXPLORATORY; the full-cache bank fails its gate at 0.845), StreamingLLM's B=256 residual replicates (−0.209 [−0.285, −0.125], bracket excludes zero). Random runs near the floor (free run 0.030, truncation 0.735), so its residual is uninformative. Unprotected Random is far below StreamingLLM on this model." | `results/diag/amendment-k/m4-qwen3-4b-256-summary.json` (+ `-ratios.json`), read 2026-10-20; all cells complete on unique keys | EXPLORATORY (disposition (a), 2026-10-06): gate FAIL shown beside every result; secondary score unusable until the 42-row human review is done; unprotected only |

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
- "0.0018 per step" in the 50 steps of the first eviction. This is **below** the floor mean of 0.006. The abstract no longer states it (owner revision, 9f6c676). §4.6's summary states the floor beside it; the Figure 4 caption does not.
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
| "Random beats recency" (unscoped), and its proposed replacement "only for prompt-evicting policies" (2026-10-17) | The pair-specific statements A13–A15: protection changes both contrasts; under protection the H2O-style difference is not detected, and a smaller StreamingLLM difference remains descriptively |
| "Random is equivalent to H2O-style under protection" / "Random beats StreamingLLM under protection" | A14 with its ±0.05 margin named; A15 as DESCRIPTIVE |
| "Readout-invisible residual" (Figure 3 caption) | "a large residual unexplained by on-reference readout", scoped to the four prompt-evictable policies at B=256 (2026-11-10). Two §5 occurrences ("the readout-invisible trajectory residual") are not yet replaced: §F5 |
| "Why does readout-style evaluation miss a 34-point loss?" (§4.6 opener: causal, unscoped) | "We next examine when next-token predictions diverge during fixed-trace replay for policies that can evict the prompt." (2026-11-10) |
| Title "Near-Lossless Under Readout, Collapse Under Free Generation" | Working title "Auditing KV-Cache Eviction: Prompt Retention, Readout, and Free-Running Reasoning". At B=256 readout loses 0.136–0.315 (A7). |

## F. Open questions this ledger cannot close yet

1. **B=1024 (protected free run + forward readout).** CLOSED 2026-10-16: A11 (a residual persists and shrinks) and A12 (no forward/reverse interaction detected).
2. **B=1024/2048 near-lossless readout.** Is there a cell where readout is within 3 points of the full cache while free running is not? This is the only form of the "false certification" story the plan allows. Status: not measured under protection.
3. **H2O-style under protection, and a fresh-uniform random baseline.** The accuracy half is answered (A13–A15, 2026-10-17). Still open:
   - the protected M/Δ/ω captures (the adapter was built and tested on CPU 2026-10-20; the smoke and campaign are not approved), so whether the remaining protected Random − StreamingLLM gap conflicts with protected metrics is unmeasured;
   - fresh-uniform (A4), which is held.
4. **Qwen3 gate disposition.** This is the owner's and supervisor's decision; see `docs/protocol/QWEN_SCORING_SENSITIVITY.md`.
5. **"Readout-invisible" in §5.** The Threats section still says "the readout-invisible trajectory residual" twice (scale axis, Llama-8B). Both describe unprotected, prompt-evictable cells. Replacing them is an owner wording decision; not done 2026-11-10.

## G. Review of fbf869f (2026-10-05): what changed in this ledger

- A5 rewritten: absolute improvements and their paired difference replace a
  reading based on percentages with different denominators.
- C2: "bounds" renamed ambiguity range; secondary scorer is v2 (0 rows changed
  vs v1 in either bank).
- Reader validation repaired (R1); every affected read re-read on the host:
  0 numeric changes in protection_b256, origin, reverse_b256, matched_b256,
  decision_b256. matched joint labels reworded by the sign-table fix (R4), same
  classification.

## H. 2026-10-17: B=1024 and M5-A entered; pair-specific framing

- A11–A16 added from FINAL reads, and A2's "B=1024 PENDING" replaced.
- B1 now carries its protected qualification; C3 has the Qwen3 coverage.
- §F1 is closed and §F3 narrowed.
- Two wordings retired: the unscoped ranking story, and its blanket
  "prompt-evicting policies" replacement. The reviewer showed the
  replacement is inaccurate, because the Random − StreamingLLM gap survives
  protection.
- The M5-A read now publishes its condition table. An independent
  standard-library reproduction matches every estimate
  (`results/diag/meeting5/m5a/reproduce_from_table.out.txt`).

## I. 2026-11-10: owner revision (9f6c676) and its follow-up

- The abstract was rewritten by the owner's patch on 91d94ef. Its numbers map
  to A6 (tighter-budget protected ratios 0.097 / 0.098, added in the
  follow-up) and A11 (B=1024 residuals and ratios 0.434 / 0.487). All are
  pinned in `paper_claims.json`.
- A11's gloss "about a 2× nominal budget ratio" was inaccurate and is
  replaced by the measured medians.
- A17 added: strict-cohort ratios for all eight M5-A arms, kept separate from
  the 100-problem S.1 ratios.
- Figure 3 and the §4.6 opener now use scoped wording (section E).

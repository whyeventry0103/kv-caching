# Meeting #3 — results and execution status

Last updated 2026-09-26 (Asia/Dubai 22:30). Machine-readable companions:
`configs/meeting3/cells.yaml` (ledger) → `results/diag/meeting3/meeting3_manifest.json`
(`python scripts/meeting3_manifest.py --check`). Protocol: PREREGISTRATION
Amendment R (R.0–R.12). Every number below was recomputed from the named
artifact when this file was written. Review evidence (gate outputs, remote test run,
parity, queue dry-run, remote-only data list):
[`meeting3_evidence/`](meeting3_evidence/README.md).

**Status, 2026-10-02 18:30 Dubai.** Llama-8B capture READ (§3f): **ω contrast
REVERSED** (−0.355 [−0.453, −0.275]); H3 HOLDS; H2 NEITHER; collateral 0.

**Status, 2026-10-02 17:00 Dubai.** The readers now sit behind one validation
boundary (`kvcache.read_validation`, after the independent review of
`f69dce3`): FINAL / PARTIAL / INVALID, labels only on FINAL. Re-read with it:
restore-v2 FINAL (unchanged), **restore-v3 FINAL (§3e: Random PARTIAL and
PROMPT-SPECIFIC; StreamingLLM INDETERMINATE and NON-SPECIFIC)**, S.1 budget
256 PARTIAL (free-run 800/800 complete, forward readout pending; no label).
Llama-8B capture 400/400 (not yet read); Qwen3-4B bank 150/200.

**Status, 2026-09-29 03:20 Dubai. Tier 2 is complete.**
- restore-v2 FINAL at 200/200: full self-restoration is **MIXED** for both
  policies (R_fs 0.338 Random, 0.297 StreamingLLM), and the cohort arms repeat
  the exact-lag labels on 24 problems (§3d).
- fork-precision FINAL: the validity gate FAILS by 0.3% (§4a).
- Storage re-measured after Tier 2: 69.449 GiB of 72.
- Tier 3 (Llama-8B) is still held; releasing it needs the owner's go-ahead
  in-session.

**Status, 2026-09-27 11:00 Dubai.** Q.1 fork control FINAL (all
TOLERANCE-DEGENERATE; the gap distribution is reported in §4). capture-v2 smoke
PASSED. The k = 256 basis is frozen and accepted. capture-b2048 is at 380/400
and resumes 2026-09-28 01:45, followed by t1-tv512/1024, restore-v2 and
fork-precision. Tier 3 (Llama-8B) is prepared and held until Tier 2 completes.

**Status, 2026-09-27 (Amendment R.17 in progress).** Target venue: proposed ICML
2027, pending supervisor confirmation.

- R.17 was registered (`ac78eab`) before any of its reads ran.
- **Task 1 onset effect: NOT ESTABLISHED** once never-compressed traces are
  excluded (R.17.1.2). Restricted B = 256 contrasts, t0 0 vs 2048:
  - Random −0.048 [−0.184, +0.075];
  - StreamingLLM +0.026 [−0.066, +0.118]. See §2a.
- **Paper (`a549250`).** The P.5 reinsertion specificity control is
  retracted (stash defect, R.5), and reinsertion is restated from the
  repaired exact-lag read. Eviction locking is reworded "detected, not
  established" (R.2).
- **Bridge (R.17.1.8):** 0 of 6 AUROC tests survive Benjamini–Hochberg. The
  pre-committed "ω and Δ predict truncation" is NOT CONFIRMED (§3a).
- **Validity gate (R.17.0.5).**
  - The fork control is still running.
  - Its replacement PASS rule (fork p95 ≤ 2 × a bf16-vs-fp32 precision
    floor) is registered.
  - The fp32 diagnosis and the floor are new jobs.
- **Queue.** capture-b2048 is held for the v2 schema (R.17.2.1). t1-tv512 and
  t1-tv1024 stay queued: their summaries carry the collapse-test windows
  (R.17.1.1).

**Execution status, 2026-09-26 22:30 Dubai (one-time daytime start at the owner's request, R.16.6).**

- **Task 1 D_acc: FINAL.** 1,200/1,200 generations, 0 provenance failures;
  the R.8.3 determinism control PASSED (200/200 token-identical at each
  budget).
  - **B = 256:** earlier onset does more damage. Damage at t0 = 0 vs 2048:
    Random +0.110 [+0.030, +0.200], StreamingLLM +0.190 [+0.080, +0.310]
    (descriptive paired contrasts).
  - That effect is carried almost entirely by the 36/100 generations whose
    full-cache trace ends before step 2048. On the 64 long generations,
    delaying onset recovers little.
  - **B = 2048:** no onset contrast excludes 0 (non-identifying, R.8.1).
  - See §2a.
- **Running (11:39 start, stops 2026-09-27 10:43):** fork-control-random and
  fork-control-slm (28 and 15 of 75 rows at 22:12). Not started:
  capture-b2048, t1-tv512, t1-tv1024.

- **Tasks 2/3 primary cell capture-b256: FINAL** (400/400, controls VALID,
  0 provenance failures). Q.3 collateral R-KV − H2O-style = +0.000
  [+0.000, +0.000] → **INCONCLUSIVE**. Q.6 ω(Random) − ω(StreamingLLM) =
  **+0.200 [+0.159, +0.234] → PREDICTED (exploratory)**. See §3a.
- **Q.5 exact-lag: FINAL** (200/200). Matched reinsertion restores nothing
  at any lag for either policy: Random NON-MONOTONE (lag 1 +0.0031
  [+0.0004, +0.0061], but NON-SPECIFIC), StreamingLLM FLAT-NULL. Only 8/25
  problems are eligible (stash size is set by prompt length), so the read is
  exploratory. See §3b.
- **End of night 2 (10:43 Dubai):**
  - dacc-t2048 **400/400** (complete); dacc-t512 **320/400**.
  - dacc-t512 stopped on the window timer (rc 124), resumable.
  - Night 2 had no aborts, tracebacks or storage blocks after the 00:31 fix.
  - Not started: dacc-t0, fork-control-random/-slm, capture-b2048,
    t1-tv512, t1-tv1024.
  - post-dacc reads only when all three D_acc arms are complete.
  - The next window opens 2026-09-27 01:45 (cron).
  - Estimated remaining GPU work, on the §6 cost estimates: about 4
    windows.

**Execution status, 2026-09-25 11:30 Dubai (after GPU night 1).**

- **Real-model gates, all PASSED.** Reports are in `results/diag/meeting3/gates/`.
  - **D_acc gate.** The gated t0=0 run is token-identical to legacy generation
    for Random and StreamingLLM. At t0=512, eviction starts at token 576 exactly
    as predicted, and tokens [0, 576) are identical to the full cache.
  - **Q.6 basis.** 100 traces, k=64, orthonormality error 6.8e-9, 340 s. The
    64 directions explain 21–41% of variance per layer.
  - **Capture smoke.** Full/full differs by exactly 0. Layer-0 Δ is exactly 0.0
    for all four policies, with 21 events each and no non-finite value.
- **Generated.** capture-b256 **255/400**, dacc-t2048 **200/400**.
  - Read by the strict readers: PARTIAL, 0 provenance failures, controls
    VALID, verdicts withheld.
  - No cell is complete yet, so there are no Task 2/3 answers yet.
  - The next window (2026-09-26 01:45) resumes capture-b256 seeds 23/89.
    The post-capture-b256 stage then reads it automatically.
- **Collateral (corrected, R.16.3).** Pooled GQA collateral is 0.0 for all
  four policies on the smoke trace and on the partial capture-b256 rows.
  Per-policy eligible/wanted/collateral totals are in
  `results/diag/meeting3/partial/`.
  - **Budget 2048: forced to zero** with 7 heads and τ = 0.01 (a wanted
    position's group mean exceeds 1/700; fewer than 700 positions can, all
    inside the top 2048). The collateral contrast there is structurally
    non-identifying.
  - **Budget 256: not forced.** A production positive control gives
    collateral 1.0, so the zeros there are an empirical property of the
    measured attention. My earlier claim that they follow from the definition
    was wrong and is withdrawn.
  - If the full cell stays at zero, the registered R-KV − H2O contrast at 256
    reads INCONCLUSIVE. That says nothing about GQA damage in general; M, Δ and
    ω are unaffected.
- **Storage.** TC-ViTok (both folders, selected in-session) deleted, 77.61 GiB
  reclaimed. The guard measures `~/kvcache-storage` + `~/kv-caching` with
  hardlinks counted once, against **72 GiB, explicitly approved by the owner
  2026-09-25 (R.16.4)**. R.15.6's "owner-delegated" wording is corrected
  there.
- **Pipeline.** Hardened after the review of cfa30cd (R.15). Acceptance table:
  `meeting3_evidence/acceptance_cfa30cd.md`.

The three Meeting #3 problems, and where each stands:

| Problem | Measured now | Queued (unattended, resumable) | Not yet runnable |
|---|---|---|---|
| **1. Divergence/damage vs cache size, exposure/horizon** | Accuracy curve 256→2048 with paired CIs, flips, realized ratios (**COMPLETE**); Q.2 exposure grid + occupancy audit (**COMPLETE**, verdict MIXED); D_acc free-running (budget, onset) grid, 1,200 generations (**FINAL**, §2a; onset effect **not established** once never-compressed traces are excluded, R.17.1.2) | Q.1 direct fork control, 150 conditions (**running**); TV at 512/1024 (summary profiles; also feed the collapse test R.17.1.1) | precision floor and fp32 fork diagnosis (R.17.0.5; being implemented) |
| **2. Attention maps: M/Δ, GQA collateral, restoration** | Q.3 capture at budget 256 (**FINAL**, 400 conditions; collateral contrast INCONCLUSIVE, M/Δ per policy with CIs, §3a); Q.5 exact-lag (**FINAL**: Random NON-MONOTONE, StreamingLLM FLAT-NULL; 8 eligible problems, exploratory, §3b) | Q.3 at 2048 **FINAL** (§3c: H3 HOLDS, H2 NEITHER, confirmatory) | restoration redesign restore-v2 **FINAL** (§3d: full self-restoration MIXED for both policies; cohort arms NON-MONOTONE / FLAT-NULL on 24 problems) |
| **3. SVD/PCA trajectory direction** | Q.6 at budget 256 (**FINAL**, PREDICTED, §3a) and at 2048 (**FINAL**, replication PREDICTED; ordering holds at every k from 8 to 256, §3c) | — | — |

---

## 1. Task 1 — damage against cache size (COMPLETE)

Artifact `results/diag/meeting3/t1_budget_curve.json` (`scripts/report_budget_curve.py`,
commit fca22f3); figure `figures/fig8_budget_curve.{pdf,png}`. MATH-500,
R1-Distill-Qwen-7B, 100 problems × 4 seeds = **400 matched conditions per
cell**, full-cache reference `math500-fc-powered-merged` (matched FC accuracy
0.8850 in every cell). Paired, problem-clustered 95% CI (2000 resamples, seed
0), Amendment R.1. Every paired total change equals the m4 summary to 4 dp.

| Policy | Budget | Realized ratio | Free acc | Total change [95% CI] | Lost | Gained | Readout change | Residual [95% CI] | Trunc | AUC-TV |
|---|---|---|---|---|---|---|---|---|---|---|
| Random | 256 | 3.16% | 0.4225 | −0.4625 [−0.545, −0.380] | 0.4675 | 0.0050 | −0.1378 | −0.3367 [−0.404, −0.262] | 0.233 | 0.1075 |
| StreamingLLM | 256 | 4.79% | 0.2375 | −0.6475 [−0.723, −0.570] | 0.6500 | 0.0025 | −0.3189 | −0.3444 [−0.405, −0.270] | 0.070 | 0.1028 |
| Random | 512 | 10.41% | 0.6800 | −0.2050 [−0.273, −0.142] | 0.2050 | 0.0000 | −0.0204 | −0.1888 [−0.252, −0.129] | 0.085 | — |
| StreamingLLM | 512 | 9.68% | 0.4775 | −0.4075 [−0.485, −0.333] | 0.4075 | 0.0000 | −0.0816 | −0.3342 [−0.394, −0.263] | 0.087 | — |
| Random | 1024 | 27.67% | 0.8000 | −0.0850 [−0.128, −0.045] | 0.0875 | 0.0025 | +0.0000 | −0.0867 [−0.133, −0.048] | 0.033 | — |
| StreamingLLM | 1024 | 24.41% | 0.6875 | −0.1975 [−0.268, −0.138] | 0.2050 | 0.0075 | −0.0051 | −0.1964 [−0.263, −0.135] | 0.045 | — |
| Random | 2048 | 58.18% | 0.8500 | −0.0350 [−0.062, −0.013] | 0.0375 | 0.0025 | −0.0026 | −0.0332 [−0.065, −0.010] | 0.030 | 0.0128 |
| StreamingLLM | 2048 | 55.11% | 0.8075 | −0.0775 [−0.125, −0.040] | 0.0875 | 0.0100 | +0.0000 | −0.0791 [−0.125, −0.042] | 0.028 | 0.0186 |

"Residual" = accuracy change unexplained by on-reference readout (never a
measured causal effect). Realized ratio = budget ÷ (mean prompt + mean
generated tokens).

**Answer.** Damage falls monotonically with cache size for both policies. The
CI of total change excludes zero at every budget, including 2048 (~55–58% of
the trace retained). Damage is almost entirely harmful flips: beneficial flip
mass is ≤ 0.010 in every cell. The readout part closes by 1024 (|readout
change| ≤ 0.005), while the residual persists through 2048. Divergence (AUC-TV)
was measured at 256 and 2048 only. At 256 the two policies' AUC-TV is similar
(0.108 vs 0.103), but their total damage differs by 0.185. This is consistent
with P.3: divergence magnitude does not predict damage within a budget.

## 2. Task 1 — exposure at matched dose (Q.2; COMPLETE, verdict MIXED)

Artifacts: `results/diag/amendment-q/onset_surface_q2.json` (pre-registered read,
re-run 2026-09-24 and **byte-identical**), `results/diag/meeting3/q2_onset_grid.json`
(`scripts/report_onset_grid.py`, commit 5e567cf); figure `figures/fig9_onset_grid.{pdf,png}`.
1,200 teacher-forced profiles (3 onset arms × 2 policies × 2 budgets × 25
problems × 4 seeds); window 256, read positions t = 2304/3328/4352. Slopes
are ×10³ (TV per 1,000 steps), paired within trace, problem-clustered 95% CI:

| Cell | Realized ratio | Problems / traces | Nominal→actual exposure, t0 = 0 / 512 / 2048 | Slope on nominal exposure (registered) | Slope on actual exposure (diagnostic) |
|---|---|---|---|---|---|
| B256 @2304 | 0.106 | 19 / 104 | 2304→2118 / 1792→1728 / 256→192 | **+4.15e−03 [+2.65e−03, +5.64e−03]** | +4.43e−03 [+2.89e−03, +5.95e−03] |
| B256 @3328 | 0.075 | 14 / 80 | 3328→3140 / 2816→2752 / 1280→1216 | **−8.88e−04 [−1.46e−03, −3.21e−04]** | −9.12e−04 [−1.46e−03, −3.80e−04] |
| B256 @4352 | 0.057 | 10 / 58 | 4352→4172 / 3840→3776 / 2304→2240 | **−1.35e−03 [−2.42e−03, −5.93e−04]** | −1.36e−03 [−2.46e−03, −5.51e−04] |
| B2048 @2304 | 0.847 | 19 / 104 | 2304→331 / 1792→332 / 256→192 | +2.35e−04 [−2.42e−05, +5.10e−04] | +3.49e−03 [−9.96e−04, +8.01e−03] |
| B2048 @3328 | 0.596 | 14 / 80 | 3328→1348 / 2816→1349 / 1280→1216 | −1.20e−04 [−4.85e−04, +2.44e−04] | −1.73e−03 [−6.53e−03, +2.79e−03] |
| B2048 @4352 | 0.459 | 10 / 58 | 4352→2380 / 3840→2381 / 2304→2240 | −7.27e−05 [−4.91e−04, +3.62e−04] | +1.71e−03 [−3.97e−03, +1.10e−02] |

**Verdict (pre-registered, full panel): MIXED.** Report the interaction and
claim no main effect. At B = 256 the exposure slope is positive at t = 2304
and negative at 3328/4352. That sign change survives the balanced
fixed-population panel (116 traces), so it is not trace-length attrition.

**Occupancy audit (new, R.8.1).** `compression_start_step` only enables the
press. Eviction starts when prompt + generated tokens first exceed the budget,
measured as the first step with TV > 0 (the invariant held: 0 traces diverge
before nominal onset). Mean eviction onset in steps (range):

- **B = 256:** 203 (65–257) for t0 = 0; 576 for t0 = 512; 2112 for t0 = 2048.
- **B = 2048:** 1976 (1729–2049) for t0 = 0 and 1975 (1728–2048) for t0 = 512, both policies; 2112 for t0 = 2048.

So at **B = 2048 the t0 = 0 and t0 = 512 arms have the same actual exposure.**
The 2048-step nominal contrast is realized as only ~140 steps, and the B = 2048
row cannot identify exposure. Only B = 256 does, and there the nominal and
actual slopes agree. The committed Q.2 artifact's `realized_ratio` omits the
prompt (B/t); the grid reports both.

**D_acc (free-running accuracy on the same grid): FINAL** (§2a; restricted re-read R.17.1.2).

## 2a. Task 1 — damage against exposure, free-running (D_acc; FINAL)

Read 2026-09-26 21:40 Dubai by the post-dacc stage.
- **Artifact:** `results/diag/meeting3/dacc.json`, status FINAL.
- **Sufficient statistics:** `dacc.conditions.tsv`, with one row per
  (onset, policy, budget, problem, seed).
- **Figure:** `results/diag/meeting3/figures/meeting3_dacc.{pdf,png}`.

Coverage, estimand and controls:
- Coverage COMPLETE: 1,200/1,200 generations (3 onsets × 2 policies ×
  2 budgets × 25 problems × 4 seeds). 0 provenance failures; identity
  consistent within each arm.
- **Estimand (R.3):** damage = Acc_FC − Acc_comp, paired by (problem, seed)
  against `math500-fc-powered-merged`, equal-problem mean, problem-cluster
  bootstrap 95% CI.
  - Truncated generations are scored incorrect. A truncation hits the 16,384
    cap *with no answer marker*; cap-hitting rows that contain a marker are
    scored normally.
  - Generations that end before their onset stay in the denominator as
    unexposed assigned observations.
- **Determinism control (R.8.3): PASSED.** The gated t0 = 0 arm is
  token-identical to the legacy grid rows for all 200 shared generations at
  each budget.
- Full-cache accuracy on the subset is 0.850.

**Registered display: the (budget, onset) grid.**
- "Realized" is budget ÷ (mean prompt 93.7 + mean generated tokens).
- Flip columns are masses over the 100 pairs.

| Policy | Budget | t0 | Realized | Comp acc | Damage [95% CI] | Harmful | Beneficial | Trunc | Unexposed |
|---|---|---|---|---|---|---|---|---|---|
| Random | 256 | 0 | 2.76% | 0.37 | +0.480 [+0.310, +0.640] | 0.49 | 0.01 | 0.30 | 0 |
| Random | 256 | 512 | 2.68% | 0.36 | +0.490 [+0.340, +0.630] | 0.49 | 0.00 | 0.32 | 4 |
| Random | 256 | 2048 | 3.27% | 0.48 | +0.370 [+0.210, +0.530] | 0.37 | 0.00 | 0.23 | 36 |
| StreamingLLM | 256 | 0 | 3.90% | 0.25 | +0.600 [+0.420, +0.750] | 0.61 | 0.01 | 0.10 | 0 |
| StreamingLLM | 256 | 512 | 3.93% | 0.27 | +0.580 [+0.400, +0.740] | 0.58 | 0.00 | 0.13 | 4 |
| StreamingLLM | 256 | 2048 | 4.38% | 0.44 | +0.410 [+0.230, +0.580] | 0.42 | 0.01 | 0.10 | 36 |
| Random | 2048 | 0 | 48.10% | 0.81 | +0.040 [−0.010, +0.120] | 0.05 | 0.01 | 0.04 | 31 |
| Random | 2048 | 512 | 49.84% | 0.84 | +0.010 [−0.020, +0.050] | 0.03 | 0.02 | 0.02 | 31 |
| Random | 2048 | 2048 | 45.70% | 0.82 | +0.030 [−0.020, +0.090] | 0.05 | 0.02 | 0.05 | 36 |
| StreamingLLM | 2048 | 0 | 47.06% | 0.76 | +0.090 [+0.030, +0.160] | 0.10 | 0.01 | 0.03 | 31 |
| StreamingLLM | 2048 | 512 | 44.60% | 0.78 | +0.070 [+0.020, +0.150] | 0.07 | 0.00 | 0.05 | 31 |
| StreamingLLM | 2048 | 2048 | 46.21% | 0.76 | +0.090 [+0.020, +0.180] | 0.10 | 0.01 | 0.01 | 36 |

At B = 2048, 31–36 of 100 generations never exceed the budget at all, and
t0 = 0 and 512 reach nearly the same achieved exposure. Onset readings there
are **non-identifying** (R.8.1).

**Onset contrasts** (descriptive, not registered).
- Extra damage from the earlier onset = Acc_comp(later) − Acc_comp(earlier),
  with the same paired equal-problem estimator (`paired_equal_problem`,
  2,000 resamples, seed 0).
- None of these tests carries a registered verdict. The family is small and
  reported whole.

| Policy | Budget | t0 0 vs 2048 | t0 512 vs 2048 | t0 0 vs 512 |
|---|---|---|---|---|
| Random | 256 | **+0.110 [+0.030, +0.200]** | **+0.120 [+0.010, +0.230]** | −0.010 [−0.090, +0.070] |
| StreamingLLM | 256 | **+0.190 [+0.080, +0.310]** | **+0.170 [+0.050, +0.310]** | +0.020 [−0.050, +0.090] |
| Random | 2048 | +0.010 [−0.020, +0.050] | −0.020 [−0.080, +0.030] | +0.030 [−0.030, +0.110] |
| StreamingLLM | 2048 | +0.000 [−0.050, +0.040] | −0.020 [−0.070, +0.030] | +0.020 [−0.030, +0.070] |

**Where the B = 256 onset effect comes from** (descriptive).
- The gated arm is bit-identical to the full cache before its onset
  (R.8.2, `dacc-gate-smoke`).
- So a generation is unexposed at t0 = 2048 exactly when its full-cache
  trajectory ends first. That is a pre-treatment property: the 36 such
  (problem, seed) pairs are identical for both policies.
- Splitting the t0 = 0 vs 2048 contrast on it:

| Policy | Stratum | Pairs | FC acc | Comp acc t0 = 0 → 2048 | Share of the 100-pair contrast | Within-stratum contrast |
|---|---|---|---|---|---|---|
| Random | short (full-cache trace ends < 2048) | 36 | 0.972 | 0.694 → 0.972 | +0.100 of +0.110 | +0.285 [+0.111, +0.465] |
| Random | long | 64 | 0.781 | 0.188 → 0.203 | +0.010 | −0.035 [−0.184, +0.092] |
| StreamingLLM | short | 36 | 0.972 | 0.556 → 0.972 | +0.150 of +0.190 | +0.479 [+0.250, +0.729] |
| StreamingLLM | long | 64 | 0.781 | 0.078 → 0.141 | +0.040 | +0.083 [+0.004, +0.171] |

**Registered re-read, R.17.1.2 (2026-09-27; not blind, because the
stratification above was seen first).**
- *Restriction:* (problem, seed) pairs whose full-cache reference exceeds
  2,304 generated tokens, a pre-treatment covariate.
- *Retained:* 60/100 pairs, full-cache accuracy 0.798 [0.623, 0.939]. 0
  retained pairs are unexposed at t0 = 2048 (no attrition).
- *Artifact:* `results/diag/meeting3/r17/dacc_restricted.json`.
- *Contrast:* Acc_comp(later onset) − Acc_comp(earlier onset), with
  equal-problem CIs.

| Policy | Budget | t0 0 vs 2048: unrestricted → restricted | t0 512 vs 2048: unrestricted → restricted |
|---|---|---|---|
| Random | 256 | +0.110 [+0.030, +0.200] → **−0.048 [−0.184, +0.075]** | +0.120 [+0.010, +0.230] → −0.009 [−0.092, +0.083] |
| StreamingLLM | 256 | +0.190 [+0.080, +0.310] → **+0.026 [−0.066, +0.118]** | +0.170 [+0.050, +0.310] → +0.061 [−0.048, +0.171] |
| Random | 2048 | +0.010 [−0.020, +0.050] → +0.013 [−0.026, +0.053] | −0.020 [−0.080, +0.030] → −0.026 [−0.105, +0.039] |
| StreamingLLM | 2048 | +0.000 [−0.050, +0.040] → +0.013 [−0.066, +0.092] | −0.020 [−0.070, +0.030] → −0.018 [−0.096, +0.066] |

**Answer to Task 1 (exposure/horizon) at this scope** (one model, MATH-500
subset of 25 problems):
- Per the pre-committed reading, the **onset effect on accuracy is not
  established** at B = 256 for either policy. Its unrestricted appearance
  comes from generations that finish before compression starts in the late
  arm.
- Among long traces, compressed in both arms, a 2,048-step delay of
  compression onset changes accuracy by an amount **not distinguishable from
  zero at this n** (60 pairs). That is not evidence of no effect.
- At B = 2048 (~46–50%), damage is small at every onset and the onset contrast
  is non-identifying.
- Divergence is not damage: the teacher-forced D_tv read (Q.2, MIXED) stands
  separately (R.3). The dose-vs-horizon collapse test on TV is R.17.1.1, which
  is waiting on the t1-tv cells.

Reproduce:

```bash
python scripts/report_dacc.py --runs-root <runs> \
    --arms 0=dacc-t0 512=dacc-t512 2048=dacc-t2048 \
    --design configs/experiments/grid_math500_dacc_t0_r3.yaml --out <out> --final
```

The descriptive contrasts come from `kvcache.coverage.paired_equal_problem` over
`dacc.conditions.tsv`: `comp_correct` keyed by (pid, seed).

## 2b. Task 1 — dose vs horizon: the collapse test (R.17.1.1) and TV at four budgets (R.17.1.3)

Read 2026-09-28 23:55 Dubai. Artifacts:
- `results/diag/meeting3/r17/collapse.json`
- `r17/auc_tv.json`
- figures `figures/r17_collapse.{pdf,png}` and
  `figures/r17_supervisor_damage_and_tv.{pdf,png}`

**Design.** At onset 0, window-mean TV (256-step window) at positions
2304/3328/4352 for B = 256/512/1024/2048, with r = B / (prompt + position).
- *Panel:* the 29 traces (10 problems) with a full window at 4352 under all
  four budgets; mean prompt 114.7.
- *Pairs:* each pair differs two-fold in budget and by 2,048 positions, with
  r matched within 7.7%.
- *Source profiles:* B = 256/2048 from the Q.2 onset-0 arm, 512/1024 from the
  t1-tv summaries.

| Pair (r) | Random: d [95% CI] | StreamingLLM: d [95% CI] |
|---|---|---|
| B256@2304 (0.106) vs B512@4352 (0.115) | +0.0361 [−0.0133, +0.0692] | +0.0229 [−0.0183, +0.0495] |
| B512@2304 (0.212) vs B1024@4352 (0.229) | +0.0177 [−0.0226, +0.0452] | +0.0184 [−0.0295, +0.0500] |
| B1024@2304 (0.423) vs B2048@4352 (0.459) | +0.0121 [−0.0057, +0.0242] | +0.0198 [−0.0260, +0.0507] |
| Range of cell means R | 0.1435 | 0.1140 |
| **Registered label** | **MIXED** | **COLLAPSE** |
| Pooled: ΔR² of budget after log r | 0.029 [0.004, 0.073] | 0.033 [0.003, 0.082] |

d = TV(smaller B, earlier position) − TV(larger B, later position).
- **StreamingLLM collapses** onto r by the registered rule.
- **Random is MIXED** only because pair 1's point estimate (0.0361) exceeds
  0.25 R (0.0359) by 0.0002. All three of its CIs include 0.
- Every point estimate is positive (the smaller cache at the earlier position
  diverges more). The pooled fit gives budget a small share of variance beyond
  log r whose CI excludes 0.
- So a residual absolute-budget effect is **suggested but not established at
  this n** (10 problems). Dose (r) carries most of the divergence.

**AUC-TV** (per-trace mean TV, onset 0, 25 problems × 4 seeds, clustered CI):

| Budget | Random | StreamingLLM |
|---|---|---|
| 256 | 0.1075 [0.0906, 0.1240] | 0.1028 [0.0831, 0.1235] |
| 512 | 0.0561 [0.0433, 0.0688] | 0.0652 [0.0506, 0.0806] |
| 1024 | 0.0279 [0.0187, 0.0379] | 0.0359 [0.0248, 0.0482] |
| 2048 | 0.0128 [0.0063, 0.0201] | 0.0186 [0.0091, 0.0291] |

## 3. Task 1 — phase locking (Q.1; recorded, control outstanding)

Recorded 2026-09-15 (Amendment Q.1, result): **Random is AMPLITUDE-DEPENDENT**.
Phase excess at I = 16 is +0.0110 [−0.0252, +0.0300] (does not exclude 0); at
I = 64, +0.1193 [+0.0446, +0.1410]; at I = 256, +0.3183 [+0.1881, +0.3361],
both peaking at phase 0. **StreamingLLM is INDETERMINATE**: no arm excludes 0,
and the peak sits at phase 9. Neither instrument check has run:

- The registered removal-anchored negative control is **NOT VALID FOR
  IDENTIFICATION** (R.2: `phase_amplitude` is rotation-invariant).
- The R.2 direct functional control (mask, then fork before physical removal)
  is **built, CPU-verified and queued** (R.11, §6).

The zero-amplitude `excess` bug is fixed (`phase-excess-v2`).

## 3a. Tasks 2/3 — attention channels and trajectory direction at budget 256 (FINAL)

Read 2026-09-26 by the post-capture-b256 stage.
- **Artifact:** `results/diag/meeting3/capture_b256.json`, status FINAL.
- **Sufficient statistics:** `capture_b256.conditions.tsv`, sha256 `3ccd9f39…`,
  recorded in the JSON.
- **Figure:** `results/diag/meeting3/figures/meeting3_capture_b256.{pdf,png}`.

Coverage and design:
- Coverage COMPLETE: 400/400 conditions (25 problems × 4 seeds × 4 policies),
  0 duplicated, 0 unexpected, 0 provenance failures, identity consistent,
  every control VALID.
- **Budget 256 = realized ratio 5.77%:** 256 ÷ (93.7 mean prompt + 4,343.5
  mean trace tokens) over the 100 teacher-forced full-cache traces, which
  range from 510 to 16,538 tokens.
- Estimator: equal-problem bootstrap (seeds within a problem, then problems),
  2,000 resamples, seed 0, 95% percentile CI.

**Registered reads**

- **Q.3 collateral: R-KV − H2O-style (ObservedAttentionPress, no recency
  reservation) = +0.000 [+0.000, +0.000] → INCONCLUSIVE.**
  - Collateral is 0 in every one of the 400 conditions.
  - Per policy, 46,451,328 evicted (position, KV group, layer, event)
    instances were checked. Some head of the group gave more than τ = 0.01 of
    its full attention to:

    | Policy | Head-wanted instances |
    |---|---|
    | Random | 3,731,830 |
    | R-KV | 1,294,665 |
    | StreamingLLM | 739,823 |
    | H2O-style | 54,932 |

  - None of them ranked outside the group-mean top-256, so none counts as
    collateral under R.4.
  - At this budget the zero is **not** forced by the definition: the positive
    control gives 1.0 (R.16.3). It is an empirical property of this model's
    attention. The group-mean ranking never buried a position that a single
    head wanted at more than 1%.
  - The registered contrast therefore cannot order R-KV against H2O-style.
    This is not evidence that GQA sharing is harmless in general: M and Δ
    below show eviction removes substantial attended mass.
- **Q.6 trajectory direction: ω(Random) − ω(StreamingLLM) = +0.200
  [+0.159, +0.234] → PREDICTED.**
  - ω = |d⊥| / |d∥| is the ratio of the hidden-state deviation outside the
    frozen full-cache k = 64 basis to the deviation inside it.
  - Random's deviation is more off-subspace than StreamingLLM's.
  - **Exploratory (R.9):** one model, one benchmark, one budget. No
    accuracy-scale claim.

**Descriptive** (per policy, mean [95% CI] over 25 problems; not tested, no
ordering claims):

| Policy | M: full-cache mass on evicted positions | Δ: shape TV on kept set | ω |
|---|---|---|---|
| Random | 0.529 [0.502, 0.550] | 0.238 [0.217, 0.257] | 3.989 [3.936, 4.039] |
| StreamingLLM | 0.268 [0.236, 0.299] | 0.160 [0.135, 0.185] | 3.789 [3.724, 3.857] |
| H2O-style | 0.215 [0.183, 0.245] | 0.152 [0.129, 0.175] | 3.689 [3.618, 3.768] |
| R-KV | 0.260 [0.226, 0.293] | 0.139 [0.118, 0.161] | 3.868 [3.802, 3.940] |

- Random evicts 1.97–2.47× the attended mass of the three structured
  policies (M 0.529 vs 0.215–0.268), and its CI is disjoint from theirs.
- Among the structured policies, H2O-style has the lowest M and R-KV the
  lowest Δ. Their CIs overlap.
- The trigger-step Δ/M ratio is descriptive only. It is measured at the first
  eviction at or after step 130, not at Q.5 reinsertion states, and licenses
  no query-drift claim. The reader JSON carries it, along with
  `wanted_any_head` and `M_event`.
  - Trigger Δ/M: Random 0.4645 [0.3907, 0.5663]; StreamingLLM 0.4572
    [0.3945, 0.5317]; H2O-style 0.9854 [0.7703, 1.2683]; R-KV 0.5305
    [0.4485, 0.6285].
  - The Q.3 cross-instrument prediction (Δ vs M at the *reinsertion* trigger)
    is **outstanding**: it was not measured at those states (R.15.5).
    R.17.2.3 measures it.

**R.17 additions (2026-09-27).** Artifacts are under
`results/diag/meeting3/r17/`: `mdelta.json`, `omega_layers.json` and
`bridge.json`.

*What Q.3 registered for M and Δ.* Per-policy M, M_event, Δ and
wanted_any_head, plus Δ/M at the trigger: all descriptive, no test (R.9). The
only registered test is the collateral contrast.

*What is stored* (0.6/0.7):
- **M and Δ:** per trace, per-(layer, KV group) sums over events, plus the
  per-(layer, group) values at the first, second and trigger events.
- **Per-query-head values** are computed and then averaged within each 7-head
  group. They are **not stored** in capture-v1.
- **Latent geometry:** per step and per layer, only the norms (‖δ∥‖, ‖δ⊥‖) at
  k = 64, all 28 layers, all four policies. They are stored as per-step sums
  plus 256 sampled positions in a per-row npz. **No projection coefficients**
  are kept, so k-sensitivity cannot be computed on b256.
- **Per-trace values** are stored for everything above.

*First-prune self-control.* Layer-0 Δ is **exactly 0.0** (maximum over all 400
rows and every event), with 0 non-finite values. The registered tolerance is
≤ 1e-5 (R.9).

*Per-layer profiles* (equal-problem mean [95% CI], full tables in
`mdelta.json` and `omega_layers.json`):
- **M.** Random peaks at layer 4, 0.6715 [0.6314, 0.7030]. StreamingLLM peaks
  at layer 14, 0.3836 [0.3414, 0.4202]. H2O-style and R-KV peak at layer 16:
  0.3197 [0.2782, 0.3566] and 0.3790 [0.3351, 0.4177]. The minimum is at layer
  0 for all four.
- **Δ.** Exactly 0 at layer 0 (the control above). It peaks at layer 16 for
  every policy: Random 0.3886 [0.3557, 0.4167], StreamingLLM 0.2703
  [0.2354, 0.3040], H2O-style 0.2534 [0.2182, 0.2861], R-KV 0.2363
  [0.2026, 0.2690].
- **ω.** Random ranges from 4.4957 [4.4305, 4.5518] at layer 8 down to 2.4172
  [2.3266, 2.5166] at layer 27. StreamingLLM ranges from 4.2607
  [4.0981, 4.4128] at layer 0 down to 2.3554 [2.2587, 2.4659] at layer 27.

*H2 / H3 (R.17.1.5; EXPLORATORY on b256, because the means were printed
before R.17; confirmatory on b2048).*
- **H2: NEITHER.**
  - spread(M) = 0.3147 against a median CI half-width of 0.0312;
  - spread(Δ) = 0.0984 against 0.0223.
  - Both spreads are large, so Δ does not mirror the accuracy-scale
    near-constant residual here.
- **H3: HOLDS.** M(H2O-style) − M(Random) = **−0.3147 [−0.3296, −0.2981]**.
  - H2O-style loses far less full-model attention mass than Random, yet has
    larger free-run damage at B = 256: total change −0.6375 vs −0.4625
    (`results/diag/amendment-k/m4-*256*`).
  - The attention-mass objective is not the damage objective, at this scope.

*Bridge (R.17.1.8).* Teacher-forced per-trace M, Δ and ω against free-run
truncation of the same (problem, seed, policy) at B = 256, from
`grid-diss256-tok`. AUROC [95% CI], two-sided bootstrap p, Benjamini–Hochberg
q over the 6 tests:

| Policy (truncated / traces) | ω | Δ | M |
|---|---|---|---|
| Random (30/100) | 0.543 [0.336, 0.726], q 0.786 | 0.659 [0.500, 0.803], q 0.165 | 0.629 [0.471, 0.771], q 0.165 |
| StreamingLLM (10/100) | 0.478 [0.291, 0.656], q 0.786 | 0.652 [0.484, 0.809], q 0.165 | 0.643 [0.464, 0.810], q 0.165 |

- **The pre-commitment is not confirmed.** ω does not predict truncation (AUROC
  ≈ 0.5), and Δ's positive AUROC does not survive BH.
- M carries no significant prediction, as pre-committed, but its point
  estimates are as large as Δ's. So "Δ predicts and M does not" is **not
  established at this n**.

*Collateral (R.17.1.6).*
- The zero is explained by R.16.3 but is not strictly forced. With a 7-to-1
  grouping, mean aggregation and a 64-of-320 cut per event (20%), a
  head-wanted evicted position is collateral only if at least 256 of the ~320
  candidates have group-mean attention above its own, which is > 1/700 each
  (> 36.6% of the group's mass, spread flat).
- The per-position rank diagnostic that would show how close real events come
  was **not recorded** in capture-v1. That is recorded here as missing and will
  not be rescued with GPU time.
- Within-group dispersion of per-head M needs per-head storage, which arrives
  with capture-b2048 (R.17.2.1).

Reproduce:

```bash
python scripts/report_capture.py --runs-root <runs> --run capture-b256 \
    --config configs/experiments/capture_math500_b256_r9.yaml --out <out> --final
```

## 3b. Task 2 — exact-lag restoration, Q.5 (FINAL)

Read 2026-09-26 01:15 Dubai by the post-exact-lag stage.
- **Artifact:** `results/diag/meeting3/exact_lag.json`, status FINAL.
- **Sufficient statistics:** `exact_lag.conditions.tsv`.
- **Figure:** `results/diag/meeting3/figures/meeting3_exact_lag.{pdf,png}`.

Coverage and design:
- Coverage COMPLETE: 200/200 (trace, policy) rows (25 problems × 4 seeds ×
  2 policies). 0 provenance failures and 0 content failures, so every read
  row passed its bitwise K/V restoration gates.
- Budget 256 (realized ratio 5.77% on these traces, §3a). Cohort = the first
  effective prune at or after step 130; reinsertion n = 64.
- **Read:** mean TV over 200 forwards from the reinsertion step, none minus
  arm (positive = the arm reduces divergence). Trace-level bootstrap exactly
  as P.5: 2,000 resamples, seed 0.
- This is recovery-after-delay: it does not isolate staleness from position
  (R.5).

**Eligibility: the read rests on 8 of 25 problems.**
- 68 of 100 rows per policy are INELIGIBLE for the registered reason "stash
  smaller than the reinsertion count" (R.10).
- The mechanism was verified on all 200 rows: stash size = prompt tokens +
  cohort step − 256, exactly. Eligibility is therefore a deterministic
  function of prompt length and the prune schedule. It agrees across the two
  policies on every trace.
- What remains:

  | Lags | Traces | Problems |
  |---|---|---|
  | 1–16 | 30 | 8 |
  | 64 | 29 | 8 |
  | 256 | 28 | 7 |

  The short traces lose their read window at the long lags.
- The registered bootstrap resamples traces, and these traces cluster
  4-per-problem in 8 problems. **The CIs below are therefore likely too
  narrow, and the read generalizes only to longer-prompt problems.** Treat the
  whole of Q.5 as exploratory at this scope.

Per-lag effects on the registered budget-matched arm, with the P.5 class and
the sweep verdict from `classify_latency_sweep` (4 decimals, the P.5 scale
exception in PAPER_FINDINGS):

**Random: NON-MONOTONE**

| Lag | Traces | Matched (registered) | Random-reinsert | Relaxed (reference only) | P.5 class |
|---|---|---|---|---|---|
| 1 | 30 | +0.0031 [+0.0004, +0.0061] | +0.0044 [+0.0010, +0.0085] | +0.0054 [+0.0021, +0.0090] | NON-SPECIFIC |
| 4 | 30 | +0.0009 [-0.0020, +0.0042] | +0.0032 [+0.0000, +0.0070] | +0.0066 [+0.0029, +0.0108] | NO RESTORATION |
| 16 | 30 | +0.0013 [-0.0018, +0.0042] | +0.0022 [-0.0019, +0.0065] | +0.0040 [+0.0007, +0.0074] | NO RESTORATION |
| 64 | 29 | +0.0036 [-0.0001, +0.0075] | +0.0033 [-0.0002, +0.0074] | +0.0008 [-0.0043, +0.0055] | NO RESTORATION |
| 256 | 28 | +0.0043 [-0.0006, +0.0094] | +0.0067 [+0.0011, +0.0129] | +0.0017 [-0.0037, +0.0073] | NO RESTORATION |

**StreamingLLM: FLAT-NULL**

| Lag | Traces | Matched (registered) | Random-reinsert | Relaxed (reference only) | P.5 class |
|---|---|---|---|---|---|
| 1 | 30 | -0.0004 [-0.0092, +0.0086] | +0.0129 [+0.0025, +0.0238] | +0.0044 [-0.0042, +0.0134] | NO RESTORATION |
| 4 | 30 | -0.0007 [-0.0094, +0.0086] | +0.0125 [+0.0023, +0.0233] | +0.0034 [-0.0051, +0.0122] | NO RESTORATION |
| 16 | 30 | -0.0011 [-0.0093, +0.0073] | +0.0117 [+0.0021, +0.0221] | +0.0013 [-0.0071, +0.0102] | NO RESTORATION |
| 64 | 29 | -0.0035 [-0.0106, +0.0040] | +0.0092 [+0.0014, +0.0177] | -0.0073 [-0.0156, +0.0012] | NO RESTORATION |
| 256 | 28 | -0.0002 [-0.0071, +0.0070] | +0.0133 [+0.0037, +0.0236] | -0.0053 [-0.0128, +0.0025] | NO RESTORATION |

- **Random: NON-MONOTONE.**
  - The matched arm reduces divergence only at lag 1: +0.0031 [+0.0004,
    +0.0061].
  - Even there, random-reinsert (+0.0044) does at least as well, so the P.5
    class is NON-SPECIFIC. Putting back *which* tokens were evicted is not
    shown to matter.
  - At lags 4–256 the matched CI includes 0 (NO RESTORATION).
  - The effect does not decay with lag. Relative to the no-intervention TV
    at the same lag (0.0552–0.0928), every matched effect is at most 5.6%.
- **StreamingLLM: FLAT-NULL.**
  - The matched arm is null at every lag (point estimates −0.0035 to −0.0002,
    all CIs include 0).
  - Descriptive only, not a registered contrast: the random-reinsert arm is
    positive at every lag, +0.0092 to +0.0133, each CI excluding 0 under the
    trace-level bootstrap.
  - So returning a random draw of *earlier*-evicted entries reduces
    divergence, while returning the most recently evicted cohort does not.
    With 8 problems and clustered traces this is a lead, not a finding.
- **Answer to Q.5 at this scope.** Exact reinsertion of the evicted cohort
  does not measurably restore the trajectory for either policy, at any lag
  from 1 to 256, **on long-prompt, low-accuracy problems**.
- **Who the read describes** (R.17 addendum A,
  `results/diag/meeting3/r17/exact_lag_eligibility.json`):

  | Set | Problems / traces | Full-cache accuracy | Full-cache truncation | Realized ratio at B = 256 |
  |---|---|---|---|---|
  | Eligible | 8 / 32 | 0.594 [0.250, 0.875] | 0.125 [0.000, 0.312] | 3.4% (mean prompt 178.5 + trace 7,352.1) |
  | Ineligible | 17 / 68 | 0.971 [0.926, 1.000] | 0.000 [0.000, 0.000] | 8.6% |

  Every problem becomes eligible under the redesign R.17.2.3 (trigger at
  step 1024).
- **Paper.** The historical P.5 specificity control, built on the defective
  stash, is retracted, and the paper now states this read instead
  (PAPER_FINDINGS §5d).

Reproduce:

```bash
python scripts/report_exact_lag.py --runs-root <runs> --run exact-lag-r5 \
    --design configs/experiments/exact_lag_math500_r5.yaml --out <out> --final
```

## 3c. Tasks 2/3 at budget 2048 on capture-v2 (FINAL; R.17 confirmatory reads)

Read 2026-09-28 02:50 Dubai by the post-capture-b2048 stage; the R.17 reads
followed. Artifacts:
- `results/diag/meeting3/capture_b2048.json`
- `r17/mdelta_capture-b2048.json`
- `r17/omega_layers_capture-b2048.json`
- `r17/capture_v2_extras_capture-b2048.json`
- figure `figures/meeting3_capture_b2048.{pdf,png}`

**Coverage and controls.** 400/400, controls VALID (layer-0 Δ exactly 0.0), 0
provenance failures. Budget 2048 is a **46.16%** realized cache (mean prompt
93.7 + trace 4,343.5).
- **31 of 100 traces per policy never exceed the budget.** With no eviction
  event, M, Δ and ω are unavailable, not zero (R.4/R.6). They are excluded by
  both the registered reader and the R.17 reads, leaving 21 problems and 69
  traces.

**Registered reads.**
- **Q.6 ω(Random) − ω(StreamingLLM) = +0.225 [+0.178, +0.273] → PREDICTED.**
  This is the replication registered in R.17.1.5 of the b256 result
  (+0.200 [+0.159, +0.234]). Its scope is exploratory: one model, one
  benchmark (R.6).
- **Q.3 collateral:** 0 in every condition. At 2048 this is **structurally
  non-identifying** (R.16.3). It is reported, not read.
- **H3 (confirmatory): HOLDS.** M(H2O-style) − M(Random) = **−0.119
  [−0.139, −0.097]**. H2O-style loses the least attended mass (0.0496
  [0.0302, 0.0713]; Random 0.1690 [0.1309, 0.2072]), yet its free-run damage
  is not the smallest.
- **H2 (confirmatory): NEITHER.**
  - spread(Δ) = 0.0273, above its median CI half-width 0.0234, though narrowly;
  - spread(M) = 0.119 against 0.024.
  - Δ varies across policies by about as much as its own uncertainty. The
    registered "mirror" (small Δ spread, large M spread) is not met, and at
    b256 it failed clearly.

**Descriptive (mean [95% CI], 21 problems).**

| Policy | M | Δ | Within-group per-head M dispersion (max − min over 7 heads) |
|---|---|---|---|
| Random | 0.1690 [0.1309, 0.2072] | 0.0877 [0.0658, 0.1109] | 0.1446 [0.1140, 0.1758] |
| StreamingLLM | 0.0922 [0.0684, 0.1186] | 0.0868 [0.0592, 0.1182] | 0.1327 [0.1045, 0.1626] |
| H2O-style | 0.0496 [0.0302, 0.0713] | 0.0717 [0.0488, 0.0975] | 0.0748 [0.0511, 0.1006] |
| R-KV | 0.0640 [0.0421, 0.0886] | 0.0604 [0.0436, 0.0787] | 0.0895 [0.0649, 0.1166] |

**k-sensitivity of ω (R.17.1.7).**
- Nested k runs from the k = 256 refit. It passed acceptance: max principal
  angle 1.46e-4 rad, mean and scale identical.
- ω(Random) − ω(StreamingLLM) at each k:

  | k | 8 | 16 | 32 | 64 | 128 | 256 |
  |---|---|---|---|---|---|---|
  | contrast | +1.079 [+0.863, +1.288] | +0.709 [+0.576, +0.839] | +0.409 [+0.328, +0.490] | +0.225 [+0.178, +0.273] | +0.125 [+0.098, +0.151] | +0.067 [+0.050, +0.082] |

- The ordering holds at every k. The magnitude shrinks as k grows, because ω
  itself does (Random 12.4 at k = 8 down to 2.24 at k = 256).

**Which component carries ω at 2048 (addendum D).** Unlike b256, no single
log component differs:
- log(‖δ⊥‖ / ‖δ∥‖) = +0.065 [+0.042, +0.090];
- log ‖δ⊥‖ = −0.003 [−0.133, +0.127];
- log ‖δ∥‖ = −0.067 [−0.187, +0.062].

## 3d. Task 2 — restoration redesigned, restore-v2 (R.17.2.3; FINAL)

Read 2026-09-29 03:16 Dubai. Artifact: `results/diag/meeting3/r17/restore_v2.json`
(`python scripts/report_r17.py --runs-root <runs> restore`). 200/200 rows
(25 problems × 4 seeds × Random/StreamingLLM), budget 256, trigger = the first
prune at or after decode step 1024 (addendum C).

**Eligibility, stated before any effect.**

| Set | Problems / traces | Full-cache accuracy | Full-cache truncation | Median trace tokens |
|---|---|---|---|---|
| Eligible (READ) | 24 / 92 | 0.844 [0.708, 0.958] | 0.042 [0.000, 0.115] | 2,933 |
| Ineligible ("no eviction at or after the trigger") | 3 / 8 | 1.000 | 0.000 | 672.5 |

- Identical for both policies. Two problems have both eligible and
  ineligible seeds, so the problem counts sum to 27.
- One READ trace (seed 47, 1,075 trace tokens) is too short for the 200-forward
  window after the trigger. It is out of the full-restoration and lag-1 reads
  (91 traces), per addendum C. Longer lags lose more (89 / 88 / 85 traces at
  16 / 64 / 256), as R.10.
- Compared with exact-lag (8 problems, full-cache accuracy 0.594), this read
  covers nearly the whole subset.

**Full self-restoration: MIXED for both policies.** R_fs = full-restoration
window-mean TV ÷ no-intervention window-mean TV over forwards [e + 1, e + 201),
with equal-problem CI (91 traces, 24 problems):

| Policy | R_fs [95% CI] | Branch |
|---|---|---|
| Random | 0.338 [0.320, 0.358] | **MIXED** |
| StreamingLLM | 0.297 [0.279, 0.317] | **MIXED** |

- Returning *every* evicted entry and stopping eviction removes about two
  thirds of the divergence over the next 200 forwards. The remaining third is
  state drift: the returned K/V are the compressed branch's own and M is 0
  (|mean M| < 1e-8 at every measured point).
- Both CIs sit well inside (0.10, 0.50). Neither DRIFT DOMINATES nor MASS
  DOMINATES.
- The remaining divergence is about a third of the no-intervention level
  (lag-1 baseline window-mean TV 0.1112 Random, 0.0981 StreamingLLM). That
  puts it far above the precision floor's mean per-step TV (0.0062, §4a).

**Channels (descriptive; mean [95% CI] at the first forward after the trigger,
e + 1, and at e + 193).**

| Policy | Arm | M at e+1 | Δ at e+1 | Δ at e+193 |
|---|---|---|---|---|
| Random | none | 0.5692 [0.5580, 0.5804] | 0.2407 [0.2293, 0.2534] | 0.2488 |
| Random | full restore | 0 | 0.1521 [0.1408, 0.1641] | 0.0870 [0.0794, 0.0956] |
| StreamingLLM | none | 0.2359 [0.2229, 0.2492] | 0.1223 [0.1111, 0.1342] | 0.1374 |
| StreamingLLM | full restore | 0 | 0.1256 [0.1137, 0.1389] | 0.0872 [0.0785, 0.0976] |

- After full restoration, Δ (shape distortion from the drifted queries) is
  not zero. It falls over the window, to about 0.087 for both policies by
  e + 193, while the no-intervention Δ stays flat or rises.
- For StreamingLLM, restoring everything does not change Δ at e + 1; it
  only diverges from the no-intervention arm later.
- This is the measurement the Q.3 cross-instrument prediction (Δ vs M at the
  reinsertion trigger) was waiting on (§3a). It is descriptive and licenses no
  causal query-drift claim.

**Cohort arms (as R.10, at the new trigger).** Effect = none − arm window-mean
TV (positive = divergence reduced), trace-level bootstrap.

| Lag | Traces | Random matched | Random random-reinsert | StreamingLLM matched | StreamingLLM random-reinsert |
|---|---|---|---|---|---|
| 1 | 91 | +0.0012 [−0.0005, +0.0028] | +0.0095 [+0.0071, +0.0118] | +0.0011 [−0.0019, +0.0042] | +0.0105 [+0.0071, +0.0141] |
| 4 | 91 | +0.0001 [−0.0015, +0.0016] | +0.0086 [+0.0062, +0.0109] | +0.0010 [−0.0022, +0.0042] | +0.0104 [+0.0065, +0.0142] |
| 16 | 89 | +0.0002 [−0.0015, +0.0021] | +0.0096 [+0.0073, +0.0121] | +0.0017 [−0.0015, +0.0049] | +0.0109 [+0.0075, +0.0146] |
| 64 | 88 | −0.0009 [−0.0030, +0.0013] | +0.0109 [+0.0082, +0.0137] | +0.0018 [−0.0017, +0.0055] | +0.0116 [+0.0076, +0.0157] |
| 256 | 85 | +0.0021 [+0.0001, +0.0042] | +0.0099 [+0.0069, +0.0129] | −0.0009 [−0.0042, +0.0026] | +0.0084 [+0.0050, +0.0122] |

- **Random: NON-MONOTONE.** The matched arm excludes 0 only at lag 256, and
  random-reinsert does more there, so it is NON-SPECIFIC. Every other lag is
  NO RESTORATION.
- **StreamingLLM: FLAT-NULL.** No lag excludes 0.
- These are the same labels as exact-lag (§3b), now on 24 problems instead of 8.
- Returning the 64 *most recently* evicted entries does nothing measurable.
  Returning 64 *randomly drawn earlier* evictions reduces divergence at every
  lag for both policies, repeating the §3b lead at full coverage.
- Every matched effect is below the precision floor's mean per-step TV
  (0.0062). The random-reinsert effects exceed it by less than 2×.

**Answer to Task 2 (restoration) at this scope.**
- The trajectory is largely restorable: putting back everything that was
  evicted removes about two thirds of the divergence.
- The restorable part is not in the most recently evicted cohort.
- What cannot be restored, about a third, is carried in the drifted state.

One model, one benchmark, budget 256; exploratory at this scope.

## 3e. Task 2 — restoration by origin, restore-v3 (Amendment S.2; FINAL)

Read 2026-10-02 16:40 Dubai by the repaired reader (`scripts/report_s.py
origin`, validation boundary of 2026-10-02). Artifacts:
`results/diag/amendment-s/origin.json`, `origin.conditions.tsv` (sha256 in
the JSON). **Status FINAL:** 200/200 rows against the frozen design, 0
malformed; provenance-v1 identity verified on every restore-v3 and restore-v2
row; every arm gate passed; the determinism control against restore-v2 holds
exactly (max |ΔTV| = 0.0 on all 91 + 91 compared conditions, eligible sets
identical).

**Eligibility** (identical for both policies): 91 READ traces from 24
problems (full-cache accuracy 0.844 [0.708, 0.958]); 8 traces never reach the
trigger and 1 is too short for the window. The Random `prompt` arm needed
padding (every head returns all of its evicted prompt entries and is padded
to the largest count with its most recently evicted trace entries): median
63 prompt entries returned per head against a median prompt of 64 tokens,
and up to 25 pad entries in the heads that had kept more of the prompt.
StreamingLLM needed no padding (every head evicts the same positions).

**Registered read.** ρ_arm = (TV_frozen − TV_arm) / (TV_frozen − TV_full),
window-mean TV over the 200 forwards after the trigger, eviction off in every
arm, equal-problem means, problem-cluster bootstrap CI (91 traces, 24
problems).

| Policy | TV frozen | TV full | TV prompt | TV trace | ρ_prompt | ρ_trace | ρ_prompt − ρ_trace | Label | Specificity |
|---|---|---|---|---|---|---|---|---|---|
| Random | 0.1005 [0.0891, 0.1132] | 0.0377 [0.0336, 0.0419] | 0.0775 [0.0694, 0.0856] | 0.0867 [0.0772, 0.0970] | **0.367 [0.310, 0.419]** | 0.220 [0.162, 0.278] | **+0.147 [+0.083, +0.200]** | **PARTIAL** | **PROMPT-SPECIFIC** |
| StreamingLLM | 0.0867 [0.0752, 0.0995] | 0.0292 [0.0253, 0.0334] | 0.0728 [0.0642, 0.0813] | 0.0711 [0.0616, 0.0803] | 0.242 [0.184, 0.295] | 0.272 [0.200, 0.340] | −0.030 [−0.090, +0.025] | **INDETERMINATE** | **NON-SPECIFIC** |

- The restorable gap TV_frozen − TV_full is 0.0629 [0.0545, 0.0722] for Random
  and 0.0575 [0.0492, 0.0665] for StreamingLLM: both CIs lie above 0, so the
  ratios are defined. Every TV here is an order of magnitude above the
  precision floor's mean per-step TV (0.0062, §4a).
- **Random:** returning the prompt entries (about 63 per head, plus pad)
  recovers 37% of what returning all ~830 stashed entries recovers; an
  equal-sized random draw of trace entries recovers 22%. Prompt entries carry
  more per entry than trace entries (the difference's CI excludes 0), and
  they do not carry most of the restorable divergence (PARTIAL, not PROMPT
  CARRIES IT). The pad is material (up to 25 of ~88 returned entries in some
  heads), so "prompt-specific" means "prompt entries plus a minority of the
  most recently evicted trace entries, against an equal-count random trace
  draw".
- **StreamingLLM:** the prompt entries recover no more than an equal number of
  trace entries (the difference's CI includes 0). The ρ_prompt CI straddles
  the 0.20 threshold, so the registered label is INDETERMINATE.
- **What this does and does not say.** This is divergence under teacher
  forcing, not accuracy. Full restoration here, unlike restore-v2's R_fs,
  is compared against a `frozen` arm in which eviction is also stopped, so
  "stopping eviction" is held constant; the remaining difference between the
  arms is what the returned entries contribute. The prompt-information
  account of the Random-vs-StreamingLLM accuracy gap is supported for Random
  only in part, and not for StreamingLLM, at this scope.

## 3f. Tasks 2/3 on DeepSeek-R1-Distill-Llama-8B (R.17.3 / addendum E; FINAL)

Read 2026-10-02 18:30 Dubai. Artifacts: `results/diag/meeting3/capture_llama8b_b256.json`
(+ `.conditions.tsv`), `r17/mdelta_capture-llama8b-b256.json`,
`r17/omega_layers_capture-llama8b-b256.json`. Both readers FINAL: 400/400
against the frozen design, 0 provenance failures against the Llama manifest
and the verified Llama basis, controls VALID (layer-0 Δ, no non-finite
values). Budget 256 is a **6.08%** realized cache (mean prompt 91.5 + trace
4,115.9). Llama-8B has 32 query heads over 8 KV heads (4 to 1). No nested
basis was fitted for Llama, so there is no k-sensitivity here.

**Registered replications (confirmatory for Llama).**

| Read | Result | Label |
|---|---|---|
| ω(Random) − ω(StreamingLLM) | **−0.355 [−0.453, −0.275]** (100 pairs, 25 problems) | **REVERSED** |
| H3: M(H2O-style) − M(Random) | −0.577 [−0.602, −0.543] | HOLDS |
| H2 | spread(M) 0.577 vs half-width 0.015; spread(Δ) 0.361 vs 0.011 | NEITHER |
| Q.3 collateral R-KV − H2O-style | 0 in every condition | INCONCLUSIVE (reported, not read) |

- **The ω contrast reverses on Llama.** As registered (R.6): reported as
  such, with no post-hoc reinterpretation of which direction the geometry
  predicted. The Qwen result (+0.200 at 256, +0.225 at 2048) does not
  generalise across architectures; the exploratory scope statement was
  correct and the claim stays exploratory, one model.
- **H3 holds on Llama too**, by a much larger margin: the attention-mass
  objective is not the damage objective on either architecture.

**Descriptive (mean [95% CI], 25 problems).**

| Policy | M | Δ | ω |
|---|---|---|---|
| Random | 0.680 [0.633, 0.717] | 0.441 [0.417, 0.459] | 3.376 [3.300, 3.446] |
| StreamingLLM | 0.128 [0.113, 0.140] | 0.091 [0.081, 0.099] | 3.731 [3.665, 3.799] |
| H2O-style | 0.104 [0.088, 0.117] | 0.081 [0.069, 0.092] | 3.823 [3.756, 3.895] |
| R-KV (port not certified on Llama) | 0.123 [0.106, 0.138] | 0.080 [0.069, 0.091] | 3.826 [3.762, 3.896] |

- On Llama, Random evicts 5–7× the attended mass of the structured policies
  and reshapes attention 5× more; on Qwen the ratios were about 2× and 1.5×.
  This is the same model on which Random's free run collapses into filler
  loops (PAPER_FINDINGS §4o). Whether the two facts are connected is not
  tested here.
- ω is lowest for Random on Llama, highest on Qwen. The latent-geometry
  ordering is therefore architecture-dependent, which is what REVERSED
  records.
- Collateral: 0 in all 400 conditions, with 2.96M evicted instances wanted
  by some head under Random (of 100.3M checked). With 4 query heads per KV
  head the group-mean bound is tighter (1/400), so this is the structurally
  expected zero (MEETING4_ANSWERS Q3).

## 4. Controls

| Control | Status |
|---|---|
| T1 paired totals equal the m4 summaries | **PASSED** (4 dp, all 8 cells) |
| Q.2 pre-registered read reproduces the committed artifact | **PASSED** (byte-identical) |
| Q.2 no divergence before nominal onset | **PASSED** (0 of 1,200 profiles; 276 never-diverged pairs all < 2,100 steps, below the first read window) |
| D_acc gate: gate None ≡ legacy; gated ≡ full cache before onset (logit level) | **PASSED** on tiny Qwen2 (mutation-tested) and on the real model: `dacc-gate-smoke`, 2026-09-25, token-identical at t0 = 0, onset at 576 exactly at t0 = 512 |
| D_acc determinism control (gated t0 = 0 vs legacy grid rows) | **PASSED**: 200/200 token-identical at each budget (`results/diag/meeting3/dacc.json`) |
| Capture: topk gather equality, layer-0 key identity at mapped positions, bitwise pre-eviction, full/full = 0, layer-0 Δ ≤ 1e-5 | **PASSED** on tiny Qwen2 and on the real model (`capture-smoke`, 2026-09-25). VALID on all 400 capture-b256 rows; layer-0 Δ maximum exactly 0.0 |
| Collateral positive control (7 heads, 320 candidates, B = 256) | **PASSED**: production tensor path and reference both return 1.0 (`tests/test_capture.py`, pinned stack, GPU-host CPU, 2026-09-27) |
| Q.1 removal-anchored negative control | **INVALID for identification** (R.2); computed for the record from the fork control's arm A |
| Q.1 direct fork control (validity gate, R.17.0.5) | **FINAL, 150/150** (`results/diag/meeting3/fork_control.json`). TOLERANCE-DEGENERATE in all six cells (INCONCLUSIVE, R.11): the replay control is exactly 0. Selection agreement 1.0. Gap distribution (`r17/fork_precision.json`): StreamingLLM `max_tv` median and p95 are 0.0 at I=16/64; Random `max_tv` median 0.048 (I=64) and 0.062 (I=256), tail in the last trace-position quartile. **Precision-floor gate: **FAIL** (Random I=256, ratio 2.006 against 2.0); fp32 diagnosis **IMPLEMENTATION** by threshold (§4a) |
| capture-v2 smoke (R.17.2.1) | **PASSED** 2026-09-27: full/full 0, layer-0 Δ 0.0, 0 non-finite, 21 events per policy. Nested-basis acceptance passed (max principal angle 1.46e-4 rad ≤ 1e-3; mean and scale identical), so nested k up to 256 is licensed. The smoke's H2O-style M equals the v1 b256 row for the same trace |
| Q.5 identity repair (true evicted K/V per layer) | **PASSED** on CPU and on every real exact-lag row: K and V bitwise gates in all 200 rows, 0 content failures |

## 4a. The validity gate (R.17.0.5): FAIL by a hair, and the precision floor

Read 2026-09-29 02:08 Dubai. Artifact: `results/diag/meeting3/r17/fork_precision.json`.
Raw output: `runs/fork-precision/fork_precision.json` on the GPU host.

**Precision floor.**
- Measured on 10 frozen-subset problems at seed 23: bf16 vs fp32, full-cache
  teacher forcing, 4,096 sampled steps.
- Per-step next-token TV: mean 0.0062, median 0.0004, p95 0.0273, p99 0.0416,
  max 0.0723.
- 95th percentile of the window maximum: 0.0501 (16-step windows), 0.0561
  (64), 0.0578 (256).

**Gate** (PASS iff fork p95 `max_tv` ≤ 2 × floor, for all six arms):

| Arm | Fork p95 max_tv | Floor p95 | Ratio | Passed |
|---|---|---|---|---|
| Random I=16 | 0.0309 | 0.0501 | 0.617 | yes |
| Random I=64 | 0.0623 | 0.0561 | 1.110 | yes |
| **Random I=256** | **0.1159** | **0.0578** | **2.006** | **no** |
| StreamingLLM I=16 | 0.0000 | 0.0501 | 0.000 | yes |
| StreamingLLM I=64 | 0.0000 | 0.0561 | 0.000 | yes |
| StreamingLLM I=256 | 0.0604 | 0.0578 | 1.045 | yes |

- **Verdict: FAIL.** One arm (Random, I=256) exceeds the fixed multiple, by
  0.3%.
- Per R.17.0.5(e): every Meeting #3 divergence number carries this floor, and
  eviction locking keeps the R.2 wording ("detected, not established").

**fp32 diagnosis (one trace, first removal at step 288, 33 compared steps).**

| | Max TV (A vs B) | Max \|logit\| | Median vocab rank of the max-difference token (arm B) | Top-1 agrees | Min top-10 overlap |
|---|---|---|---|---|---|
| bf16 | 0.0530 | 0.344 | 115 | all | 9 |
| fp32 | 1.26e-6 | 1.6e-5 | 10,051 | all | 10 |

- **Registered label: IMPLEMENTATION.** The fp32 max TV of 1.26e-6 exceeds the
  pre-set 1e-6 threshold; the |logit| criterion (≤ 1e-3) is met.
- **The implementation checks the branch requires**, recorded from the code
  (`kvcache.fork_control`):
  - *Positions:* both arms receive identical `cache_position`.
  - *RoPE:* compaction gathers already-rotated cached keys, with no
    re-rotation; new tokens are rotated at the same absolute position in both
    arms.
  - *Mask:* the same builder puts −inf on the masked columns for every query
    head of the group; arm A has those columns removed instead.
  - *Kernel path:* both arms use the same model and the same default attention
    implementation.
  - **The only difference found is the key/value length, and so the
    reduction length.**
- The gap shrinks by about four orders of magnitude in fp32, and in fp32 the
  largest remaining difference sits deep in the vocabulary tail (rank ~10k).
  Both are consistent with reduction-order rounding. The label stays as
  registered.

**What the floor means for the Meeting #3 divergence numbers.**
- The floor is bf16 against fp32, so it bounds precision noise from above for
  bf16-against-bf16 comparisons.
- Its mean per-step TV (0.0062) is:
  - about 1/17 of budget-256 divergence (AUC-TV 0.1075 Random);
  - about half of budget-2048 divergence (0.0128 Random, 0.0186 StreamingLLM);
  - above the exact-lag restoration effects (~0.003).
- So the budget-2048 TV magnitudes and the restoration differences are **not
  established above precision at this floor**. The budget-256 divergence is.

## 5. Held units and resolved decisions

- **capture-b2048: held (`blocked:`, R.17)** before its first row, until the
  v2 schema (per-query-head M/Δ, nested-k norms) is implemented, tested and
  smoke-gated (R.17.2.1).
- **t1-tv512 / t1-tv1024: queued, unchanged.**
  - Their R.12 summary profiles keep 256-step window means aligned at
    multiples of 256. The collapse test's windows at 2304/3328/4352 are
    windows 9/13/17, on the same panel, seeds and interval as the Q.2 onset-0
    arm.
  - So no re-specification is needed (R.17.1.1). Each window is used only if
    n_steps ≥ position + 256.
- **Old Q.5 lag sweep (`restoration-lag*`): held.** Its press is invalid
  (R.5). Replaced by exact-lag (R.10), with a redesign registered (R.17.2.3).
- **Storage definition: resolved.** The guard measures `~/kvcache-storage` +
  `~/kv-caching`, with hardlinks counted once, against 72 GiB with a 1 GiB
  margin. The owner explicitly approved this on 2026-09-25 (R.16.4). Footprint
  on 2026-09-26 23:10: 69.30 GiB.

## 6. Queue (01:45–10:45 Dubai cron windows; out-of-window starts only when the owner asks, R.16.6)

| Queue job | Measures | Status | Cost |
|---|---|---|---|
| `dacc-gate-smoke`, `q6-basis-fit`, `capture-smoke` | gates | **PASSED** | — |
| `capture-b256` | Q.3 + Q.6 primary, 400 conditions | **FINAL** (§3a) | — |
| `exact-lag-r5` | Q.5, 200 (trace, policy) | **FINAL** (§3b) | — |
| `dacc-t2048`, `dacc-t512`, `dacc-t0` | D_acc, 1,200 generations | **FINAL** (§2a) | ~7.2 h each on 2 GPUs (measured) |
| `fork-control-random`, `fork-control-slm` | Q.1 direct control, 75 conditions each | **RUNNING** | ~11 GPU-h each (est.) |
| `t1-tv512`, `t1-tv1024` | TV at 512/1024 (summary profiles) | queued | ~19 h on 2 GPUs (est.) |
| `capture-b2048` | Q.3 + Q.6 secondary; H2/H3/ω confirmatory | **held** (R.17.2.1) | ~12 h (est.) |

The scheduled readers, exactly as `scripts/postprocess_meeting3.py` runs them
(each gets `--runs-root`; every reader except the budget curve also gets `--final`):

```bash
ssh kvcache-gpu 'cd ~/kv-caching && R=~/kvcache-storage/runs && \
  .venv/bin/python scripts/report_capture.py --runs-root $R --run capture-b256 \
    --config configs/experiments/capture_math500_b256_r9.yaml \
    --out results/diag/meeting3/capture_b256.json --final; \
  .venv/bin/python scripts/report_capture.py --runs-root $R --run capture-b2048 \
    --config configs/experiments/capture_math500_b2048_r9.yaml \
    --out results/diag/meeting3/capture_b2048.json --final; \
  .venv/bin/python scripts/report_dacc.py --runs-root $R \
    --arms 0=dacc-t0 512=dacc-t512 2048=dacc-t2048 \
    --legacy 256=grid-diss256-tok 2048=grid-diss2048-tok \
    --design configs/experiments/grid_math500_dacc_t0_r3.yaml \
    --out results/diag/meeting3/dacc.json --final; \
  .venv/bin/python scripts/report_exact_lag.py --runs-root $R --run exact-lag-r5 \
    --design configs/experiments/exact_lag_math500_r5.yaml \
    --out results/diag/meeting3/exact_lag.json --final; \
  .venv/bin/python scripts/report_fork_control.py --runs-root $R \
    --run fork-control-random --run fork-control-slm \
    --design configs/experiments/fork_control_random_r11.yaml \
    --design configs/experiments/fork_control_slm_r11.yaml \
    --out results/diag/meeting3/fork_control.json --final; \
  .venv/bin/python scripts/report_budget_curve.py --runs-root $R \
    --tv-runs 512=t1-tv512 1024=t1-tv1024 \
    --out results/diag/meeting3/t1_budget_curve.json'
```

Each `--final` read exits 3 while its cell is incomplete, and that is expected.
The R.17 reads run with `python scripts/report_r17.py --runs-root $R <read>`.

Registered reads:
- D_acc per R.3/R.8, plus R.17.1.2.
- Q.3: collateral R-KV − H2O-style (R.4); H2/H3 per R.17.1.5.
- Q.6: ω(Random) − ω(StreamingLLM), PREDICTED / NULL / REVERSED, exploratory
  (R.9).
- Q.5 per R.10, and the redesign per R.17.2.3.
- Q.1: fork-control TOLERANCE-DEGENERATE under R.11, and PASS/FAIL under the
  R.17.0.5 floor.
- Collapse test: R.17.1.1.
- Bridge: R.17.1.8.

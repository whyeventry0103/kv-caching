# Meeting #3 — results and execution status

Last updated 2026-09-26 (Asia/Dubai 00:45). Machine-readable companions:
`configs/meeting3/cells.yaml` (ledger) → `results/diag/meeting3/meeting3_manifest.json`
(`python scripts/meeting3_manifest.py --check`). Protocol: PREREGISTRATION
Amendment R (R.0–R.12). Every number below was recomputed from the named
artifact when this file was written. Review evidence (gate outputs, remote test run,
parity, queue dry-run, remote-only data list):
[`meeting3_evidence/`](meeting3_evidence/README.md).

**Execution status, 2026-09-26 00:45 Dubai (GPU night 2 in progress; early start at the owner's request, R.16.6).**

- **Tasks 2/3 primary cell capture-b256: FINAL** (400/400, controls VALID,
  0 provenance failures). Q.3 collateral R-KV − H2O-style = +0.000
  [+0.000, +0.000] → **INCONCLUSIVE**. Q.6 ω(Random) − ω(StreamingLLM) =
  **+0.200 [+0.159, +0.234] → PREDICTED (exploratory)**. See §3a.
- **Running:** exact-lag-r5 (seed 89, GPU0; 150/200 before it) and
  dacc-t2048 (seed 23, GPU1; 200/400 before it). Not started: dacc-t512,
  dacc-t0, fork-control-random/-slm, capture-b2048, t1-tv512, t1-tv1024.
  Each postprocess stage reads its cell when complete.

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
| **1. Divergence/damage vs cache size, exposure/horizon** | Accuracy curve 256→2048 with paired CIs, flips, realized ratios (**COMPLETE**); Q.2 exposure grid + occupancy audit (**COMPLETE**, verdict MIXED) | D_acc free-running (budget, onset) grid, 1,200 generations; Q.1 direct fork control, 150 conditions; TV at 512/1024 (summary profiles, R.12) | — |
| **2. Attention maps: M/Δ, GQA collateral, restoration** | Q.3 capture at budget 256 (**FINAL**, 400 conditions; collateral contrast INCONCLUSIVE, M/Δ per policy with CIs, §3a) | Q.3 capture at 2048, 400 conditions (collateral structurally non-identifying there, R.16.3); Q.5 exact-lag restoration, 200 (trace, policy) × 5 lags × 4 arms | — |
| **3. SVD/PCA trajectory direction** | Q.6 basis (frozen, k = 64) + projection at budget 256 (**FINAL**, ω contrast PREDICTED, exploratory, §3a) | Q.6 at budget 2048 (same capture) | — |

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

**D_acc (free-running accuracy on the same grid): QUEUED.** Details in §6.

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

Reproduce:

```bash
python scripts/report_capture.py --runs-root <runs> --run capture-b256 \
    --config configs/experiments/capture_math500_b256_r9.yaml --out <out> --final
```

## 4. Controls

| Control | Status |
|---|---|
| T1 paired totals equal the m4 summaries | **PASSED** (4 dp, all 8 cells) |
| Q.2 pre-registered read reproduces the committed artifact | **PASSED** (byte-identical) |
| Q.2 no divergence before nominal onset | **PASSED** (0 of 1,200 profiles; 276 never-diverged pairs all < 2,100 steps, below the first read window) |
| D_acc gate: gate None ≡ legacy; gated ≡ full cache before onset (logit level) | **PASSED** on tiny Qwen2 with the pinned stack (mutation-tested); real-model `dacc-gate-smoke` **PASSED** 2026-09-25 (token-identical at t0 = 0; onset at 576 exactly at t0 = 512) |
| D_acc determinism control (gated t0 = 0 vs legacy grid rows) | QUEUED: needs `dacc-t0` (not yet started); reader built |
| Capture: topk gather equality, layer-0 key identity at mapped positions, bitwise pre-eviction, full/full = 0, layer-0 Δ ≤ 1e-5 | **PASSED** on tiny Qwen2 for all four policies (wrong position map caught by mutation); real-model `capture-smoke` **PASSED** 2026-09-25 (full/full 0, layer-0 Δ 0.0); the same controls are VALID on all 400 capture-b256 rows |
| Q.1 removal-anchored negative control | **INVALID for identification** (R.2); computed for the record from the fork control's arm A (QUEUED) |
| Q.1 direct fork control | tiny Qwen2: B-vs-B exactly 0, A−B < 1e-4, next selections identical, fires when masking is off — **PASSED on CPU**; real run QUEUED |
| Q.5 identity repair (true evicted K/V per layer) | tiny Qwen2: layer-0 stash equals full keys bitwise; the old tail-stash bug is refused (mutation test) — **PASSED on CPU**; real run QUEUED |

## 5. Blocked units and open owner decisions

- **T1.tv512 / T1.tv1024 (TV at the intermediate budgets): unblocked by R.12.**
  Full per-step profiles would need ~196 MiB, against 146 MiB usable (68.857
  GiB used of 70, 1 GiB margin), of which ~117 MiB is committed to queued cells.
  The registered read needs only per-trace mean TV, so the cells run as
  registered but store ~3 KB summary profiles (~1.2 MB total). Queued last as
  `t1-tv512` / `t1-tv1024`, ~19 h on 2 GPUs (est.).
- **Old Q.5 lag sweep (`restoration-lag*`): held as `blocked:`** because its
  press is invalid (R.5). The exact-lag replacement (R.10) is queued as
  `exact-lag-r5`.
- **Storage definition.** R.8.5: the guard measures `~/kvcache-storage`
  only. Counting `~/kv-caching` (5.85 GiB alone, but mostly hardlinks into the uv
  cache, so +0.33 GiB when hardlinks are counted once; corrected 2026-09-25,
  see `meeting3_evidence/storage.md`) would leave 0.81 GiB
  of headroom, below the 1 GiB margin, and block everything. The owner should
  confirm which trees the 70 GiB covers.

## 6. What runs unattended, in order (01:45–10:45 Dubai windows)

| Queue job | Measures | Cost (measured or estimated) | Gate |
|---|---|---|---|
| `dacc-gate-smoke` | D_acc parity + onset on the 7B model | ~10 min | writes `dacc-gate-smoke-PASS.json` |
| `q6-basis-fit` | frozen full-only basis, 28 layers, k = 64 | ~15 min (est.) | header `q6_basis.json` (npz sha256) |
| `capture-smoke` | full/full + 4 policies on 1 trace | ~15–30 min | writes `capture-smoke-PASS.json` |
| `capture-b256` | **Q.3 + Q.6 primary**, 400 conditions | ~12 h on 2 GPUs (est.) | requires capture-smoke |
| `exact-lag-r5` | Q.5 exact-lag restoration, 200 (trace, policy) | ~5 h on 2 GPUs (est.) | requires capture-smoke |
| `dacc-t2048` → `dacc-t512` → `dacc-t0` | D_acc, 400 generations each | ~7.2 h each on 2 GPUs (measured) | require dacc-gate-smoke |
| `fork-control-random`, `fork-control-slm` | Q.1 direct control, 75 conditions each | ~11 GPU-h each, one GPU each (est.) | none (live checks) |
| `capture-b2048` | Q.3 + Q.6 secondary | ~12 h (est.) | requires capture-smoke |
| `t1-tv512`, `t1-tv1024` | T1 AUC-TV at 512/1024 (R.12 summary profiles) | ~19 h on 2 GPUs (est.) | none |

Readers (CPU, run after each cell completes; each decides readiness itself):

```bash
ssh kvcache-gpu 'cd ~/kv-caching && R=~/kvcache-storage/runs && \
  .venv/bin/python scripts/report_dacc.py --runs-root $R \
    --arms 0=dacc-t0 512=dacc-t512 2048=dacc-t2048 \
    --legacy 256=grid-diss256-tok 2048=grid-diss2048-tok \
    --out results/diag/meeting3/dacc.json && \
  .venv/bin/python scripts/report_capture.py --runs-root $R --run capture-b256 \
    --out results/diag/meeting3/capture_b256.json && \
  .venv/bin/python scripts/report_exact_lag.py --runs-root $R --run exact-lag-r5 \
    --out results/diag/meeting3/exact_lag.json && \
  .venv/bin/python scripts/report_fork_control.py --runs-root $R \
    --run fork-control-random --run fork-control-slm \
    --out results/diag/meeting3/fork_control.json'
```

Registered reads: D_acc per R.3/R.8; Q.3 contrast collateral(R-KV) −
collateral(ObservedAttentionPress), with a negative CI excluding 0 supporting
the Q.3 ordering; Q.6 ω(Random) − ω(StreamingLLM), labelled PREDICTED / NULL /
REVERSED, exploratory (R.9); Q.5 per-lag P.5 effects and the DECAY /
FLAT-NULL / NON-MONOTONE sweep verdict (R.5, R.10); Q.1 fork control PASS /
FAIL, or EXACT / TOLERANCE-DEGENERATE under R.11.

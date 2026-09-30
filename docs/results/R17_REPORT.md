# R.17 / Amendment S report (P.4) — 2026-09-30

Target venue: proposed ICML 2027, pending supervisor confirmation. Every number
below is copied from the named artifact; CIs are problem-clustered 95%
bootstrap intervals (2,000 resamples, seed 0) unless stated. Full tables:
`docs/results/MEETING3_RESULTS.md`; protocol: `docs/protocol/PREREGISTRATION.md`.

## First: what changes a paper claim

| Item | Outcome | Consequence for the paper |
|---|---|---|
| **0.5 validity gate** (`results/diag/meeting3/r17/fork_precision.json`) | **FAIL** by a hair: Random I=256 fork p95 0.1159 against 2× floor 0.0578, ratio 2.006 vs 2.0; the other five arms pass. Floor: mean per-step TV 0.0062, p95 0.0273. fp32 diagnosis IMPLEMENTATION by threshold (1.26e-6 > 1e-6; the only implementation difference is K/V length). | Every divergence number carries the floor. Budget-256 divergence is ~17× the floor mean and stands; budget-2048 divergence (0.0128 / 0.0186) and the exact-lag effects (~0.003) are **not established above numerics**. Locking stays "detected, not established" (R.2). |
| **Reinsertion (0.1)** | P.5 specificity control retracted (stash defect, R.5); exact-lag: no restoration at any lag, 8/25 problems | Paragraph rewritten `a549250`; PAPER_FINDINGS §5d |
| **Locking wording (0.2)** | per R.2 | "detected at two amplitudes and not yet validated" `a549250` |
| **D_acc restricted (1.2)** | onset effect **NOT ESTABLISHED**: Random −0.048 [−0.184, +0.075], StreamingLLM +0.026 [−0.066, +0.118] on 60/100 long pairs | No onset/exposure claim; the unrestricted +0.110 / +0.190 is attributed to never-compressed traces |
| **Amendment S (new, 2026-09-30)** | arXiv:2609.03430: the prompt is the fragile part of the cache; **none of our presses protects the prompt** (verified) | Until the prompt-protected cells are read, the residual is **open to a prompt-loss reading**. Three branches registered (S.1), including reframing around prompt loss. |

## Every numbered item

| Item | What | Status | Outcome (artifact) |
|---|---|---|---|
| 0.1 | reinsertion retraction/rewrite | DONE `a549250` | above |
| 0.2 | locking wording | DONE `a549250` | above |
| 0.3 | Q.2 ratio note | DONE `ac78eab` | corrected ratios 0.106/0.075/0.057 and 0.847/0.596/0.459 |
| 0.4 | cleanup a–d | DONE `afbc9a3` | figure default dir out of the hashed tree; R.16.6 sentence; `**` in tex; email claim corrected |
| 0.5 | fork control arms, distribution, fp32, floor | DONE `5cceb6c` | all six arms TOLERANCE-DEGENERATE (INCONCLUSIVE); gate FAIL (above) |
| 0.6 | M/Δ report | DONE | `r17/mdelta.json`, `r17/mdelta_capture-b2048.json` |
| 0.7 | storage facts | DONE | per-head values not stored in v1 (stored in v2); no projection coefficients in v1 |
| 1.1 | collapse test | DONE `bb35522` | StreamingLLM **COLLAPSE**; Random **MIXED** by 0.0002 on one pair (all pair CIs include 0); ΔR² of budget after log r 0.029 [0.004, 0.073] / 0.033 [0.003, 0.082]; 10 problems (`r17/collapse.json`) |
| 1.2 | restricted D_acc | DONE | NOT ESTABLISHED (above) (`r17/dacc_restricted.json`) |
| 1.3 | AUC-TV CIs | DONE | Random 0.1075 / 0.0561 / 0.0279 / 0.0128 at 256/512/1024/2048 (`r17/auc_tv.json`) |
| 1.4 | figures | DONE, viewed | `results/diag/meeting3/figures/r17_*.png`; heatmap TV panel still lacks the 512/1024 onset-0 cells (known gap) |
| 1.5 | H2/H3 | DONE | b256 exploratory: H2 NEITHER, H3 HOLDS −0.315 [−0.330, −0.298]; **b2048 confirmatory: H3 HOLDS −0.119 [−0.139, −0.097], H2 NEITHER** |
| 1.6 | collateral | DONE | 0 in all 800 conditions; INCONCLUSIVE at 256, structurally non-identifying at 2048 |
| 1.7 | per-layer ω, k sensitivity | DONE | ordering holds at every k 8–256 at b2048 (+1.079 … +0.067) (`r17/capture_v2_extras_capture-b2048.json`) |
| 1.8 | bridge AUROC with BH | DONE | 0/6 survive (min q 0.165); pre-commitment NOT CONFIRMED (`r17/bridge.json`) |
| 2.1 | capture-b2048 on v2 schema | DONE | ω +0.225 [+0.178, +0.273] PREDICTED (21 problems / 69 pairs; 31/100 traces per policy never compressed) |
| 2.2 | t1-tv respec | not needed | summaries carry the collapse windows |
| 2.3 | restore-v2 | DONE `4c9627c` | R_fs Random 0.338 [0.320, 0.358], StreamingLLM 0.297 [0.279, 0.317]: **MIXED** both; cohort arms NON-MONOTONE / FLAT-NULL on 24 problems (FC acc 0.844) |
| 2.4 | fork-precision | DONE `5cceb6c` | gate FAIL (above) |
| 3 | Llama-8B capture | **RELEASED 2026-09-30** (`e069540`), runs in the next window | H2/H3/ω replication registered (addendum E) |
| P.1 | mechanism draft | UPDATED with every landed read (still `\iffalse`, 67 pinned literals) | audit clean; PDF ×2, 26 pages, 0 errors, 1 pre-existing overfull |
| P.2 | ICML page budget | DONE | `docs/strategy/ICML2027_PAGE_BUDGET.md` |
| P.3 | five gates | lint/test/dryrun/audit pass locally; parity holds at each sync | see commits |
| S.1 | prompt-protected decomposition | smoke PASSED 02:07; grid-pp256 391/800; readout/reverse and budget 1024 queued | nothing read |
| S.2 | restoration by origin | 100/200 | nothing read |
| S.4 | Qwen3-4B | configs + registration done; **held on the 14B deletion** | — |

## GPU hours (sum of recorded per-row wall-clock, excludes model load and teacher-forcing setup; measured on the host 2026-09-30)

| Run | Rows | GPU-h |
|---|---|---|
| dacc-t0 / t512 / t2048 | 3 × 400 | 15.00 / 15.06 / 13.22 |
| capture-b256 / capture-b2048 | 2 × 400 | 12.08 / 12.40 |
| exact-lag-r5 | 200 | 2.62 |
| fork-control-random / -slm | 2 × 75 | 6.07 / 6.02 |
| fork-precision (from the log: 01:45–02:08) | 1 | ≈ 0.4 |
| t1-tv512 / t1-tv1024 | 2 × 200 | 6.33 / 6.33 |
| restore-v2 | 200 | 6.74 |
| **Meeting #3 + R.17 total** | | **≈ 102** |
| grid-pp256 (partial, S.1) | 391 | 15.62 |
| restore-v3 (partial, S.2) | 100 | 1.74 |

**Tier 3 cost (estimate from capture-b256):** ≈ 12 GPU-h for the capture plus
≈ 0.1 h for the basis fit; ≈ 0.06 GiB.

## Storage (host, `~/kvcache-storage` + `~/kv-caching`, hardlinks once, against 72 GiB)

| When | GiB |
|---|---|
| 2026-09-26 23:10 (before Tier 2) | 69.30 |
| 2026-09-29 03:15 (after Tier 2) | 69.449 |
| 2026-09-30 00:18 (before S) | 69.450 |
| S cells so far | + ≈ 0.02 (grid-pp256 20 MB, restore-v3 0.3 MB) |
| Pending: delete the 14B weights (−28 GB) then Qwen3-4B (+≈ 8 GB) | owner action |

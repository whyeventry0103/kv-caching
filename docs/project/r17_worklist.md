# R.17 worklist (started 2026-09-27)

Target venue: proposed ICML 2027, pending supervisor confirmation.

This is the working checklist for the 2026-09-27 instruction set (Tiers 0–3,
P.1–P.4). Every new read is registered as PREREGISTRATION Amendment R.17
before it runs or is read. Each item ends in a check; the status column says
what was observed, not what was intended.

| Item | What | Done when | Status |
|---|---|---|---|
| hold | capture-b2048 held before its first row | remote dry-run shows BLOCKED | DONE, released onto capture-v2 after host tests (`5b46a69`) |
| R.17 | registration of every read | committed before any read | DONE `ac78eab`; addenda A-D |
| 0.1 | reinsertion paragraph retracted/rewritten | PDF x2, claims pinned | DONE `a549250` |
| 0.2 | locking wording per R.2 | PDF, PAPER_FINDINGS old/new | DONE `a549250` |
| 0.3 | Q.2 ratio note | dated note at the table | DONE `ac78eab` |
| 0.4 | cleanup a-d | each checked | DONE `afbc9a3` |
| 0.5 | fork control arms / distribution / fp32 diagnosis / floor | FINAL read + fork-precision job | DONE `5cceb6c`: fork FINAL (all TOLERANCE-DEGENERATE); gate FAIL (Random I=256 ratio 2.006 vs 2.0); fp32 diagnosis IMPLEMENTATION; floor measured |
| 0.6 | M/Delta report | mdelta.json | DONE |
| 0.7 | storage facts | code + rows | DONE (t1-tv summaries cover the collapse windows) |
| 1.1 | collapse test | needs t1-tv rows | DONE `bb35522`: SLM COLLAPSE, Random MIXED |
| 1.2 | D_acc restricted | dacc_restricted.json | DONE: NOT ESTABLISHED |
| 1.3 | AUC-TV CIs | auc_tv.json | DONE at all four budgets `bb35522` |
| 1.4 | figures | PNG viewed | DONE (supervisor, heatmaps); collapse plot after 1.1 |
| 1.5 | H2/H3 b256 | mdelta.json | DONE (exploratory): NEITHER / HOLDS; b2048 confirmatory PENDING |
| 1.6 | collateral | written | DONE |
| 1.7 | per-layer omega; k status | omega_layers.json | DONE; nested k comes with b2048 |
| 1.8 | bridge | bridge.json | DONE: NOT CONFIRMED |
| 2.1 | b2048 on capture-v2 | host tests pass; chain queued | DONE (queued: q6-basis-fit-k256 -> capture2-smoke -> capture-b2048) |
| 2.2 | t1-tv | no re-spec needed | queued |
| 2.3 | restore-v2 | host tests pass; queued | DONE: FINAL 200/200, read 2026-09-29; R_fs MIXED both policies; cohort NON-MONOTONE / FLAT-NULL (MEETING3_RESULTS §3d) |
| 2.4 | fork-precision | host tests pass; queued | DONE: FINAL, read `5cceb6c` |
| 3 | Llama-8B capture | config + footprint | configs + manifest DONE; footprint ~0.06 GiB; storage after Tier 2 69.449/72 GiB (2026-09-29 03:15); three jobs still `blocked:` pending owner go-ahead to release |
| P.1 | mechanism draft | pinned, not compiled | DONE `ddadc57`; the section 4.4 discrepancy is recorded |
| P.2 | ICML page budget | doc | DONE `ddadc57` |
| P.3/P.4 | gates, report | | after the GPU cells |

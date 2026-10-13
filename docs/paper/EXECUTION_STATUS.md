# Execution status (one row per work package)

Specification: `docs/paper/EXECUTION_PLAN.md` (v1.0, baseline `1b644dc`).
Claims: `docs/paper/CLAIM_LEDGER.md`.
Approvals pending: `docs/paper/PROPOSALS_FOR_APPROVAL.md`.

Snapshot **2026-10-05, about 20:00 Dubai** (host clock 19:39 when checked). The bounded repair of the independent review of `fbf869f` is applied. Both GPUs are idle and no jobs are running; the next cron window opens at 01:45.

| WP | State | Artifacts | Remaining |
|---|---|---|---|
| WP0 freeze + single status | Documentation landed. **The paper remains a draft.** | This file; the ledger; paper edits (title, 13 bracket-wording edits, sub-floor TV qualification, abstract now states the interaction) | Rewrite the abstract's structure around the final question after B=1024 (do not just append caveats) |
| WP1 existing queue | **In progress** | Protected reverse b256 (`reverse_b256.json`, FINAL) | grid-pp1024 (474/800) then readout-pp1024 (0/800), both in the normal window. Then `report_s.py protection --budget 1024`, `consequence`, and the practical-margin read at 1024 |
| WP2 Qwen3 scoring | **Audit done (v2); human review and disposition open** | `grading_sensitivity.py` v2; `bank_audit_*` (v2), `*_v1.*`; `human_review_v2.tsv` (42 rows) | A person fills in the review sheet. Owner and supervisor choose option (a)/(b)/(c); the memo and the review both recommend (a) as a documented exception |
| WP3 no-GPU analysis | **Done, repaired** | `decision_b256`, `matched_b256`, `reverse_b256` (re-read after R1/R4: 0 numeric changes); `RELATED_WORK_MATRIX.md` (with a verbatim evidence record) | Recheck novelty gate N after B=1024 |
| WP4 new capability | **CPU-complete (2026-10-06); needs only the approved smoke (P2)** | `protected_selectors.py`; runner adapters `run_grid_protected.py` and `run_replay_protected.py` (they refuse real runs while held or without the smoke marker); provenance family `protected_selectors`; held configs, dry-run in `make dryrun` | Smoke runner written and dry-run (`protected_selectors_smoke.py`); eager attention fixed for protected H2O; the gate accepts only an identity-matched PASS record. Real-generation smoke = P2, needs approval |
| WP5 protected selectors | **Held** | Manifest, held configs | After P2 (smoke) and P3 (timing pilot) |
| WP6 confirmation | **Held** | Manifests; `sample_size.json` (endpoint-specific, includes the interaction); `cost_estimate.json` (corrected readout pricing, seeds parameterised, overheads) | Choose the primary endpoint, then N. Qwen3 at 300×4 is about 600–1,100 GPU-h, which does not fit the nightly windows |
| WP7 writing | **Drafting** | `DRAFT_METHODS_RELATED_FIGURES.md` | Figures 2 and 5 from existing B=256 data |

## Decisions (made by the owner on 2026-10-06)

1. **Qwen3 gate:** option (a). Exploratory; FAIL kept; secondary only after the human review sheet is done.
2. **P1** (B=1024 reverse, ≤20 GPU-h), **P2** (smoke, ≤2 GPU-h) and **P3** (pilot, ≤6 GPU-h): approved and queued.
3. **GSM8K:** pinned and downloaded (1,319 rows). Preparation only.
4. **Priority:** the three Qwen3 jobs are held until `grid-pp1024` and `readout-pp1024` are complete. Lift the `blocked:` lines then.

**Queue order from 2026-10-07 01:45:**
1. grid-pp1024 (seeds 23 and 89 remaining)
2. readout-pp1024 and reverse-readout-pp1024
3. protsel-smoke
4. the pilots

Simulated on the host: with 22 GB free the scheduler waits; with free GPUs it launches grid-pp1024.

## Review of fbf869f: repairs and their effect

| Item | Repair | Evidence | Effect on results |
|---|---|---|---|
| R1 reader validation | Values are checked before any `bool()` call, and readout flag/value consistency is checked. Bank rows are validated as an allowed superset, with duplicate detection. `matched` validates free-run rows against the frozen design (missing → PARTIAL, duplicate → INVALID), lists declared metric exclusions, and carries a dependency fingerprint. The same coercion was fixed in `report_s.py` (protection, origin) and is named separately | `tests/test_protocol_audit_cli.py`: 12 real-CLI cases, 10 failing before the fix; `test_s_acceptance` gains 2 cases, also failing before | Re-read on the host: **0 numeric changes** in protection_b256, origin, reverse_b256, matched_b256, decision_b256; TSVs byte-identical |
| R2 secondary grader | v2: positive unordered wording required; the multi-box check reads the whole suffix | 5 new tests fail on v1 and pass on v2 | **0 rows changed** in either bank |
| R3 planning and cost | Readouts priced at their source trace; seeds parameterised; interaction endpoint added; overheads, smoke and pilot made explicit; scaling corrected to 6× (was "3×") | `cost_estimate.json`, `sample_size.json` | WP5 now 74–142 GPU-h (was 73–94). WP6 at 100×2: 100–188 GPU-h (Qwen3) |
| R4 sign table | All four quadrants covered | 7 tests | Random − H2O label unchanged (MISMATCH: first policy higher accuracy, more loss) |
| Interpretation | Paired difference of absolute protection improvements added | `reverse_b256.json` `difference_of_improvements` | Random +0.078 [+0.007, +0.149]; StreamingLLM −0.007 [−0.081, +0.069], not detected. "Protection works mainly forward" retired |

## 2026-10-06 additions (review of 8e1e81a)

- **Eager attention.** Protected families map to their base policy for the attention backend. `h2o_style_pp_v1` now loads eager attention in both adapters (tests go through the real adapters).
- **Smoke gate.** `verify_smoke` requires `passed: true`, `status: "PASS"`, no failures, an identity equal to the one computed at launch, and coverage of the run's families, model, budgets and interval. The review's empty and failed markers are refused (tests, including a subprocess launch through the adapter). Markers are looked up under `KVCACHE_BASE_RUNS_ROOT`, where `launch.sh` points the gates; the adapters had used the per-run `REMOTE_RUNS_ROOT`, which was wrong.
- **Smoke runner.** `scripts/protected_selectors_smoke.py`, config `protsel_smoke_wp4.yaml` (held). Includes a live identity control against kvpress top-k. Launcher command `protselsmoke`.
- **B=1024 reverse reader.** `report_protocol_audit.py reverse --budget 1024` gives the protected 2×2 alone and withholds the protected-vs-unprotected comparison (no base arm). CLI tests cover FINAL and PARTIAL.
- **Gates.** Full local gate: 725 passed, 79 skipped. Host CPU torch tests: 52 passed. Re-read of all five stored reads: FINAL, 0 numeric changes.
- **Queue contention, 2026-10-06 01:45.** A co-tenant held about 26 GB on each 48 GB GPU, leaving 22.2 and 22.7 GB free. `grid-pp1024` needs 24 GB free (a 7B protected job measured about 15–17 GB, so the threshold carries about 7 GB of margin), so the scheduler launched the Qwen3 grid (needs 20 GB) on both GPUs for the whole window. `grid-pp1024` is still at 474/800 and Qwen3 is at 281/400. Owner decision needed: see the final message of 2026-10-06.

## Snapshot 2026-10-13 22:30 Dubai

- **WP1.** grid-pp1024 796/800 (the shared-disk floor blocked launches 7–12 Oct). Readouts and the pilot are next in the queue.
- **WP4.** Real-generation smoke PASS (P2). Approved GPU time used so far: about 0.3 GPU-h.
- **Report.** Progress report for the reviewer: https://claude.ai/artifact/Uy6XizAJpZ2kPvgLTZ4Ydc


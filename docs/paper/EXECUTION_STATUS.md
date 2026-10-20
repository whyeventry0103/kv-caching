# Execution status

Specifications:
- `docs/paper/EXECUTION_PLAN.md` (v1.0, baseline `1b644dc`);
- `docs/paper/MEETING5_REVIEW_AND_PLAN.md` (Meeting 5 review of `fd77d40`, adopted 2026-10-14).

Claims: `docs/paper/CLAIM_LEDGER.md`. Approvals and caps: `docs/paper/PROPOSALS_FOR_APPROVAL.md` and `configs/campaigns.yaml`.

## Current state (2026-10-20, 12:30 Dubai)

| Item | State | Evidence / next |
|---|---|---|
| Meeting 5 G0 closeout | **Complete** | Smoke bundle + validator (VALID from a fresh clone). The 21-record smoke rerun passed under P2 (2026-10-15). Cumulative caps enforced before launch. B=1024 interaction inference frozen and read (§5p). Qwen ordering explicit and released. Latent-basis wording corrected |
| WP1 protected B=1024 | **Forward and reverse FINAL** | Forward (2026-10-15): protection_b1024, consequence (A), decision_b1024 (PAPER_FINDINGS §5o). Reverse (2026-10-16): reverse_b1024 FINAL, P1 800/800 in 12.91 of 20 GPU-h; the registered interaction t test + Holm does not reject for either policy (Holm p 0.64); the trajectory contrast excludes 0 under both readouts (§5p). Paper-text edits pending |
| P2 smoke | **PASS (2026-10-15 13:19)** | 21 records, 0 failures, 0.549 of 2 GPU-h; the 2026-10-13 15-record run stays bundled as history |
| P3 timing pilot | **Complete** | Grid, readout, reverse 16/16 each, 1.39 of 6 GPU-h; rows excluded from every claim |
| Qwen3 | **Complete; read 2026-10-20 (EXPLORATORY)** | Unique expected keys: bank 200/200, grid / readout / reverse 400/400 each. StreamingLLM REPLICATES (residual −0.209); Random near floor (free run 0.030, truncation 0.735). Gate FAIL 0.845 kept; disposition (a). The 42-row human scoring review is still unassigned. PAPER_FINDINGS §5r |
| Fork-precision gate | **FAIL (scoped)** | Constrains next-token TV claims only, not accuracy (ledger §D) |
| Meeting 5 CPU preparation | **Bundle ready for review (2026-10-14)** | Protected R-KV `rkv_pp_v1`: own provenance, gate, smoke and held adapters; real-model CPU parity PASS (`results/diag/meeting5/rkv_pp_cpu_parity/`). Finite-channel instrument (`task_information.py`, `information_tasks.py`) and reader CLI. Protocol draft v0.1 (`docs/protocol/MEETING5_INFORMATION_PROTOCOL.md`). Related work rechecked from primary sources; Garcia 2026 precedents the protection × policy cross. Inventory, strict-fit cohort (97), AIME inventory, frozen GSM8K/BBH graders and manifests. M5-A reader (PARTIAL today). Not yet written: information-probe GPU runner, pid-filter adapter for CAPMATCH, protected capture adapter |
| Meeting 5 approval request | **Decided 2026-10-14** | A1–A3 approved (caps 1/28/28), queued after P1/P2/P3 with the M5 production runner (`run_grid_m5.py`). A4/A5 held. Margin ±0.03 (±0.05 sensitivity). BBH pinned (preparation only). Review of 6b409001 repaired: `docs/paper/MEETING5_REVIEW_6b40900_ACCEPTANCE.md` |
| Review of 946c76a | **Two execution issues repaired (CPU)** | Smoke gates derive the target, layers and lengths independently; the M5 runner uses atomic pair publication and recovery. `docs/paper/MEETING5_REVIEW_946c76a_ACCEPTANCE.md` |
| M5 grids | **Complete, validated** | grid-rkvpp256-m5 400/400 (19.24 of 28 GPU-h), grid-h2opp256-m5 400/400 (15.49 of 28 GPU-h; 5 launches failed at 0 h on 2026-10-15 from the gate bug fixed in 4f50d28) |
| M5-A strict read | **FINAL (2026-10-17); condition table published** | Both registered interactions reject after Holm. Protected Random − H2O-style −0.005: not shown within ±0.03, within ±0.05 sensitivity. Protected Random − StreamingLLM +0.054: descriptive only, not equivalent. The 388-row condition table is bound to the read by sha256 and reproduces it independently. Paper (§4.5 added; abstract, intro and §4.2 now pair-specific; threats) and claim ledger (A11–A16) updated. PAPER_FINDINGS §5q; `results/diag/meeting5/m5a/m5a_read_strict.json`. Whole-cohort read held (capacity comparators absent). G1 half met (protected captures not implemented) |
| Information calibration | **APPROVED 2026-10-17; NOT RUN (sync blocker, now cleared)** | The host was unreachable 2026-10-17 → 2026-10-20 and kept `5327072` (held/blocked). Synced 2026-10-20; first eligible window 2026-10-21 01:45. Campaign M5C-CAL, cap 4 GPU-h. `docs/paper/STATUS_REPORT_20261020.md` |
| Protected M/Δ/ω capture (M5A-CAPTURE-PP) | **Adapter built and tested (CPU, 2026-10-20); smoke and campaign PROPOSED, not approved** | `kvcache.capture_protected` (protected selection read from the press's own record, verified bitwise, inside the unchanged capture replay) and `scripts/run_capture_pp.py`; 12 tests pass on the host CPU, including exact identity controls. Cohort frozen: 23 strict-fit capture problems × 4 seeds = 92 traces. Proposal: smoke cap 1 GPU-h, campaign about 13.8 GPU-h (cap 16). `docs/paper/CAPTURE_PP_SMOKE_TIMING_PROPOSAL.md` |
| M5-C confirmation contrasts | **Amendment T (2026-10-20, before any information data)** | Random − StreamingLLM added beside the preserved Random − H2O-style (Holm over 2). All calibration comparisons exploratory |
| GSM8K timing | **PROPOSED (cap 3 GPU-h); blocked on a pid-filter adapter (CPU)** | 24-problem calibration split × 2 seeds, full cache; 0.82–2.76 GPU-h from the measured 0.0253 s/token. `docs/paper/GSM8K_TIMING_PROPOSAL.md` |
| Review of 5616630 | **R1–R4 repaired (CPU); calibration still HELD** | Calibration measurements validated against a frozen schedule; actual inputs bound by an input manifest; C6 floor rule (pre-data); compression read bound to FINAL S.1 evidence and relabelled as nominal-budget ratios. Feasibility without weights passed (83/40 reserve passages; all 64 variants render). `docs/paper/MEETING5_REVIEW_5616630_ACCEPTANCE.md` |
| Old WP5/WP6 | **Superseded** | Replaced by Meeting 5 manifests. The old manifests are kept as history |

## History (dated; superseded statements are kept as they were)

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


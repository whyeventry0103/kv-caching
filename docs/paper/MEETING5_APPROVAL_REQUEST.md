# Meeting 5 — one itemized approval request for new GPU work

**Decision recorded 2026-10-14 (owner, in-session): A1–A3 APPROVED at caps
1 / 28 / 28 GPU-h. A4 (fresh-uniform) and A5 (capacity comparators) are NOT
approved and stay held. The BBH revision is pinned for preparation only. The
equivalence margin is ±0.03, with ±0.05 as sensitivity.** The original request follows
unchanged.

**Prepared 2026-10-14. Nothing in this document is approved, and every config
it names is `held: true`.** It answers review §13 and §15 step 3: one concrete
budget request, with exact counts, prerequisites, worst-case allocated time and
storage.

**Where the numbers come from.** All hours come from
`results/diag/meeting5/ledger.json`, which
`scripts/plan_meeting5_cost.py --ledger` reproduces from
`configs/meeting5/m5_manifest.yaml` and the archived inventory. The rates are
measured, not estimated:
- a cell costs rows × mean per-row `wallclock_s`;
- the band runs from the fastest to the slowest mean of the six complete
  comparable 7B/MATH-500/B = 256 cells: **130.5 s / 162.8 s / 216.4 s per
  row**;
- each launch adds 0.05 h for model loading, plus 10% for retries.

## A. Requested now (M5-A, discovery cohort: 100 problems × seeds 11/23/47/89, B = 256)

| Item | What | Rows | GPU-h lower / central / upper | Proposed hard cap | Prerequisite |
|---|---|---:|---|---:|---|
| M5A-RKVPP-SMOKE | Protected R-KV real-model smoke on 3 traces: identity, replay and generation (9 records) | — | 0.26 (measured analogue: the protected-selector smoke took 0.2584 h) | **1** | CPU parity PASS (done, see §C) |
| M5A-H2OPP | H2O-style / current-attention, prompt protected | 400 | 16.06 / 20.01 / 26.55 | **28** | P2 smoke PASS under the 21-record design (P2 is approved and pending) |
| M5A-RKVPP | R-KV (certified port), prompt protected | 400 | 16.06 / 20.01 / 26.55 | **28** | M5A-RKVPP-SMOKE PASS |
| M5A-UFPP | Fresh-uniform random, prompt protected. A separately named sensitivity arm, not a repair of historical Random | 400 | 16.06 / 20.01 / 26.55 | **28** | P2 smoke PASS |
| M5A-CAPMATCH | Unprotected comparators at the protected arm's effective budget for the 3 relaxed problems (algebra/1349 @ 408, prealgebra/1646 @ 417, precalculus/1199 @ 280) × 4 policies × 4 seeds. None exist | 48 | 2.08 / 2.55 / 3.34 | **4** | A pid-filter grid adapter (**not yet written**: `run_grid.py` selects by subset count, and editing it would change pending cells' generation identity) |
| **Total** | | **1,248** | **50.52 / 62.84 / 83.25** | **89** | |

**Essential versus optional.**
- **Essential (review §8.1):** M5A-H2OPP, M5A-RKVPP and the smoke. Together:
  32.4 / 40.3 / 53.4 GPU-h, cap 57.
- **Sensitivity:** M5A-UFPP, against the fixed-slot Random objection (H5).
- **Needed only for the whole-cohort read:** M5A-CAPMATCH. The primary read
  is the 97-problem strict-fit cohort, which needs no comparator rows.

A reduced approval works: approve the essential pair first, and the others
later or never.

**Calendar.**
- The nominal maximum is 2 GPUs × 8 h 58 min ≈ 17.9 allocated GPU-hours per
  night. Shared-disk contention blocked about five windows in October
  (10-07 to 10-12).
- The essential pair therefore needs roughly 3 to 5 nights, and the full
  request 4 to 7. It is not promised as fewer.
- Order of work:
  1. finish the approved P1/P2/P3;
  2. the R-KV-pp smoke;
  3. H2O-pp and R-KV-pp;
  4. fresh-uniform;
  5. CAPMATCH, once the adapter exists and has been dry-run.

**Storage.** Measured comparable cells take 15–26 MB per 400 rows, token IDs
included. The 1,248 rows need about 0.1 GB. Our footprint is 50.45 GB of the
72 GiB allocation. The shared disk had 35 GB free at 00:12 on 2026-10-14,
close to the 30 GiB floor; the floor guard stays as is.

**Enforcement.** Each approved item would become its own campaign in `configs/campaigns.yaml`
with the cap above. The ledger (`kvcache.campaign_budget`) reserves before a
launch, bounds the process, settles from the launcher's end record, and keeps
an unreconciled crash charged. Normal 01:45–10:43 Dubai windows only.

## B. Not requested now (listed so the scope is complete)

| Item | Why not now |
|---|---|
| Forward readout for H2O-pp and R-KV-pp (800) | After the G1 accuracy read (review §8.1: accuracy first) |
| Protected M/Δ/ω captures | Needs a protected capture adapter (not written) |
| GSM8K protected replication (2,000 at 200 × 2) | No 7B GSM8K timing exists, so it cannot be priced honestly. The first step is a small full-cache calibration timing on the frozen 24-problem calibration split |
| BBH protected replication (2,000) | The revision needs a **human pin** (proposed `9ee07bd…`); the loader refuses until then |
| Information-channel calibration (320 prefix executions, proposed cap 4 GPU-h) | The runner (`scripts/run_information_probe.py`) is not written; the generator and reader are. A separate request will follow when the runner dry-runs |
| Unprotected B = 1024 reverse; WP6 | Deferred per review §16 |

## C. CPU evidence already in hand (no GPU used)

**Protected R-KV (`rkv_pp_v1`).**
- 7 wrapper tests pass on the host CPU with torch and kvpress.
- 15 gate/provenance tests pass in the CPU gate.
- Real-model CPU parity on the pinned 1.5B model and the fc-q15b bank, over 3
  traces (prompts 344 / 52 / 15 tokens), status PASS:
  - with protection off, all 112 prunes per trace give byte-identical K/V to
    base R-KV, and the TV profiles are identical;
  - with protection on, the prompt plus the recent-8 window are retained at
    every prune;
  - the target is 408 for the 344-token prompt and 256 otherwise;
  - all 28 layers are pruned, and pre-eviction TV is 0.
  - Evidence: `results/diag/meeting5/rkv_pp_cpu_parity/`.
- The family has its own provenance (`rkv_pp`), its own smoke marker and gate,
  and runner adapters that refuse while held. It is not in the protected-
  selector registry.

**Information-channel instrument.** The finite channel reproduces the
review §10.3 table exactly:

| Channel | MI (bits) | Correct probability |
|---|---|---|
| identity | 2 | 1 |
| uniform | 0 | 0.25 |
| wrong permutation | 2 | 0 |
| always-OTHER | 0 | 0 |

The reader CLI is FINAL / PARTIAL / INVALID-checked. The generator is tested
for equal lengths, counterbalancing, leakage and block splits.

**Inventory.**
- All 7 reuse cells are complete: 400/400 frozen keys, no duplicates.
- The strict-fit cohort is 97 problems, frozen in
  `configs/meeting5/strict_fit_b256.json`.
- AIME: Qwen-7B has full cache 240/240 and Random / StreamingLLM / R-KV at
  B = 2048 at 120/120 each, with readouts 93/120 defined per policy. The
  Llama-8B published-setup R-KV is 47/120, incomplete. There is no H2O-style
  arm and no protected arm, so AIME has no matched factorial.

**Graders.** The shared MATH grader marks `\boxed{1,250}`, `\boxed{18.00}`
and `\boxed{72 clips}` wrong for GSM8K. A frozen GSM8K numeric grader, applied
at read time, and a BBH single-option grader are implemented and tested.
Problem manifests are frozen for MATH-500 confirmation, GSM8K and BBH.

**M5-A reader.** `report_policy_replication.py` on today's data reads
**PARTIAL** for both cohorts (h2o+P and rkv+P absent) and prints no numbers,
as designed.

## D. The decision asked of the owner

1. Approve, reduce or decline items A1–A5. For each approved item, confirm the
   cap.
2. Pin, or decline, the BBH revision `9ee07bd481feebf959a6b59d61ea57bdcf30964d`
   (blob `936cd6d8…`).
3. Confirm the frozen ±0.05 equivalence margin (protocol §4.3), or set another
   **before** any M5 outcome is read.

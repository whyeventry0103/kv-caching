# Status report: Meeting 5 and the information investigation (2026-10-20)

**Observation time.**
- *Server:* inspected 2026-10-20 12:21–12:30 Dubai (+04:00). SSH timed out
  from the evening of 2026-10-17 (the first failed sync followed `ea16cd2`)
  until this inspection.
- *Repository:* the head before this report was `ea16cd2` (origin and backup).

**Evidence.**
- `results/diag/meeting5/status_20261020/status_inspection_20261020.json` and
  its `status_inspection_20261020.py.txt`: ledger, launches, scheduler counters,
  unique-key coverage, decode rates, calibration prerequisites.
- The result artifacts cited inline.

This is a status-and-analysis report. No GPU work was started for it, and no
cap, window or approval changed.

## 1. The scientific answer

Central question: does access to task-relevant information explain why
Random can be more accurate than StreamingLLM, despite the ordering of M, Δ
and ω?

**Answer: unknown.** No information measurement exists on any condition,
protected or not, and no protected M/Δ/ω exists. Every piece of evidence
needed to test the explanation is still missing. Specifically:

- **What we know about Random vs StreamingLLM accuracy.**
  - *Unprotected B=256* (Qwen-7B, MATH-500, 100 problems): Random is ahead
    (paper §4.2).
  - *Protected, strict cohort* (97 problems whose prompt fits; 388 conditions
    per arm): Random 0.683, StreamingLLM 0.629. The protected contrast is
    **+0.0541**, with descriptive paired 95% interval [+0.0232, +0.0902].
    - Verified today: the read re-run on the host is identical in estimates,
      sources, dependency fingerprint (`7ebd4c4a…`) and condition table to the
      committed `results/diag/meeting5/m5a/m5a_read_strict.json`.
    - The registered test is the **interaction**: −0.137, t −4.02 (df 96),
      rejecting after Holm. Protection substantially changes the contrast.
      The positive protected contrast itself is DESCRIPTIVE, not a registered
      superiority claim. It is not shown within either equivalence margin.
  - *Qwen3-4B, unprotected* (EXPLORATORY; bank gate FAIL): Random 0.030 and
    StreamingLLM 0.215. Random is far behind, mostly through non-termination
    (truncation 0.735; §5r). Together with Llama-8B, the unprotected ordering
    is model-dependent.
- **Under protection and matched capacity.** The gap persists on the strict
  cohort, where capacity is matched because every problem fits at B=256.
  The whole-cohort, capacity-matched comparison has not been run: its 48
  comparator rows are neither approved nor generated, and need a pid-filter
  adapter that is not implemented.
- **Information measured on protected conditions?** **No.** The calibration
  is approved but has not run (§2), and no confirmatory information
  measurement exists.
- **Has the information investigation explained, associated with, or failed
  to reach the gap?** Not reached. There is no information data at all.
- **What is missing, specifically:**
  1. protected M/Δ/ω on matched conditions;
  2. a passing information calibration, then a confirmation with a frozen
     contrast;
  3. the earlier-information → later-answer bridge;
  4. equal-budget interventions;
  5. any link to natural MATH problems.

**Random − H2O-style is a different outcome and is kept separate.**
- Its registered interaction is −0.186 (t −5.45; rejects after Holm).
- Under protection a difference is not detected (−0.005 [−0.036, +0.031]).
  It is within ±0.05, but not shown within the ±0.03 primary margin.
- The unprotected Random-vs-H2O-style "mismatch" therefore has no protected
  accuracy counterpart to explain.

## 2. Mutual-information stages

| Stage | Status | Evidence |
|---|---|---|
| Task generator and production runner | **implemented, CPU-tested** | `kvcache.information_tasks`, `scripts/run_information_probe.py`; `tests/test_information_runner.py` (23 tests) |
| Frozen input manifest, provenance, resume checks | **implemented, CPU-tested; host feasibility checked** | Input manifest bound into the design and every row; changed-input and incomplete-block resume refused before model load. Host check 2026-10-16: 83 eligible passages against 40 needed; 64 variants with 1,024-token bodies; manifest `1f527da6…` |
| Real-model controls and smoke prerequisites | **prerequisites met; probe controls not yet executed on the 7B** | Both protected-family smoke PASS markers are valid (`smoke_gate_problems: []`, 2026-10-20 12:22). The probe's own controls (full/full, clone agreement, no readout eviction, end-to-end) execute inside the calibration |
| Calibration execution | **approved, queued, NOT RUN** | Owner approval 2026-10-17: campaign `M5C-CAL`, cap 4 GPU-h (`ea16cd2`). **Blocker: sync.** The host was unreachable 2026-10-17 → 2026-10-20 and still carried `5327072` (config `held: true`, job `blocked:`). No run directory exists. The sync is done with this report; the first eligible window is 2026-10-21 01:45 Dubai. **No new decision is required** |
| Calibration analysis against C1–C6 (+ floor) | not started (no data) | Reader: `report_information_probe.py --require-calibration-pass` |
| Confirmation design and sample size | **not designed** | Protocol §6.7: precision target 0.1-bit half-width; N is frozen after calibration |
| Earlier-information → later-answer bridge | **specified, not implemented; held until calibration PASS (G2)** | Protocol §6.9 |
| Equal-budget target / nuisance / placebo interventions | **specified, not implemented; held until G2** | Protocol §7 |
| Connection to natural MATH accuracy | **not designed** | — |

**What the quantity measures.** It is exact output-channel MI between the
randomized target fact S and the declared reader's answer distribution (four
symbols plus OTHER), with the nuisance N marginalized.
- It is information through that reader, **not** total information inside
  the cache.
- Higher MI does not guarantee higher accuracy: a consistent permutation
  carries 2 bits at 0 correct. Correct probability, log loss and OTHER mass
  are therefore always reported beside it.
- Calibration results are instrument development, never confirmatory.

**Registered contrasts.**
- The registered M5-C paired contrast is **Random − H2O-style**
  (`configs/meeting5/info_calibration.yaml`: `paired_contrasts`).
- **Random − StreamingLLM is not a registered information contrast.** If it
  is read from the calibration it is exploratory.
- Making it a confirmatory contrast needs a dated pre-confirmation amendment.
  It would be motivated by the M5-A result and labelled as chosen after that
  result. That is a decision for the owner (see §6).

## 3. Is the proposed explanation supported?

| Claim | Status | Evidence |
|---|---|---|
| A. Random has a protected accuracy advantage over StreamingLLM | **Measured (descriptive):** +0.054 [+0.023, +0.090] on the strict cohort; the registered interaction rejects | `m5a_read_strict.json` (+ condition table) |
| B. Random still has higher protected M, Δ and ω on matched conditions | **Unmeasured.** Only unprotected captures exist: on the 25 capture problems at teacher-forced B=256, Random − StreamingLLM M +0.261, Δ +0.078, ω +0.200, accuracy +0.12 (joint label NOT ESTABLISHED). Combining these with protected accuracy is not permitted | `results/diag/paper_audit/matched_b256.json`, `matched_omega_b256.json` |
| C. Random preserves more accessible task-relevant information | **Unmeasured** | calibration not run |
| D. Earlier access predicts a later, separately generated answer beyond retention, age, occupancy and proxies | **Unmeasured** (not implemented) | protocol §6.9 |
| E. Equal-budget relevance interventions change outcomes as predicted | **Unmeasured** (not implemented) | protocol §7 |
| F. Transfer to natural reasoning problems | **Unmeasured** | — |

**Definitions** (PREREGISTRATION Q.3/Q.6):
- *M = 1 − Σ_{k∈S} a_full(k).* The full-cache attention mass on evicted
  positions: a selection loss.
- *Δ = TV(a_comp, a_full|S).* The shape distortion of attention over the
  retained set: a state-drift measure.
- *ω = ‖δ⊥‖ / ‖δ∥‖.* The ratio of hidden-state displacement outside a 64-PC
  full-cache subspace to the displacement inside it. It is a ratio, not a
  bounded fraction, and a larger ω does not by itself mean larger total
  displacement or more answer-relevant damage.

## 4. Workstreams

| Task | Implementation | Measured coverage (unique keys) | Analysis | Result | Blocker | Next action | Approval needed |
|---|---|---|---|---|---|---|---|
| Protected H2O-style and R-KV accuracy | done | 400/400 each | M5-A strict FINAL | H2O-style 0.688, R-KV 0.644 protected | — | none | — |
| Strict-cohort analysis, interaction and equivalence tests | done | 388/arm | FINAL, verified today | Both interactions reject; H2O-style within ±0.05 not ±0.03; StreamingLLM +0.054 descriptive | — | none | — |
| Whole-cohort capacity-matched comparators | pid-filter adapter **not built** | 0/48 | PARTIAL by design | — | adapter and approval | build the adapter (CPU), then request ≤48 rows | yes |
| Protected M/Δ/ω capture adapter, smoke, timing, campaign | **not built** | 0 | — | — | implementation | build on CPU, then a smoke/timing proposal | yes (smoke and campaign) |
| GSM8K / BBH timing, calibration and replication | loaders, graders, pinned data, frozen manifests | 0 generations | — | — | no approval; unpriced | full-cache timing on the 24-problem GSM8K calibration split | yes |
| Fresh-uniform vs historical Random | config prepared (M5A-UFPP) | 0 | — | — | held (A4) | — | yes |
| Qwen3 bank, grid, forward and reverse readout | done | 200/200; 400/400 ×3 | **read today** (§5r) | StreamingLLM REPLICATES; Random near floor (EXPLORATORY) | — | none | — |
| Qwen3 bank gate and human scoring review | — | — | gate FAIL 0.845 | stands | owner/supervisor must assign the 42-row review | assign a human reviewer | owner |
| B=1024 forward/reverse and combined interpretation | done | 800/800 per stage | FINAL (§5o, §5p) | residual persists; no interaction detected | — | none | — |
| Paper text, abstract, figures, ledger, catalogue | §4.5 added; pair-specific text; ledger A11–A16, C3; catalogue current | — | audit clean | — | — | figures for M5-A/§4.5 not yet drawn; abstract rewrite at evidence freeze | — |
| Server sync, activity, budgets | synced with this report | GPUs idle (outside window, nothing launchable on the old commit) | — | Ledger: P1 12.909/20, P2 0.549/2, P3 1.388/6, smoke 0.187/1, H2O-pp 15.489/28, R-KV-pp 19.24/28, M5C-CAL 0/4 | the 3-day link outage (resolved) | calibration at 2026-10-21 01:45 | — |

## 5. Gaps closed in this pass

- **Qwen3 S.4 read.** The data had been complete since 2026-10-18; it is now
  read, archived and recorded (PAPER_FINDINGS §5r, PREREGISTRATION, ledger
  C3), with nominal ratios.
- **Strict-cohort condition table.** Already published at `5327072`:
  - `results/diag/meeting5/m5a/m5a_read_strict.conditions.tsv`: 388 rows of
    pid, seed and each of the 8 arms' correct and truncated flags;
  - sha256 `da50c268…`, recorded in the read;
  - reproduction:
    `python results/diag/meeting5/m5a/reproduce_from_table.py.txt <tsv> <json>`
    (standard library only; output `reproduce_from_table.out.txt`: ALL MATCH).

  Re-verified identical today.

## 6. Next steps, in scientific priority

**Already authorized (no decision needed).**
- *M5-C calibration:* cap 4 GPU-h, estimated about 2.5 GPU-h (336k decode
  steps at the measured 0.0265 s/token). It runs from 2026-10-21 01:45. It
  answers only whether the instrument works. It cannot, by itself, answer the
  central question.

**New proposals, in priority order.**
1. **Protected capture adapter, then a smoke/timing run, then a protected
   capture** (the order follows the definitions in §3).
   - *Why first:* without protected M/Δ/ω there is no established protected
     mismatch for information to explain (claim B).
   - *Cost:* CPU implementation first. The measured unprotected capture of
     the same shape took 12.08 h of row wallclock for 400 conditions, so a
     four-policy protected capture is about 12–16 GPU-h. Random-protected and
     StreamingLLM-protected alone are about 6–8 GPU-h.
2. **Register the information contrast that matches the remaining question.**
   Add Random − StreamingLLM as a pre-confirmation contrast by a dated
   amendment, labelled as chosen after the M5-A result, before any
   confirmation data. It costs no GPU and is a decision for the owner. The
   calibration's own Random − StreamingLLM numbers remain exploratory either
   way.
3. **GSM8K full-cache timing on the 24-problem calibration split, then
   price the matched replication.**
   - *Why:* A is measured on one model and one benchmark only, and Qwen3
     shows the unprotected ordering is model-dependent.
   - *Cost:* at most about 2.5 GPU-h (24 problems × 2 seeds × the 8,192
     token cap × 0.023 s/token); typically far less.

**Is MI the most promising next experiment?**
- *Not as the next step for the central question.* The landed results
  removed the H2O-style gap under protection, and a smaller StreamingLLM gap
  remains. Whether that remaining gap even conflicts with M/Δ/ω under
  protection is unmeasured, so the protected capture is the logically prior
  test.
- *Still worth running.* The MI calibration is cheap and already
  authorized, so it should run: it settles whether the instrument is usable.
- *What it can and cannot show.* Even a passing calibration and a positive
  confirmation would show retrieval through a declared reader on a synthetic
  task. It would not explain natural reasoning accuracy without the bridge
  (§6.9) and the interventions (§7). Nothing here guarantees that MI explains
  the gap, or any publication outcome.

## Post-sync verification (2026-10-20 13:24 +04:00)

- *Host checkout:* `5d372e2` (`.sync-state`, synced 09:23:50 UTC). It
  contains the approval: `held: false`, `campaign: M5C-CAL` on the job, and
  the campaign defined with cap 4.
- *Scheduler dry-run at tomorrow's window*
  (`nightly_autorun.py --dry-run --force --now 2026-10-21T01:50:00`):
  - it selects `info-calibration-m5c` 0/640 on one GPU (seed shard 0);
  - preflight `check_remote.sh` PASS;
  - "campaign M5C-CAL: remaining 4.000 GPU-h -> would reserve 1 x 4.000";
  - nothing launched.
- *Note on the bound:* the dry-run prints the window bound (8.91 h). A real
  launch sets `KVCACHE_MAX_HOURS = min(window, granted reservation)`, so the
  process is bounded by the 4 GPU-h cap.
- *If the run outgrows the cap:* the run would read PARTIAL, and finishing it
  would need a new request (approval request §6).

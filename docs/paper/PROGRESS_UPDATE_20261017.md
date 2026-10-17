# Progress update for the reviewer (2026-10-17, state at 17:00–17:10 Dubai, +04:00)

Since the review of 5616630, it covers the strict M5-A read, every campaign,
the combined B=1024 finding, the Meeting 5 questions, and the information
calibration request.

**Basis.** Every number below comes from a committed artifact or from the
archived status inspection, taken on the GPU host at 2026-10-17 17:06:14 +04:00
(`results/diag/meeting5/status_20261017/status_inspection.json`, script
`status_inspection.py.txt` beside it).
- *Source state:* the host runs commit `a470f0b` (`.sync-state`, synced
  2026-10-16 06:46 UTC). This update adds docs, catalogue registrations and
  archived outputs only.
- *Authorization:* no GPU work was started for it, and no cap, window or
  approval changed.

## 1. M5-A strict-cohort read: FINAL

The read is `results/diag/meeting5/m5a/m5a_read_strict.json`, written on the
host at 2026-10-17 17:00:21 +04:00 (dependency fingerprint `9ae31273…`).
Reproduce with:

```
python scripts/report_policy_replication.py --runs-root <runs> --cohort strict
```

**Validation, before cohort selection.** Every source run was validated in
full against its own config's design:

| Source run | Design | Present (unique expected) | Missing / dup / foreign / malformed |
|---|---|---|---|
| grid-diss256-tok (Random, StreamingLLM unprotected) | 800 | 800 | 0 / 0 / 0 / 0 |
| grid-pp256 (Random, StreamingLLM protected) | 800 | 800 | 0 / 0 / 0 / 0 |
| grid-diss256-h2o (H2O-style unprotected) | 400 | 400 | 0 / 0 / 0 / 0 |
| grid-h2opp256-m5 (H2O-style protected) | 400 | 400 | 0 / 0 / 0 / 0 |
| grid-diss256-rkv (R-KV unprotected) | 400 | 400 | 0 / 0 / 0 / 0 |
| grid-rkvpp256-m5 (R-KV protected) | 400 | 400 | 0 / 0 / 0 / 0 |

Controls: the four legacy provenance controls passed, and the M5 rows carry
the expected identity and frozen prompt hash. Validation failures: **none**.

**Denominators and exclusions.**
- *Cohort:* the frozen strict-fit cohort (`configs/meeting5/strict_fit_b256.json`,
  frozen 2026-10-14 before any M5 outcome): 97 problems × 4 seeds = **388
  conditions in every arm**.
- *Excluded by design:* the 3 problems whose prompt leaves fewer than 64 trace
  slots at B=256: algebra/1349 (effective budget 408), prealgebra/1646 (417)
  and precalculus/1199 (280).
- *Truncation:* truncated rows stay in the denominator, scored incorrect, and
  none is scored correct. Inspection counts per arm (correct / truncated, of
  388), which reproduce the read's accuracies exactly:

| | Unprotected | Protected |
|---|---|---|
| Random | 169 / 87 | 265 / 12 |
| H2O-style | 99 / 40 | 267 / 17 |
| StreamingLLM | 95 / 24 | 244 / 32 |
| R-KV | 208 / 48 | 250 / 46 |

**Accuracy.** Equal-problem means, problem-cluster bootstrap 95% CI, 2,000
resamples, seed 0:

| Policy | Unprotected | Protected |
|---|---|---|
| Random (historical fixed-slot) | 0.436 [0.356, 0.513] | 0.683 [0.598, 0.771] |
| H2O-style / current-attention | 0.255 [0.188, 0.325] | 0.688 [0.611, 0.765] |
| StreamingLLM | 0.245 [0.183, 0.314] | 0.629 [0.541, 0.714] |
| R-KV (certified port) | 0.536 [0.451, 0.624] | 0.644 [0.559, 0.729] |

**Contrasts** (Random − policy, matched keys) and the protection-by-policy
interaction:

| | Unprotected | Protected | Interaction |
|---|---|---|---|
| Random − H2O-style (scientific) | +0.180 [+0.121, +0.240] | −0.005 [−0.036, +0.031] | −0.186 [−0.250, −0.113] |
| Random − StreamingLLM (scientific) | +0.191 [+0.134, +0.247] | +0.054 [+0.023, +0.090] | −0.137 [−0.201, −0.070] |
| Random − R-KV (descriptive) | −0.101 [−0.157, −0.046] | +0.039 [−0.003, +0.082] | +0.139 [+0.072, +0.209] |

**Registered interaction test** (frozen 2026-10-14: problem-level two-sided t
test, Holm over the two scientific pairings, α = 0.05):

| Pairing | t (df) | p | Holm p | Verdict |
|---|---|---|---|---|
| Random − H2O-style | −5.45 (96) | 3.8e−7 | 7.7e−7 | rejects |
| Random − StreamingLLM | −4.02 (96) | 1.2e−4 | 1.2e−4 | rejects |

**Equivalence of the protected contrasts.** TOST on each protected contrast,
Holm over the pairings; the joint claim is intersection–union.

| Pairing | TOST interval (1 − 2α) | ±0.03 primary: p (Holm) | ±0.05 sensitivity: p (Holm) |
|---|---|---|---|
| Random − H2O-style | [−0.034, +0.023] | 0.076 (0.151): **not shown within margin** | 0.005 (0.010): **within margin** |
| Random − StreamingLLM | [+0.026, +0.082] | 0.922 (0.922): not shown within margin | 0.596 (0.596): not shown within margin |
| Joint (both scientific pairings) | — | fails | fails |

**What this does and does not establish.**
- *Random − H2O-style under protection: no difference detected* (−0.005).
  Equivalence at the owner's ±0.03 margin is **not established**, because the
  TOST interval reaches −0.034. It holds only at the ±0.05 sensitivity
  margin. This is never written as "equal".
- *Random − StreamingLLM under protection: a difference is established*
  (+0.054, CI excludes 0). It is about a quarter of the unprotected +0.191,
  and it is not equivalent at either margin.
- *Both interactions are established.* Protection changes the Random contrast
  for both scientific pairings.
- *R-KV is descriptive only.* Its ordering against Random reverses sign under
  protection; the protected CI includes 0.
- *Scope:* one model, MATH-500, B=256, the 97-problem strict cohort.
  H2O-style is current-attention, not canonical H2O; R-KV is our certified
  port.

**Whole-cohort read: HELD.** It is PARTIAL by construction until the 48
capacity-matched comparators (M5A-CAPMATCH) exist. They are not approved and
not generated, and they need a pid-filter grid adapter that is not
implemented. It is never labelled capacity-matched before that.

## 2. Campaigns

All times are Dubai (+04:00). The ledger is `logs/campaign_ledger.json` on the
host, as summarized in `status_inspection.json`.

| Campaign / job | Validated completion | GPU-h used / cap | Failures | Current activity | Depends on | Next action |
|---|---|---|---|---|---|---|
| P1 `reverse-readout-pp1024` | 800/800, `reverse_b1024.json` FINAL (2026-10-16 00:31) | 12.909 / 20 | none (4 launches, all rc 0) | complete | — | none; the 7.091 h left is unused, not carried over |
| P2 `protsel-smoke` | 21-record PASS 2026-10-15 13:19; its gate is still valid at 17:06 today | 0.549 / 2 | none. The 2026-10-13 15-record run is superseded (stale scope), not failed | complete | — | none |
| P3 `protsel-pilot-{grid,readout,reverse}` | 16/16 rows per stage by scheduler count; timing-only rows, excluded from every claim, no read registered | 1.388 / 6 | none (3 launches, rc 0) | complete | P2 | timings available for costing |
| M5A-RKVPP-SMOKE `rkvpp-smoke` | 9-record PASS 2026-10-15 15:14; gate still valid today | 0.187 / 1 | none | complete | — | none |
| M5A-H2OPP `grid-h2opp256-m5` | 400/400, validated in the M5-A read | 15.489 / 28 | 5 launches failed at 0 h on 2026-10-15 15:13–15:33 (gate bug, fixed in `4f50d28`); then 2 launches rc 0 | complete | P2 | none |
| M5A-RKVPP `grid-rkvpp256-m5` | 400/400, validated in the M5-A read | 19.240 / 28 | none (2 launches, rc 0) | complete | M5A-RKVPP-SMOKE | none |
| Qwen3 `fc-qwen3-4b` (exploratory; no campaign cap) | 200/200 unique expected keys, 0 dup / 0 foreign | not ledgered; row wallclock 7.42 h (a lower bound) | 0 (scheduler counter) | complete | — | none |
| Qwen3 `grid-diss256-qwen3-4b` | **381/400** unique expected keys, 0 dup / 0 foreign / 0 torn. Every policy × seed has 50/50 except Random seed 89 at 31/50 | not ledgered; row wallclock 42.50 h (a lower bound) | 0 | idle (window closed 10:43); last row 06:33 today | `fc-qwen3-4b` | resumes at 01:45 tomorrow |
| Qwen3 `readout-diss256-qwen3-4b` | **169/400** unique expected keys (Random s11 50, s47 34; StreamingLLM s11 50, s47 35; seeds 23 and 89 not started), 0 dup / 0 foreign | not ledgered (readout rows carry no wallclock) | 0 | idle; launched 03:30 today on GPU 1 | `fc-qwen3-4b` | resumes at 01:45 |
| Qwen3 `reverse-readout-diss256-qwen3-4b` | 0/400 (no run directory) | — | — | waiting | the Qwen3 grid at 400/400 | starts after the grid completes |
| `info-calibration-m5c` | not run | 0 / none (no campaign yet) | — | blocked: not approved | both smoke PASS markers (valid) | the owner's decision on the request in §5 |

The Qwen3 work is the exploratory S.4 work approved before campaign caps
existed. It has no campaign entry, so its GPU time is not ledgered, and row
wallclock understates allocated time. The Qwen3 bank gate FAIL (0.845) and
disposition (a) stand, and the 42-row human scoring review is still
unassigned.

**Still held, with no campaign:**
- fresh-uniform protected arm (A4);
- capacity comparators (A5 / M5A-CAPMATCH);
- GSM8K and BBH generation;
- protected forward readouts (M5A-READOUT);
- protected captures (M5A-CAPTURE-PP);
- the information calibration.

## 3. Combined B=1024 finding (protected arms) and the paper text still needed

**Finding.** Sources: PAPER_FINDINGS §5o/§5p; `protection_b1024.json`,
`reverse_b1024.json`, `realized_compression.json`.

| | Random | StreamingLLM |
|---|---|---|
| Forward protected residual | −0.047 [−0.086, −0.015] | −0.040 [−0.070, −0.013] |
| Registered label | PERSISTS, SHRINKS | PERSISTS, SHRINKS |
| Trajectory contrast, compressed readout (common support) | −0.037 [−0.076, −0.004] | −0.039 [−0.073, −0.011] |
| Trajectory contrast, full-cache readout (common support) | −0.040 [−0.080, −0.006] | −0.034 [−0.067, −0.006] |
| Registered interaction (t test, Holm) | +0.003, Holm p 0.64, does not reject | −0.005, Holm p 0.64, does not reject |
| Nominal-budget ratio, median | 0.434 | 0.487 |

- *Random's SHRINKS:* the Δresidual lower bound is only +0.001.
- *Budget ratio:* about 2× compression, not 4×. 31/400 conditions are at
  ratio 1, a lower bound on never-compressed.
- *The residual persists under either readout:* at protected B=1024, an
  accuracy change of about −0.03 to −0.05 that on-reference readout does not
  explain appears whichever readout defines it. The readout-change contrasts
  are about 0.
- *Interaction:* none detected. Without a registered equivalence test this is
  not "no interaction".
- *Contrast with B=256:* there, the protected directions disagree
  (interaction +0.130 / +0.127).
- *Limits:* the claim is operational, not causal. The common support excludes
  undefined readouts, and StreamingLLM's worst-case interaction bound lies
  entirely below 0.

**Paper text still needed** (`kv_readout_vs_trajectory_paper.tex` and
`docs/paper/CLAIM_LEDGER.md`; none of it is done):

1. *Abstract (L71).* Replace "the protected budget-1024 cells are pending"
   with the forward and reverse result and its ≈2× nominal ratio. The abstract
   is also flagged (L72–73) for a full rewrite from the claim ledger at the
   evidence freeze.
2. *§4.2 "collapse, and random beats recency" (L283 ff.).* Scope "random beats
   recency" to prompt-evictable policies, and add the protected four-policy
   result from §1 above: the advantage over H2O-style is not detected; the
   advantage over StreamingLLM is +0.054; both interactions are established.
3. *§4.3 forward decomposition (L294 ff.) and §4.4 reverse replay (L334 ff.).*
   Add the protected rows at B=256 and B=1024, including that the B=256
   directions disagree while the B=1024 interaction is not detected.
4. *Budget reporting.* Every absolute budget carries its nominal-budget
   ratio, with the ratio-1 lower-bound wording.
5. *Threats (§5, L657 ff.).* Protection scope; H2O-style naming;
   the R-KV-port scope; "not detected" ≠ equivalence (and the ±0.03 vs ±0.05
   result for H2O-style); one model / one benchmark.
6. *Claim ledger.*
   - A2 still marks B=1024 PENDING and C3 still says Qwen is held behind
     B=1024. Both are stale.
   - New entries are needed for §5p (B=1024 forward/reverse) and §5q (M5-A).
   - B1 (the unprotected Random-vs-H2O-style mismatch on the 25 capture
     problems) must state two things. Under protection, the Random − H2O-style
     accuracy difference is not detected on the 97-problem strict cohort
     (within ±0.05, not shown within ±0.03). And no protected capture exists,
     so the protected mismatch is unmeasured.
7. *Uncompiled R.17 draft (`\iffalse`, L469).* It says the Amendment S cells
   are "running at the time of writing", which is stale. Update it if the
   draft is adopted.

## 4. Meeting 5 questions

| Question | Implemented | Measured | Analysed | Held / missing |
|---|---|---|---|---|
| **Protected-policy accuracy (M5-A)** | Protected selector and protected R-KV presses, both with real-model smoke PASS; M5 production runner with atomic pair publication; strict and whole readers | All 8 arms at B=256: 4 protected + 4 unprotected, 400 conditions each | **Strict read FINAL** (§1) | Whole-cohort read (needs M5A-CAPMATCH and a pid-filter adapter; neither approved nor built); fresh-uniform A4 |
| **Protected metric captures (M/Δ/ω)** | **Not implemented.** The protected capture adapter is not written | none | none | M5A-CAPTURE-PP (after G1). Unprotected captures are never substituted. G1 is therefore half met: the accuracy half is done, the capture half is not |
| **Dataset replication (M5-B)** | GSM8K loader and strict grader (pinned `740312a`, data prepared on the host). BBH loader and strict option grader, pinned by the owner (`PINNED_COMMIT 9ee07bd…`, blob `936cd6d…`, 250 examples, data fetched). Frozen problem manifests (GSM8K 24 calibration / 200 confirmation; BBH 25 / 200; MATH-500 200 confirmation + 200 reserve, outside the discovery cohort) | none for M5-B. The pre-existing AIME-24 B=2048 three-policy results are reusable but are not the protection factorial. Second-model check: Qwen3 is in progress (§2), exploratory | none | GSM8K/BBH generation (not approved; unpriced — there is no 7B GSM8K timing yet; the first step would be a full-cache timing on the 24-problem calibration split) |
| **Information calibration (M5-C)** | Task generator, finite-channel reader, runner core and GPU path, controls (full/full, clone agreement, no readout eviction, end-to-end), criteria C1–C6 plus the floor rule, and the R1–R3 repairs | CPU fixtures only; host feasibility without weights (§5) | none on real data | **The calibration run itself** (approval requested, §5); the confirmation run and the bridge after that |

*Discrepancy to note.* `configs/meeting5/problem_manifests.json` still labels
the BBH `revision` as "TODO(human): proposed 9ee07bd…". The binding pin is the
loader constant set on the owner's 2026-10-14 decision. The manifest is
frozen, so the stale label is reported here, not edited.

## 5. Information-calibration approval request

The full request is `docs/paper/MEETING5_CALIBRATION_APPROVAL_REQUEST.md`. In
brief:

- **Scope:** 4 blocks × 16 variants × 5 arms (full cache plus the four
  protected policies) at B = 256. That is 320 prefix executions plus 8
  full/full controls, and 640 rows. Calibration rows are never confirmatory.
- **Cap: ≤ 4 GPU-hours** in a new campaign `M5C-CAL`, cumulative allocated
  time, normal windows only, one GPU. **Estimate about 2.5 GPU-h**, from
  336k decode steps at the slowest measured protected rate (0.0265 s/token).
  The probe's own loop has not been timed on the GPU, so the cap is the bound.
- **Prerequisites, all met:**
  - both smoke PASS markers are valid (checked 17:06 today);
  - 83 of the 40 needed reserve passages are available, and the pinned
    tokenizer renders all 64 variants;
  - the criteria were frozen pre-data, including the C6 floor;
  - the R1–R3 repairs pass the review's own reproduction cases (malformed →
    INVALID; floor → FAIL; changed filler → refused before model load).
- **On approval, three config changes** land together: the campaign entry,
  the job's `campaign:` line plus removal of `blocked:`, and `held: false`.
  All gates run bare before the sync. Without the `campaign:` line the cap
  would not be enforced.
- **Planned response:**
  - INVALID → instrument defect: fix with a regression test; it does not use
    the redesign;
  - PARTIAL → resume under the same identity;
  - FAIL → report it, then allow one pre-registered redesign under a separate
    approval; a second FAIL abandons the instrument (no MI claim);
  - PASS → freeze the confirmation N and request that run separately.

**Not launched.** The job remains `blocked:` and the config `held: true`.

## Artifacts in this update

- `results/diag/meeting5/m5a/m5a_read_strict.json`: the M5-A strict read.
- `results/diag/meeting5/status_20261017/status_inspection.json` (+
  `status_inspection.py.txt`): ledger, scheduler counters, per-arm counts,
  Qwen3 unique-key coverage, decode throughput, calibration prerequisites.
- `results/diag/paper_audit/reverse_b1024.json`,
  `results/diag/amendment-s/protection_b1024.json`,
  `results/diag/amendment-s/realized_compression.json`: the B=1024 reads
  (committed earlier).
- `docs/results/PAPER_FINDINGS.md` §5q; `docs/protocol/PREREGISTRATION.md`
  (M5-A result entry); `docs/paper/EXECUTION_STATUS.md`;
  `docs/paper/MEETING5_CALIBRATION_APPROVAL_REQUEST.md`.

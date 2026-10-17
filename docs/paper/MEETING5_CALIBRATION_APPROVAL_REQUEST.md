# Approval request: M5-C information-channel calibration (2026-10-17)

**Status: APPROVED by the owner in-session, 2026-10-17** (option "Approve
≤4 GPU-h").
- *Changes landed together:* campaign `M5C-CAL` with cap 4 GPU-h in
  `configs/campaigns.yaml`; `campaign: M5C-CAL` on the queue job, with
  `blocked:` removed; `held: false` in the config.
- *Not yet launched:* the first eligible window is 2026-10-18 01:45 Dubai.
- *Still separate decisions:* the confirmation run and the reasoning bridge.

The text below is the request as approved.

## 1. What would run

The four-block calibration of the finite-channel instrument, as specified in
protocol §6 (`docs/protocol/MEETING5_INFORMATION_PROTOCOL.md`) and
`configs/meeting5/info_calibration.yaml`:

| Item | Value |
|---|---|
| Model | `r1_distill_qwen_7b` (pinned revision; eager attention) |
| Task | 4 blocks × 16 (S, N) variants; body 1,024 tokens; facts at body positions 64 and 768, order alternating by block |
| Arms | full cache, `random_pp`, `streamingllm_pp`, `h2o_style_pp_v1`, `rkv_pp_v1`; B = 256, compression interval 64, eviction seed 0 |
| Executions | 320 prefix executions plus 8 full/full controls; 1,280 cloned readouts plus 320 end-to-end readouts |
| Rows | 640 (`readout/<block>-s0.jsonl`, each block written atomically) |
| Filler | MATH-500 confirmation-reserve full-cache traces (outside the discovery cohort and the confirmation set) |
| Storage | about 0.01 GB |

Calibration records are instrument development. They are never confirmatory
and never enter a claim.

## 2. Cap and cost basis

- **Cap: ≤ 4 GPU-hours,** cumulative allocated time. It includes loading,
  failures and retries, is enforced before launch by
  `kvcache.campaign_budget`, and allows normal 01:45–10:43 Dubai windows only.
  It runs on one GPU, as one shard (eviction seed 0).
- **Estimate: about 2.5 GPU-hours.** 328 prefix executions × (1,024 body
  decode steps + one prompt prefill) is about 336k single-token forward
  passes. At the slowest measured protected decode rate (0.0265 s/token,
  `grid-h2opp256-m5`; 0.0237 for `grid-rkvpp256-m5`, 0.0232 for `grid-pp256`),
  that is about 2.5 h. The 1,600 short readouts and the cache clones add
  minutes.
  - Source: `results/diag/meeting5/status_20261017/status_inspection.json`,
    `decode_throughput`.
  - *Uncertainty, stated once:* the probe's own per-token loop
    (`TransformersBackend`) has not been timed on the GPU. The cap, not the
    estimate, bounds the run.

## 3. Prerequisites and their evidence

| Prerequisite | Evidence | State |
|---|---|---|
| Both protected-family real-model smokes PASS and still valid (the runner refuses otherwise) | `smoke_gates()` returned no problems on the host at 2026-10-17 17:06 (`status_inspection.json` → `calibration_prereqs`) | met |
| Real reserve filler suffices, and the pinned tokenizer renders the frozen task (no weights) | `run_information_probe.py --check-inputs` on the host, 2026-10-16: 83 eligible passages against 40 needed; 0 tokenizer problems; 4 blocks / 64 variants with 1,024-token bodies; 0 leak-screen skips; input-manifest hash `1f527da6…` | met |
| Acceptance criteria frozen before data | C1–C6 frozen 2026-10-15; the C6 floor rule (`floor_mi_bits` 0.1) was added 2026-10-16, before any calibration data (protocol §6.8/§9) | met |
| Repairs from the review of 5616630 | R1 measurement validation against a frozen schedule, with criteria bound to the design. R2 input manifest bound into the design and every row, with changed-input resume refused before model load. R3 floor rule. Before/after: `results/diag/meeting5/review_5616630/`; acceptance: `docs/paper/MEETING5_REVIEW_5616630_ACCEPTANCE.md` | met (CPU) |
| Validity separated from acceptance | `report_information_probe.py --require-calibration-pass` exits 4 unless FINAL and PASS; `information_calibration.accepted()` | met (CPU) |
| No other job displaced | Queue order puts the calibration after the M5 grids (complete) and before the Qwen3 jobs; the owner may reorder | to confirm |

## 4. Exact changes on approval (each a reviewable diff, dry-run first)

1. `configs/campaigns.yaml`: add campaign `M5C-CAL` with `cap_gpu_hours: 4`,
   the approval date and its scope.
2. `configs/nightly_queue.yaml`, job `info-calibration-m5c`: add
   `campaign: M5C-CAL` and remove the `blocked:` line. Without the campaign
   line the cap would not be enforced, so both changes land together.
3. `configs/meeting5/info_calibration.yaml`: `held: false`. The design hash
   covers the config, so this changes the design identity; no calibration row
   exists, so nothing is orphaned.
4. Gates `make lint test dryrun audit`, each run bare; commit; push;
   `make sync-apply`. The job then launches only in a normal window.

## 5. How it will be read

```
report_information_probe.py --rows <run>/readout --design <run>/design.json \
    --out <read.json> --require-calibration-pass
```

The criteria evaluated are the ones frozen in `design.json`. A different
criteria file makes the read INVALID.

## 6. Planned response to every outcome (fixed now, before data)

| Outcome | Meaning | Response |
|---|---|---|
| **INVALID** (exit 2): a failed control, malformed or missing measurement, manifest or identity mismatch | An instrument or data defect, not a calibration result | No criterion is interpreted. Root-cause it, fix it with a regression test, and record a dated note. A rerun inside the remaining cap needs that note; if the cap is spent, it needs a new request. It does not use the redesign allowance |
| **PARTIAL** (exit 3): rows missing, e.g. the window ended | Incomplete | Resume in the next window under the same identity (complete blocks are kept; an incomplete block refuses). No read until FINAL |
| **FINAL, calibration FAIL** | The instrument does not have usable range at this operating point | Report the failed criteria and their values plainly. **One** redesign is allowed on calibration data (gate G2), stated as a dated pre-data amendment before it runs: C1/C2 → task or answer format; C5 → operating point (offsets, body length, budget); C6 at floor → a less harsh operating point; C6 at ceiling → a harsher one. C4 is investigated as a possible instrument defect first. The redesign's run is a **separate approval**. If it also fails, the instrument is abandoned: it is reported as a retrieval diagnostic and no MI claim is made |
| **FINAL, calibration PASS** | The channel can carry a confirmatory question | Freeze the confirmation N against the 0.1-bit half-width target and request the confirmation run separately. Calibration rows stay out of every claim |

Never, in any branch:
- tune the design toward a policy ranking (equal, informative compressed arms
  are a valid null);
- reuse calibration blocks in confirmation;
- relabel a FAIL.

## 7. Not requested here

- the confirmation run and the §6.9 bridge;
- GSM8K and BBH generation;
- fresh-uniform (A4) and the capacity comparators (A5 / M5A-CAPMATCH);
- protected captures;
- any change to existing caps or windows.

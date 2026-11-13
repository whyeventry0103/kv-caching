# Proposal: GSM8K full-cache timing and calibration (2026-10-20)

**Status: PROPOSED.** Nothing here is approved or launched. No GSM8K
generation has ever run.

## Why now

Every policy result is on one benchmark (MATH-500), and the protected results
are on one model. The Qwen3-4B read (§5r, EXPLORATORY) shows the unprotected
Random-ahead ordering failing on a second model, which makes the
dataset/model scope of every ordering claim the open question. The first
replication step is to price GSM8K honestly.

## What runs (smallest informative step)

- *Problems:* the frozen GSM8K **calibration split**: 24 problems from
  `configs/meeting5/problem_manifests.json`, pinned revision `740312a…`.
  They are hash-selected before any outcome and never enter confirmation.
- *Model:* Qwen-7B (R1-Distill), full cache only, at the project's sampling
  (temperature 0.6, top-p 0.95).
- *Token cap:* the proposed GSM8K cap is 8,192 (Meeting 5 plan §9.3), subject
  to this calibration's truncation check. Any cap change before the matched
  replication is a pre-outcome amendment.
- *Seeds:* 11 and 23.
- *Read:* read-time scoring with the frozen GSM8K grader
  (`kvcache.task_graders.gsm8k_grade`; gold parsing tested separately).
- *Reported:*
  - full-cache accuracy;
  - truncation rate;
  - the token-length distribution;
  - measured seconds per token;
  - how often a trace would exceed B=256 and B=1024, which shows whether
    eviction is even active.

## Cost (from measurements)

The Qwen-7B full-cache bank on MATH-500 measured 0.0253 s per generated token
(2,000 rows, 7.49M tokens, 52.61 h); its median trace is 2,422 tokens. GSM8K
lengths have not been measured, so:

| Assumption | GPU-h for 24 × 2 traces |
|---|---|
| Every trace hits the 8,192 cap (upper bound) | 2.76 |
| GSM8K traces as long as the MATH median | 0.82 |

**Proposed cap: 3 GPU-h,** including model loading. Normal windows only.

## Problem-ID adapter (built 2026-11-10; the CPU blocker is cleared)

The existing loaders select problems by a count or a seeded sample, not by
ID, so the frozen calibration split needed an additive adapter.
- *Selection:* `src/kvcache/manifest_problems.py` resolves a split to its
  frozen IDs and refuses:
  - a manifest revision that differs from the pin;
  - a missing, empty or repeated split;
  - overlap with another split;
  - a loaded pool of the wrong size;
  - any absent ID.
- *Runner:* `scripts/run_bank_manifest.py`. It is held, and generates full
  cache only by importing `build_bank.py`'s path. `run_grid.py`,
  `build_bank.py` and `datasets.py` are byte-identical.
- *Publication (2026-11-10):* the Meeting 5 runner's recoverable pair path.
  - Each row and its token-ID sidecar are journalled together and published
    idempotently; recovery runs before the model loads; resume counts complete
    pairs only.
  - `kvcache.manifest_publish` reuses `kvcache.m5_publish` unchanged, scoped
    to `gsm8k`. That module is part of the M5 generation identity, so it is
    not edited.
  - CPU regression: a stop after the result write resumes to exactly one
    matching pair for each of the 48 keys, both when the envelope survives
    (no regeneration) and when it is lost (the orphan row is quarantined and
    regenerated once).
- *Reader:* `scripts/read_gsm8k_timing.py` + `kvcache.gsm8k_timing`.
  - It is FINAL only with exactly one matching row and sidecar for all 48
    keys. It refuses on any missing, extra or repeated key, a torn line, a
    pending envelope, or a trace hash or token-count mismatch.
  - It scores the stored traces with the frozen `gsm8k_grade`; the row's
    `correct` field is reported only as a disagreement count.
  - Golds come from the pinned dataset and must match the committed
    resolution record.
  - It also reports truncation, the length distribution, s/token, GPU-h, and
    whether B=256 or B=1024 would bind.
- *Tests:* 16 in `tests/test_gsm8k_timing_pilot.py`.
- *Config:* `configs/experiments/bank_gsm8k_timing_m5.yaml` (`held: true`,
  `scale_authorized: false`). It is on `make dryrun`; there is no queue job
  and no campaign.
- *Checks:*
  - 17 CPU tests (`tests/test_manifest_problems.py`);
  - a dry-run resolves 24 IDs × 2 seeds = 48 generations at revision
    `740312a…`;
  - on the GPU host, CPU only and offline, `--resolve-only` selects all 24
    from the 1,319-row pinned pool, with non-empty gold answers
    (`results/diag/meeting5/gsm8k_timing/split_resolution.json`).
- *Not yet built:* the compressed-grid counterpart that M5A-CAPMATCH needs.
- *Launch preflight (2026-11-10, read-only):* see `docs/project/context.md`
  (2026-11-10 pilot-path bullet).
  It would reuse the same selection module.
- *To run, after approval:*
  1. a campaign (cap 3 GPU-h);
  2. a queue job and a `launch.sh` alias;
  3. `held: false` and `scale_authorized: true`;
  4. shared-disk headroom above the 30 GiB launch floor (20 GB free on
     2026-11-10).

## Scheduler integration (owner patch, 2026-11-10)

- Queue job `bank-gsm8k-timing-m5`: `blocked`, no campaign. Progress counts
  unique, expected row + sidecar pairs.
- `M5B-GSM8K-TIMING` (cap 3 GPU-h) is under `proposed_campaigns`, which the
  scheduler does not read.
- The `bank_manifest` launcher alias refuses without that campaign's capped
  reservation.

## Replication paths (prepared 2026-11-10; design and cost PENDING)

The paths are ready; the design is not. Both runners refuse real execution
until the config is unheld, authorized and `design_status: final`, and the
press family's own gate passes.
- *Compressed free runs:* `scripts/run_grid_manifest.py` uses the frozen
  confirmation IDs. Presses come from the config's single registered family
  (`kvcache.manifest_grid`): unprotected, S.1 protected, protected selector,
  or `rkv_pp_v1`. Each family uses its unchanged builder and gate. Rows are
  published as pairs through the M5 recoverable path.
- *Forward readout:* `scripts/run_readout_manifest.py` runs `run_replay.main`
  unchanged, after swapping in three things: the frozen-ID loader, the
  family's replay press factory, and a reference loader that reads live
  result files only. It refuses unless the full-cache reference bank has
  exactly one pair per expected key.
- *Reads:* `kvcache.manifest_reads`. Coverage is validated per expected key,
  and scores come from the registered `task_scoring.rescore`.
- *Readout convention (proposed; confirm with the design):* an undefined
  boundary is excluded and counted; an answer span that reaches
  `max_answer_tokens` counts as truncated, so incorrect.
- *Templates:* `grid_gsm8k_template_m5b.yaml` and
  `readout_gsm8k_template_m5b.yaml`, both held and pending, with placeholder
  presses, budgets and counts. Both are on `make dryrun`.
- *Tests:* 25 in `tests/test_manifest_replication_paths.py`.
- *Not built:* the full-cache confirmation bank config (the reference),
  reverse readout, and the decomposition read. All wait for the design.

## Approval and the BBH path (2026-11-13)

- *GSM8K pilot APPROVED* by the owner in-session (cap 3 GPU-h, normal window,
  after the real preflight): campaign `M5B-GSM8K-TIMING` active, job unblocked,
  config released together (PREREGISTRATION, same date). Calibration and
  costing only; never policy evidence.
- *Prompt fit (pre-outcome, CPU, tokenizer only):*
  `scripts/probe_manifest_prompts.py` →
  `results/diag/meeting5/replication/prompt_fit.json`. At B=256 every frozen
  calibration and confirmation problem of both datasets fits `P+64 <= B`
  (GSM8K 224/224, BBH 225/225), so `B_eff = B` everywhere and the protected and
  unprotected arms have exactly equal capacity with no relaxed budgets. Median
  prompt: GSM8K 62 tokens (confirmation), BBH 152; median trace slots at
  B=256: 194 and 104.
- *BBH calibration path (HELD):* `configs/experiments/bank_bbh_timing_m5.yaml`
  (25 frozen calibration problems × seeds 11/23 = 50, cap 4,096), queue job
  `bank-bbh-timing-m5` (`blocked`), its own capped launch alias
  `bank_manifest_bbh`, and the proposed campaign `M5B-BBH-TIMING` (2 GPU-h,
  under `proposed_campaigns`). The manifest runners load BBH from the
  blob-verified pinned file (`kvcache.manifest_problems.load_frozen_pool`);
  the timing reader and replication reads dispatch `bbh_grade` by benchmark.
  Resolution record (host, offline):
  `results/diag/meeting5/bbh_lde5_timing/bbh_split_resolution.json`.
  Tests: `tests/test_bbh_replication_path.py`.

## Pre-reasoned calibration branches (written 2026-11-13, before the pilot ran)

The pilot is calibration only. Its read (`timing_read.json`) selects the
design response below; no branch looks at any compressed outcome, and no
budget or dataset is chosen by which policy wins. The same rules apply to the
BBH pilot (cap 4,096).

1. **Truncation at the cap** (full cache, in-denominator).
   - ≤ 5% of the 48 traces: keep the cap (GSM8K 8,192) for every arm.
   - > 5%: a dated pre-outcome amendment raises the cap for **every** arm and
     the reference together (plan 9.3; never one arm alone), priced before
     approval.
2. **Is B=256 informative?** Use the share of full-cache traces with
   `P + T > B_eff` (`manifest_capacity.predicted_exposure`; every problem has
   `B_eff = B`).
   - ≥ 80%: B=256 is the primary budget, as planned.
   - 50–80%: keep B=256, and report the unexposed share as a low-exposure
     stratum in every table.
   - < 50%: B=256 is a low-exposure regime for this dataset. Report that
     plainly; do not move to a smaller budget without a separate,
     pre-outcome amendment.
   - B=1024 is secondary, and only where its share is ≥ 50%.
3. **Full-cache accuracy.** No branch. A near-ceiling reference (≥ 0.95)
   leaves little room for loss, which is reported as such; it is not a
   reason to drop the dataset.
4. **Grader sanity (instrument check, not a branch).** Any trace the frozen
   grader marks `no_answer` while its row is not truncated is inspected
   before the read is used. A systematic extraction failure is an instrument
   bug and stops the design (CLAUDE.md §2.8).
5. **Sample size.** Planning N stays 200 × 2 seeds; the pilot cannot measure
   a policy contrast's SD. Planning source: the M5-A MATH-500 strict cohort,
   restricted to seeds 11/23.
   - Problem-level SDs: protected 0.17–0.22, unprotected 0.31–0.40,
     interaction 0.36–0.43.
   - 95% half-widths at N=200: 0.024–0.030 protected, 0.043–0.055
     unprotected, 0.050–0.060 interaction.
   - 80%-power detectable effects (Holm over 2): about 0.04–0.05 protected,
     0.07–0.09 unprotected, 0.08–0.09 interaction.
   - The MATH-500 effects (unprotected gaps 0.18–0.19, interactions −0.14 to
     −0.19) are above these; the protected gaps there (−0.005 to +0.054) are
     at or below them, so a protected null is not evidence of equivalence
     unless TOST says so.

## What comes after (each a separate decision)

1. **Price the matched GSM8K replication.** Use the measured rate and
   lengths: the protected four-policy contrast plus full cache first, then
   the full protection factorial (Meeting 5 plan §9.2). The planning N is
   200 × 2 and is not guaranteed adequate.
2. **BBH logical-deduction.** The same steps on its 25-problem calibration
   split, with a proposed cap of 4,096.

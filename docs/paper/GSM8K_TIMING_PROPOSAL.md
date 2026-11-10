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

## What comes after (each a separate decision)

1. **Price the matched GSM8K replication.** Use the measured rate and
   lengths: the protected four-policy contrast plus full cache first, then
   the full protection factorial (Meeting 5 plan §9.2). The planning N is
   200 × 2 and is not guaranteed adequate.
2. **BBH logical-deduction.** The same steps on its 25-problem calibration
   split, with a proposed cap of 4,096.

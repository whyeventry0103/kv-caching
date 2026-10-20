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

## Blocker before it can run (CPU work, no approval needed)

`scripts/run_grid.py` selects problems by subset **count**, not by ID. The
frozen calibration split is a hashed ID list, so it needs a pid-filter grid
adapter. This is the same adapter the whole-cohort capacity comparators
(M5A-CAPMATCH) need.
- *Design constraint:* build it as an additive runner path. Changing
  `run_grid.py` would alter the generation identity of completed cells.
- *Readiness:* the adapter, its tests and the config come first; the timing
  request follows.

## What comes after (each a separate decision)

1. **Price the matched GSM8K replication.** Use the measured rate and
   lengths: the protected four-policy contrast plus full cache first, then
   the full protection factorial (Meeting 5 plan §9.2). The planning N is
   200 × 2 and is not guaranteed adequate.
2. **BBH logical-deduction.** The same steps on its 25-problem calibration
   split, with a proposed cap of 4,096.

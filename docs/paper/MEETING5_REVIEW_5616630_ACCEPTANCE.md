# Review of 5616630: repairs R1–R4, the P1 reverse read, and the calibration feasibility check (2026-10-16)

**Scope.** This pass covers the four findings of `MEETING5_REVIEW_5616630.md`
and the review's concrete actions 1–7. It is not a general audit:
- no generation was rerun and no GPU time was used;
- no approval or cap changed;
- the information calibration stays **HELD**.

The accepted B=1024 forward estimates, their condition table and the
registered forward labels are untouched.

## Results, kept separate

- **Software acceptance: passed.** The review's reproduction script and the
  new CLI/execute regression tests are covered below.
- **Scientific result, new:** the registered P1 reverse read at B=1024 (see
  below and PAPER_FINDINGS §5p).
- **Instrument feasibility (no weights): passed** on the host.

## Before / after (the review's reproduction script)

Evidence is in `results/diag/meeting5/review_5616630/`:
- `review_5616630_repro_before.jsonl`: the review's script
  (`review_5616630_repro_script.py.txt`), unmodified, in a clean worktree
  at `5616630`. It reproduces the review's observed output exactly.
- `review_5616630_repro_after.jsonl`: the archived
  `review_5616630_repro_script_adapted.py.txt` on the repaired
  tree. It differs from the original only in that the changed-filler resume
  catches the new refusal, so the script reaches its remaining cases, and in
  that ROOT points to the repo root.

| Case | Before (5616630) | After |
|---|---|---|
| Forward residuals and Δresidual (both policies) | reproduce exactly | reproduce exactly |
| Valid execution fixture | exit 0, FINAL, PASS | exit 0, FINAL, PASS |
| Changed filler, same config, resume | rows_written 0, skipped 2, design unchanged | **REFUSED** (design.json differs: actual selected inputs changed), nothing rewritten |
| 63 of 64 end-to-end measurements removed | exit 0, FINAL, PASS | **exit 2, INVALID** |
| Final end-to-end value NaN | exit 0, FINAL, PASS | **exit 2, INVALID** |
| Every compressed retained fraction −1 | exit 0, FINAL, PASS (C5 −1) | **exit 2, INVALID** |
| Full + four compressed arms all at floor | exit 0, FINAL, PASS (C6 range ≈ 2 bits) | exit 0, FINAL, **calibration FAIL [C6]** |
| Compression read, one row per cell, no S.1 evidence | exit 0, all eight summaries written | **exit 3, PARTIAL, no numbers** |

## R1: calibration measurements are validated before PASS

- *Shared validation.* `kvcache.task_information.measurement_problem` runs for
  every production row inside the shared reader, before any aggregation:
  - `n_prunes` is a non-negative integer, and 0 on the full arm;
  - `fact_spans` equal the design's frozen schedule;
  - both retained fractions are finite and in [0, 1] on every compressed row,
    and null on the full arm;
  - every row of the scheduled end-to-end query (the LAST query, as the
    producer measures it) carries a finite non-negative difference, and no
    other row carries one.

  Violations make the read INVALID (exit 2). Booleans, NaN, negatives and
  missing values are all refused.
- *Frozen schedule.* The design freezes `measurement_schedule`
  (`kvcache.information_probe.measurement_schedule`, built from the rendered
  variants).
- *Coverage.* A compressed arm that never pruned keeps no position maps. Its
  zero prune count is recorded as the evidence that both spans are fully
  retained (1.0), so retention cannot silently drop out of the mean.
- *Criteria binding.* The design embeds the calibration criteria and their
  fingerprint, and the reader evaluates those. `--calibration-criteria` must
  name identical criteria; a loosened file is INVALID. The report carries
  `dependencies` (input-manifest and criteria fingerprints).
- *Validity vs acceptance.* Malformed data → INVALID. A complete valid channel
  that misses a criterion → FINAL with calibration FAIL (the no-eviction
  fixture still FAILs C5/C6). Downstream acceptance is
  `information_calibration.accepted(read)` (FINAL and PASS);
  `--require-calibration-pass` exits 4 otherwise.
- *Defensive evaluator.* The evaluator also refuses subsets, NaN and missing
  retention rather than skipping them, and it refuses a criteria set that is
  not exactly the frozen keys.

## R2: the frozen design binds the actual inputs

- *Manifest.* `kvcache.information_tasks.input_manifest` is built before any
  weights load and records:
  - each filler passage (source, length, token-id hash);
  - each block (aliases, order, filler segments, skipped sources);
  - token-id hashes of every rendered instruction, body and query per variant;
  - the symbol ids.

  It is stored as `input_manifest.json`. Its hash is in the design and every
  row's provenance (`input_manifest_sha256`), and the reader verifies the
  stored manifest against it.
- *Resume.* A resume refuses before model loading or any write when:
  - the design or manifest differs;
  - rows exist without their design and manifest;
  - a stored row has another identity;
  - a stored block is incomplete or duplicated (`finished_blocks`).
- *Acceptance through `execute`.* CPU tests show:
  - an unchanged-input resume is byte-identical;
  - changed filler refuses, both after a complete run and after a partial run,
    with the model loader instrumented to fail if called and the run directory
    byte-identical afterwards;
  - an unchanged partial run completes only the missing block;
  - a truncated stored block refuses;
  - a manifest with the same sources but different token content hashes
    differently.

## R3: C6 floor (pre-data correction)

C6 now also requires at least one compressed arm at ≥ `floor_mi_bits` = 0.1
bits (5% of the 2-bit maximum). The full-to-compressed, compressed-only and
all-arm ranges are reported separately. No policy order is required: a test
confirms that two equal, informative compressed arms PASS.

This is recorded as a dated pre-data correction in the information protocol
§6.8/§9 and the config comment.

## R4: compression summary bound to S.1 evidence; labels narrowed

- *Evidence binding.* `scripts/report_realized_compression.py` publishes only
  when, for each budget:
  - the S.1 protection read is FINAL;
  - its conditions table matches its hash;
  - its full dependency fingerprint equals the one recomputed now
    (`report_s.protection_dependencies`, as `consequence` does).

  Missing evidence gives PARTIAL (exit 3); stale evidence or malformed rows
  give INVALID (exit 2). In neither case are numbers published.
- *Row checks.* Rows must be exactly the S.1 design's conditions. Every stored
  ratio is cross-checked against min(1, B / (prompt_len + n_tokens)) from the
  recorded prompt length.
- *Re-read on the host: FINAL* (fingerprints match at both budgets). Every
  value equals the earlier artifact. Labels are now:
  - nominal-budget ratio;
  - `n_ratio_one`, a lower bound on never-compressed conditions (31/400 at
    B=1024);
  - effective-budget ratios for the 12 relaxed protected conditions per
    policy at B=256 (means 0.1271 / 0.1389 against nominal 0.1269 / 0.1387;
    medians unchanged).
- *Docs.* PAPER_FINDINGS §5o and PREREGISTRATION were corrected visibly.

## Actions 5–7

- **5. Feasibility without weights** (host, `run_information_probe.py
  --check-inputs`, pinned tokenizer from the project cache, offline):
  - 83 eligible reserve passages against 40 needed;
  - no tokenizer problems;
  - 4 blocks and 64 variants rendered, every body exactly 1,024 tokens;
  - 0 leak-screen skips;
  - fact spans 64–69 and 768–773, alternating order by block;
  - input-manifest hash `1f527da6…`.
- **6. P1 reverse read:** FINAL. The registered t test + Holm does not reject
  for either policy (Holm p 0.64 each). The trajectory contrast excludes 0
  under both readout directions, and the readout-change contrasts are about 0
  (PAPER_FINDINGS §5p). The Qwen hold is released per its registered
  condition, behind both M5 grids in queue order.
- **7. Queue:** grid-rkvpp256-m5 is running (358/400 at 00:17); the H2O-style
  grid follows in queue order. Protected M/Δ/ω captures remain separate
  future work.

## Gates

`make lint`, `make test`, `make dryrun` and `make audit` were each run bare,
followed by `git diff --exit-code -- paper_numbers.json` after regeneration.
The catalogue collision ratchet was raised from 164 to 172 after examination
(documented in the test).

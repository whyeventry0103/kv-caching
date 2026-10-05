# Proposals for approval (prepared 2026-10-05; nothing here is authorized)

**Costs.** From `results/diag/paper_plan/cost_estimate.json`. They are estimates built from measured historical rates plus stated assumptions (readout at 0.6–1.0× its source trace's free-run rate; new-family overhead 1.0–1.5×; 0.1 h setup per job launch). They are not caps with measured safety coverage.

**Order recommended by the review of fbf869f:**
1. finish the authorized B=1024 work;
2. P1;
3. P2 and P3 to measure;
4. only then a campaign.

## P1 — protected B=1024 reverse readout (highest priority)

- **Question.** Does the post-protection forward/reverse interaction (+0.130 / +0.127 at B=256) persist at a less extreme budget?
- **Design.** `configs/experiments/reverse_readout_pp1024_s1x.yaml` (held, not queued).
  - Source traces: the protected budget-1024 free runs (`grid-pp1024`), read under the full cache.
  - Scope: 100 problems × 4 seeds × 2 policies = 800 rows.
  - Runner: `scripts/run_replay.py` unchanged; dry-run is in `make dryrun`.
- **Dependency.** `grid-pp1024` complete (474/800 at 11:07).
- **Frozen before generation:**
  - Primary: protected interaction per policy on the common support. EP estimator, joint 2000-resample problem bootstrap, seed 0.
  - Secondary: worst-case bounds, both one-direction contrasts, the difference of protection improvements (needs the unprotected B=1024 reverse arm, which is NOT proposed — say so if the comparison is wanted).
  - Missingness: undefined readouts stay in the population, with bounds.
  - Multiplicity: two policies, Holm.
  - Stopping: read only when 800/800 FINAL.
- **Cost.** 800 readouts of b1024 traces. At the measured b1024 free-run rate (87.3 s/row) × 0.6–1.0: **≈ 11.6–19.4 GPU-h + ~0.2 h setup, ≈ 1 night.** Storage ≈ 6 MB.
- **Approval needed.** Yes (new cell outside S.1).

## P2 — WP4 real-generation smoke (needed before any WP5 row)

- **Design.**
  - Traces: 3 (longest, median and shortest prompt).
  - Families × arms: 3 families × (1 free run + 1 forward readout), plus the `h2o_style_pp_v1` identity control (`forced_prompt_len=0` reproduces `h2o` selections). 21 rows.
  - Checks at every prune: prompt (and buffer) retained, no duplicate positions, exact target cardinality.
  - On pass: write `protected-selectors-smoke-PASS.json`.
- **Cap.** 2 GPU-h.
- **Code still to write.** The smoke runner (pattern: `scripts/pp_smoke.py`). CPU work, not started.

## P3 — engineering pilot (timing)

- **Design.** 8 fixed problems × 1 seed × 2 new families × (free, forward, reverse) = 48 rows. Excluded from every claim. Purpose: per-row wallclock including per-prune Python selection.
- **Cap.** 6 GPU-h.
- **Output.** Replaces the 1.0–1.5× assumption in `configs/paper/cost_inputs.yaml`.

## Campaign estimates (decide after P1–P3; all HELD)

| Campaign | GPU-h (low–high) | Nights at full capacity |
|---|---|---|
| WP5 protected selectors, Qwen-7B, B=256, 100×4 | 74–142 | 4.1–7.9 |
| WP6 Qwen3-4B, 100×2 | 100–188 | 5.6–10.4 |
| WP6 Qwen3-4B, 300×2 | 299–561 | 16.6–31.2 |
| WP6 Qwen3-4B, 300×4 | 596–1121 | 33–62 |
| WP6 GSM8K (Qwen-7B), 100×2 | 17–63 | 1.0–3.5 |
| WP6 GSM8K (Qwen-7B), 300×4 | 98–371 | 5.4–20.6 |

**Scaling.** At fixed stages, conditions scale with problems × seeds: 300×4 is **6×** the conditions of 100×2, not 3×.

**Precision is endpoint-specific** (`sample_size.json` → `endpoint_summary`), at 300×4:

| Endpoint (Random / StreamingLLM) | Half-width | Power for a 5-pt effect | Problems for ±0.025 (normal approx.) |
|---|---|---|---|
| Protected forward residual | 0.024 / 0.026 | 97% / 96% | 279 / 326 |
| Protected interaction | 0.031 / 0.037 | 89% / 75% | 473 / 663 |
| Protected free-run gap | 0.042 / 0.045 | 63% / 56% | 853 / 998 (exceeds MATH-500) |

**Choose the primary endpoint first, then the design.** These numbers borrow Qwen-7B/MATH-500 variance and are not promises for Qwen3 or GSM8K.

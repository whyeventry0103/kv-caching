# Acceptance report for the validation-and-reporting repair pass (2026-10-02)

Scope: the independent review of `f69dce3` (reviewed 2026-10-01 UTC). One
bounded CPU pass: readers repaired behind one validation boundary, the five
reproduced failures tested through the real CLI, existing raw data re-read,
claims corrected. No new experiment family, no historical generation identity
changed, nothing deleted, no storage change, no out-of-window run. The
approved S.1 / S.2 / Llama / Qwen jobs kept running on their schedule.

## 1. Reproduced failure → fix → production-path test → re-read result

| # | Reviewed failure | Reproduced as | Fix | Test (real CLI, `tests/test_s_acceptance.py`) | Re-read on the host |
|---|---|---|---|---|---|
| 1 | `origin` labelled PROMPT CARRIES IT from one Random row with no provenance and failed gates, exit 0 | fixture mutation: one row of 200 | `read_validation`: coverage against the frozen design over every row; provenance-v1 identity per row; arm gates and TV domain per READ row; INVALID exit 2; ratios withheld | `test_review_failure_1_one_row_without_provenance_or_gates_is_invalid` | restore-v3: **FINAL**, 200/200, 0 malformed, all controls passed |
| 2 | `determinism_control` passed NaN vs finite and inf vs inf | unit + CLI mutation | non-finite or out-of-domain values fail closed, named in the output | `test_review_failure_2_nan_and_inf_fail_the_determinism_gate` | max \|ΔTV\| = 0.0 on 91 + 91 conditions, eligible sets identical |
| 3 | `protection` took a row count for coverage; a duplicate undefined key replaced a missing one and the read was FINAL | fixture mutation | coverage over every row incl. undefined; duplicates and missing keys named; INVALID exit 2 | `test_review_failure_3_duplicate_undefined_key_is_invalid_not_final` | S.1 b256: grid_pp / grid_base / readout_base COMPLETE, readout_pp PARTIAL → **PARTIAL, no label** |
| 4 | missing prompt sidecars → empty exclusion list, sensitivity became primary | fixture mutation | prompt lengths required for every pid from both grids and must agree; INVALID otherwise | `test_review_failure_4_missing_prompt_metadata_fails_closed` | prompt_metadata control passed (100/100 pids) |
| 5 | both protected channels at 0 reached branch C (prompt loss) | fixture mutation | floor/ceiling → UNINFORMATIVE (exploratory floor 0.05, dated, not registered); consequence reaches no branch; NOT DETECTED wording; branch C "consistent with … not established" | `test_review_failure_5_floored_channels_are_uninformative_not_prompt_loss` | n/a until the readout lands |
| — | stale FINAL outputs reusable after a failed/changed read | fixture | outputs retired before every write; `consequence` recomputes the input hash and refuses a changed input | `test_stale_final_is_retired_and_consequence_refuses_changed_inputs` | — |
| — | R.17 restoration reader without the boundary | — | same boundary + exit contract | `test_r17_restore_reader_has_the_same_boundary` | restore-v2: **FINAL**, numbers unchanged |
| — | R.17 capture reads (mdelta, omega-layers, capture-v2) without the boundary | — | `_validated_capture`: design coverage, provenance against the config's own manifest and verified basis, live-control flags; labels withheld unless FINAL | covered by the b256/b2048/Llama re-reads below | b256 FINAL, b2048 FINAL (labels unchanged), Llama FINAL |

Exit contract: 0 FINAL; 3 PARTIAL (0 with `--partial`); 2 INVALID. A
PARTIAL read writes its numbers and no label.

## 2. Re-read status of existing raw data (CPU only; no GPU regenerated)

| Read | Expected | Observed | Analysed | Status | Controls | Output (hash in file) |
|---|---|---|---|---|---|---|
| restore-v2 (R.17.2.3) | 200 | 200 | per policy: 92 READ (91 with the full-restoration window, 1 whose window does not fit) + 8 ineligible (corrected 2026-10-03; was mis-typed "109 ineligible") | FINAL | provenance 200/200, gates | `results/diag/meeting3/r17/restore_v2.json` |
| restore-v3 (S.2) | 200 | 200 | 91 READ, 8 + 1 ineligible per policy | FINAL | provenance v3 + v2, gates, determinism exact | `results/diag/amendment-s/origin.json`, `origin.conditions.tsv` |
| S.1 budget 256 | 400 per cell × 4 row sets | grid_pp 800, grid_base 800, readout_base 800, readout_pp 0 | 400 free pairs per press, 0 readout pairs | PARTIAL | reference, prompt metadata, legacy provenance | `protection_b256.json`, `.conditions.tsv` |
| capture-b256 / b2048 (R.17.1.5) | 400 | 400 | 400 / 276 | FINAL | provenance, live controls | `r17/mdelta*.json` |
| capture-llama8b-b256 (R.17.3) | 400 | 400 | 400 | FINAL | provenance (Llama manifest, verified basis), controls VALID | `capture_llama8b_b256.json`, `r17/*_capture-llama8b-b256.json` |

Legacy rows (the S.1 grids and readouts, the original Meeting 3 cells) carry
`runmeta` and `versions` but no provenance-v1 identity or manifest hash; the
reader reports that limitation and matches them to the full-cache bank by
key. Historical generation identities are untouched.

## 3. Claims: supported, corrected, still uncertain

**Supported by the re-read data**
- The budget-256 free-run accuracies (0.4225 / 0.2375 / 0.2475 / 0.5200) and
  the readouts reproduce from the raw rows, 3,200 rows re-scored with 0
  mismatches against the project's grader.
- Random's advantage over StreamingLLM and H2O-style is carried by the
  readout channel (paired CI excludes 0); the paired residual difference is
  not detected (CI [−0.08, +0.09]).
- S.2: for Random, the prompt-plus-padding arm (all 91 eligible rows padded
  with 1–25 recent trace entries) outperforms an equal-size random draw of
  trace entries (ρ_prompt − ρ_trace +0.147 [+0.083, +0.200]) and does not
  carry most of the restorable divergence (ρ_prompt 0.367 [0.310, 0.419]); a
  per-entry advantage of prompt entries alone is not isolated. For
  StreamingLLM a difference is not detected (−0.030 [−0.090, +0.025]).
- Llama-8B: the ω contrast **reverses** (−0.355 [−0.453, −0.275]), so the
  Qwen ordering does not generalise to this second model and setup (the
  cause is not isolated); the H3 attention-mass contrast holds (−0.577
  [−0.602, −0.543]) but carries no damage interpretation there (H2O-style
  not run free on Llama; Random, which loses most mass, has the lowest
  accuracy); H2 NEITHER; collateral 0.

**Corrected wording (MEETING4_ANSWERS, PAPER_FINDINGS §5i–5j)**
- "not a bug" → "the numbers reproduce; nothing we checked is a bug", with
  the grader-consistency limitation stated; 3,200 rows, not 2,800.
- Residual = descriptive decomposition; policy differences with paired CIs.
- Rankings on the matched 25-problem and 2048-eligible keys added.
- restore-v2's R_fs returns entries AND stops eviction; only S.2's frozen arm
  holds the stop constant.
- ω: "association not established", never "shown not to predict"; ΔR² in the
  collapse test is descriptive.
- Collateral 0 ≠ head agreement; the Llama 4-to-1 bound is tighter (1/400).
- "Random" = kvpress `RandomPress` with per-prune reseeding; the retention
  simulation is committed with torch version, device (CUDA and CPU), dtype
  and seed (`docs/results/meeting4_evidence/retention.json`). "Nothing
  permanently" removed: an evicted entry stays absent unless restored.

**Still uncertain (not established at this n, or not yet read)**
- Whether the residual survives prompt protection (S.1: readout pending at
  256; budget 1024 queued). No consequence branch is reachable yet.
- Whether the prompt-information account explains the accuracy gap: S.2
  supports it in part for Random only; S.1 is the accuracy-scale test.
- Any generality of the ω ordering (it holds on one model and reverses on
  the second).
- The Qwen3-4B cells (bank 150/200; the bank gate [0.85, 0.99] is read
  before any cell).

## 4. Jobs

| Job | State |
|---|---|
| grid-pp256 | 800/800 |
| readout-pp256, reverse-readout-pp256, grid/readout-pp1024 | queued (next window 01:45 Dubai) |
| restore-v3 | 200/200, READ |
| capture-llama8b-b256 | 400/400, READ |
| fc-qwen3-4b → grid/readout/reverse | bank 150/200, rest queued |

## 5. Gates and remotes

Software gates: `make lint`, `make test` (617 passed, 77 skipped),
`make dryrun`, `make audit`, catalogue in sync, `make parity` OK after sync.
These are code and bookkeeping checks. They do not change the experimental
validity gate (R.17.0.5), which still reads **FAIL** for Random at interval
256 (ratio 2.006 against 2.0); its caveat on every divergence number stands. Pushed to
`origin/main` and `backup/main`; both resolve to the commit named in the
final message.


---

# Second review (of `49c572c`), closed 2026-10-03

The independent review of `49c572c` confirmed the five original fixes and
reproduced the new numerical results, and found three remaining boundary
gaps plus six wording issues. Same scope as before: CPU only, no GPU change,
no new experiment family, nothing deleted, no out-of-window start.

## Acceptance table

Every row was reproduced through the real CLI on the reviewed readers first
(14 of the 15 new tests fail on `49c572c`; the 15th is the valid-fixture
positive control), then fixed. All tests are in `tests/test_s_acceptance.py`.

| # | Counterexample | Fix | Test(s) | Real-data re-read |
|---|---|---|---|---|
| 1a | one READ row → `FAILED: injected`: exit 0, FINAL, PROMPT CARRIES IT | `read_validation.restoration_row_problems`: status must be one the producer emits | `test_issue1_injected_failed_status_is_invalid` | origin FINAL |
| 1b | an exclusion the trace contradicts (3,000-token trace "never reaches the trigger") | exclusions re-derived from each row's trace length, cohort step and the config's trigger / interval / window | `test_issue1_illegitimate_exclusion_is_invalid` | all 18 real exclusions verified legitimate |
| 1c | eligible-set mismatch with v2 recorded but not enforced | in a complete read the v3 READ set must equal v2's eligible set (subset if incomplete); part of `passed` | `test_issue1_eligible_set_mismatch_in_a_complete_read_is_invalid` | sets equal for both policies |
| 1d | failed v2 `full_restore` gate accepted by origin | every v2 comparator row through the same validator the v2 reader uses | `test_issue1_failed_v2_comparator_gate_is_invalid` (origin and `report_r17 restore`) | restore-v2 rows control passed |
| 2a | torn JSON after a FINAL: exit 1, old FINAL left in place | output (JSON and TSV) retired before analysis; failures caught, INVALID evidence written, exit 2 (both readers) | `test_issue2_truncated_json_after_final_leaves_no_final`, `test_issue2_schema_failure_is_invalid_evidence_not_a_crash` | — |
| 2b | consequence FINAL after a prompt sidecar was deleted | one dependency fingerprint (`read_validation.dependency_fingerprint`): rows, token sidecars, full-cache bank, v2 comparators, configs, frozen pid lists and manifest, models.yaml, analysis code; protection stores it, consequence recomputes it and names the changed group | `test_issue2_consequence_refuses_changed_dependencies[deleted_prompt_sidecar, changed_full_cache_comparator]`, `test_issue2_changed_analysis_code_invalidates_stored_reads`, `test_issue2_origin_fingerprint_covers_v2_and_full_cache` | protection_b256 and origin carry fingerprints |
| 3 | R.17 capture read FINAL / H3 HOLDS with `layer0_delta_max = 1.0` | `_validated_capture` reuses `report_capture.control_failures` (the strict reader's own) and adds domain checks on every consumed metric | `test_issue3_capture_controls_and_metrics_fail_closed[layer0_failure, nonfinite_metric, out_of_domain_M, negative_omega]`, `test_issue3_complete_capture_fixture_is_final` | see below |

**Found while re-reading (issue 3).** The first real-data run of the new
metric checks marked capture-b2048 INVALID: 148 rows carried slightly
negative attention-mass sums. M = 1 − exp(lse_kept − lse_all); at budget
2048 nearly all mass is kept, so the two log-sum-exps agree to fp32 rounding
and M lands a hair below 0. The worst value was −2.7e-6 per measurement;
none occur at budget 256 or on Llama. I counted them first and then set a
per-measurement tolerance of 1e-5, reusing the registered layer-0 Δ
tolerance (R.9). This was a choice made after seeing the data, and it is
stated as one. Values are never clipped, and anything beyond the tolerance
(a planted −1.0, or a mean of 1.2) is INVALID
(`test_issue3_mass_rounding_tolerance_accepts_fp32_noise_not_real_violations`).

## Real-data re-read (CPU, 2026-10-03), against the pre-repair outputs

| Read | Status | Numeric fields compared | Changed | Labels changed |
|---|---|---|---|---|
| S.2 origin | FINAL | 491 | 0 | 0 |
| S.1 protection b256 | PARTIAL (readout 789/800), no label | — | — | — |
| R.17 restore (v2) | FINAL | 597 | 0 | 0 |
| mdelta b256 / b2048 / Llama | FINAL / FINAL / FINAL | 1,416 / 1,416 / 1,608 | 0 | 0 |
| omega-layers b256 / b2048 / Llama | FINAL / FINAL / FINAL | 307 / 307 / 347 | 0 | 0 |
| capture-v2 extras b2048 | FINAL | 70 | 0 | 0 |
| capture-v2 extras Llama | not applicable (no nested basis fitted for Llama) | — | — | — |

Every estimate and label is unchanged; the repair changed what the readers
accept, not what they compute.

## Wording corrected (MEETING3_RESULTS §3e/§3f, PAPER_FINDINGS §5i/§5j, MEETING4_ANSWERS, the mechanism draft, this report)

1. S.2 Random: "the prompt-plus-padding arm outperforms an equal-size random
   trace draw"; every one of the 91 eligible rows was padded (1–25 entries),
   so a per-entry prompt advantage is not isolated.
2. S.2 StreamingLLM: "a difference is not detected", not "the prompt is not
   special".
3. Llama ω: "does not generalise to this second model and setup"; the
   comparison does not isolate architecture as the cause.
4. H3 on Llama is an attention-mass contrast only. On the 25 capture
   problems Llama Random scores 0.00 and StreamingLLM 0.16 (free run), and
   H2O-style was never run free on Llama; for the measured pair, mass and
   damage rank the same way. The damage-objective reading stays a Qwen
   statement.
5. Software gates passing is distinct from the experimental validity gate,
   which still FAILS (Random I=256, 2.006 vs 2.0).
6. restore-v2 counts: per policy 92 READ (91 with the full window) + 8
   ineligible, not "109 ineligible".

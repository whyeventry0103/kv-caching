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
| restore-v2 (R.17.2.3) | 200 | 200 | 91 READ + 109 ineligible per policy | FINAL | provenance 200/200, gates | `results/diag/meeting3/r17/restore_v2.json` |
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
- S.2: for Random, prompt entries recover more per entry than trace entries
  (ρ_prompt − ρ_trace +0.147 [+0.083, +0.200]) and do not carry most of the
  restorable divergence (ρ_prompt 0.367 [0.310, 0.419]); for StreamingLLM the
  prompt is not special (−0.030 [−0.090, +0.025]).
- Llama-8B: H3 holds (−0.577 [−0.602, −0.543]); the ω contrast **reverses**
  (−0.355 [−0.453, −0.275]); H2 NEITHER; collateral 0.

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
- Any generality of the ω ordering (now one architecture).
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

`make lint`, `make test` (617 passed, 77 skipped), `make dryrun`,
`make audit`, catalogue in sync, `make parity` OK after sync. Pushed to
`origin/main` and `backup/main`; both resolve to the commit named in the
final message.

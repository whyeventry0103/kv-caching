# Acceptance: review of cfa30cd

- **Tested implementation:** commit `58d4dc8` (generation-side part in `0a86cbe`).
- **Pinned stack:** torch 2.7.1, transformers 4.56.2, kvpress 0.5.3, CPU on the
  GPU host. Implementation ids per family are in `remote_pytest.txt`.
- **Totals:** 162 acceptance/regression tests PASSED and 0 FAILED by name; full
  suite 570 passed.

Every row below tests the **production path**, never a detached helper:
- real reader subprocesses through `postprocess_meeting3.run_stage`;
- the scheduler's `count_progress` / `launch_candidates` / `stage_done`;
- the real runners on a tiny model.

| # | Finding | Fix | Production-path test | Observed |
|---|---|---|---|---|
| A | exact-lag and fork readers had no entry point; scheduled subprocess exited 0, did nothing | `__main__` restored; every script checked | `test_every_stage_finalizes_through_its_real_reader[exact-lag]`, `[fork-control]` run the files as subprocesses | both FINAL; outputs, TSV and stage record written |
| A | only capture was tested end to end | all six stages through real subprocesses with provenance-correct synthetic trees | `test_every_stage_finalizes_through_its_real_reader[capture-b256, capture-b2048, dacc, exact-lag, fork-control, t1-tv]` | 6/6 FINAL, figures present where required, manifest refreshed, stage accepted |
| B | scheduler accepted any parseable marker (`{}` → done) | typed gates `kvcache.gates` in `count_progress`; expectation from committed specs | `test_invalid_dacc_smoke_markers_never_satisfy_the_gate[absent, empty, malformed, empty-object, failed, no-identity, mismatched]`, `test_a_stale_pass_from_other_generation_code_is_rejected` | none satisfy; dependents not launched |
| B | basis header trusted without its bytes | basis gate verifies npz, hash and fit identity | `test_a_corrupted_basis_behind_a_valid_header_is_not_a_finished_fit`, `test_a_basis_of_another_identity_is_rejected_and_a_true_one_accepted` | corrupt/foreign basis rejected; `capture-smoke` not launched |
| B | smoke identity not checked | capture-smoke identity must include the verified basis bytes | `test_capture_smoke_identity_must_include_the_verified_basis` | wrong basis sha rejected; later basis corruption invalidates the smoke |
| B | real markers | same gates on the GPU host overnight | real `dacc-gate-smoke`, `q6-basis-fit`, `capture-smoke` | all accepted; dependents launched (log) |
| B | provenance `{anything: "same"}` accepted | `provenance-v1` required fields + expected identity + frozen input manifest | `test_consistent_but_meaningless_provenance_is_invalid`, `test_row_provenance_requires_schema_fields_identity_and_inputs`, `test_foreign_inputs_or_no_expected_identity_are_invalid` | INVALID, no verdict |
| B | input identity not checked at resume | runners check the manifest at generation and resume | `test_a_resumed_row_with_foreign_inputs_is_refused`, `test_a_changed_reference_bank_is_refused_at_generation` | REFUSED |
| C | exact-lag counterexample (lags `{}`, stash `{passed: false}`) was FINAL | full content checks | `test_exact_lag_row_checks_reject_the_reviewed_counterexample`; stage tests `[lag-payload-missing]`, `[restoration-gate-false]` | INVALID, terminal, no figure |
| C | fork: infinite control → tolerance ∞ → PASS | evidence validated before classification | `test_malformed_fork_evidence_is_invalid_before_classification`; stage test `[fork-infinite-control]` | INVALID-EVIDENCE, terminal, not passed |
| C | D_acc control RUN treated as success | NOT RUN / RAN-INCOMPLETE / FAILED / PASSED; FAILED keeps its registered consequence | `test_determinism_ran_is_not_passed`, `test_cli_final_is_nonzero_without_the_determinism_control`; stage test `[dacc]` requires PASSED | as specified |
| C | duplicates/unexpected gave PARTIAL | INVALID for duplicated/unexpected/malformed/mismatched; PARTIAL only for missing | capture/D_acc/fork/exact-lag reader tests; `test_a_missing_condition_stays_partial_and_is_retried` | as specified |
| D | stale FINAL + failing reader → FINAL | fresh attempt directory; status from the current report and exit code only | `test_an_old_final_followed_by_a_reader_failure_is_not_accepted` | READER-FAILED, old report untouched, not accepted, retried |
| D | terminal marker before manifest; manifest exit not persisted | stage record → manifest → terminal marker; exit persisted | `test_manifest_refresh_failure_is_retryable_and_never_terminal` | MANIFEST-FAILED, not accepted; retry finalizes; history kept |
| D | acceptance ignored outputs, code and inputs | `stage_done` checks output hashes, analysis identity and input-row digest | `test_changed_rows_or_analysis_code_invalidate_acceptance` | re-run required |
| D | inconclusive fork promoted as success | FINAL analysis with separate `gate_outcome`; manifest shows both | stage test `[fork-control]`; manifest `_evidence_status` | analysis FINAL, gate reported separately |
| — | scheduler retries without duplicating generation | CPU stages retried each tick; GPU jobs resume only needy seeds | scheduler stage tests; `queue_dryrun_tick.txt` | next window resumes capture-b256 seeds 23 and 89 only |
| — | restarts | persistence-layer restarts (R.13) | `test_restart_after_each_condition_and_after_the_last_row`, `test_old_pass_then_failure_leaves_no_accepted_pass` | pass |

## Inspection against the requirements (not only test counts)

- **The scheduler's production call path** was inspected. `count_progress`
  handles `marker` jobs with `gate` through `kvcache.gates.gate_status`, and
  `postprocess` jobs through `postprocess_meeting3.stage_done`.
  `launch_candidates` and `select_job` skip `cpu_only` jobs. `main()` calls
  `run_cpu_stages` before the window check.
- **All six scheduled reader commands** respond to `--help` and run end to end
  as subprocesses.
- **Real data.** On last night's rows the strict readers report capture-b256
  255/400 PARTIAL and dacc 200/1200 PARTIAL. Both have 0 provenance failures,
  VALID controls and withheld verdicts. No valid generation was invalidated by
  this pass: no generation-source file changed after the window started.

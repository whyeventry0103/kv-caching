# Execution status (one row per work package)

Specification: `docs/paper/EXECUTION_PLAN.md` (v1.0, baseline `1b644dc`).
Claims: `docs/paper/CLAIM_LEDGER.md`.
Approvals pending: `docs/paper/PROPOSALS_FOR_APPROVAL.md`.

Snapshot **2026-10-05, about 20:00 Dubai** (host clock 19:39 when checked). The bounded repair of the independent review of `fbf869f` is applied. Both GPUs are idle and no jobs are running; the next cron window opens at 01:45.

| WP | State | Artifacts | Remaining |
|---|---|---|---|
| WP0 freeze + single status | Documentation landed. **The paper remains a draft.** | This file; the ledger; paper edits (title, 13 bracket-wording edits, sub-floor TV qualification, abstract now states the interaction) | Rewrite the abstract's structure around the final question after B=1024 (do not just append caveats) |
| WP1 existing queue | **In progress** | Protected reverse b256 (`reverse_b256.json`, FINAL) | grid-pp1024 (474/800) then readout-pp1024 (0/800), both in the normal window. Then `report_s.py protection --budget 1024`, `consequence`, and the practical-margin read at 1024 |
| WP2 Qwen3 scoring | **Audit done (v2); human review and disposition open** | `grading_sensitivity.py` v2; `bank_audit_*` (v2), `*_v1.*`; `human_review_v2.tsv` (42 rows) | A person fills in the review sheet. Owner and supervisor choose option (a)/(b)/(c); the memo and the review both recommend (a) as a documented exception |
| WP3 no-GPU analysis | **Done, repaired** | `decision_b256`, `matched_b256`, `reverse_b256` (re-read after R1/R4: 0 numeric changes); `RELATED_WORK_MATRIX.md` (with a verbatim evidence record) | Recheck novelty gate N after B=1024 |
| WP4 new capability | **CPU integration done; not yet runnable end to end** | `protected_selectors.py`; runner adapters `run_grid_protected.py` and `run_replay_protected.py` (they refuse real runs while held or without the smoke marker); provenance family `protected_selectors`; held configs, dry-run in `make dryrun` | Smoke runner (CPU, not written); real-generation smoke (P2, needs approval) |
| WP5 protected selectors | **Held** | Manifest, held configs | After P2 (smoke) and P3 (timing pilot) |
| WP6 confirmation | **Held** | Manifests; `sample_size.json` (endpoint-specific, includes the interaction); `cost_estimate.json` (corrected readout pricing, seeds parameterised, overheads) | Choose the primary endpoint, then N. Qwen3 at 300×4 is about 600–1,100 GPU-h, which does not fit the nightly windows |
| WP7 writing | **Drafting** | `DRAFT_METHODS_RELATED_FIGURES.md` | Figures 2 and 5 from existing B=256 data |

## Decisions that need a person

1. **Qwen3 gate disposition** (memo §4).
2. **P1, B=1024 reverse readout:** about 12–19 GPU-h, roughly one night. The review rates it the highest-priority new experiment.
3. **P2 smoke and P3 timing pilot:** ≤2 and ≤6 GPU-h, needed before any WP5 costing.
4. **GSM8K:** pin the revision (`docs/protocol/GSM8K_REVISION_PROPOSAL.md`; candidate `740312a…`) and authorize the download.

## Review of fbf869f: repairs and their effect

| Item | Repair | Evidence | Effect on results |
|---|---|---|---|
| R1 reader validation | Values are checked before any `bool()` call, and readout flag/value consistency is checked. Bank rows are validated as an allowed superset, with duplicate detection. `matched` validates free-run rows against the frozen design (missing → PARTIAL, duplicate → INVALID), lists declared metric exclusions, and carries a dependency fingerprint. The same coercion was fixed in `report_s.py` (protection, origin) and is named separately | `tests/test_protocol_audit_cli.py`: 12 real-CLI cases, 10 failing before the fix; `test_s_acceptance` gains 2 cases, also failing before | Re-read on the host: **0 numeric changes** in protection_b256, origin, reverse_b256, matched_b256, decision_b256; TSVs byte-identical |
| R2 secondary grader | v2: positive unordered wording required; the multi-box check reads the whole suffix | 5 new tests fail on v1 and pass on v2 | **0 rows changed** in either bank |
| R3 planning and cost | Readouts priced at their source trace; seeds parameterised; interaction endpoint added; overheads, smoke and pilot made explicit; scaling corrected to 6× (was "3×") | `cost_estimate.json`, `sample_size.json` | WP5 now 74–142 GPU-h (was 73–94). WP6 at 100×2: 100–188 GPU-h (Qwen3) |
| R4 sign table | All four quadrants covered | 7 tests | Random − H2O label unchanged (MISMATCH: first policy higher accuracy, more loss) |
| Interpretation | Paired difference of absolute protection improvements added | `reverse_b256.json` `difference_of_improvements` | Random +0.078 [+0.007, +0.149]; StreamingLLM −0.007 [−0.081, +0.069], not detected. "Protection works mainly forward" retired |

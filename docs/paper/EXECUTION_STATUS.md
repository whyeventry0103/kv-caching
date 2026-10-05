# Execution status (one row per work package)

Specification: `docs/paper/EXECUTION_PLAN.md` (v1.0, baseline `1b644dc`).
Claims: `docs/paper/CLAIM_LEDGER.md`.
Snapshot: **2026-10-05 13:30 Dubai.** The remotes were fetched; origin and backup were both at `1b644dc` with no newer commits.
Server at 11:07: both GPUs idle (outside the 01:45–10:43 window); storage 49.6 GiB of 72.

| WP | Owner | Depends on | State | Artifacts | Command | Actual resource use | Next action |
|---|---|---|---|---|---|---|---|
| WP0 freeze + single status | Claude | — | **DONE** | This file; `CLAIM_LEDGER.md` (incl. §D fork-precision scope, §E retired wording); paper edits (title, abstract qualification, 13 causal "bracket" edits, sub-floor TV qualification); `PREREGISTRATION.md` dated entry | `make audit`; pdflatex ×2 (0 errors, 1 pre-existing overfull, 27 pp) | CPU | — |
| WP1 existing queue | Claude (scheduler) | cron window | **IN PROGRESS** | reverse-readout-pp256 READ (`results/diag/paper_audit/reverse_b256.json`, FINAL) | `report_protocol_audit.py --runs-root <runs> reverse --budget 256` | GPU: existing authorization only; nothing new launched | Tonight: grid-pp1024 (474/800 → 800), then readout-pp1024 (0/800). Then `report_s.py protection --budget 1024` and `consequence`. A B=1024 *reverse* cell is not queued (see blockers) |
| WP2 Qwen3 scoring | Claude implements; owner + supervisor decide | — | **DONE (audit)**; disposition **WAITING ON OWNER** | `src/kvcache/grading_sensitivity.py`, `scripts/audit_qwen_scoring.py`, `docs/protocol/QWEN_SCORING_SENSITIVITY.md`, `results/diag/qwen_scoring/bank_audit_{qwen3_4b,r1_qwen7b}.*` | `audit_qwen_scoring.py --runs-root <runs> --bank qwen3_4b` (with the scheduler's REMOTE_* env) | CPU on the host | Owner picks (a)/(b)/(c) in memo §4 (recommended: a). Have a person check the 12 rescued + 12 ambiguous rows |
| WP3 no-GPU analysis | Claude | — | **DONE** | `decision_b256.json` (practical margins, shrinkage CI), `matched_b256.json` (matched keys), `RELATED_WORK_MATRIX.md` (incl. novelty gate N), PAPER_FINDINGS §5l and §5m | `report_protocol_audit.py decision` / `matched` | CPU | Recheck N after B=1024 |
| WP4 new capability | Claude | — | **CODE READY, SMOKES HELD** | `src/kvcache/presses/protected_selectors.py` (`h2o_style_pp_v1`, `uniform_fresh_pp_v1`, `uniform_fresh_pp_r64_v1`); `tests/test_protected_selectors.py` (11 tests: 9 pass locally, all 11 pass on the server CPU stack together with prompt_protect) | — | CPU | Write the runner adapters and register them in dryrun. Add the provenance family. Real-generation smoke (needs authorization: GPU) |
| WP5 protected-selector extension | — | WP4 smokes, authorization | **HELD** | `configs/paper/protected_selector_q7_math_b256_v1.yaml` | — | — | Approve or decline: about 73–94 GPU-h (4–5 nights at full capacity), 0.1 GiB |
| WP6 confirmation | — | WP5, gate N, Qwen3 disposition, GSM8K pin | **HELD** | `configs/paper/confirm_*.yaml`, `results/diag/paper_plan/{sample_size,cost_estimate}.json` | `plan_paper_sample_size.py sample-size` / `cost` | CPU | Freeze N: the plan's N=100×2 gives a 95% half-width of about 0.05; reaching about 0.025 needs N≈300×4 (about 3× the cost) |
| WP7 writing | Claude drafts; authors own | evidence freeze | **DRAFTING** | `docs/paper/DRAFT_METHODS_RELATED_FIGURES.md` | — | — | Figures 2/5 can be built from existing B=256 data now |

## Blockers and genuine gates (each needs a person, not more code)

1. **Qwen3 gate disposition.** Memo §4 in `QWEN_SCORING_SENSITIVITY.md`. Owner and supervisor decide.
2. **New GPU work.** WP4 smokes and the WP5/WP6 campaigns need one approval of a concrete budget (`cost_estimate.json`).
3. **GSM8K.** `configs/benchmarks.yaml` pins no revision (a human pins it). There is no loader yet, and the dataset is absent from the host (a download needs explicit permission).
4. **B=1024 reverse readout.** It is not in the registered S.1 design. A forward/reverse comparison at 1024 needs a new held cell (~800 rows) and the same approval as item 2.

## Surprises this session

1. **The replay directions disagree after protection** (A6). The ~80% shrinkage is direction-specific.
2. **The evidence2026 "residual" was at unmatched budgets** (2048 vs 256). It was corrected in the paper and recorded in §5m.
3. **The secondary scorer moves the anchor model too** (+0.035). The format mismatch is not specific to Qwen3.

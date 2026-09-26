# Meeting #3 — review evidence

Recorded 2026-09-24 (19:30 UTC / 23:30 Dubai), before the first GPU window that
runs any of the new work. The primary results document is
[`../MEETING3_RESULTS.md`](../MEETING3_RESULTS.md); the machine-readable ledger
is `configs/meeting3/cells.yaml` → `results/diag/meeting3/meeting3_manifest.json`.

## Status by unit (read these words literally)

- **Experimentally completed:** measured on the real model, read, and written up.
- **Queued:** implemented and tested, but the real-model run has not happened.
- **Running:** none at the time of this commit.

| Unit | Status | What exists |
|---|---|---|
| T1 accuracy / readout / residual vs budget 256–2048, paired CIs, flips, realized ratios | **EXPERIMENTALLY COMPLETE** | `results/diag/meeting3/t1_budget_curve.json`, `figures/fig8_budget_curve.*` |
| Q.2 exposure grid (D_tv) + occupancy audit | **EXPERIMENTALLY COMPLETE** (verdict MIXED) | `results/diag/amendment-q/onset_surface_q2.json`, `results/diag/meeting3/q2_onset_grid.json`, `figures/fig9_onset_grid.*` |
| Q.1 interval arms (phase locking) | **ANALYZED** 2026-09-15 (earlier session); its instrument controls are queued | `results/diag/amendment-q/shape_sawtooth_i*.json` |
| D_acc (R.3/R.8) | IMPLEMENTED, TESTED (CPU, tiny model, mutation-tested), **QUEUED** behind a real-model smoke | `src/kvcache/generation.py`, `scripts/run_grid.py`, `scripts/dacc_gate_smoke.py`, `scripts/report_dacc.py` |
| Q.3 attention channels + Q.6 latent split (R.4/R.6/R.9) | IMPLEMENTED, TESTED, **QUEUED** behind basis fit + smoke | `src/kvcache/capture.py`, `scripts/run_basis_fit.py`, `scripts/run_capture.py`, `scripts/report_capture.py` |
| Q.5 exact-lag restoration + identity repair (R.5/R.10) | IMPLEMENTED, TESTED, **QUEUED** | `src/kvcache/exact_lag.py`, `scripts/run_exact_lag.py`, `scripts/report_exact_lag.py` |
| Q.1 direct fork control + legacy gate (R.2/R.11) | IMPLEMENTED, TESTED, **QUEUED** | `src/kvcache/fork_control.py`, `scripts/run_fork_control.py`, `scripts/report_fork_control.py` |
| T1 TV at 512/1024 (R.1/R.12) | IMPLEMENTED (summary storage), TESTED, **QUEUED** | `src/kvcache/replay.py`, `configs/experiments/replay_ext_math500_b{512,1024}_r1.yaml` |

"Tested" means CPU tests on a tiny random-init Qwen2 against the pinned
torch 2.7.1 / transformers 4.56.2 / kvpress 0.5.3 on the remote host. It does
**not** mean validated on the 7B model: that is what the queued smoke gates
(`dacc-gate-smoke`, `capture-smoke`) do, and none of them has run yet.

## Second repair pass (review of cfa30cd, 2026-09-25)

See **`acceptance_cfa30cd.md`**: every finding, its fix, its production-path test and the observed result. Tested implementation `58d4dc8`; implementation ids in `remote_pytest.txt`.

## Repair pass for the external review of 4e38224 (2026-09-25)

| Finding | Fix | Commit | Regression tests |
|---|---|---|---|
| 1. Row-count completion | exact coverage vs the frozen design (`kvcache.coverage`), one identity, controls, eligibility separate; FINAL-only verdicts, `--final` exit 3; all four new readers | 391e8fd | `tests/test_report_capture.py`, `test_report_dacc.py`, `test_exact_lag.py`, `test_fork_control.py` |
| 2. Observation-weighted means | `equal_problem_bootstrap` / `paired_equal_problem` (4-vs-1 seed → 0.5) | 391e8fd | same, plus balanced parity with the old helper |
| 3. Restart/gate provenance | R.13 identity on rows, basis and gates; atomic gates retire stale PASS; smoke rebuilt from persisted rows; verified basis | ad30309 | `tests/test_provenance.py`, `tests/test_capture_scripts.py` (restart after each condition / after last row / old PASS then FAIL / changed config / corrupt basis) |
| 4. Fork PASS on logits only | `logit_gate` (R.11 unchanged) + `selection_gate` + coverage → `overall` | 391e8fd | `test_selection_disagreement_fails_even_within_logit_tolerance` |
| 5. No automatic analysis | `cpu_only` postprocess stages in the queue (reader → figures → manifest → terminal marker), evidence-derived ledger status; T1 units 200 | 391e8fd | `tests/test_postprocess.py`, scheduler stage tests |
| Scope: Δ/M at Q.5 states | kept OUTSTANDING; no query-drift causal claim | R.14 §5 | reader note |
| Restoration identity values | K and V checked across layers; reinserted values gated; gate reports recorded | 391e8fd | deep-layer and reinsert-value mutation tests |
| Storage facts | measured both scopes; correction of the 11.2 GiB figure | this commit | `storage.md` |

"Tested" is still CPU (tiny model, pinned stack). No real-model gate has run as
of this commit; the first window is 2026-09-25 01:45 Dubai.

## Files in this directory

| File | What it shows |
|---|---|
| `local_lint.txt`, `local_test.txt`, `local_dryrun.txt` | `make lint` / `make test` / `make dryrun`, each run bare, exit 0. Locally torch is absent, so the torch tests skip here. |
| `remote_pytest.txt` | The same tree on the remote, pinned stack: every Meeting #3 test (incl. the R.13/R.14 repairs and scheduler) by name (130 PASSED, 0 FAILED), then the full suite (538 passed, 0 skipped). |
| `parity.txt` | `make parity`: commit, source tree and all 111 `results/diag` artifacts agree between local and remote. |
| `storage.md` | Measured footprint under both storage scopes, queued outputs, and the correction of the 11.2 GiB figure. |
| `r16_operations.md` | R.16.6/R.16.7 operational record: the preflight-halt log lines and recovery, both out-of-window starts, the stage-record collision failure and fix, the host stage-record listing. |
| `manual_ticks_20260925.sh.txt`, `manual_ticks_20260926.sh.txt` | Verbatim copies of the two owner-requested tick loops (sha256 in `r16_operations.md`). |
| `queue_dryrun_tick.txt` | Dry run of the 2026-09-25 01:45 orchestrator tick: the six CPU postprocess stages report what they wait on, the old Q.5 cells are reported BLOCKED with their reason, and `dacc-gate-smoke` is selected. |

## Mutation checks

Each check below showed that the instrument catches the defect it exists for.

- `tests/test_onset_gate.py`: a decode-step counter shifted by one.
- `tests/test_capture.py`: a wrong position map (the layer-0 key identity check fires).
- `tests/test_exact_lag.py`: the pre-R.5 tail stash (refused).
- `tests/test_fork_control.py`: masking disabled (the A−B difference fires).

## Remote-only evidence, not in this repository

These stay on the GPU host (`~/kvcache-storage/runs`). They are raw per-row or
per-step outputs, too large to commit, and gitignored by policy. Every number
in the committed artifacts was computed from them by a committed script.

| Run | Size | Used by |
|---|---|---|
| `math500-fc-powered-merged` | 62 MiB | full-cache reference for every contrast |
| `grid-diss256-tok`, `grid-diss512`, `grid-diss1024`, `grid-diss2048-tok` | 45 / 34 / 26 / 24 MiB | T1 paired CIs (`scripts/report_budget_curve.py`) |
| `replay-onset-t0`, `-t512`, `-t2048` | 197 / 192 / 174 MiB | Q.2 verdict and grid (`scripts/report_onset_surface.py`, `scripts/report_onset_grid.py`) |
| `replay-ext-diss256` | 392 MiB | AUC-TV at 256/2048 (Amendment P artifact) |

Also remote-only: model weights, datasets, caches, orchestrator and run logs
(`~/kv-caching/logs`), and every output of the queued cells. As queued cells
finish, their compact reader outputs will be committed under
`results/diag/meeting3/`, together with the smoke PASS/FAIL reports.

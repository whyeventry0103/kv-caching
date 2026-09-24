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

## Files in this directory

| File | What it shows |
|---|---|
| `local_lint.txt`, `local_test.txt`, `local_dryrun.txt` | `make lint` / `make test` / `make dryrun`, each run bare, exit 0. Locally torch is absent, so the torch tests skip here. |
| `remote_pytest.txt` | The same tree on the remote, pinned stack: every Meeting #3 torch test by name (37 PASSED, 0 FAILED), then the full suite (504 passed, 0 skipped). |
| `parity.txt` | `make parity`: commit, source tree and all 111 `results/diag` artifacts agree between local and remote. |
| `queue_dryrun_tick.txt` | Dry run of the 2026-09-25 01:45 orchestrator tick: the old Q.5 cells are reported BLOCKED with their reason, and `dacc-gate-smoke` is selected. |

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

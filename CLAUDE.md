# kv-caching

Research scaffold for separating **readout distortion** from **trajectory
distortion** in KV-cache compression for reasoning LLMs. Current stage: most
experimental results are in and the active work is the paper
(`kv_readout_vs_trajectory_paper.tex`, findings log in `PAPER_FINDINGS.md`)
plus remaining queued runs (R-KV, AIME, Llama-8B — see
`configs/nightly_queue.yaml` and recent commit history). Scientific claims and
analysis choices are **frozen in `PREREGISTRATION.md`**: do not change metrics,
thresholds, seeds, decoding parameters, or model/dataset pins without an
explicit amendment recorded there.

## The rule that shapes everything

This machine is local-only: source, configs, docs, lint, and CPU unit tests.
Models, datasets, caches, and all experiments run on a remote GPU server.
`remoteserver.md` is the authoritative operations and safety policy;
`docs/WORKFLOW.md` describes the selected loop ("Policy B", rsync-first).
Never attempt generation, model/dataset downloads, or anything requiring
loaded weights locally — the code guards against it, and the guards are
intentional.

## Commands (verified against Makefile and pyproject.toml)

```bash
make lint     # ruff check . && black --check .   (88 cols, py311)
make test     # pytest -q   (CPU-only, no network, no weights)
make dryrun   # every runner script x config planned in --dry-run mode
```

- The full local gate before any sync is `make lint test dryrun`.
- The venv is `.venv/`; the Makefile auto-prefers `.venv/bin/*` when present.
- pytest config: `pythonpath=["src"]`, `testpaths=["tests"]`, `addopts=-ra`.
  Tests use synthetic tensors and hand-computable values; they must pass on a
  machine with no GPU, no model weights, and no network.
- Sync to remote: `bash scripts/sync_source.sh` (dry-run preview by default),
  then `bash scripts/sync_source.sh --apply`. It never uses `--delete` and
  never transfers `.git`, environments, data, caches, runs, logs, or results.

## Layout

- `src/kvcache/` — the package; all computation and scientific logic.
  Notable modules: `bank.py` (reference banks), `replay.py`, `generation.py`,
  `grading.py`, `canonicalize.py`, `determinism.py`, `stats.py`,
  `io_schema.py`, `token_ids.py`; subpackages `presses/` (compression-policy
  registry + R-KV port) and `metrics/` (`add.py`, `tdp.py`, `attribution.py`,
  `reverse_replay.py`, ...).
- `scripts/` — thin orchestration entry points and remote launchers. Has its
  own `scripts/CLAUDE.md`; read it before working there.
- `configs/` — shared registries (`models.yaml`, `benchmarks.yaml`,
  `presses.yaml`, `sampling.yaml`, `remote.yaml`, `nightly_queue.yaml`) and
  `configs/experiments/` for per-run configs.
- `tests/` — mirrors module names (`tests/test_<module>.py`).
- `docs/` — design notes and probes (`METRICS.md`, `RKV_PORT.md`,
  `KVPRESS_API_PROBE.md`, `WORKFLOW.md`, ...).
- Root markdown (`PREREGISTRATION.md`, `PAPER_FINDINGS.md`,
  `COAUTHOR_REVIEW.md`, `SUBMISSION_STRATEGY.md`, `context.md`) — the paper's
  paper trail; treat as project state, not disposable notes.

## Conventions actually in use

- `from __future__ import annotations`, full type hints, 88-column limit
  (black + ruff with `E,F,I,B,UP`). Ruff exempts only `scripts/*.py` from
  `E402,I001` because of the sys.path bootstrap there.
- Module docstrings state the *scientific rationale* for the module (see
  `src/kvcache/determinism.py`), not a restatement of the code.
- torch and numpy are imported lazily inside functions with `ImportError`
  fallbacks, so the package imports and the test suite runs on CPU-only
  machines. Preserve this when adding code — a top-level `import torch` in
  `src/kvcache` breaks the local test gate's reason for existing.
- Metric and scoring logic has a **single source in the package**; scripts
  import it. Commit 193e12f exists because a script's scoring drifted from
  the package's.
- Experiment configs are named `{stage}_{benchmark}_{variant}.yaml` and carry
  `stage:`, `remote_only:`, `pilot:` flags plus `estimated_generation_count`.
- New metrics get tests with small hand-computed cases via `pytest.approx`,
  including edge cases that return `None` (see `tests/test_add_decomposition.py`).

## Verifying a change before calling it done

1. If it touches analysis, metrics, decoding, seeds, or model choice: check
   it against `PREREGISTRATION.md` first. If it conflicts, the change needs an
   amendment there, not a quiet edit.
2. `make lint` — run it before committing, not after; two recent commits are
   pure lint fixes for overlong comment lines that slipped through.
3. `make test`.
4. `make dryrun` — every runner/config pair must plan cleanly. If you add a
   runner script or an experiment config, add a corresponding `--dry-run`
   line to the Makefile `dryrun` target, or it is unguarded.
5. Anything beyond dry-run is remote-only and goes through the sync +
   `scripts/check_remote.sh` + `scripts/launch_tmux.sh` path in
   `docs/WORKFLOW.md`. Do not shortcut it.

## Known pitfalls (evidence on disk)

- `TODO(human)` sentinels: `models.py` and `datasets.py` refuse to run when a
  model/dataset `revision` is unpinned. This is deliberate — a human must pin
  revisions; never substitute a guessed revision to un-stick a run.
- Press eligibility is tiered in `src/kvcache/presses/__init__.py`:
  `EXECUTABLE`, `PREFILL_ONLY`, `PORT_REQUIRED`, `DEFERRED`. `rkv` is promoted
  to executable only by a smoke-marker file written when every smoke check
  passes; `h2o` requires eager attention. Do not move a press between tiers
  by hand.
- Heavy scripts refuse to run without `--i-understand`; downloads additionally
  require `--allow-model-download` / `--allow-dataset-download`; launchers
  refuse without a verified integer `KVCACHE_PERSONAL_FREE_GB` (shared-server
  quota; filesystem-wide `df` is not sufficient). These gates have caught at
  least one real bug (commit 5c68acd) — never weaken one to make a job start.
- Output rows are append-only JSONL; `--resume` logic heals a partial trailing
  line. When touching bank/grid output code, keep writes atomic per line and
  keep resume keyed on (benchmark, problem id, seed).
- The repo moved off `~/Desktop` (macOS TCC permission lockout, 2026-06-14).
  Do not relocate it back or hardcode Desktop paths; `~/Desktop/kv-caching`
  may be a stale symlink.

## Terminology discipline (binding, from PREREGISTRATION.md)

- The residual is "accuracy change unexplained by on-reference readout" —
  never a measured causal effect of trajectory distortion.
- The Qwen3-4B comparison is "cross-training-paradigm within the Qwen
  family", never "cross-architecture".
- Every absolute cache budget is reported with its realized compression ratio.
- Benjamini-Hochberg correction is mandatory for families of problem-level
  tests.

## Handoff operations (added 2026-07-06 — the results-landing playbook era)

The queue (`configs/nightly_queue.yaml`, 22 jobs) finishes the science unattended.
The session's job while it drains is the **morning loop**:

1. Read the status email (or `nightly_report.py --dry-run` on the remote).
2. `ssh kvcache-gpu 'cd ~/kv-caching && .venv/bin/python scripts/harvest_results.py'`
   — runs the right report for every newly complete unit, archives artifacts under
   `<runs>/harvest/<unit>/`, prints the digest.
3. Open **`RESULTS_PLAYBOOK.md`** at that unit: fill the number slots, pick the
   branch matching the digest, apply the listed tex/figure/docs edits verbatim.
4. Rebuild the PDF (TinyTeX pdflatex ×2; check `^!` and Overfull), gates, commit,
   `make sync-apply`. Copy each harvest `summary.json` to `results/diag/`.

Never read a partial cell into the paper — harvest decides readiness.

Additional hard-won rules not covered above:

- **Run `make lint` bare.** `make lint 2>&1 | tail -1 && ...` takes tail's exit code;
  a broken lint sailed through a commit chain exactly this way once.
- **After any `scp` to the remote, finish with `make sync-apply`** or the watcher's
  preflight hash check fails that night.
- **Eyeball every regenerated figure PNG** (Read tool). Every figure batch to date
  had one collision that only visual inspection caught. New palette hues must pass
  the dataviz validator first.
- Remote link drops for minutes-to-hours: short ConnectTimeouts, background
  `until ssh ...; do sleep 60; done` probes, don't hammer.
- Standing phrasing landmines (also in PAPER_FINDINGS §5 area): "H2O-style /
  ObservedAttentionPress (no recency reservation)" — never "H2O collapses", no HOTT
  contradiction; "non-negative on 94/100 (44 win / 50 tie / 6 loss)" — never "beats
  on 94/100"; Table 3 TV windows are the per-trace estimator (pooled = 0.118, larger);
  R-KV numbers characterize *our certified port at 7B*, not the authors' setup;
  underwhelming pre-registered outcomes (e.g. no silent-flip regime at 2048) are
  reported plainly — the honesty trail is the paper's moat.
- **Decisions already made** (don't relitigate): R-KV+AIME drop-dead 2026-08-01
  (paper survives the cut); TV-matched-noise control, logic benchmarks, Qwen3
  deliberately not queued (reasons in context.md 2026-07-05 bullet).
- **Endgame after the queue drains**: venue conversion executing
  SUBMISSION_STRATEGY.md §6 (paste-ready intro, Fig-1↔ladder swap) +
  BEST_POSSIBLE_PAPER.md Phase 8 (notation diet: cut TDP from main text, retire
  symbol soup); anonymization scrub (personal email in `scripts/nightly_report.py`;
  internal strategy docs excluded from the artifact); arXiv stake ~mid-August.

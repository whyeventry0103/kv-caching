# complete_setup.md — the full research-project workflow, generalized

Written 2026-07-15, distilled from the kv-caching project after ~6 weeks of
operation. This is the **complete, transferable setup** for any AI/ML research
project run the same way: local-only development on a laptop, all compute on a
shared remote GPU server, unattended overnight experiment queues, automatic
morning email reports, and a paper-trail discipline that survives model/session
handoffs. Replace `$PROJECT` with your project name; everything else transfers.

Internal ops doc — exclude from any public release artifact.

---

## 0. The philosophy (read this first, it explains every rule below)

1. **Two machines, one direction.** The laptop holds source, configs, docs,
   tests. The GPU server holds models, datasets, runs, results. Code flows
   laptop → server via rsync; result summaries flow back via scp. The laptop
   NEVER runs generation or loads weights (guard it in code); the server NEVER
   edits source (all edits happen locally and sync over).
2. **Everything result-bearing is resumable, append-only, and keyed.** Any run
   can be killed at any moment (nightly window ends, OOM, link drop) and
   relaunched without loss or duplication.
3. **Guards are load-bearing.** Every footgun that once fired gets a mechanical
   guard (a refusal flag, a dry-run default, a disk check, a gate script). You
   never weaken a guard to make something run.
4. **State lives in files, never in a chat context or your head.** A person (or
   LLM session) with zero prior context must be able to resume from the repo
   alone.
5. **Nothing partial enters the paper.** A harvester script — not a human
   eyeballing progress — decides when a result cell is complete enough to read.

---

## 1. Repository layout

```
$PROJECT/
├── CLAUDE.md               # operating spec for assistant sessions (see §12)
├── Makefile                # THE entry points: lint/test/dryrun/sync/sync-apply/launchers
├── pyproject.toml          # deps + ruff/black config (88 cols; scripts/ exempt E402,I001)
├── src/$PROJECT/           # ALL computation & scientific logic (importable package)
│   ├── generation.py, grading.py, stats.py, io_schema.py, determinism.py, ...
│   └── (module docstrings state the SCIENTIFIC rationale, not just behavior)
├── scripts/                # thin orchestration ONLY — no science logic here
│   ├── CLAUDE.md           # directory-local conventions (runner contract, §4)
│   ├── run_*.py            # runners: --config/--dry-run/--resume/--seeds contract
│   ├── nightly_autorun.py  # queue orchestrator (§9)
│   ├── nightly_report.py   # morning email (§10)
│   ├── harvest_results.py  # readiness-gated result harvester (§11)
│   ├── sync_source.sh      # Policy-B rsync (§5)
│   ├── launch_tmux.sh / launch.sh / setup_remote.sh / check_remote.sh
│   └── make_paper_figures.py  (figure method: docs/FIGURES_PLAYBOOK.md)
├── configs/
│   ├── models.yaml, benchmarks.yaml, sampling.yaml   # PINNED revisions (§7)
│   ├── experiments/*.yaml  # one config per run, named {stage}_{bench}_{variant}
│   └── nightly_queue.yaml  # the ordered job queue (§9) — the queue IS the plan
├── tests/                  # mirrors module names; CPU-only, no network, no weights
├── docs/                   # WORKFLOW.md, probe docs, FIGURES_PLAYBOOK.md
├── results/diag/           # gitignored archive of harvested summary JSONs
├── figures/                # committed rendered figures (pdf+png pairs)
├── PREREGISTRATION.md, PAPER_FINDINGS.md, RESULTS_PLAYBOOK.md, context.md  (§12)
└── remoteserver.md         # ops/safety policy for YOUR server (authoritative)
```

Rules that keep this healthy:
- Scoring/parsing/statistics live in `src/`, imported by `scripts/` — the moment
  a script grows logic, move it into the package (script-level logic WILL drift
  from package logic; it happened here and cost a parity audit).
- `results/`, `runs/`, `logs/`, datasets, caches, environments: gitignored and
  rsync-excluded. Git carries source + docs + figures + archived summaries'
  *paper-trail* only.

## 2. Bootstrap a new project (one-time, ~half a day)

1. `git init`, create the layout above, copy the Makefile/pyproject/sync script
   from the previous project and rename env-var prefixes (`$PROJECT_...`).
2. Local venv: `python -m venv .venv && .venv/bin/pip install -e .[dev]`
   (dev = ruff, black, pytest; heavy deps like torch go in the REMOTE env only,
   and are imported lazily inside functions so CPU tests never need them).
3. SSH config on the laptop (`~/.ssh/config`): a named host alias with key auth:
   ```
   Host $PROJECT-gpu
     HostName <ip>  User <user>  IdentityFile ~/.ssh/<key>
   ```
   The sync script additionally multiplexes (ControlMaster/ControlPersist=120,
   ControlPath under /tmp) so a password/hop is negotiated at most once.
4. Remote: clone nothing — `make setup` runs `setup_remote.sh --i-understand`
   which creates the remote project dir + venv from the synced source, installs
   the pinned stack, and creates the runs root (`$PROJECT-storage/runs`, on the
   big disk, OUTSIDE the code dir).
5. Pin everything: model + dataset revisions in `configs/*.yaml` with a
   `TODO(human)` sentinel pattern — loader code REFUSES unpinned revisions, and
   only a human ever fills a pin (never let an assistant guess one to unblock).
6. Write `remoteserver.md`: who shares the box, disk etiquette, GPU count, the
   personal-quota env var (§7), allowed hours, contact for conflicts.

## 3. Local development loop

The three gates, run **bare** — never inside a pipe (a `| tail` swallows the
exit code; a broken lint once sailed through a commit chain exactly this way):

```bash
make lint     # ruff check . && black --check .
make test     # pytest -q  — CPU-only, fast (<5s), no network, no weights
make dryrun   # EVERY runner × config in --dry-run (validates plans, no compute)
```

- `make dryrun` grows one line per new (runner, config) pair. A runner absent
  from the dryrun target is untested by definition — adding the line is part of
  adding the runner, not optional.
- Test style: small hand-computed cases with `pytest.approx`, including `None`/
  edge cases; every bug fix lands WITH its regression test in the same commit,
  the test naming the incident in a comment.
- Style: `from __future__ import annotations`, full type hints, 88-col
  black/ruff (`E,F,I,B,UP`); `scripts/*.py` start with a `sys.path` bootstrap
  and are exempted from `E402,I001` — deliberately, don't "fix" it.

## 4. The runner contract (every experiment script)

```
run_x.py --config configs/experiments/foo.yaml [--dry-run] [--resume]
         [--seeds 11,23] [--i-understand] [--allow-model-download] [--allow-dataset-download]
```

- `--dry-run` validates the config and prints the full plan (counts, models,
  seeds) with **zero** model/dataset/network/CUDA touch. It is the default
  posture; heavy execution requires `--i-understand` AND `$PROJECT_REMOTE=1`
  in the environment, so the laptop physically cannot run one by accident.
- `--resume` skips completed keys and heals a partial trailing JSONL line left
  by a kill. The key is the row's identity tuple — e.g. (benchmark, problem_id,
  seed) — and "complete" means the row AND all its sidecars exist.
- `--seeds` is a comma-subset of the config's seeds: per-GPU sharding. Each
  shard owns disjoint per-seed output files so two GPUs never write the same
  file. EVERY runner must accept it — the nightly launcher passes it
  unconditionally (a runner without it fails rc=2 and gets marked stuck; this
  exact bug cost a night here).
- Output rows are built by shared package builders (one `build_result_row`
  function, one schema version string) — never hand-assembled dicts.
- Scale guard: configs must declare `pilot: true` or `scale_authorized: true`;
  runners refuse otherwise, and refuse a "pilot" whose row count isn't the
  pilot size. Configs carry `estimated_generation_count` so the plan is audited.

## 5. Sync: laptop → server ("Policy B")

```bash
make sync         # DRY-RUN preview (default — always look first)
make sync-apply   # the real rsync
```

`sync_source.sh` mechanics (keep ALL of these when porting):
- `rsync -avzh` over a multiplexed ssh; **never `--delete`** (a bad local state
  can then never destroy remote source).
- Excludes: `.git`, environments, data, caches, `runs/`, `logs/`, `results/`,
  generated files. Source and docs only.
- On success it writes a **`.sync-state`** stamp on the remote. The nightly
  automation preflights this stamp — so the iron rule: after ANY manual `scp`
  of code to the remote, finish with `make sync-apply`, or the watcher may
  refuse to launch that night.
- The remote never has local edits (if you must hotfix on the server, copy the
  fix back locally, commit, and re-sync the same hour).

## 6. Git conventions

- Commit messages carry the **why** — the unit/decision and its rationale, what
  failed, what was verified. `git log` doubles as the lab notebook's index.
- Every result read into the paper = its own commit naming the result unit.
  Every bug fix = its own commit naming root cause + the regression test.
  Never smuggle an unrelated fix into a result commit.
- Cleanups get separate commits AFTER the task lands; near experiment code,
  prefer additive changes (new flag/config/marker) over mutations — if you
  record per-row source hashes, gratuitous diffs are a provenance cost.
- `results/diag/*.json` (archived summaries) stay untracked; the tracked
  paper-trail is the dated notes in `PAPER_FINDINGS.md` quoting them.

## 7. Remote safety guards (each one earned its existence)

- **Personal disk quota env var** (`$PROJECT_PERSONAL_FREE_GB`): launchers
  refuse without a verified integer — on a shared box, global `df` is
  meaningless and one runaway run can fill the disk for everyone.
- **`--i-understand`** on anything heavy; **`--allow-model-download` /
  `--allow-dataset-download`** on first fetches (and queue jobs that need them
  pass them via `extra_args`, so a download can never happen implicitly).
- **Pinned revisions everywhere** (model/dataset/package). The remote stack is
  version-pinned; upgrading it is a project (isolated env, probe doc, go/no-go
  gates), not a `pip install -U`.
- **tmux for anything > ~1 minute** (`launch_tmux.sh`: persistent log under
  `logs/<run>.log`, survives SSH disconnect). Foreground SSH is for dry-runs
  and probes only. Expect the link to flake: retry loops with
  `-o ConnectTimeout=15`, keepalives on long transfers
  (`-o ServerAliveInterval=10 -o ServerAliveCountMax=3`), and background
  `until ssh ...; do sleep 60; done` probes rather than hammering.

## 8. Manual runs (before automation exists)

`make smoke` → tiny end-to-end run proving the whole path (model loads, rows
appended, resume works). Then pilots (small n, `pilot: true`), then powered runs
— each stage a config, each preceded by its dry-run, each launched via
`launch_tmux.sh --i-understand <run-name> <command> <config> [flags]`.
`check_remote.sh` is the preflight (env sane, disk quota var set, config parses
on the remote).

## 9. Nightly automation (the heart of unattended progress)

**The queue** (`configs/nightly_queue.yaml`): ordered list of jobs —
```yaml
version: 1
enabled: true          # kill-switch: false -> orchestrator exits immediately
jobs:
  - name: my-run       # also the tmux session prefix + state key
    command: grid      # dispatches to a runner via the launcher
    config: configs/experiments/grid_x.yaml
    run_name: my-run   # output dir under the runs root
    total: 800         # expected row count when complete
    seeds: [11, 23, 47, 89]
    min_free_gb: 28    # per-job OOM/disk threshold
    progress_kind: grid   # how to count progress: grid|readout|marker subtree
    requires: other-run   # dependency gate (null = none); markers gate too
    extra_args: ["--allow-model-download"]   # optional
```
Appending a job needs no code change. The queue file **is** the plan of record.

**The orchestrator** (`nightly_autorun.py`, idempotent, cron-fired):
- Runs only inside the night window (e.g. 01:45–10:45 local server time — set
  it to the shared box's social contract); stops launching <1h before close.
- Each tick: parse live tmux sessions → which GPUs are busy; count each job's
  rows (`progress_kind` subtree, excluding sidecars); pick launch candidates =
  first incomplete jobs whose `requires` are complete, sharding **needy seeds**
  across free GPUs via `--seeds` (disjoint files → no write races).
- **Tail-fill**: a job whose remaining seeds are all in flight does NOT block
  the queue — the next job fills the otherwise-idle GPU. (An improvised version
  without this idled a GPU for hours; design and test it, 20+ unit tests.)
- **Stuck detection**: persisted state (`logs/nightly_state.json`) counts
  consecutive no-progress launches per job; at 3 the job is skipped and named
  in the log/email. After fixing the cause, RESET its `fails` counter (back up
  the state file first) or it stays skipped forever.
- Everything is launched `--resume`, so window-end kills are free.

**Cron (on the remote)**:
```cron
45,55 1 * * *  cd ~/$PROJECT && .venv/bin/python scripts/nightly_autorun.py >> logs/nightly_autorun.log 2>&1
*/10 2-10 * * * cd ~/$PROJECT && .venv/bin/python scripts/nightly_autorun.py >> logs/nightly_autorun.log 2>&1
55 10 * * *    cd ~/$PROJECT && .venv/bin/python scripts/nightly_report.py  >> logs/nightly_report.log 2>&1
5 11 * * 1     cd ~/$PROJECT && .venv/bin/python scripts/scoop_sweep.py     >> logs/scoop_sweep.log 2>&1
```

## 10. The morning email + scoop watch

`nightly_report.py` (fires once after the window) emails via SMTP (credentials
in an untracked json on the remote; **remember the address is PII — scrub
before any public release**). Sections that proved their worth:
- LAST NIGHT: per-job row deltas + completions.
- QUEUE: progress bars per job; TOTAL % + ETA at last night's rate.
- TONIGHT: the launcher's actual plan (same candidate function — no drift
  between what's reported and what will run) with per-seed progress.
- DEADLINES: D-countdowns to drop-dead dates.
- HEALTH: stuck jobs, GPU free memory, disk. Silence is never success — the
  report states health explicitly.
It also writes a snapshot (per-job counts) so the next report can diff.
`scoop_sweep.py`: weekly arXiv API query for your keywords → email + a dated
log in `notes/` — you find concurrent work before reviewers do.

## 11. The harvester (results enter the paper through ONE door)

`harvest_results.py` defines analysis **Units**: named bundles of required runs
with expected counts and an analysis kind →
`Unit("my-cell", {"run-a": ("grid", 800), "run-b": ("readout", 800)}, "m4")`.
Each invocation: prints NOT READY units with exact missing counts, runs the
right report for newly-complete units, writes artifacts under
`<runs>/harvest/<unit>/`, drops a `harvested.marker` (idempotent; `--force
NAME` to redo; `--dry-run` for readiness only), and prints a compact digest of
exactly the numbers the paper needs. **The morning loop**: read email → run
harvester → for each READY unit, open the results playbook at that unit, fill
number slots, apply the pre-written branch — never read a partial cell, never
force a branch the numbers don't pick.

## 12. The paper-trail file system (what makes handoffs survivable)

| File | Role | Mutation rule |
|---|---|---|
| `PREREGISTRATION.md` | frozen protocol: metrics, thresholds, seeds, pins, pre-committed outcome criteria | append-only dated amendments (A, B, C, ...); a quiet edit is a protocol violation |
| `PAPER_FINDINGS.md` | dated result records with report-exact numbers + framing/terminology discipline (the "never say X" list) + archive paths | new dated subsection per read, same session as the read |
| `RESULTS_PLAYBOOK.md` | for every PENDING result: expected shape, pre-drafted paper text with `{SLOT}`s, decision branches incl. the unwanted outcomes, STOP rules separating instrument bugs from findings | written BEFORE results land, while context is loaded |
| `context.md` | dated ★-bullets, newest first: exact numbers, row counts, commits, THE literal next action | top bullet updated every session end / stage completion |
| `CLAUDE.md` | operating spec for assistant sessions: ground truth, IF-THEN working rules, worked examples w/ commit hashes | evolves; conflicts resolve user > CLAUDE.md > instinct |
| `remoteserver.md`, `docs/WORKFLOW.md`, `docs/FIGURES_PLAYBOOK.md` | ops policy, the loop, the figure method | as needed |

Two disciplines make this system actually work:
- **Pre-reason judgment while you have context**: when a future result will
  demand a call, write the branches NOW with observable numbers selecting each
  branch — including a branch for the outcome you don't want. The successor
  fills slots instead of exercising fresh judgment.
- **Expected-shape gates**: every pending number gets a sanity band written in
  advance; a value outside the band is an instrument bug until the instrument's
  own controls are re-checked (here that meant: eyeball raw generations before
  interpreting a 0.000 cell — it turned out real, but checking first is the rule).

## 13. Data & determinism rules

- Results are append-only JSONL; identity keys make resume/dedup trivial;
  side-car files (e.g. exact token IDs) enable bitwise-faithful replay.
- Truncated/failed generations stay in every denominator, explicitly scored —
  never silently dropped (a classic silent selection effect).
- Buy cheap validity evidence deliberately: an exact re-run replication of a
  full grid, a negative control on null data, a self-control inside the
  instrument (e.g. divergence must be exactly 0 before the intervention), and
  gates before scale-ups (parity harness + real-generation smoke; the smoke
  caught a bug the synthetic harness could not see).
- Never patch result rows. Buggy-instrument data is invalidated wholesale and
  regenerated; derived reports over valid rows are just re-run.
- Display convention: every number in prose/figures is the mechanical rendering
  (`f"{v:.3f}"`) of the artifact value — no hand-rounding, ever; re-derive at
  write time rather than copying your own earlier statement.

## 14. New-project day-one checklist

```
[ ] layout + Makefile + pyproject copied & renamed; git init; first commit
[ ] venv local (dev tools) — heavy deps lazily imported, remote-only
[ ] ssh alias + key; sync_source.sh env vars renamed; make sync (dry) clean
[ ] remoteserver.md written; personal-quota env var chosen & guarded
[ ] setup_remote.sh run; remote venv pinned; runs root on big disk
[ ] pins: models/datasets/packages (TODO(human) sentinels wired to refusals)
[ ] first runner w/ full contract (--config/--dry-run/--resume/--seeds/guards)
[ ] make dryrun line added; schema builders + row tests
[ ] make smoke end-to-end on remote (tmux, log, resume verified by kill+relaunch)
[ ] PREREGISTRATION.md frozen BEFORE the first result-bearing run
[ ] nightly_queue.yaml + autorun + report ported; cron installed; one
    supervised night watched end-to-end before trusting it
[ ] harvest_results.py Units for the first cells; RESULTS_PLAYBOOK entries
    with branches + STOP rules; context.md started
[ ] CLAUDE.md operating spec seeded from the previous project's
```

## 15. The expensive lessons (why the rules above exist)

1. Piped gates lie (`make lint | tail` returns tail's exit code) → run gates bare.
2. A runner missing the launcher's `--seeds` contract = a lost night → the
   contract is universal, and new job TYPES get their first launch supervised.
3. `int(budget)` assumptions crash on legitimate `"full"` sentinels → sentinel
   values are part of the schema; test them.
4. The synthetic parity harness passed while the live press was silently
   misconfigured → always smoke-test at the real boundary (generation), not a
   proxy.
5. Double-BOS: the chat template already emits BOS → `add_special_tokens=False`
   + a decode↔encode round-trip check; all prior pilot data invalidated, not
   patched.
6. Every figure batch shipped one defect only eyes caught → viewing rendered
   artifacts is a required step (full catalog: docs/FIGURES_PLAYBOOK.md).
7. A stale claim orbit: a new result made "order of magnitude / 0.56–0.75"
   false in four places → when a number lands, grep the docs for every claim
   near its topic.
8. Link flaps constantly: ConnectTimeout everywhere, keepalives on transfers,
   background wait-then-act loops, and never two speculative fixes between checks.
9. Hand-rounded ties (0.2475→"0.248") diverge from mechanical rendering →
   pin the display convention on day one.
10. STUCK counters persist across nights → fixing a job's bug isn't done until
    its state-file counter is reset and the next launch verified.

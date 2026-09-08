# Session handoff — kv-caching project

> **STATUS 2026-09-08: HISTORICAL. Superseded by `docs/project/context.md` §0,
> whose 2026-09-08 bullet is the current state.** Kept as the record of how this
> session was resumed after the Claude Pro lapse, and because its landmine list
> (§8) is the only place several mechanical gotchas are written down.
>
> **Three of its claims were checked against disk and are wrong or stale — do
> not propagate them:**
> 1. §10 item 7 says it is unclear whether the `CLAUDE.md:98` embedded-email fix
>    landed. It had not. It has now (commit `9b41d11`); the claim was false and
>    `scripts/nightly_report.py` has zero address literals.
> 2. §8 landmine 21 and §10 item 6 put the anonymized-mirror footnote at `.tex`
>    line ~493. It is at **634**, and the paragraph asserts the mirror **twice**,
>    not once. `docs/operations/PRE_RELEASE_SCRUB.md` item 2 now carries the grep
>    that finds it instead of a line number.
> 3. §10 item 1 and §13 say the mechanism clause is an open decision that Claude
>    must not make. It was **decided and propagated on 2026-09-08** (candidate A,
>    sharpened) on the owner's explicit instruction. See
>    `docs/project/MECHANISM_CLAUSE_CANDIDATES.md` (Decision section) and
>    PAPER_FINDINGS §4x. Zero `TODO(M.3-RESTATEMENT)` markers remain.
>
> Every number in this document was re-verified against its artifact on
> 2026-09-08 and all of them held. Its instruction to trust artifacts over its
> own prose was correct and is what caught the three items above.

**Written:** 2026-08-28, by a different Claude (chat assistant, not Claude Code)
reconstructing state from a full prior Claude Code session transcript. The
Claude Pro subscription on the account running Claude Code has lapsed and all
prior sessions are closed. This document exists so a brand-new Claude Code
session with zero context can pick up exactly where the last one stopped.

## How to use this document

1. Read this whole file first, before touching the repo or the remote.
2. Then read the four authoritative docs listed in "Authority order" below,
   **fresh from disk** — do not trust any number in *this* document as final.
   This document is a map and a checkpoint, not a source of truth. The
   project's own standing rule is "the artifact wins over the prose," and
   that applies to this handoff exactly as it applies to any other document.
3. Where this document states a number, treat it as "this is what I
   understood as of 2026-08-28, verify against the artifact before citing it
   anywhere, especially in the paper."
4. The one thing you can trust without re-verifying: the *mechanical* facts —
   file paths, script names, commit hashes, cron schedule, gotchas. Those are
   not going to have quietly changed since the account went idle, and they
   are the part that doesn't live conveniently in any one file already.

---

## 1. What this project is

A measurement/analysis paper (not a new compression method) explaining why
two literatures disagree about KV-cache compression on reasoning models:
method papers report near-lossless compression on **readout-shaped**
evidence (trace held fixed), while free-running benchmarks report severe
degradation. The paper decomposes the accuracy change under compression into
a **readout** term (measured) and a **residual** (defined as the remainder —
never a measured causal effect). Primary model: DeepSeek-R1-Distill-Qwen-7B.
Benchmarks: MATH-500, AIME-24. Target venue: ICLR.

Owner: Muhammad Ali Shah, final-year CS undergrad. This is a supervised
university research project with regular supervisor meetings; "Meeting #1"
and "Meeting #2" in the docs refer to those.

---

## 2. Authority order for facts — read these, in this order, before acting

```
repo code (disk is ground truth)
  > docs/protocol/PREREGISTRATION.md   frozen; changes only via dated amendments A through P (and counting)
    > docs/results/PAPER_FINDINGS.md   results record + binding framing rules; §4-CURRENT is current truth,
                                        §4-LOG is an archival read-in log (dated entries, do not read as current)
      > docs/results/RESULTS_PLAYBOOK.md   pre-reasoned read-ins for pending results
        > docs/project/context.md      dated narrative bullets, newest first — §0 only is current state
          > CLAUDE.md                  operating specification for this repo
```

Also read, because they were created or heavily edited in the final session
and this handoff does not attempt to reproduce their content:

- `docs/project/MECHANISM_CLAUSE_CANDIDATES.md` — three draft wordings (A/B/C)
  for the paper's central mechanism clause, with a recommendation. **Not yet
  chosen.** This is the single biggest open item — see §10.
- `docs/operations/PRE_RELEASE_SCRUB.md` — a checklist, explicitly marked
  NOT RUN, of what needs doing before any public release.
- `docs/README.md` — the index over the whole `docs/` tree.
- `results/diag/README.md` — documents the artifact status-tag naming
  convention (`_INVALID_<reason>`, `_PRE-AMENDMENT-K`, `_RESTRICTED`).

Later dated amendments in `PREREGISTRATION.md` supersede earlier ones. Check
the date before citing anything from it.

---

## 3. Where everything lives

**Local repo:** `/Users/apple/dev/kv-caching` is canonical.
`~/Desktop/kv-caching` is a symlink to the same inode — confirmed directly,
not assumed. Either path works; prefer the `~/dev` one since that's what the
repo's own docs consistently use now.

**GitHub, both private:**
- `origin` → `whyeventry0103/kv-caching`
- `backup` → `whyeventry0103/kv-caching-backup`

Verified via `gh repo view` (not assumed) that both are `"visibility":"PRIVATE"`
as of the last session. Re-verify if anything about repo visibility ever
seems in question before pushing — a push to a private repo is a backup
operation; a push to a public one is publication, and those are different
decisions.

**Remote GPU server:** SSH alias `kvcache-gpu`. 2× RTX 6000 Ada (~49 GB each,
sm_89), driver 570.207 / CUDA 12.8, ~503 GB RAM. Project checked out at
`~/kv-caching` on the remote; storage under
`~/kvcache-storage/{data,cache,runs}`. Environment via `uv`, Python 3.11.

**Remote git is NOT a live clone.** `sync_source.sh` never transfers `.git`,
so the remote's git checkout is permanently frozen at whatever commit it was
originally cloned from. `make parity`'s commit check does **not** run
`git rev-parse HEAD` on the remote (that would always mismatch) — it compares
`.sync-state`'s `source_commit` field, written atomically with the synced
files, and separately reports the frozen clone hash labelled as such. This
was a real bug, fixed late in the last session (see §8, item 4).

---

## 4. Repo layout — annotated

```
kv-caching/
├── CLAUDE.md                     operating spec — read it
├── LICENSE                       MIT, added late in the project
├── README.md                     public-facing root readme
├── Makefile                      lint / test / dryrun / audit / parity / sync targets
├── pyproject.toml
├── requirements.txt / -legacy-tf452.txt / -next.txt
├── paper_numbers.json            GENERATED — the catalogue every paper number is checked against
├── paper_claims.json             GENERATED — pins ~79+ load-bearing numbers to the SPECIFIC quantity
│                                  they assert (value-matching alone can't separate two numbers that
│                                  round alike; this pins intent, not just value)
├── kv_readout_vs_trajectory_paper.{tex,pdf,aux,log,out}
├── configs/
│   ├── models.yaml, benchmarks.yaml, presses.yaml, sampling.yaml, remote.yaml
│   ├── nightly_queue.yaml        the orchestrator's queue — as of the last session, ALL 45 JOBS COMPLETE,
│   │                             nothing queued, GPUs idle
│   └── experiments/              one YAML per run — ~70 configs, includes shape_p2v2_restricted.yaml
│                                  (used for the final Meeting-#2 phase read) and replay_ratio_matched_m3.yaml
├── docs/
│   ├── README.md                 index over this tree
│   ├── protocol/PREREGISTRATION.md
│   ├── results/{PAPER_FINDINGS,RESULTS_PLAYBOOK,RESULTS}.md, RESULTS.html
│   ├── operations/{remoteserver,WORKFLOW,REMOTE_SETUP,complete_setup,
│   │                INTEGRATION_CHECKLIST,FIGURES_PLAYBOOK,PRE_RELEASE_SCRUB}.md
│   ├── project/{context,scoop-watch,MECHANISM_CLAUSE_CANDIDATES}.md
│   ├── design/{KVPRESS_API_PROBE,M4_REPLAY_DESIGN,METRICS,RKV_PORT,STACK_MIGRATION_M2_5}.md
│   └── strategy/                 INTERNAL — see §10, item on docs/strategy/ visibility.
│                                  4 files tracked+pushed (BEST_POSSIBLE_PAPER, COAUTHOR_REVIEW,
│                                  PROPOSED_REFRAME, SUBMISSION_STRATEGY), 2 gitignored/local-only
│                                  (publication_strategy, research_landscape). Inconsistent, unresolved.
├── figures/                      fig0_protocol through fig6_aime, .pdf + .png each
├── results/
│   ├── diag/                     the artifact catalogue everything is checked against
│   │   ├── README.md             status-tag naming convention, see §2
│   │   ├── amendment-k/          K-series: additivity, resolution, R-KV published config, shape, add, grader
│   │   ├── amendment-p/          P-series: M.3, restoration, onset, shape (Meeting #2 work)
│   │   ├── m4-*-summary.json     CANONICAL cell summaries — read from amendment-k/, see landmine #6
│   │   ├── m4-*-summary_PRE-AMENDMENT-K.json   the OLD pre-Amendment-K values, renamed not deleted
│   │   └── (various report/audit artifacts)
│   └── runs/                     raw JSONL generation output, per-run directories
├── scratch/coherence-audit/      manifests from the last coherence-audit pass — NOT gitignored-checked,
│                                 treat as scratch, safe to regenerate/delete
├── scripts/                      thin orchestration only — has its own scripts/CLAUDE.md, read it
│   └── (see §7 and §8 for the ones that matter operationally)
├── src/kvcache/                  the actual science — presses, metrics, decomposition, stats, replay
└── tests/                        mirrors src/kvcache module names
```

---

## 5. Environment & stack

Canonical (post-M2.5 migration):
```
python 3.11, torch 2.7.1+cu126, transformers 4.56.2, tokenizers 0.22.2,
kvpress 0.5.3, datasets 2.21.0, accelerate 1.7.0,
numpy 2.2.6, scipy 1.15.3, sympy 1.14.0
```
Legacy environment preserved on the remote as `.venv-legacy-tf452` — do not
use it for anything result-bearing; it exists only so the pre-migration stack
is reproducible if ever needed.

Why the stack moved: stock kvpress 0.2.10 compressed prefill only, not
decoding — useless for this project's regime. `DecodingPress` first shipped
in kvpress 0.5.0, which requires transformers ≥4.56.

Pinned model revisions live in `configs/models.yaml`. Never substitute a
guessed revision — `TODO(human)` sentinels in `models.py`/`datasets.py` exist
specifically to force a human to pin them.

---

## 6. The local gate

```bash
make lint      # ruff check . && black --check .   (88 cols, py311)
make test      # pytest -q, CPU-only, no weights    (313 pass as of last session)
make dryrun    # every runner script × every config, in --dry-run mode
make audit     # rebuilds paper_numbers.json, then runs BOTH:
               #   audit_paper_numbers.py --strict   (every literal has a source)
               #   audit_paper_claims.py --strict     (79+ pinned numbers assert the right QUANTITY)
make parity    # three-way: git commit (via .sync-state, not remote git), source-tree hash
               #   (root-anchored excludes now — see landmine #4), all results/diag/ artifacts by sha256
```

**Run every gate bare.** `make lint 2>&1 | tail -1` takes tail's exit code,
not make's — this exact mistake has bitten this project before. Never pipe a
gate through anything.

`rederive_artifacts.py` is remote-only (needs the raw JSONL rows, which don't
exist locally) — it re-runs every producing report against raw rows on the
remote and diffs field-by-field against the committed artifact. Last full run:
29/29 (later grew as more artifacts were added) identical, catching one real
defect (a stale compute-hours allowlist) along the way.

Sync: `make sync` (dry-run preview) → `make sync-apply` (real). `sync_source.sh`
never uses `--delete` and never transfers `.git`, environments, data, caches,
runs, logs, or `results/` (root-only, see landmine #4). Any change to source
or config must be synced before it can be launched remotely.

---

## 7. Remote orchestration: cron, queue, launch mechanics

**Cron (Dubai time, UTC+4):** fires at 01:45, then ticks every 10 minutes
through 10:50. This is the standing nightly window — do not change it without
a reason. It runs `scripts/nightly_autorun.py` against
`configs/nightly_queue.yaml`.

**Queue state as of the last session:** `nightly_queue.yaml` has 45 jobs, and
**all 45 are complete**. Nothing is queued. Both GPUs idle. If you need to run
something new, you are adding a fresh entry, not resuming an existing one.

**How a job actually launches:**
- `scripts/launch_tmux.sh --i-understand <session> <stage> <config>` is the
  entry point; it calls `scripts/launch.sh`, which dispatches to one of
  several commands (grid, readout, replay, bank, shape, restoration, etc.).
- The orchestrator (`nightly_autorun.py`) **always appends `--resume --seeds
  <list>`** to every launch it makes. Any new runner script MUST accept these
  flags, even if it ignores them, or it will die `rc=2` before writing a
  single row and get silently marked "stuck" by the orchestrator (see
  landmine #1).
- **tmux session naming is load-bearing.** The orchestrator only recognises
  sessions named `<run_name>-g<idx>` as "ours" — that's how it decides a GPU
  is occupied and avoids double-launching. A manually launched session with
  any other name (this has happened — e.g. `rextA`/`rextB`) is **invisible**
  to the orchestrator, which will then launch a duplicate job onto the same
  GPU at the next cron tick. See landmine #2.
- `KVCACHE_RUN_NAME` env var and the tmux session name must match — an
  earlier bug had `launch_tmux.sh` silently overwriting `KVCACHE_RUN_NAME`
  with the session name, so a caller's intended run name was ignored. Fixed,
  but worth knowing the failure mode existed.
- Storage/free-space guard: launchers refuse without a verified integer
  `KVCACHE_PERSONAL_FREE_GB`. Personal budget is **≈70 GB**; last measured
  usage was **68.4 GB, i.e. only ~1.6 GB headroom**. Disk reclaim was
  attempted and came back nearly empty-handed — see landmine #10. **If you
  are about to queue any new GPU-heavy job, check free space first**; it will
  likely need clearing (from the model/uv cache, never from `results/`) before
  it can run.
- Download guards: `--allow-model-download` / `--allow-dataset-download`
  required explicitly, never implied.

---

## 8. Hard-won lessons / landmines — read before touching anything

Numbered for reference, not in strict chronological order:

1. **The orchestrator always adds `--resume --seeds <list>`.** A hand
   dry-run that omits these flags tests something different from what
   actually launches. Every new runner needs a regression test asserting it
   accepts these flags — one earlier fix's first regression test passed on
   the *broken* code because it derived its expected-command list from the
   queue rather than from `launch.sh`'s dispatch table; it was later
   corrected to derive from the dispatch table so all 9 commands are checked,
   not just the ones currently queued.

2. **Tmux session names must be `<run_name>-g<idx>`.** Anything else is
   invisible to the orchestrator's occupancy check and risks a duplicate
   launch burning GPU time.

3. **`pkill -f <pattern>` can match its own shell's command line**,
   including the `pkill -f <pattern>` invocation itself if the pattern is
   loose. Bracket the pattern (e.g. `[r]ederive_artifacts` instead of
   `rederive_artifacts`) to avoid self-matching. This has caused at least one
   self-inflicted SSH session kill.

4. **Exclude patterns must be root-anchored, not substring-matched.**
   `sync_source.sh`'s exclude for `results/` was correctly anchored to the
   repo root. `source_hash.py`'s exclude for the same name was **not**
   anchored, so it matched `docs/results/` too — silently excluding
   `PAPER_FINDINGS.md` and friends from the parity hash. Parity could report
   OK while `PAPER_FINDINGS.md` differed between local and remote by dozens
   of lines. Fixed once in `sync_source.sh`, recurred later in
   `source_hash.py` — the lesson: a fix in one place doesn't mean the pattern
   was checked everywhere else it appears. If you add a new exclude anywhere,
   check both files.

5. **Generated files get fixed at the generator, never by hand.**
   `docs/results/RESULTS.md` is produced by `scripts/export_results.py`. A
   hand-edit to the output file is not a fix — it's a change with a countdown
   timer on it, silently reverted the next time the exporter runs.

6. **Nine `m4-*-summary.json` files exist at two paths with materially
   different values, by design — read from `amendment-k/`, not top-level.**
   `results/diag/m4-*-summary.json` (top level) are the **pre-Amendment-K**
   values, measured against the unrestricted 0.8825 full-cache bank.
   `results/diag/amendment-k/m4-*-summary.json` are the **canonical,
   current** values, each measured against its own matched reference
   population (0.885 etc., per Amendment K's population-discipline fix).
   Both `build_paper_numbers.py` and `rederive_artifacts.py` read only from
   `amendment-k/`. The top-level copies were renamed to
   `*_PRE-AMENDMENT-K.json` (via `git mv`, so history follows) precisely so
   nobody accidentally globs the wrong directory again. One of the
   differences is a genuine **sign flip** (H2O-2048 readout_change: +0.0155
   pre-Amendment-K vs −0.0051 canonical) — a sharp illustration of why this
   mattered. All ten `PAPER_FINDINGS.md` citations of the old top-level path
   are **inside the archival fence** (`AUDIT:ARCHIVAL-BEGIN`/`-END`) and were
   correct when written — they were deliberately left untouched rather than
   rewritten, per the project's standing rule that dated archival entries are
   never retroactively edited (a forward-pointer note at the fence explains
   how to resolve an old citation to the renamed file instead).

7. **`results/diag/` basename uniqueness is enforced, with a principled
   exemption, not a path allowlist.** "Role names" — `summary.json`,
   `table.md`, `records.json`, which describe a file's *role* inside its own
   per-unit directory and are never cited bare in prose — are exempt.
   "Identity names" — e.g. `m4-rkv-256-summary.json`, cited bare in text —
   must be globally unique. Adding a fourth role name requires a conscious
   edit to the test, not silent passing.

8. **Display precision is `.3f` of the artifact float, with one documented
   exception.** The Task 2 restoration family (matched-arm effects) lives at
   the 4th decimal — at `.3f`, Random's point estimate literally equals its
   own displayed upper CI bound, making a correct null look like a malformed
   interval. That family is displayed at `.4f`, with the exception
   **documented directly beside the pinned convention** in
   `PAPER_FINDINGS.md` so nobody "corrects" it back later. Do not extend this
   exception elsewhere without equally strong justification.

9. **Never re-derive a number from a rounded intermediate.** The artifact
   float `−0.248547` rounds correctly to `−0.249` at `.3f`. A prior draft had
   carried the 4-decimal intermediate `−0.2485` and rounded *that* to
   `−0.248` — wrong. Always round from the raw artifact float, never from an
   already-rounded value someone wrote down earlier (including in this very
   document — verify, don't propagate).

10. **Storage headroom is nearly exhausted, and reclaim has nothing left to
    give.** At last check: 68.4 GB used of ~70 GB budget. The HuggingFace
    model cache had **zero** redundant blobs (every blob is snapshot-
    referenced). The uv package cache is **~98% hardlinked into the venvs**
    — clearing it frees almost nothing (measured: ~39 MiB). The only
    meaningful reclaim path, if one is ever needed, is dropping an entire
    model (e.g. the 14B model, ~28 GB) — that is a decision, not routine
    cleanup. **Do not delete** the invalidated-but-retained runs
    (`restoration-diss256`, `replay-ratio-matched-m3` — the v1, invalidated
    versions, kept deliberately as permanent record of the instrument bugs)
    or `pilot-readout-v2`/`pilot-replay-compression-v1` (referenced only from
    *inside* `m4-pilot-v1/summary.json`, invisible to a filename-based
    orphan search).

11. **A prose quantity is invisible to `audit --strict`.** The gate checks
    literal numbers, not English. "Five to six times the baseline" passed
    the gate while being false (measured: 6.49× and 5.49×, and the range
    excludes 6.49). A separate framing/human review pass is the only thing
    that catches this class of error — the numeric audit structurally
    cannot.

12. **Instrumented presses need `build_press(..., instrumented=True)`, not a
    bare `DecodingPress`.** Two builders (`build_ratio_matched_press`,
    `build_restoration_press`) were built on a plain `DecodingPress` that
    lacked `enabled`/`profile_step`. Consequence: the restoration trigger
    (which reads `profile_step`) would never fire, silently degenerating all
    four arms to the null; and `press_enabled(press, False)` — which is what
    keeps the full-cache reference forward uncompressed — would be ignored,
    compressing the reference too. The existing unit tests didn't catch this
    because they inject a fake base object with `profile_step` pre-set.
    Fixed; any future instrumented-press builder needs the same care and an
    end-to-end (not just fake-base) test.

13. **A monitor/waiter watching `*.json` under a profiles directory will
    double-count once sidecars exist.** Restoration and ratio-matched runs
    write `.restoration.json` / `.ratio_trace.json` sidecars beside the
    `.json` profile files. A bare glob counts both, corrupting progress
    numbers (this actually happened — reported progress was 2× the truth for
    a while) and, worse, a profile *reader* that picks up a sidecar crashes
    (`KeyError: 'estimate'`, since sidecars don't have that key). Fixed with
    one shared predicate, `kvcache.replay.is_profile_file`, used by every
    reader — if you write a new reader, use it.

14. **`plan_reinsertion`'s eviction pool must be drawn from the whole live
    cache, not filtered against the reinsertion stash.** An earlier version
    intersected two different index spaces (stash indices vs. live-cache
    indices) under the mistaken belief this was needed to avoid double-
    selecting a position — it isn't, because a stashed (evicted) entry is by
    definition not in the live cache, so the intersection was a no-op in
    intent but had the real effect of numerically excluding live cache
    positions that happened to share a number with a stash index. This
    biased the `matched` restoration arm against evicting recent entries.
    Fixed; regression test asserts the eviction pool is never filtered by
    stash-index overlap.

15. **The eviction RNG stream must be independent of the reinsertion RNG
    stream**, seeded identically across arms, so `matched` and `random` evict
    the *same* set and differ *only* in which entries return. Drawing both
    choices from one stream made the eviction set depend on how many draws
    the reinsertion choice had already consumed — a subtle, silent bias.
    Fixed via a separate `evict_rng`.

16. **A "ratio-matched" press must compute its target from the LOGICAL
    context (cumulative evicted + current cache), never from the current
    cache size alone.** `keys.shape[2]` is the *output* of the previous
    compression, so computing `target_size = ratio × keys.shape[2]` creates
    a feedback loop that collapses to the floor and never recovers — this
    happened for real (M.3 v1: 100% of 377,104 events pinned to the
    `min_budget` floor of 32, across all 200 traces). The fix reconstructs
    the true context as `cumulative_evicted_per_layer + current_cache_size`
    (accumulated **per layer**, since `compress` fires once per layer per
    event — a single counter would multiply the total by layer count). The
    unit test that should have caught this originally used a fake base whose
    `compress` returned keys unchanged (never actually evicting), so the
    feedback loop could never form in the test — the test was replaced with
    one that actually evicts.

17. **Restoration reads must anchor on the ACTUAL step the intervention
    landed, not the nominal pre-registered trigger step.** The press can
    only intervene during a compression event that already has a full
    stash, so a nominal trigger of (e.g.) 130 can land at 192 or 256
    depending on prompt length. Anchoring the read window at the nominal
    step instead of the actual `applied_step` would dilute the effect by a
    trace-dependent factor. `restoration_read` takes `applied_steps` per
    trace and anchors there, recording which anchoring convention was used.

18. **Data from a buggy instrument is invalidated wholesale and re-run,
    never patched in place.** Both major Meeting-#2 powered cells
    (restoration, M.3) had a v1 that turned out to be measuring the wrong
    thing due to a real bug in the intervention code, not in the analysis.
    Both were invalidated entirely and re-run as v2 under a new run name; the
    invalid v1 output was kept on disk under its original name as permanent
    record (never deleted, never silently superseded) with an `_INVALID_
    <reason>` suffix on any derived artifact. If you find a bug in an
    instrument that already produced a "result," this is the pattern:
    diagnose, fix, re-run fresh, keep the old output labelled invalid, and
    write up both the bug and — if applicable — any wrong hypothesis that
    was floated about it before the re-run clarified things (this happened:
    a hypothesis about why v1's control arm was so large turned out to be
    wrong once v2 landed, and that wrong hypothesis is recorded on purpose).

19. **A pre-registered read condition is written into `PREREGISTRATION.md`
    BEFORE any computation, every time.** This is not bureaucracy — it is
    what makes "the branch wasn't picked after seeing the numbers" a
    checkable claim rather than an assertion. Every powered analysis in this
    project's history follows: write the branch conditions → run the
    analysis → apply whichever branch the numbers actually hit → report
    plainly, including if it's a "does not survive" or "refuted" outcome.
    **Never invent a favourable-sounding branch after seeing the data, and
    never skip writing the condition down first.**

20. **`--dry-run` on the orchestrator ignores tmux session state entirely**
    (checked at one point: line ~484 of `nightly_autorun.py` at the time),
    so a dry-run cannot validate session-naming fixes — you have to test the
    real function directly.

21. **The anonymized-repo-mirror footnote in the `.tex` (around line
    ~493 as of the last edit) promises something that does not exist.**
    The paper's repo-URL footnote asserts the double-blind submission
    carries an anonymized mirror. No such mirror has been built. This is
    now item 2 of `docs/operations/PRE_RELEASE_SCRUB.md`. It has two
    mutually exclusive correct resolutions depending on an open decision
    (see §10) — don't build the mirror reflexively; the right fix might be
    to delete the footnote instead.

---

## 9. Exact state as of session end — commit chain and what landed

**Local HEAD:** `78fce25`
**`origin/main`:** `8c1026b`
**`backup/main`:** `8c1026b`

Local is **3 commits ahead of both remotes, unpushed**:

| Commit | What it is | Pushed? |
|---|---|---|
| `8c1026b` | Push authorization was scoped to exactly this point (push both remotes, disk reclaim, orphan annotation, pre-release scrub checklist written) | **Yes — both remotes** |
| `09be3b3` | The 7 settled Meeting-#2 paper edits (Task 2 null, Task 3 budget-effect, Task 4 degeneracy, StreamingLLM withdrawal, both instrument invalidations, the wrong-hypothesis record, the windowing-bug disclosure) | **No** |
| `076a062` | 4 fixes from an independent framing review of the above (display precision exception for Task 2, both specificity ratios stated explicitly, per-subsection realized-ratio mentions, the paired M.3 site restated with no causal connective) | **No** |
| `78fce25` | Phase 4: `docs/project/MECHANISM_CLAUSE_CANDIDATES.md` written — 3 candidate wordings for the thesis mechanism clause, no `.tex` edit | **No** |

**Do not assume push authorization carries forward.** It was explicitly
scoped to the `8c1026b` push only. Ask before pushing `09be3b3` onward,
even though `origin`/`backup` are both confirmed private.

**Gates at last check, all bare:** lint clean, 313 tests passed, dryrun
clean, `audit --strict` clean, `parity` clean (commit line, source tree
including `docs/results/`, all artifacts — first fully clean parity in the
project's history, achieved by fixing landmine #4).

**Paper page count:** 29 pages (was 26 before the Meeting-#2 edits landed),
against ICLR's 9-page limit. Unresolved — see §10.

---

## 10. What is open — prioritized

### Blocking / highest priority

1. **The mechanism-clause wording is not chosen.** All the evidence exists.
   `docs/project/MECHANISM_CLAUSE_CANDIDATES.md` has three candidates:
   - **A — regime-anchored** (recommended by the last session): makes no
     causal claim, cheapest to propagate (clause swap at 4 sites, one
     retitle).
   - **B — mechanism-forward**: leads with the licensed within-cycle
     eviction-locking result. The last session's own review found this
     candidate loses to a reviewer attack (leading with a phase statistic
     while the 34-point headline loss has no attached mechanism looks like
     a bait-and-switch), and further found that StreamingLLM's restricted
     peak is phase 1, not phase 0 like Random's — so a wording claiming
     exact eviction-step locking is one step off for one of the two
     policies. Weakest candidate.
   - **C — explicitly two-level**: most faithful, cannot be misread, but
     costs a rewrite at every one of the 7 marked sites and makes the
     thesis sentence carry a structural distinction rather than a finding.
   - The recommendation is **A**, retitling §4.4, **unless** `fig4_tvprofile`
     gets re-plotted with a ratio-matched overlay (see next item), in which
     case **C** becomes the better fit.
   - A genuine open question the last session flagged: whether the clause
     should be **cut entirely** rather than restated, since "≈zero at the
     moment of eviction" is arguably an instrument-validity property (true
     by construction under teacher forcing) rather than a finding, and the
     paper already uses it elsewhere as a validity control.
   - **This is a human decision, explicitly not Claude's to make.** Do not
     pick a candidate and commit it without being told to.
   - All 7 `TODO(M.3-RESTATEMENT)` markers in the `.tex` are still in place
     (verified: they're non-rendering LaTeX comments, stripping them leaves
     the file byte-identical, confirmed by a PDF rebuild). Grep for that
     exact token to find every site. Propagating a chosen candidate is
     **not** a find-and-replace — the last session's own note says sites 5
     and 7 (Threats to Validity, the instrument-invalidation paragraphs)
     carry argumentative weight that needs a full read under the new
     wording, not a mechanical substitution.

2. **`fig4_tvprofile` likely needs re-plotting, not just re-captioning.**
   Its x-axis is literally "decoding step relative to first eviction" —
   which, at fixed budget, is a deterministic function of the realized
   compression ratio. The figure's shape visually asserts a horizon
   reading regardless of what the caption says. The proposed fix uses data
   that **already exists**: overlay the ratio-matched arm on the same axes
   (one curve rises, the other flattens/reverses). This is a plotting task,
   not new GPU work.

3. **Push authorization for `09be3b3`, `076a062`, `78fce25`.** Not
   inherited from the earlier authorization. Ask explicitly.

### High priority, not blocking

4. **Page budget: 29 pages against ICLR's 9.** Structural decision about
   what stays in the body vs. moves to an appendix vs. gets cut. Everything
   still pending (mechanism clause propagation) will likely *add* material
   before this can be finalized, so this probably comes after item 1.

5. **`docs/strategy/` visibility is inconsistent and unresolved.** 4 files
   tracked and already pushed to both remotes' history
   (`BEST_POSSIBLE_PAPER.md`, `COAUTHOR_REVIEW.md`, `PROPOSED_REFRAME.md`,
   `SUBMISSION_STRATEGY.md`); 2 gitignored/local-only
   (`publication_strategy.md`, `research_landscape.md`). Since the repo is
   currently private, exposure is zero — but if/when the repo is ever made
   public, the 4 already-tracked files need either an accepted-as-public
   decision or a history rewrite (not just a future `.gitignore` entry,
   which would not remove them from existing history). Logged in
   `PRE_RELEASE_SCRUB.md` as item 3.

6. **The anonymized-mirror footnote (`.tex` ~line 493)** — see landmine #21.
   Depends on an ICLR double-blind-submission-copy scope decision that has
   not been made. The owner has previously stated they do **not** want the
   work anonymous in general — but whether that extends to the ICLR
   submission copy specifically (as opposed to the public preprint/repo) is
   a separate, unconfirmed question. Resolve the scope question first, then
   either build the mirror or delete the footnote — do not do either
   reflexively.

### Lower priority / tech debt

7. **`CLAUDE.md:98`'s claim that a personal email is embedded in
   `scripts/nightly_report.py` is false**, verified independently at least
   twice (the file has zero address literals; the recipient comes from
   `KVCACHE_REPORT_TO`/`KVCACHE_REPORT_MX` env vars with no compiled-in
   default). Unclear from the available history whether the one-line fix at
   `CLAUDE.md:98` itself ever actually landed, as opposed to just being
   flagged. **Check this file directly before trusting either way.**

8. **~155 result-precision literals in the `.tex` remain sourced but not
   pinned to their specific quantity** (as opposed to the ~79+ that are
   pinned via `paper_claims.json`). Not known wrong — every one traces to
   *some* catalogue entry and every backing artifact re-derives — just
   weaker than the pinned set. `make audit` prints the current count.

9. **`docs/operations/PRE_RELEASE_SCRUB.md` exists and is marked NOT RUN.**
   Execute only as one deliberate step, if and when the repo is actually
   made public — not routine.

---

## 11. Experimental status — everything is done, with the final numbers

**All experimental work is complete.** No GPU runs are queued or pending.
Every question from Meeting #2 has an answer. This section summarizes the
final state of each; **re-verify every number against
`docs/results/PAPER_FINDINGS.md` and the cited artifact under
`results/diag/amendment-p/` before using any of it** — this is a summary of
a long transcript, not a re-derivation.

### Meeting #2, Task 1 — per-step divergence / eviction-locking

Extended replay pass: 800 profiles (25 problems × 4 seeds × 4 presses ×
budgets {256, 2048}) — first replay data ever collected for R-KV. Budgets
must not be pooled (the phenomenon is essentially absent at 2048).

**⚠ Citation hazard: there are multiple, non-interchangeable StreamingLLM
phase-amplitude numbers across the project's history.** In order:
1. Original P.2 pilot (n=100, single sample): excess ≈ +0.0384
   [+0.0013, +0.0560] — excluded zero narrowly.
2. Independent replication sample: +0.0195 [−0.0212, +0.0499] — did **not**
   exclude zero. Per the project's standing rule (never re-cut a
   pre-registered verdict in our favour), (1) was **withdrawn**, not
   averaged with (2).
3. A separate whole-trace measurement taken directly on the M.3 ratio-
   matched cell's own traces (a different underlying dataset from the P.2
   replay cells): +0.102 [0.065, 0.128], peak phase 6.
4. The final, restricted-window ("Branch A") result — the one that should
   actually be cited for the current paper text: **+0.076 [0.031, 0.099],
   peak phase 1.**
Do not conflate these. Verify which one any given piece of prose is
citing before trusting it.

Random's number is comparatively simple: replicates cleanly across samples,
final restricted-window result **+0.252 [0.184, 0.271], peak phase 0** (the
prune step exactly).

R-KV: marginal (+0.062 [0.0016, 0.078], peak phase 9 — not aligned to the
prune step). H2O-style: null in every sample.

**Framing that must travel with any of this:** within a prune cycle,
divergence is genuinely eviction-locked for Random — that's licensed as a
mechanism. **Across** cycles, the rising envelope is a compression-ratio
artifact per M.3, below — that is explicitly **not** licensed as a horizon
mechanism. Both hold simultaneously and must never be cited one without the
other.

### Meeting #2, Task 2 — restoration (does reinserting evicted entries help?)

**Final answer: no.** v1 was invalidated (index-space bug, landmine #14);
v2 is the valid result. Instrument gate: 800/800 pass, every reinserted
entry bitwise identical to what was evicted.

Matched-arm effect (positive would mean divergence reduced):
- Random: **−0.0013 [−0.0019, −0.0007]**, baseline divergence 0.0383
- StreamingLLM: **−0.0015 [−0.0022, −0.0009]**, baseline divergence 0.0445

Both exclude zero but are ~3% of baseline — write as *indistinguishable
from no help*, never as measured harm. The relaxed arm (allowed to grow the
cache) does no better, so it isn't a budget-match artifact.

The interesting result is the **specificity control**: reinserting the same
*number* of randomly chosen (rather than the actual just-evicted) entries
**increases** divergence by **6.49× baseline (Random)** / **5.49× baseline
(StreamingLLM)** — state both ratios explicitly, never as a vague range.
The cache is acutely sensitive to *which* entries occupy it; restoring the
specific ones a policy just evicted is nearly neutral.

Scope: one model, MATH-500, budget 256, one trigger, `reinsert_fraction`
0.25, 200-step read window. Not a general claim about reinsertion.

### Meeting #2, Task 3 — does AUC-TV predict the residual?

**Final answer: BUDGET EFFECT ONLY**, a pre-registered branch, taken as
written. Pooled Spearman ρ = **−0.857 [−0.893, −0.643]** over 7 cells looks
strong but is carried entirely by the budget-256-vs-2048 contrast. Within
budget 256 alone: **ρ = −0.200 [−0.400, +0.800]** over only 4 cells — does
not exclude zero. Write as "not established at this n," never "refuted" —
the within-budget test is underpowered, and rank correlation is a weaker
claim than proportionality (linearity through the origin was never tested).

### Meeting #2, Task 4 — JSD / top-1 / rank displacement onset

**Final answer: the pre-registered crossing test is degenerate by
construction**, not just unreadable in one direction. Pre-onset TV is
exactly 0.0 in teacher-forced replay, so the crossing rule reduces to
"first non-zero step" — and TV and JSD share a non-zero support on 382 of
400 profiles (not universally, as an earlier draft over-claimed). No JSD
onset-sensitivity claim is available in either direction from this test.

One genuine finding survives: rank displacement is **coarser** at onset
(median first-crossing 280–451 steps after onset) than TV/JSD (median 192)
— the opposite of what an earlier docstring asserted from mechanism rather
than data. That docstring has been corrected in place.

### M.3 — the ratio-matched control (the big structural result)

**Final answer, "Branch A"/FLATTENS: the rising divergence staircase is
mostly a compression-ratio artifact, not a licensed horizon/compounding
mechanism.** v1 was invalidated (feedback-loop bug, landmine #16); v2 is
valid. Read condition (above-floor coverage etc.) was pre-registered and met
(measured 83.5% vs. 83.9% predicted).

Drift (last-quarter mean ÷ first-quarter mean; >1 means divergence rises
across the trace):
- Random: fixed budget **1.201 [1.061, 1.339]** → ratio-matched
  **0.662 [0.614, 0.709]**
- StreamingLLM: fixed budget **1.299 [1.100, 1.520]** → ratio-matched
  **0.739 [0.680, 0.805]**

Both take the FLATTENS branch. Three qualifiers that must travel with this
result, always:
1. Holding the ratio fixed *requires* letting the absolute cache grow
   (32 → 626 tokens observed). This does **not** show horizon length is
   irrelevant at fixed cache size — that experiment was never run.
2. The comparison is restricted to the above-floor window, which trims out
   the early trace where most of the rise happens — whole-trace drift is
   considerably larger (2.93/2.26) than the restricted-window figures above.
3. The ratio-matched arm doesn't merely flatten, it **reverses** (drift
   below 1 for both presses) — divergence *falls* across the trace when the
   budget grows with context.

**This is the result that gates Task 1's eviction-locking interpretation**
(see the framing note under Task 1, above) and is the reason the mechanism
clause needs restating rather than left as originally written.

---

## 12. Relationship to the Notion workspace and the other Claude

The project owner also maintains a Notion workspace ("KV Cache Research")
that is kept in sync by a **separate** Claude instance — a chat assistant
(not Claude Code), running in claude.ai, with access to Notion but not to
this repo or the remote server directly. That Claude has been building out a
Cell register, Decision log, Work queue, and other pages tracking exactly
the kind of state this handoff document summarizes.

**Claude Code does not need to touch Notion.** The owner relays results back
to the Notion-side Claude by pasting Claude Code's own reports into that
other chat. If you (the new Claude Code session) produce a report, structure
it the way the prior session did — clear headers, exact numbers with CIs,
explicit statement of what changed and what's still open — since that's the
format the owner has been copy-pasting across.

One thing worth knowing: the Notion-side Claude has repeatedly caught real
transcription errors by insisting on re-deriving numbers from artifacts
rather than trusting prose (including catching an error in its own earlier
work, twice). If a number you report gets pushed back on, the correct
response is to re-check the artifact, not to defend the number.

---

## 13. Immediate next action

There is no GPU work to resume. The very next useful thing to do, in order:

1. Confirm this handoff's state summary against disk — `git log --oneline
   -5`, `git status`, `make lint test dryrun audit parity` (all bare), and
   read `docs/project/context.md` §0 fresh, since it may have been updated
   after this document was written if any work happened between sessions.
2. Read `docs/project/MECHANISM_CLAUSE_CANDIDATES.md` in full.
3. **Do not pick a candidate.** Report back to the owner that this is the
   open decision, exactly as framed in §10 item 1, and wait for a choice
   before touching the `.tex`.
4. If the owner has already made that decision by the time you read this
   (check `context.md` §0 first), propagate it to all 7
   `TODO(M.3-RESTATEMENT)` sites via a careful read of each, not a
   find-and-replace, then rebuild the PDF and report the new page count.

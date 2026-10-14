# Review of 946c76a: bounded repair of the two remaining execution issues (2026-10-14)

**Scope.** This pass covers only the two issues the review left open. It is not
a general audit. No GPU was used, no historical run was regenerated, and no
approval changed. The essential pair and smoke keep their caps of 1 / 28 / 28
GPU-h.

## Results, kept separate as the review asked

- **Software acceptance: passed.** Details below.
- **Real-model smoke acceptance: not yet run.** The approved smokes run in queue
  order. The available REAL smoke measurements (2026-10-13 selector smoke)
  were revalidated with the tightened checks on the host and show **no
  measurement problem**: 15 records, 28 layers from frozen model metadata,
  relaxed target 417 for the 353-token prompt. Their identity and 15-record
  coverage were already stale, and none of their measurements or identities
  were rewritten.
- **Scientific results: none new.**

## Before / after (the review's reproduction script)

Evidence is in `results/diag/meeting5/review_946c76a/`. The "before" run is the
review's script unmodified at `946c76a`. In the "after" run, only the
interruption hook moved: the repaired runner no longer calls
`append_token_ids`, so the same interruption point (row published, sidecar
not yet) is injected in `kvcache.m5_publish._publish_one`.

| Case | Before (946c76a) | After |
|---|---|---|
| Selector gate, valid measured control | verified | verified |
| Selector, generation `protect.n_prunes` = NaN | verified | refused: "protect.n_prunes is not a valid count: nan" |
| Selector, coordinated target 255 | verified | refused: "target/sizes 255/255/[255] differ from the derived effective budget 256" |
| Selector, 0 layers / 0 pruned | verified | refused: "layers 0/0 differ from the pinned model's 28" |
| R-KV-pp gate, the same four cases | verified ×4 | verified (control), then refused ×3 with the same reasons |
| Interrupted between row and sidecar | 1 row / 0 sidecars; `--resume` fails with `FileExistsError`; still 1 / 0 | 1 / 0; `--resume` exits 0; 100 rows / 100 sidecars, one pair per key |

**Earlier acceptance still holds:** the 37 acceptance outputs of the previous
pass reproduce byte-identically (`review_6b40900_repro_after.json`).

**Archive hashes (SHA-256):**

| File | SHA-256 |
|---|---|
| before | `42049d25628c34a09a5dfbc6a41a7aa348e93dee22b3d71a24b45ec9860b7e3a` |
| after | `6033e52d49a9dcdad65c10e1b62ab8c157803a6bb1486cd2a939ef3846f8225b` |
| script | `1cf934e1f5f24b53ac22a4cea4d56eb91477269884a087600a1c761fe16ae61b` |
| adapted script | `fb0c50f0e9fd4e913b390d5ab532b49922f3822ab974fa467aebd4b51b54353d` |

## Issue 1 — the smoke gates now derive what a record cannot vouch for

`kvcache.smoke_validation` now checks the nested values too:
- `protect` fields and every post-prune size must be exact non-negative
  integers. NaN or a bool is never a count.
- A replay or generation must record at least one prune.

It also compares each record with values it derives independently:
- **Effective target:** the family's own rule, applied to the smoke config's
  budget and the frozen trace's prompt length. The expected target, actual
  target and post-prune sizes must all equal it.
- **Layer coverage:** taken from `configs/meeting5/model_metadata.json`. That
  file holds `config.json` facts frozen at the pinned revision, with the file's
  hash; there is no weight load. The gate refuses if the frozen revision
  differs from `configs/models.yaml`.
- **Replay length:** must be min(configured prefix, trace length).
- **Generation length:** must lie within the configured cap.
- **Press prompt length:** must equal the trace's.

**Fixture correction (not a weakening).** The synthetic fixture claimed 1,536
replayed tokens on traces about 1,030 long. The fixture was corrected; the
check stayed as it is.

**Tests:** `tests/test_smoke_derived_checks.py`. It covers the review's three
mutations plus seven neighbouring cases, for both families, and keeps the
consistent positive controls.

## Issue 2 — atomic pair publication and recovery (`kvcache.m5_publish`, M5 path only)

**How a condition is published.**
1. Its row and sidecar are built and validated.
2. Both are written in one atomic per-condition journal envelope (`m5-pair-v1`,
   with a pair hash).
3. Each is then published idempotently: an identical record already present is
   accepted; a different record **refuses** as a conflict.
4. The envelope is kept, renamed `*.published.json`.

**What happens on start (resume).** Only the seed shards the process owns are
touched, and this runs before the model loads:
- Pending envelopes are re-validated against the run identity and published,
  **without a new generation**.
- Torn lines and pre-journal orphans (a row without its sidecar, or the
  reverse) are first copied verbatim into `m5_quarantine/` with a reason and
  time, then removed from the live file. The condition is then regenerated
  under control.
- Token IDs are never rebuilt from text.

**Progress counting.** M5 grid jobs use the new progress kind `grid_pairs`,
which counts complete pairs only. An orphan row completes nothing.

**Tests:** `tests/test_m5_publish.py` covers:
- interruption before publication, between row and sidecar, and after both
  (none regenerates the interrupted condition, and each ends with one matching
  pair per key);
- a second resume (no-op);
- orphan rows and orphan sidecars (quarantined, then regenerated);
- conflicts (a different published record, a foreign identity, a tampered
  envelope);
- shard ownership (a seed-11 process leaves seed 23's files and envelope
  untouched);
- scheduler counting.

The legacy runners, historical runs and the approved P1/P2/P3 paths are
unchanged.

## Gates

See the commit message for this pass.

# Operational evidence for the R.16.6 / R.16.7 fixes (collected 2026-09-27)

These are primary records taken from the GPU host (`~/kv-caching/logs/nightly_autorun.log`,
the tick scripts, and a listing of the stage-record directory), plus the local commits.
Nothing below is reconstructed from memory.

## R.16.6 — figures written into the hashed source tree halted launches

Postprocessing published stage figures into `figures/`, which the preflight
check hashes. Every tick that tried to launch then refused:

```
29924-remote source changed after rsync; apply a fresh local sync
29925-expected source hash: a6d0856cfaa687aeb264bca1bb40213da7e28c39787440762a5d8bd5dec07dad
29926-actual source hash:   ee2e5654e4d74d3b5612786b1e8812bc5be8fc457b43ad867b32c17d42c3e5d6
29928:[2026-09-26 00:15:15] ABORT: preflight failed -> no launch
30254-remote source changed after rsync; apply a fresh local sync
30255-expected source hash: a6d0856cfaa687aeb264bca1bb40213da7e28c39787440762a5d8bd5dec07dad
30256-actual source hash:   ee2e5654e4d74d3b5612786b1e8812bc5be8fc457b43ad867b32c17d42c3e5d6
30258:[2026-09-26 00:25:18] ABORT: preflight failed -> no launch
```

After the fix (`910d4dc`: figures go to `results/diag/meeting3/figures/`,
regression `test_no_stage_writes_into_the_source_tree`), the misplaced files
were moved, not deleted, and the next tick passed:

```
30275:[2026-09-26 00:31:48] preflight check_remote.sh: PASS
30276:[2026-09-26 00:31:48] launched exact-lag-r5-g0 (gpu 0, seeds [89])
```

- **Running jobs were unaffected.** The preflight gates only new launches.
- **Lost time:** launches were refused on two ticks (00:15 and 00:25).
- **Latent hazard, closed 2026-09-27 (R.17 0.4b).** The standalone figure
  script's default `--out-dir` still pointed at `figures/`. It now defaults to
  `results/diag/meeting3/figures/`, pinned by
  `test_standalone_figure_script_defaults_outside_the_source_tree`.

## R.16.6 — the owner-requested out-of-window starts

Every start ran the unchanged orchestrator through a tmux loop using
`--max-hours-override`, which bypasses only the window check. Copies of the
scripts are in this directory with their sha256:

| Start | Script | sha256 |
|---|---|---|
| 2026-09-25 20:24 Dubai ("can u start the runs right now") | `manual_ticks_20260925.sh.txt` (host `~/manual_ticks.sh`) | `a5f5c8bb…51d8` |
| 2026-09-26 11:39 Dubai ("one time bypass from our nightly runs") | `manual_ticks_20260926.sh.txt` (host `~/manual_ticks_0926.sh`) | `286e487f…4480` |
| 2026-09-28 20:51 Dubai ("can you start the runs right now") | `manual_ticks_20260928.sh.txt` (host `~/manual_ticks_0928.sh`) | `b152a9d7…e8e` |

The log shows 113 `[manual tick …]` lines in total. The first loop ended with
`[manual ticks done 01:45] cron takes over`. The second runs until 2026-09-27
01:40, and every job it launches stops at 10:43.

## R.16.7 — stage-record basename collision

Before `e29c8d4` the record path was `stages/f"{stage}.json"`
(`git show e29c8d4^:scripts/postprocess_meeting3.py`, line 374). The D_acc
record `stages/dacc.json` therefore shared its basename with the reader output
`dacc.json`, and the full suite failed on real data:

```
FAILED tests/test_paper_numbers_properties.py::test_identity_bearing_artifact_basenames_are_unique_under_results_diag
AssertionError: identity-bearing basenames must resolve to one file under results/diag/, found 1:
  dacc.json -> ['meeting3/dacc.json', 'meeting3/stages/dacc.json']
1 failed, 534 passed, 48 skipped
```

- **Regression test:** `test_stage_records_never_share_a_basename_with_published_outputs`
  failed for `dacc` before the fix (`1 failed, 5 passed`) and passes after.
- **Suite after the fix:** 541 passed, 48 skipped. The uniqueness assertion is
  unchanged.

Stage-record directory on the host after the rename. The old names are left in
place because no remote deletion is authorized, and nothing reads them:

```
-rw-rw-r-- 1 sk10741 sk10741 1467 2026-09-26_01:25 capture-b256.json
-rw-rw-r-- 1 sk10741 sk10741 1467 2026-09-26_22:30 capture-b256.stage.json
-rw-rw-r-- 1 sk10741 sk10741 1755 2026-09-26_21:40 dacc.json
-rw-rw-r-- 1 sk10741 sk10741 1755 2026-09-26_22:30 dacc.stage.json
-rw-rw-r-- 1 sk10741 sk10741 1842 2026-09-26_01:25 exact-lag.json
-rw-rw-r-- 1 sk10741 sk10741 1842 2026-09-26_22:30 exact-lag.stage.json
```

`make parity` after the republish: `PARITY OK (commit, source tree, and result
artifacts agree)` (at `470a9bc`).

## Figure layout (R.16.6)

Viewing the rendered PNGs found three collisions: capture tick labels, exact-lag
error bars and legend, and D_acc error bars and legend. All three were fixed in
`48de0fa`, `8d4eef4` and `e29c8d4`. Each FINAL stage was then republished from
unchanged rows. Its TSV sha256 is unchanged, and every output hash equals its
stage record.

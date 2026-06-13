# Workflow

The authoritative operations and safety policy is
[`remoteserver.md`](../remoteserver.md). Read it before any remote action.

## Policy B loop

Policy B (rsync-first) is explicitly selected:

1. Edit source, configs, tests, and docs locally.
2. Run `make lint test dryrun` locally.
3. Preview with `bash scripts/sync_source.sh`.
4. Apply with `bash scripts/sync_source.sh --apply`.
5. Export the configured remote roots, `KVCACHE_REMOTE=1`, and a verified
   integer `KVCACHE_PERSONAL_FREE_GB`.
6. Run `scripts/check_remote.sh` before every expensive job.
7. Launch through `scripts/launch.sh --i-understand ...`, preferably in tmux or
   the site's scheduler.
8. Pull only small JSON/JSONL/CSV summaries with `scripts/sync_results.sh`.
9. Commit and push stable source changes locally.

The sync script records the local commit, dirty status, and timestamp in a
remote `.sync-state`. It never uses `--delete` and never transfers `.git`,
environments, data, caches, runs, logs, results, or generated Python files.

Local is authoritative for code/config/tests/docs. Remote is authoritative for
datasets/caches/checkpoints/logs/runs. Git is authoritative for committed
history. A local edit is not active remotely until rsync completes successfully.

# Workflow

The authoritative operations and safety policy is
[`remoteserver.md`](../remoteserver.md). Read it before any remote action.

## Policy A loop

Policy A (Git-first) is the default:

1. Edit source, configs, tests, and docs locally.
2. Run `make lint test dryrun` locally.
3. Commit and push `main`.
4. On the remote, verify a clean worktree and run
   `git pull --ff-only origin main`.
5. Export the configured remote roots and `KVCACHE_REMOTE=1`.
6. Export a verified integer `KVCACHE_PERSONAL_FREE_GB`; shared `df` output is
   not a personal quota measurement.
7. Run `scripts/check_remote.sh` before every expensive job.
8. Launch through `scripts/launch.sh --i-understand ...`, preferably in tmux or
   the site's scheduler.
9. Pull only small JSON/JSONL/CSV summaries with `scripts/sync_results.sh`.

`TODO(human)`: toggle to Policy B only by updating both `remoteserver.md` and
`configs/remote.yaml`, and record the local diff and commit for each important
rsync-first run.

Local is authoritative for code/config/tests/docs. Remote is authoritative for
datasets/caches/checkpoints/logs/runs. Git is authoritative for committed
history. A local edit is not active remotely until the remote pull is verified.

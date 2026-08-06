# Workflow

The authoritative operations and safety policy is
[`docs/operations/remoteserver.md`](remoteserver.md). Read it before any remote action.

## Policy B loop

Policy B (rsync-first) is explicitly selected:

1. Edit source, configs, tests, and docs locally.
2. Run `make lint test dryrun` locally.
3. Preview with `bash scripts/sync_source.sh`.
4. Apply with `bash scripts/sync_source.sh --apply`.
5. Export the configured remote roots, `KVCACHE_REMOTE=1`, and a verified
   integer `KVCACHE_PERSONAL_FREE_GB`.
6. Run `scripts/check_remote.sh` before every expensive job.
7. Launch every model download, dataset download, generation, replay, or other
   long-running command through `scripts/launch_tmux.sh --i-understand ...`.
   Direct foreground execution is reserved for dry-runs and short probes.
8. Pull only small JSON/JSONL/CSV summaries with `scripts/sync_results.sh`.
9. Commit and push stable source changes locally.

The sync script records the local commit, dirty status, and timestamp in a
remote `.sync-state`. It never uses `--delete` and never transfers `.git`,
environments, data, caches, runs, logs, results, or generated Python files.
It opens one temporary multiplexed SSH connection per invocation, so password
authentication is requested at most once even though verification uses several
SSH/rsync operations.

For passwordless daily use, install a passphrase-protected SSH key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_kvcache -C "kv-caching remote"
ssh-copy-id -i ~/.ssh/id_ed25519_kvcache.pub sk10741@10.224.32.112
ssh-add --apple-use-keychain ~/.ssh/id_ed25519_kvcache  # macOS
```

Then add this local `~/.ssh/config` entry:

```sshconfig
Host kvcache-gpu
    HostName 10.224.32.112
    User sk10741
    IdentityFile ~/.ssh/id_ed25519_kvcache
    IdentitiesOnly yes
    AddKeysToAgent yes
    UseKeychain yes
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

The sync script defaults to `kvcache-gpu` after this alias is tested. Override
with `KVCACHE_SSH_TARGET` only when needed. Never disable host-key checking or
commit private keys.

Each tmux launch treats `REMOTE_RUNS_ROOT/<SESSION>` as that run's root. The
frozen result convention is therefore:

```text
<REMOTE_RUNS_ROOT>/<SESSION>/<bench>/<model>/<press>-<budget>-s<seed>.jsonl
```

This preserves the schema's per-run `runs/{bench}/...` layout while preventing
smoke, pilot, and full-grid cells from appending into the same files.

Example guarded smoke launch:

```bash
bash scripts/launch_tmux.sh \
  --i-understand smoke-fullcache bank \
  configs/experiments/smoke_fullcache.yaml \
  --allow-model-download --allow-dataset-download

tmux capture-pane -pt smoke-fullcache -S -100
tail -f logs/smoke-fullcache.log
```

Local is authoritative for code/config/tests/docs. Remote is authoritative for
datasets/caches/checkpoints/logs/runs. Git is authoritative for committed
history. A local edit is not active remotely until rsync completes successfully.

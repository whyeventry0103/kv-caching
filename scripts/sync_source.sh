#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${KVCACHE_SSH_TARGET:-sk10741@10.224.32.112}"
REMOTE_PROJECT_ROOT="${KVCACHE_REMOTE_PROJECT_ROOT:-/home/sk10741/kv-caching}"
MODE="${1:-}"

if [[ "$MODE" != "" && "$MODE" != "--apply" ]]; then
  echo "usage: $0 [--apply]" >&2
  exit 2
fi

RSYNC_FLAGS=(-avzh)
if [[ "$MODE" != "--apply" ]]; then
  RSYNC_FLAGS+=(-n)
fi

echo "Action: sync local source to remote using Policy B"
echo "Source: $PROJECT_ROOT/"
echo "Destination: $REMOTE:$REMOTE_PROJECT_ROOT/"
echo "Mode: $([[ "$MODE" == "--apply" ]] && echo APPLY || echo DRY-RUN)"
echo "No --delete is used. Remote data, caches, runs, logs, and environments are excluded."

if [[ "$MODE" == "--apply" ]]; then
  ssh "$REMOTE" "mkdir -p '$REMOTE_PROJECT_ROOT'"
else
  ssh "$REMOTE" "test -d '$REMOTE_PROJECT_ROOT'" || {
    echo "remote project directory is missing; create it before dry-run" >&2
    exit 1
  }
fi

rsync "${RSYNC_FLAGS[@]}" \
  --itemize-changes \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='data/' \
  --exclude='kvcache-storage/' \
  --exclude='runs/' \
  --exclude='logs/' \
  --exclude='results/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='*.pyc' \
  --exclude='*.npy' \
  --exclude='*.safetensors' \
  "$PROJECT_ROOT/" \
  "$REMOTE:$REMOTE_PROJECT_ROOT/"

if [[ "$MODE" == "--apply" ]]; then
  COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  if command -v shasum >/dev/null; then
    DIRTY="$(git -C "$PROJECT_ROOT" status --porcelain | shasum -a 256 | awk '{print $1}')"
  else
    DIRTY="$(git -C "$PROJECT_ROOT" status --porcelain | sha256sum | awk '{print $1}')"
  fi
  TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  STATE="sync_method=rsync
source_commit=$COMMIT
status_hash=$DIRTY
synced_at=$TIMESTAMP
source=$PROJECT_ROOT"
  printf '%s\n' "$STATE" | ssh "$REMOTE" "cat > '$REMOTE_PROJECT_ROOT/.sync-state'"
  echo "Applied source sync and wrote $REMOTE_PROJECT_ROOT/.sync-state"
else
  echo "Dry-run only. Review the file list, then rerun with --apply."
fi

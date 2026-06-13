#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${KVCACHE_SSH_TARGET:-kvcache-gpu}"
REMOTE_PROJECT_ROOT="${KVCACHE_REMOTE_PROJECT_ROOT:-/home/sk10741/kv-caching}"
MODE="${1:-}"
CONTROL_DIR="/tmp/kvcache-ssh-$UID"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"
CONTROL_PATH="$CONTROL_DIR/%C"
SSH_OPTIONS=(
  -o ControlMaster=auto
  -o ControlPersist=120
  -o "ControlPath=$CONTROL_PATH"
)
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=120 -o ControlPath=$CONTROL_PATH"

close_control_connection() {
  ssh "${SSH_OPTIONS[@]}" -O exit "$REMOTE" >/dev/null 2>&1 || true
}
trap close_control_connection EXIT

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
echo "SSH multiplexing is enabled; this sync should request the password at most once."

ssh "${SSH_OPTIONS[@]}" -MNf "$REMOTE"

if [[ "$MODE" == "--apply" ]]; then
  ssh "${SSH_OPTIONS[@]}" "$REMOTE" "mkdir -p '$REMOTE_PROJECT_ROOT'"
else
  ssh "${SSH_OPTIONS[@]}" "$REMOTE" "test -d '$REMOTE_PROJECT_ROOT'" || {
    echo "remote project directory is missing; create it before dry-run" >&2
    exit 1
  }
fi

rsync "${RSYNC_FLAGS[@]}" \
  -e "$RSYNC_SSH" \
  --itemize-changes \
  --exclude='.git/' \
  --exclude='.claude/' \
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
  TREE_HASH="$(
    PYTHONPATH="$PROJECT_ROOT/src" python3 -m kvcache.source_hash "$PROJECT_ROOT"
  )"
  REMOTE_TREE_HASH="$(
    ssh "${SSH_OPTIONS[@]}" "$REMOTE" \
      "cd '$REMOTE_PROJECT_ROOT' && PYTHONPATH=src python3 -m kvcache.source_hash ."
  )"
  [[ "$TREE_HASH" == "$REMOTE_TREE_HASH" ]] || {
    echo "REFUSED: remote source hash does not match transferred local source" >&2
    exit 1
  }
  TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  STATE="sync_method=rsync
source_commit=$COMMIT
status_hash=$DIRTY
source_tree_hash=$TREE_HASH
synced_at=$TIMESTAMP
source=$PROJECT_ROOT"
  printf '%s\n' "$STATE" |
    ssh "${SSH_OPTIONS[@]}" "$REMOTE" "cat > '$REMOTE_PROJECT_ROOT/.sync-state'"
  echo "Applied source sync and wrote $REMOTE_PROJECT_ROOT/.sync-state"
else
  echo "Dry-run only. Review the file list, then rerun with --apply."
fi

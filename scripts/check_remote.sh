#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG=""
RUN_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?missing value for --config}"; shift 2 ;;
    --run-name) RUN_NAME="${2:?missing value for --run-name}"; shift 2 ;;
    *) echo "usage: $0 --config FILE --run-name NAME" >&2; exit 2 ;;
  esac
done

[[ -n "$CONFIG" && -f "$PROJECT_ROOT/$CONFIG" ]] || {
  echo "expected config does not exist: $CONFIG" >&2
  exit 1
}
[[ -n "$RUN_NAME" ]] || { echo "--run-name is required" >&2; exit 2; }
[[ "${KVCACHE_REMOTE:-}" == "1" ]] || {
  echo "REFUSED: health checks are remote-only; set KVCACHE_REMOTE=1" >&2
  exit 2
}

cd "$PROJECT_ROOT"
echo "=== IDENTITY ==="
hostname
whoami
pwd
echo "=== GIT ==="
git status --short
git rev-parse --short HEAD
echo "=== TMUX ==="
tmux ls 2>/dev/null || true
echo "=== RELEVANT PROCESSES ==="
ps -ef | grep -E 'python|torchrun|accelerate|deepspeed|ray' | grep -v grep || true
echo "=== DISK ==="
df -h .
du -sh "${REMOTE_RUNS_ROOT:?REMOTE_RUNS_ROOT is required}" 2>/dev/null || true
quota -s 2>/dev/null || echo "quota command unavailable or no quota reported"
echo "Declared personal free GB: ${KVCACHE_PERSONAL_FREE_GB:-UNSET}"
echo "=== MEMORY ==="
free -h
echo "=== GPU ==="
nvidia-smi

: "${KVCACHE_PERSONAL_FREE_GB:?set personal free GB; df reports shared capacity}"
MIN_PERSONAL_FREE_GB="${KVCACHE_MIN_PERSONAL_FREE_GB:-5}"
if (( KVCACHE_PERSONAL_FREE_GB < MIN_PERSONAL_FREE_GB )); then
  echo "REFUSED: personal headroom is below ${MIN_PERSONAL_FREE_GB} GB" >&2
  exit 2
fi
FILESYSTEM_FREE_KB="$(df -Pk "$REMOTE_RUNS_ROOT" | awk 'NR == 2 {print $4}')"
MIN_FILESYSTEM_FREE_GB="${KVCACHE_MIN_FILESYSTEM_FREE_GB:-30}"
MIN_FILESYSTEM_FREE_KB=$((MIN_FILESYSTEM_FREE_GB * 1024 * 1024))
if (( FILESYSTEM_FREE_KB < MIN_FILESYSTEM_FREE_KB )); then
  echo "REFUSED: runs filesystem has less than ${MIN_FILESYSTEM_FREE_GB} GiB free" >&2
  exit 2
fi
[[ -d "${REMOTE_DATA_ROOT:?REMOTE_DATA_ROOT is required}" ]] || {
  echo "missing REMOTE_DATA_ROOT" >&2
  exit 1
}
[[ -d "${REMOTE_CACHE_ROOT:?REMOTE_CACHE_ROOT is required}" ]] || {
  echo "missing REMOTE_CACHE_ROOT" >&2
  exit 1
}
if [[ "${KVCACHE_ALLOW_EXISTING_RUN:-}" == "1" ]]; then
  echo "RESUME: reusing existing run dir $REMOTE_RUNS_ROOT/$RUN_NAME"
else
  [[ ! -e "$REMOTE_RUNS_ROOT/$RUN_NAME" ]] || {
    echo "run-name clash: $REMOTE_RUNS_ROOT/$RUN_NAME" >&2
    exit 1
  }
fi
if ps -eo pid=,comm=,args= | awk -v self="$$" -v run="$RUN_NAME" '
  $1 != self &&
  $2 ~ /^(python([0-9.]*)?|torchrun|accelerate|deepspeed|ray)$/ &&
  index($0, run) { found = 1 }
  END { exit !found }
'; then
  echo "an active process already contains run name: $RUN_NAME" >&2
  exit 1
fi

SYNC_METHOD="$(awk '/^primary_sync_method:/ {print $2}' configs/remote.yaml)"
if [[ "$SYNC_METHOD" == "rsync" ]]; then
  [[ -f .sync-state ]] || {
    echo "missing .sync-state; apply scripts/sync_source.sh from the local machine" >&2
    exit 1
  }
  echo "=== RSYNC STATE ==="
  cat .sync-state
  EXPECTED_TREE_HASH="$(awk -F= '/^source_tree_hash=/ {print $2}' .sync-state)"
  [[ -n "$EXPECTED_TREE_HASH" ]] || {
    echo "missing source_tree_hash in .sync-state; resync from local" >&2
    exit 1
  }
  ACTUAL_TREE_HASH="$(PYTHONPATH=src python3 -m kvcache.source_hash .)"
  [[ "$ACTUAL_TREE_HASH" == "$EXPECTED_TREE_HASH" ]] || {
    echo "remote source changed after rsync; apply a fresh local sync" >&2
    echo "expected source hash: $EXPECTED_TREE_HASH" >&2
    echo "actual source hash:   $ACTUAL_TREE_HASH" >&2
    exit 1
  }
  echo "Remote Git status may be dirty under Policy B:"
  git status --short
else
  [[ -z "$(git status --porcelain)" ]] || {
    echo "remote worktree is dirty" >&2
    exit 1
  }
  UPSTREAM="$(git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null)" || {
    echo "no upstream configured; cannot verify Git-first synchronization" >&2
    exit 1
  }
  [[ "$(git rev-parse HEAD)" == "$(git rev-parse "$UPSTREAM")" ]] || {
    echo "HEAD differs from locally known upstream; git pull --ff-only first" >&2
    exit 1
  }
fi
echo "Health check passed. Confirm dataset/model manifests for this config before launch."

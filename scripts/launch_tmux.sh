#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIRM="${1:-}"
SESSION="${2:-}"
COMMAND="${3:-}"
CONFIG="${4:-}"
if [[ $# -ge 4 ]]; then
  shift 4
fi
EXTRA_ARGS=("$@")

echo "Action: preflight and start a detached tmux job"
echo "Session: $SESSION"
echo "Command kind: $COMMAND"
echo "Config: $CONFIG"

if [[ "$CONFIRM" != "--i-understand" ]]; then
  echo "REFUSED: costly tmux launch requires --i-understand" >&2
  exit 2
fi
[[ "${KVCACHE_REMOTE:-}" == "1" ]] || {
  echo "REFUSED: tmux launch is remote-only; set KVCACHE_REMOTE=1" >&2
  exit 2
}
[[ "$SESSION" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "session must contain only letters, numbers, underscore, or hyphen" >&2
  exit 2
}
[[ -n "$COMMAND" && -n "$CONFIG" ]] || {
  echo "usage: $0 --i-understand SESSION COMMAND CONFIG [CLI_ARGS...]" >&2
  exit 2
}
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p logs
bash scripts/check_remote.sh --config "$CONFIG" --run-name "$SESSION"

RUNNER="$PROJECT_ROOT/logs/$SESSION.runner.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  for name in \
    KVCACHE_REMOTE KVCACHE_PERSONAL_FREE_GB KVCACHE_MIN_PERSONAL_FREE_GB \
    KVCACHE_MIN_FILESYSTEM_FREE_GB REMOTE_DATA_ROOT REMOTE_CACHE_ROOT \
    REMOTE_RUNS_ROOT ENVIRONMENT_NAME UV_CACHE_DIR PIP_CACHE_DIR HF_HOME \
    TORCH_HOME CUDA_VISIBLE_DEVICES GPU_MODEL GPU_COUNT; do
    if [[ -v "$name" ]]; then
      printf 'export %s=%q\n' "$name" "${!name}"
    fi
  done
  printf 'export KVCACHE_RUN_NAME=%q\n' "$SESSION"
  printf 'cd %q\n' "$PROJECT_ROOT"
  printf 'exec bash scripts/launch.sh --i-understand %q %q' "$COMMAND" "$CONFIG"
  if (( ${#EXTRA_ARGS[@]} )); then
    printf ' %q' "${EXTRA_ARGS[@]}"
  fi
  printf '\n'
} >"$RUNNER"
chmod 700 "$RUNNER"

tmux new-session -d -s "$SESSION" "bash '$RUNNER'"
echo "Started tmux session: $SESSION"
echo "Log: $PROJECT_ROOT/logs/$SESSION.log"
echo "Monitor: tmux capture-pane -pt $SESSION -S -100"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIRM="${1:-}"
COMMAND="${2:-}"
CONFIG="${3:-}"
if [[ $# -ge 3 ]]; then
  shift 3
fi
EXTRA_ARGS=("$@")

echo "Action: launch remote experiment command '$COMMAND'"
echo "Project: $PROJECT_ROOT"
echo "Config: $CONFIG"
echo "This may consume GPU time and write under REMOTE_RUNS_ROOT."

if [[ "$CONFIRM" != "--i-understand" ]]; then
  echo "REFUSED: costly launch requires --i-understand" >&2
  exit 2
fi
[[ "${KVCACHE_REMOTE:-}" == "1" ]] || {
  echo "REFUSED: launch is remote-only; set KVCACHE_REMOTE=1" >&2
  exit 2
}
[[ -n "$COMMAND" && -n "$CONFIG" ]] || {
  echo "usage: $0 --i-understand COMMAND CONFIG" >&2
  exit 2
}
[[ -f "$PROJECT_ROOT/$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 1; }

case "$COMMAND" in
  bank) PY_SCRIPT="scripts/build_bank.py" ;;
  grid) PY_SCRIPT="scripts/run_grid.py" ;;
  replay) PY_SCRIPT="scripts/run_replay.py" ;;
  metrics) PY_SCRIPT="scripts/run_metrics.py" ;;
  mechanism) PY_SCRIPT="scripts/run_mechanism.py" ;;
  figures) PY_SCRIPT="scripts/make_figures.py" ;;
  probe-decoding) PY_SCRIPT="scripts/probe_decoding.py" ;;
  *) echo "unknown command: $COMMAND" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
VENV_PATH="${ENVIRONMENT_NAME:-.venv}"
[[ -x "$VENV_PATH/bin/python" ]] || {
  echo "missing venv Python: $PROJECT_ROOT/$VENV_PATH/bin/python" >&2
  exit 1
}
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
export PYTHONUNBUFFERED=1
export HF_HOME="${REMOTE_CACHE_ROOT:?REMOTE_CACHE_ROOT is required}/huggingface"
export TORCH_HOME="$REMOTE_CACHE_ROOT/torch"
export XDG_CACHE_HOME="$REMOTE_CACHE_ROOT/xdg"
export REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:?REMOTE_DATA_ROOT is required}"
BASE_RUNS_ROOT="${REMOTE_RUNS_ROOT:?REMOTE_RUNS_ROOT is required}"
export KVCACHE_BASE_RUNS_ROOT="$BASE_RUNS_ROOT"

RUN_NAME="${KVCACHE_RUN_NAME:-$(basename "$CONFIG" .yaml)-$(date -u +%Y%m%dT%H%M%SZ)}"
export REMOTE_RUNS_ROOT="$BASE_RUNS_ROOT/$RUN_NAME"
mkdir -p "$REMOTE_RUNS_ROOT" "$PROJECT_ROOT/logs"
echo "Run name: $RUN_NAME"
echo "Output root: $REMOTE_RUNS_ROOT"
echo "Personal free-space declaration: ${KVCACHE_PERSONAL_FREE_GB} GB"
printf 'Command: %q %q --config %q --i-understand' \
  "$VENV_PATH/bin/python" "$PY_SCRIPT" "$CONFIG"
if (( ${#EXTRA_ARGS[@]} )); then
  printf ' %q' "${EXTRA_ARGS[@]}"
fi
printf '\n'

{
  "$VENV_PATH/bin/python" "$PY_SCRIPT" --config "$CONFIG" --i-understand \
    "${EXTRA_ARGS[@]}"
  echo "Finished: $RUN_NAME"
} 2>&1 | tee "$PROJECT_ROOT/logs/$RUN_NAME.log"

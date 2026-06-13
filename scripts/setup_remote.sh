#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIRM="${1:-}"
ENV_PATH="${ENVIRONMENT_NAME:-.venv}"
PYTHON_BIN="${REMOTE_PYTHON_BIN:-python3.11}"

echo "Action: create venv '$PROJECT_ROOT/$ENV_PATH' with $PYTHON_BIN"
echo "Action: install exactly the pinned packages from $PROJECT_ROOT/requirements.txt"
echo "No system packages, CUDA drivers, or system Python will be changed."
echo "Shared-filesystem df output does not establish the user's personal quota."

if [[ "$CONFIRM" != "--i-understand" ]]; then
  echo "REFUSED: environment installation requires --i-understand" >&2
  exit 2
fi
command -v "$PYTHON_BIN" >/dev/null || {
  echo "$PYTHON_BIN is required; do not install system Python from this script" >&2
  exit 1
}
[[ -f "$PROJECT_ROOT/requirements.txt" ]] || {
  echo "missing requirements.txt" >&2
  exit 1
}
: "${KVCACHE_PERSONAL_FREE_GB:?set from quota or storage-owner information}"
MIN_SETUP_FREE_GB="${KVCACHE_MIN_SETUP_FREE_GB:-8}"
if (( KVCACHE_PERSONAL_FREE_GB < MIN_SETUP_FREE_GB )); then
  echo "REFUSED: setup requires at least ${MIN_SETUP_FREE_GB} GB personal headroom" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m venv "$ENV_PATH"
"$ENV_PATH/bin/python" -m pip install -r requirements.txt
"$ENV_PATH/bin/python" -c \
  'from importlib.metadata import version; print({p: version(p) for p in ("torch", "transformers", "kvpress")})'

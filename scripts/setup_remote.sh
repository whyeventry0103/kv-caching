#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIRM="${1:-}"
ENV_PATH="${ENVIRONMENT_NAME:-.venv}"
UV_BIN="${UV_BIN:-uv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

echo "Action: use uv to provision Python 3.11 in '$PROJECT_ROOT/$ENV_PATH'"
echo "Action: install exactly the pinned packages from $PROJECT_ROOT/$REQUIREMENTS_FILE"
echo "No system packages, CUDA drivers, or system Python will be changed."
echo "Shared-filesystem df output does not establish the user's personal quota."

if [[ "$CONFIRM" != "--i-understand" ]]; then
  echo "REFUSED: environment installation requires --i-understand" >&2
  exit 2
fi
command -v "$UV_BIN" >/dev/null || {
  echo "uv is required; expected /snap/bin/uv on this remote" >&2
  exit 1
}
[[ -f "$PROJECT_ROOT/$REQUIREMENTS_FILE" ]] || {
  echo "missing requirements file: $REQUIREMENTS_FILE" >&2
  exit 1
}
: "${KVCACHE_PERSONAL_FREE_GB:?set from quota or storage-owner information}"
MIN_SETUP_FREE_GB="${KVCACHE_MIN_SETUP_FREE_GB:-8}"
if (( KVCACHE_PERSONAL_FREE_GB < MIN_SETUP_FREE_GB )); then
  echo "REFUSED: setup requires at least ${MIN_SETUP_FREE_GB} GB personal headroom" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
"$UV_BIN" venv --python 3.11 "$ENV_PATH"
"$UV_BIN" pip install --python "$ENV_PATH/bin/python" -r "$REQUIREMENTS_FILE"
"$ENV_PATH/bin/python" -c \
  'from importlib.metadata import version; print({p: version(p) for p in ("torch", "transformers", "tokenizers", "kvpress", "accelerate", "numpy")})'

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME=""
CONFIRM_LARGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) RUN_NAME="${2:?missing run name}"; shift 2 ;;
    --i-understand-large-transfer) CONFIRM_LARGE=1; shift ;;
    *) echo "usage: $0 --run-name NAME [--i-understand-large-transfer]" >&2; exit 2 ;;
  esac
done

: "${SSH_ALIAS:?SSH_ALIAS is required}"
: "${REMOTE_RUNS_ROOT:?REMOTE_RUNS_ROOT is required}"
[[ -n "$RUN_NAME" ]] || { echo "--run-name is required" >&2; exit 2; }

SOURCE="$SSH_ALIAS:$REMOTE_RUNS_ROOT/$RUN_NAME/"
DESTINATION="$PROJECT_ROOT/results/$RUN_NAME/"
echo "Action: pull JSON/CSV summaries from $SOURCE"
echo "Destination: $DESTINATION"
echo "Large artifacts, checkpoints, arrays, traces, and logs are excluded."
mkdir -p "$DESTINATION"

SIZE_KB="$(ssh "$SSH_ALIAS" \
  "find '$REMOTE_RUNS_ROOT/$RUN_NAME' -type f \\( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' \\) -printf '%s\n'" \
  | awk '{s+=$1} END {printf "%.0f", s/1024}')"
if [[ "${SIZE_KB:-0}" -gt 1048576 && "$CONFIRM_LARGE" -ne 1 ]]; then
  echo "REFUSED: selected summaries exceed 1 GiB; use --i-understand-large-transfer" >&2
  exit 2
fi

rsync -avh --prune-empty-dirs \
  --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' \
  --exclude='*' "$SOURCE" "$DESTINATION"

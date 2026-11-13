#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Meeting 5 review G0.2: a launch the scheduler charged to a campaign carries
# KVCACHE_LAUNCH_ID. Record its start now and its end on ANY exit (including
# the refusals below), so the campaign ledger settles actual elapsed GPU time.
# Absent the variable, nothing changes.
if [[ -n "${KVCACHE_LAUNCH_ID:-}" ]]; then
  LAUNCH_REC_DIR="$PROJECT_ROOT/logs/launches"
  mkdir -p "$LAUNCH_REC_DIR"
  LAUNCH_START="$(date +%s)"
  printf '{"launch_id": "%s", "start_epoch": %s}\n' "$KVCACHE_LAUNCH_ID" \
    "$LAUNCH_START" > "$LAUNCH_REC_DIR/$KVCACHE_LAUNCH_ID.json"
  _kvcache_write_end() {
    local rc=$?
    # 2026-10-15: the run block below is piped into tee, so its exit code is set
    # in a subshell and $? here is tee's (always 0). The block records the
    # RUNNER's true exit code next to the launch record; prefer it.
    if [[ -f "$LAUNCH_REC_DIR/$KVCACHE_LAUNCH_ID.rc" ]]; then
      rc="$(cat "$LAUNCH_REC_DIR/$KVCACHE_LAUNCH_ID.rc")"
      rm -f "$LAUNCH_REC_DIR/$KVCACHE_LAUNCH_ID.rc"
    fi
    printf '{"launch_id": "%s", "start_epoch": %s, "end_epoch": %s, "rc": %d}\n' \
      "$KVCACHE_LAUNCH_ID" "$LAUNCH_START" "$(date +%s)" "$rc" \
      > "$LAUNCH_REC_DIR/$KVCACHE_LAUNCH_ID.json.tmp"
    mv "$LAUNCH_REC_DIR/$KVCACHE_LAUNCH_ID.json.tmp" \
      "$LAUNCH_REC_DIR/$KVCACHE_LAUNCH_ID.json"
  }
  trap _kvcache_write_end EXIT
fi
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
  bank_manifest)
    if [[ "$CONFIG" != "configs/experiments/bank_gsm8k_timing_m5.yaml" \
       || "${KVCACHE_LAUNCH_ID:-}" != M5B-GSM8K-TIMING-* \
       || -z "${KVCACHE_MAX_HOURS:-}" ]]; then
      echo "REFUSED: GSM8K timing requires its capped scheduler reservation" >&2
      exit 2
    fi
    PY_SCRIPT="scripts/run_bank_manifest.py"
    ;;
  bank_manifest_bbh)
    if [[ "$CONFIG" != "configs/experiments/bank_bbh_timing_m5.yaml" \
       || "${KVCACHE_LAUNCH_ID:-}" != M5B-BBH-TIMING-* \
       || -z "${KVCACHE_MAX_HOURS:-}" ]]; then
      echo "REFUSED: BBH timing requires its capped scheduler reservation" >&2
      exit 2
    fi
    PY_SCRIPT="scripts/run_bank_manifest.py"
    ;;
  grid) PY_SCRIPT="scripts/run_grid.py" ;;
  replay) PY_SCRIPT="scripts/run_replay.py" ;;
  rkvsmoke) PY_SCRIPT="scripts/rkv_smoke.py" ;;
  daccsmoke) PY_SCRIPT="scripts/dacc_gate_smoke.py" ;;
  basisfit) PY_SCRIPT="scripts/run_basis_fit.py" ;;
  capture) PY_SCRIPT="scripts/run_capture.py" ;;
  capturepp) PY_SCRIPT="scripts/run_capture_pp.py" ;;
  capture2) PY_SCRIPT="scripts/run_capture_v2.py" ;;
  exactlag) PY_SCRIPT="scripts/run_exact_lag.py" ;;
  restore2) PY_SCRIPT="scripts/run_restore_v2.py" ;;
  restore3) PY_SCRIPT="scripts/run_restore_v3.py" ;;
  grid_pp) PY_SCRIPT="scripts/run_grid_pp.py" ;;
  replay_pp) PY_SCRIPT="scripts/run_replay_pp.py" ;;
  ppsmoke) PY_SCRIPT="scripts/pp_smoke.py" ;;
  grid_protsel) PY_SCRIPT="scripts/run_grid_protected.py" ;;
  replay_protsel) PY_SCRIPT="scripts/run_replay_protected.py" ;;
  protselsmoke) PY_SCRIPT="scripts/protected_selectors_smoke.py" ;;
  rkvppsmoke) PY_SCRIPT="scripts/rkv_pp_smoke.py" ;;
  grid_m5) PY_SCRIPT="scripts/run_grid_m5.py" ;;
  infoprobe) PY_SCRIPT="scripts/run_information_probe.py" ;;
  forkcontrol) PY_SCRIPT="scripts/run_fork_control.py" ;;
  forkprecision) PY_SCRIPT="scripts/run_fork_precision.py" ;;
  metrics) PY_SCRIPT="scripts/run_metrics.py" ;;
  mechanism) PY_SCRIPT="scripts/run_mechanism.py" ;;
  figures) PY_SCRIPT="scripts/make_figures.py" ;;
  shape) PY_SCRIPT="scripts/run_shape_analysis.py" ;;
  onset-surface) PY_SCRIPT="scripts/run_onset_surface.py" ;;
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

# Optional wall-clock cap: KVCACHE_MAX_HOURS=5 stops the run cleanly after 5h
# (SIGTERM, then SIGKILL 120s later if a generation is still in flight). Completed
# rows are already on disk; rerun the identical command (with --resume in EXTRA_ARGS
# and the same KVCACHE_RUN_NAME) to continue from exactly where it stopped.
RUN_CMD=("$VENV_PATH/bin/python" "$PY_SCRIPT" --config "$CONFIG" --i-understand
  "${EXTRA_ARGS[@]}")
if [[ -n "${KVCACHE_MAX_HOURS:-}" ]]; then
  echo "Timer: will stop after ${KVCACHE_MAX_HOURS}h (resumable)"
  RUN_CMD=(timeout --kill-after=120s --signal=TERM "${KVCACHE_MAX_HOURS}h"
    "${RUN_CMD[@]}")
fi

{
  set +e
  "${RUN_CMD[@]}"
  RC=$?
  set -e
  if [[ -n "${KVCACHE_LAUNCH_ID:-}" ]]; then
    printf '%d\n' "$RC" > "$LAUNCH_REC_DIR/$KVCACHE_LAUNCH_ID.rc"
  fi
  if [[ $RC -eq 124 || $RC -eq 137 || $RC -eq 143 ]]; then
    echo "STOPPED by timer (rc=$RC): $RUN_NAME"
    echo "Progress saved. Rerun the SAME command (keep --resume + KVCACHE_RUN_NAME) to continue."
  elif [[ $RC -ne 0 ]]; then
    echo "FAILED (rc=$RC): $RUN_NAME"
  else
    echo "Finished: $RUN_NAME"
  fi
} 2>&1 | tee "$PROJECT_ROOT/logs/$RUN_NAME.log"

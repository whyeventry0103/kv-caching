#!/usr/bin/env bash
set -euo pipefail
export KVCACHE_REMOTE=1
export KVCACHE_PERSONAL_FREE_GB=72
export REMOTE_DATA_ROOT=/home/sk10741/kvcache-storage/data
export REMOTE_CACHE_ROOT=/home/sk10741/kvcache-storage/cache
export REMOTE_RUNS_ROOT=/home/sk10741/kvcache-storage/runs
export ENVIRONMENT_NAME=.venv
export CUDA_VISIBLE_DEVICES=0
export KVCACHE_RUN_NAME=protsel-smoke
export KVCACHE_MAX_HOURS=1.24
export KVCACHE_ALLOW_EXISTING_RUN=1
cd /home/sk10741/kv-caching
exec bash scripts/launch.sh --i-understand protselsmoke configs/experiments/protsel_smoke_wp4.yaml --resume --seeds 11

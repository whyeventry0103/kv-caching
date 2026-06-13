# Remote Setup

First fill every `TODO(human)` in `configs/remote.yaml` and the matching Project
Configuration block in `remoteserver.md`. Do not infer infrastructure values.

This is a shared server. `df -h` reports filesystem-wide free space and does not
establish the user's personal quota. The user has confirmed at least 70 GB of
personal free space. Continue to monitor actual usage before every job. Do not place
model weights, Hugging Face caches, or run banks in the home allocation unless
their measured sizes fit with the configured safety headroom. The 7B bf16 model
alone is expected to exceed that remaining allocation, so obtain a larger
quota-backed project/scratch path before model download.

On the remote:

```bash
export KVCACHE_REMOTE=1
export REMOTE_DATA_ROOT=...
export REMOTE_CACHE_ROOT=...
export REMOTE_RUNS_ROOT=...
export GPU_MODEL=...
export GPU_COUNT=...
export KVCACHE_PERSONAL_FREE_GB=70
bash scripts/setup_remote.sh --i-understand
bash scripts/check_remote.sh \
  --config configs/experiments/grid_compression.yaml \
  --run-name grid-smoke
```

Setup creates the pinned Python 3.11 `.venv` only. Conda is not installed. It
does not modify system Python, CUDA, drivers, or system packages. Model and
dataset access requires a separate explicit human decision and the relevant
environment guard.

Confirm Python and quota before setup:

```bash
python3.11 --version
python3 --version
quota -s 2>/dev/null || true
du -sh "$HOME"
du -sh "$HOME"/.[!.]* "$HOME"/* 2>/dev/null | sort -h | tail -30
```

Before scientific execution, verify every press named `verify-or-port` against
the pinned KVPress version and implement the remote backend adapters currently
designed to refuse execution.

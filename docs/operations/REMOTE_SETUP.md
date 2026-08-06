# Remote Setup

First fill every `TODO(human)` in `configs/remote.yaml` and the matching Project
Configuration block in `docs/operations/remoteserver.md`. Do not infer infrastructure values.

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

Setup uses `/snap/bin/uv` to provision Python 3.11 in `.venv`. Conda and system
Python 3.11 are not installed. It does not modify system Python, CUDA, drivers,
or system packages. Model and
dataset access requires a separate explicit human decision and the relevant
environment guard.

Confirm uv and quota before setup:

```bash
uv --version
quota -s 2>/dev/null || true
du -sh "$HOME"
du -sh "$HOME"/.[!.]* "$HOME"/* 2>/dev/null | sort -h | tail -30
```

Before scientific execution, verify every configured KVPress baseline against
the pinned release and implement the H2O/RKV adapters that currently refuse.

Before any model or dataset download, query metadata and replace the
`TODO(human)` revisions in `configs/models.yaml` and
`configs/benchmarks.yaml` with exact commit SHAs:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
print("r1", api.model_info("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B").sha)
print("qwen3", api.model_info("Qwen/Qwen3-4B").sha)
print("math500", api.dataset_info("HuggingFaceH4/MATH-500").sha)
PY
```

The guarded pilot supports a tokenizer-only probe and a full-cache bank:

```bash
.venv/bin/python scripts/build_bank.py \
  --config configs/experiments/pilot_fullcache.yaml \
  --probe-template --allow-model-download --i-understand

bash scripts/launch_tmux.sh \
  --i-understand pilot-fullcache bank \
  configs/experiments/pilot_fullcache.yaml \
  --allow-model-download --allow-dataset-download
```

Both commands remain remote-only and require `KVCACHE_REMOTE=1`. Review the
template output before authorizing the model-weight pilot. All model downloads,
dataset downloads, and generations must use the detached tmux launcher.

The tmux session name is also the stable run directory under
`REMOTE_RUNS_ROOT`; use a new descriptive session name for every run.

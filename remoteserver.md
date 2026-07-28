# Local and Remote Server Operations Guide

This document is a project-neutral operating manual for a codebase developed on
a local computer and executed on a remote Linux server. It is intended for
humans, Claude Code, Codex, and other coding agents.

Read this file before editing files, syncing code, installing dependencies,
launching long-running jobs, or interpreting remote results.

The examples use placeholders. Replace the values in **Project Configuration**
before using any command.

---

## 1. Mental Model

There are two separate machines and usually two separate copies of the project:

1. **Local machine**
   - Primary location for source editing, documentation, review, and Git work.
   - Usually a laptop or desktop.
   - May not contain large datasets, model weights, caches, or generated runs.

2. **Remote server**
   - Primary location for GPU/CPU-intensive execution.
   - Usually Linux with SSH access.
   - Stores large datasets, caches, checkpoints, logs, and experiment outputs.

The two filesystems are independent. Editing a file locally does not update the
remote copy. Editing a file remotely does not update the local copy.

The word **sync** always means an explicit transfer using Git, `rsync`, `scp`, or
another named mechanism.

---

## 2. Project Configuration

Fill this block in for the new project. Keep the completed values near the top
of the file so another agent can immediately understand the setup.

```text
PROJECT_NAME=kv-caching

LOCAL_PROJECT_ROOT=/Users/apple/dev/kv-caching   # moved off ~/Desktop 2026-06-14 (macOS TCC lockout); ~/Desktop/kv-caching may be a symlink to here

SSH_ALIAS=kvcache-gpu
REMOTE_USER=sk10741
REMOTE_HOST=10.224.32.112             # hostname: ADUAED16811WKLX
REMOTE_PROJECT_ROOT=/home/sk10741/kv-caching

REMOTE_DATA_ROOT=/home/sk10741/kvcache-storage/data
REMOTE_CACHE_ROOT=/home/sk10741/kvcache-storage/cache
REMOTE_RUNS_ROOT=/home/sk10741/kvcache-storage/runs

REMOTE_SHELL=bash
REMOTE_OS=Ubuntu 22.04.5 LTS
REMOTE_PYTHON=3.11

GPU_COUNT=2
GPU_MODEL=NVIDIA RTX 6000 Ada Generation 48GB
CUDA_VERSION=12.8

ENVIRONMENT_TYPE=uv                 # /snap/bin/uv; provisions user-space Python 3.11
ENVIRONMENT_NAME=.venv

PERSONAL_STORAGE_FREE_GB_ESTIMATE=70 # user-confirmed personal free space
MIN_PERSONAL_FREE_GB=5               # launcher refuses below this value
MIN_FILESYSTEM_FREE_GB=30            # independent shared-filesystem floor

PRIMARY_SYNC_METHOD=rsync           # Policy B, explicitly selected by human
PRIMARY_BRANCH=main
```

### Stack migration addendum (2026-06-13)

The original `transformers==4.52.4` / `kvpress==0.2.10` stack was verified to be
prefill-only and could not execute the project's long-output decoding-compression
regime. A controlled isolated migration validated and promoted:

```text
torch==2.7.1
transformers==4.56.2
tokenizers==0.22.2
kvpress==0.5.3
```

The previous environment is retained remotely as `.venv-legacy-tf452`, and its
direct dependency list is retained as `requirements-legacy-tf452.txt`. New
full-cache and compressed rows must use the promoted stack for exchangeability.

Recommended convention:

```text
Local source of truth: source code, configuration, tests, documentation
Remote source of truth: datasets, caches, checkpoints, logs, run artifacts
Git source of truth: committed source history
```

If the project intentionally uses a different ownership model, document it
explicitly.

---

## 3. Placeholder Reference

Commands in this guide use these placeholders:

| Placeholder | Meaning |
|---|---|
| `<LOCAL_PROJECT_ROOT>` | Absolute project path on the local machine |
| `<SSH_ALIAS>` | Preferred SSH alias from `~/.ssh/config` |
| `<REMOTE_USER>` | Remote Linux username |
| `<REMOTE_HOST>` | Remote hostname or IP |
| `<REMOTE_PROJECT_ROOT>` | Project directory on the remote |
| `<REMOTE_DATA_ROOT>` | Remote datasets/assets directory |
| `<REMOTE_CACHE_ROOT>` | Remote package/model cache directory |
| `<REMOTE_RUNS_ROOT>` | Remote output/checkpoint/log directory |
| `<ENV_NAME>` | Virtual environment or Conda environment |
| `<SESSION>` | Short unique tmux session name |
| `<RUN_NAME>` | Stable name for one training/evaluation run |
| `<GPU_IDS>` | Physical GPUs exposed to a command, such as `0` or `0,1` |

Never paste a command containing unresolved angle-bracket placeholders.

---

## 4. Recommended Directory Layout

### Local

```text
<LOCAL_PROJECT_ROOT>/
├── .git/
├── README.md
├── remoteserver.md
├── configs/
├── scripts/
├── src/
├── tests/
├── docs/
└── results/                 # Small synced summaries only, if desired
```

### Remote

```text
<REMOTE_PROJECT_ROOT>/
├── .git/
├── configs/
├── scripts/
├── src/
├── tests/
├── logs/
└── runs/                    # Checkpoints and generated outputs

<REMOTE_DATA_ROOT>/
├── datasets/
├── models/
└── assets/

<REMOTE_CACHE_ROOT>/
├── huggingface/
├── torch/
└── other-tool-caches/
```

Keep large data outside the Git repository. Avoid embedding machine-specific
absolute paths in source code; use environment variables or config files.

---

## 5. SSH Setup

### 5.1 Preferred SSH configuration

On the local machine, configure `~/.ssh/config`:

```sshconfig
Host <SSH_ALIAS>
    HostName <REMOTE_HOST>
    User <REMOTE_USER>
    IdentityFile ~/.ssh/<PRIVATE_KEY_FILE>
    ServerAliveInterval 60
    ServerAliveCountMax 5
    TCPKeepAlive yes
    Compression yes
```

Then connect with:

```bash
ssh <SSH_ALIAS>
```

Without an alias:

```bash
ssh <REMOTE_USER>@<REMOTE_HOST>
```

### 5.2 Test connectivity

```bash
ssh <SSH_ALIAS> 'hostname; whoami; pwd'
```

### 5.3 First connection

Verify the host fingerprint through a trusted channel before accepting it.
Never suppress host-key checking merely to make automation easier.

### 5.4 SSH keys and secrets

- Never commit private SSH keys.
- Never place passwords, API keys, or access tokens in this document.
- Prefer SSH keys protected by a passphrase.
- Use an SSH agent or operating-system keychain where appropriate.
- Store application secrets in remote environment files excluded by Git.

---

## 6. Determining Where a Command Runs

This is the most important operational distinction.

### Local command

Run in the local terminal:

```bash
cd <LOCAL_PROJECT_ROOT>
git status
```

### Interactive remote command

```bash
ssh <SSH_ALIAS>
cd <REMOTE_PROJECT_ROOT>
git status
```

### One-off command sent from local to remote

```bash
ssh <SSH_ALIAS> 'cd <REMOTE_PROJECT_ROOT> && git status'
```

### How to identify the machine

```bash
hostname
whoami
pwd
```

Before a destructive or expensive command, run those checks if the active
machine is not obvious.

---

## 7. Shell Quoting Rules

Remote commands introduce two layers of shell parsing:

1. the local shell;
2. the remote shell.

Use single quotes around a remote command when local variable expansion is not
desired:

```bash
ssh <SSH_ALIAS> 'cd <REMOTE_PROJECT_ROOT> && echo "$HOME"'
```

Here `$HOME` expands remotely.

Use double quotes only when deliberately interpolating a local variable:

```bash
RUN_NAME=my_run
ssh <SSH_ALIAS> "cd <REMOTE_PROJECT_ROOT> && ls runs/$RUN_NAME"
```

Avoid composing large, complex remote commands through nested quoting. Put
repeatable logic in a version-controlled script and invoke that script remotely.

---

## 8. Source-Control Workflow

Git and file synchronization solve different problems.

### Git is best for

- source code;
- configuration;
- tests;
- documentation;
- reproducible scripts;
- reviewable history.

### Git is not appropriate for

- datasets;
- caches;
- large model weights;
- generated checkpoints;
- secrets;
- temporary logs.

### Recommended Git workflow

Local:

```bash
cd <LOCAL_PROJECT_ROOT>
git status --short
git diff
git add <FILES>
git commit -m "<MESSAGE>"
git push origin <PRIMARY_BRANCH>
```

Remote:

```bash
cd <REMOTE_PROJECT_ROOT>
git status --short
git pull --ff-only origin <PRIMARY_BRANCH>
```

Use `--ff-only` to prevent an accidental merge commit on the execution server.

### Dirty remote worktree

Do not pull blindly if the remote has modifications:

```bash
cd <REMOTE_PROJECT_ROOT>
git status --short
git diff
```

Determine whether the changes are:

- intentional remote edits that need to be copied or committed;
- generated files that should be ignored;
- or stale local modifications.

Never discard remote changes without explicit authorization.

---

## 9. Syncing with rsync

`rsync` is the preferred direct-transfer tool because it is incremental,
preserves timestamps, and supports dry runs and exclusions.

### 9.1 Always dry-run broad syncs

```bash
rsync -avhn \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'data/' \
  --exclude 'runs/' \
  --exclude 'logs/' \
  <LOCAL_PROJECT_ROOT>/ \
  <SSH_ALIAS>:<REMOTE_PROJECT_ROOT>/
```

Flags:

```text
-a  archive mode
-v  verbose
-h  human-readable sizes
-n  dry run; make no changes
-z  compression; useful on slow links, often unnecessary on fast LANs
-P  progress plus partial-transfer support for large files
```

Remove `-n` only after reviewing the dry-run output.

### 9.2 Sync selected source directories

```bash
rsync -avh \
  <LOCAL_PROJECT_ROOT>/src/ \
  <SSH_ALIAS>:<REMOTE_PROJECT_ROOT>/src/

rsync -avh \
  <LOCAL_PROJECT_ROOT>/scripts/ \
  <SSH_ALIAS>:<REMOTE_PROJECT_ROOT>/scripts/

rsync -avh \
  <LOCAL_PROJECT_ROOT>/configs/ \
  <SSH_ALIAS>:<REMOTE_PROJECT_ROOT>/configs/
```

### 9.3 Sync one file

```bash
rsync -avh \
  <LOCAL_PROJECT_ROOT>/path/to/file \
  <SSH_ALIAS>:<REMOTE_PROJECT_ROOT>/path/to/file
```

Ensure the destination directory exists first:

```bash
ssh <SSH_ALIAS> 'mkdir -p <REMOTE_PROJECT_ROOT>/path/to'
```

### 9.4 Pull results from remote

```bash
mkdir -p <LOCAL_PROJECT_ROOT>/results/<RUN_NAME>

rsync -avh \
  <SSH_ALIAS>:<REMOTE_RUNS_ROOT>/<RUN_NAME>/metrics.json \
  <LOCAL_PROJECT_ROOT>/results/<RUN_NAME>/
```

Pull only the files needed for analysis or reporting. Avoid downloading
multi-gigabyte checkpoints unless required.

### 9.5 Large-file transfer

```bash
rsync -avhP \
  <SSH_ALIAS>:<REMOTE_RUNS_ROOT>/<RUN_NAME>/checkpoint.pt \
  <LOCAL_PROJECT_ROOT>/checkpoints/
```

Verify available disk space at both ends before starting.

### 9.6 Trailing-slash semantics

These commands are different:

```bash
rsync -avh source/ destination/
```

Copies the **contents** of `source`.

```bash
rsync -avh source destination/
```

Copies the `source` directory itself inside `destination`.

### 9.7 Avoid `--delete` by default

`--delete` removes destination files not present at the source. It is dangerous
when the remote contains runs or edits absent locally.

Only use it after:

1. confirming the source is authoritative;
2. excluding remote-only directories;
3. running a dry run;
4. reviewing every deletion.

---

## 10. Syncing with scp

Use `scp` for simple one-file transfers:

```bash
scp <LOCAL_PROJECT_ROOT>/config.yaml \
  <SSH_ALIAS>:<REMOTE_PROJECT_ROOT>/config.yaml
```

Pull a file:

```bash
scp <SSH_ALIAS>:<REMOTE_PROJECT_ROOT>/output.json \
  <LOCAL_PROJECT_ROOT>/results/
```

Prefer `rsync` for directories, interrupted transfers, repeated synchronization,
or anything requiring exclusions.

---

## 11. Recommended Sync Policy

Use one of these two policies and document the choice.

### Policy A: Git-first

1. Edit locally.
2. Test locally where possible.
3. Commit and push.
4. Pull with `--ff-only` on remote.
5. Run remotely.
6. Pull small results with `rsync`.

This is the safest and most reproducible policy.

### Policy B: Rsync-first during rapid iteration

1. Edit locally.
2. Run `rsync --dry-run`.
3. Sync only changed source/config files.
4. Run remotely.
5. Once stable, commit locally.

This is faster but makes it easier for Git history and remote execution state to
diverge. Record the commit hash and local diff for every important run.

### Never mix directions casually

If both local and remote copies were edited, compare before syncing:

```bash
ssh <SSH_ALIAS> 'cd <REMOTE_PROJECT_ROOT> && git diff'
git diff
```

Copy remote edits into a temporary local location first rather than overwriting
the local file:

```bash
rsync -avh \
  <SSH_ALIAS>:<REMOTE_PROJECT_ROOT>/path/to/file \
  /tmp/remote-file-copy

diff -u <LOCAL_PROJECT_ROOT>/path/to/file /tmp/remote-file-copy
```

---

## 12. Bootstrapping a New Remote Project

### 12.1 Create directories

```bash
ssh <SSH_ALIAS> '
  mkdir -p <REMOTE_PROJECT_ROOT>
  mkdir -p <REMOTE_DATA_ROOT>
  mkdir -p <REMOTE_CACHE_ROOT>
  mkdir -p <REMOTE_RUNS_ROOT>
'
```

### 12.2 Clone with Git

```bash
ssh <SSH_ALIAS> '
  git clone <GIT_REPOSITORY_URL> <REMOTE_PROJECT_ROOT>
'
```

Or create an empty directory and use `rsync` for a non-Git project.

### 12.3 Create the environment

Python virtual environment:

```bash
ssh <SSH_ALIAS>
cd <REMOTE_PROJECT_ROOT>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda:

```bash
conda create -n <ENV_NAME> python=<PYTHON_VERSION>
conda activate <ENV_NAME>
pip install -r requirements.txt
```

`uv`:

```bash
uv sync
```

Record the actual environment method in **Project Configuration**.

### 12.4 Verify the environment

```bash
which python
python --version
python -m pip --version
python -c "import sys; print(sys.executable)"
```

For GPU projects:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

---

## 13. Environment Variables

Use environment variables for machine-specific paths and secrets.

Example:

```bash
export PROJECT_ROOT=<REMOTE_PROJECT_ROOT>
export DATA_ROOT=<REMOTE_DATA_ROOT>
export CACHE_ROOT=<REMOTE_CACHE_ROOT>
export RUNS_ROOT=<REMOTE_RUNS_ROOT>
```

Application-specific caches:

```bash
export HF_HOME=<REMOTE_CACHE_ROOT>/huggingface
export TORCH_HOME=<REMOTE_CACHE_ROOT>/torch
export XDG_CACHE_HOME=<REMOTE_CACHE_ROOT>/xdg
```

### Persistent versus per-job variables

Stable, non-secret machine paths may be added to `~/.bashrc`.

Run-specific values should be set in:

- a launcher script;
- a `.env` file excluded by Git;
- a job scheduler script;
- or the tmux command.

### Confirm values

```bash
env | sort
echo "$DATA_ROOT"
```

Do not print secrets into shared logs.

---

## 14. Scripts and Launchers

Repeatable commands should live in `scripts/` rather than chat history.

Recommended scripts:

```text
scripts/
├── setup_remote.sh
├── check_remote.sh
├── launch_train.sh
├── launch_eval.sh
├── summarize_run.sh
└── sync_results.sh
```

Every launcher should:

1. use `set -euo pipefail`;
2. resolve or validate the project root;
3. activate the expected environment;
4. export required variables;
5. create output and log directories;
6. print the command and configuration;
7. fail if required files are missing;
8. write logs to a stable path;
9. propagate the underlying command’s exit status.

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-<REMOTE_PROJECT_ROOT>}"
RUN_NAME="${1:?usage: $0 RUN_NAME}"

cd "$PROJECT_ROOT"
mkdir -p "runs/$RUN_NAME" "logs"

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/train.py \
  --config "configs/$RUN_NAME.yaml" \
  --output "runs/$RUN_NAME" \
  2>&1 | tee "logs/$RUN_NAME.log"
```

Validate shell scripts:

```bash
bash -n scripts/launch_train.sh
```

If ShellCheck is installed:

```bash
shellcheck scripts/launch_train.sh
```

---

## 15. tmux for Long-Running Jobs

Use `tmux` for jobs that must survive SSH disconnection.

### 15.1 Basic concepts

- **Server:** the background tmux process.
- **Session:** a named workspace.
- **Window:** a tab inside a session.
- **Pane:** a split terminal inside a window.

### 15.2 Create an interactive session

```bash
tmux new -s <SESSION>
```

Detach:

```text
Ctrl-b, then d
```

Reattach:

```bash
tmux attach -t <SESSION>
```

### 15.3 Create a detached job

```bash
tmux new-session -d -s <SESSION> \
  "cd <REMOTE_PROJECT_ROOT> && bash scripts/launch_train.sh <RUN_NAME>"
```

### 15.4 List sessions

```bash
tmux ls
```

No tmux server is not automatically an error:

```bash
tmux ls 2>/dev/null || true
```

### 15.5 Inspect output without attaching

```bash
tmux capture-pane -pt <SESSION> -S -100
```

Follow the persistent log instead when available:

```bash
tail -f logs/<RUN_NAME>.log
```

### 15.6 Session naming

Use short, unique, descriptive names:

```text
train_modelA_seed42
eval_modelA_validation
download_dataset_name
```

Avoid generic names such as `run`, `test`, or `train`.

### 15.7 Stop a session

Graceful stop from inside the session:

```text
Ctrl-c
```

Kill a specific tmux session only when necessary:

```bash
tmux kill-session -t <SESSION>
```

Killing the tmux session may not terminate detached child processes. Verify:

```bash
ps -ef | grep -E '<SCRIPT_NAME>|<RUN_NAME>' | grep -v grep
```

### 15.8 Do not rely on tmux output as the only log

tmux history is bounded and disappears with the session. Always send stdout and
stderr to a file:

```bash
command 2>&1 | tee logs/<RUN_NAME>.log
```

With pipelines, use `set -o pipefail` so a failed command is not masked by a
successful `tee`.

---

## 16. Standard Pre-Launch Health Check

Run before every expensive job:

```bash
cd <REMOTE_PROJECT_ROOT>

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
du -sh <REMOTE_RUNS_ROOT> 2>/dev/null || true

echo "=== MEMORY ==="
free -h

echo "=== GPU ==="
nvidia-smi
```

Also validate:

- the expected config exists;
- the output directory is correct;
- required datasets and checkpoints exist;
- no run with the same name is active;
- local code has been synced;
- sufficient disk space remains for checkpoints and logs.

---

## 17. GPU Semantics

### 17.1 Physical versus logical GPU IDs

This command exposes physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python script.py --device cuda:0
```

Inside the process, the only visible GPU is remapped to logical `cuda:0`.

This exposes physical GPUs 2 and 3:

```bash
CUDA_VISIBLE_DEVICES=2,3 python script.py
```

Inside the process:

```text
logical cuda:0 -> physical GPU 2
logical cuda:1 -> physical GPU 3
```

### 17.2 Check GPU use

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

Process details:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

### 17.3 Single-GPU pattern

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py --config configs/example.yaml
```

### 17.4 PyTorch DDP pattern

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  scripts/train.py --config configs/example.yaml
```

The code must explicitly support distributed training. Do not assume that
launching with `torchrun` makes an arbitrary script correct.

### 17.5 Effective batch size

For data-parallel training:

```text
effective batch size =
per-device batch size
× gradient accumulation steps
× number of data-parallel processes
```

Record all four values in run metadata.

---

## 18. Running a Long Job Safely

### 18.1 Create output directories before launch

```bash
mkdir -p \
  <REMOTE_RUNS_ROOT>/<RUN_NAME> \
  <REMOTE_PROJECT_ROOT>/logs
```

### 18.2 Use a detached tmux session

```bash
tmux new-session -d -s <SESSION> "
  set -o pipefail
  cd <REMOTE_PROJECT_ROOT>
  source .venv/bin/activate
  export PYTHONUNBUFFERED=1
  export DATA_ROOT=<REMOTE_DATA_ROOT>
  export CACHE_ROOT=<REMOTE_CACHE_ROOT>
  CUDA_VISIBLE_DEVICES=<GPU_IDS> \
    python scripts/train.py \
      --config configs/<CONFIG_FILE> \
      --output <REMOTE_RUNS_ROOT>/<RUN_NAME> \
      2>&1 | tee logs/<RUN_NAME>.log
"
```

For complex launches, invoke a checked-in launcher script instead of embedding
everything in the tmux command.

### 18.3 Verify startup

After 10–60 seconds:

```bash
tmux capture-pane -pt <SESSION> -S -100
tail -100 logs/<RUN_NAME>.log
nvidia-smi
df -h <REMOTE_PROJECT_ROOT>
```

Confirm:

- model or workload loaded;
- expected GPU processes exist;
- the correct GPUs are active;
- no traceback or out-of-memory error appears;
- progress is increasing;
- logs are being written.

---

## 19. Monitoring

### Process

```bash
ps -ef | grep '<RUN_NAME>' | grep -v grep
```

### Log

```bash
tail -100 logs/<RUN_NAME>.log
tail -f logs/<RUN_NAME>.log
```

### Error scan

```bash
grep -nE \
  'Traceback|RuntimeError|Error|Killed|No space|out of memory|OOM|NCCL' \
  logs/<RUN_NAME>.log || true
```

### Disk

```bash
df -h <REMOTE_PROJECT_ROOT>
du -sh <REMOTE_RUNS_ROOT>/<RUN_NAME>
```

### GPU

```bash
nvidia-smi
```

### tmux

```bash
tmux capture-pane -pt <SESSION> -S -80
```

Do not infer that a job succeeded merely because its tmux session ended. Check
its exit evidence, output files, and final log lines.

---

## 20. Completion Audit

For every important run, verify:

1. no process remains unexpectedly;
2. the log contains the expected completion marker;
3. no fatal error appears;
4. expected output files exist;
5. output files have plausible sizes;
6. summary metrics can be parsed;
7. disk space remains safe;
8. important artifacts have checksums;
9. small results are synced locally;
10. run metadata records the exact source revision.

Example:

```bash
echo "=== OUTPUTS ==="
find <REMOTE_RUNS_ROOT>/<RUN_NAME> -maxdepth 2 -type f -ls

echo "=== COMPLETION ==="
grep -nE 'Training complete|Evaluation complete|Finished' \
  logs/<RUN_NAME>.log | tail

echo "=== ERRORS ==="
grep -nE \
  'Traceback|RuntimeError|Killed|No space|out of memory' \
  logs/<RUN_NAME>.log || true

echo "=== HASH ==="
sha256sum <REMOTE_RUNS_ROOT>/<RUN_NAME>/<IMPORTANT_ARTIFACT>
```

---

## 21. Run Metadata

Every serious run should record:

```text
run name
date/time and timezone
Git commit hash
dirty-worktree status
config path and config hash
command
environment name
Python/package versions
host
GPU model and count
CUDA_VISIBLE_DEVICES
random seed
dataset version or manifest
input checkpoint and hash
output artifact and hash
final metrics
notes or anomalies
```

Useful commands:

```bash
git rev-parse HEAD
git status --short
sha256sum configs/<CONFIG_FILE>
python --version
python -m pip freeze
nvidia-smi
```

Avoid relying only on a mutable `latest` directory. Give runs stable,
human-readable names.

---

## 22. Checksums and Artifact Integrity

Create a SHA-256 checksum:

```bash
sha256sum <FILE>
```

On macOS:

```bash
shasum -a 256 <FILE>
```

Verify a copied file by comparing the source and destination hashes.

For a set of artifacts:

```bash
find <DIRECTORY> -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS
```

Keep checksum files beside irreplaceable outputs or backups.

---

## 23. Disk Management

### Check free space

```bash
df -h
df -h <REMOTE_PROJECT_ROOT>
```

### Find large directories

```bash
du -xh --max-depth=1 <REMOTE_PROJECT_ROOT> | sort -h
du -xh --max-depth=1 <REMOTE_DATA_ROOT> | sort -h
```

### Find large files

```bash
find <REMOTE_PROJECT_ROOT> -type f -size +1G -printf '%s %p\n' \
  | sort -n
```

Do not delete files merely because they are large. Classify them:

- reproducible cache;
- replaceable intermediate checkpoint;
- unique best/final checkpoint;
- dataset;
- log;
- user-owned unrelated data.

Require explicit approval before deleting unique artifacts or unrelated data.

### Suggested thresholds

Define project-specific limits based on expected checkpoint and temporary-file
sizes. A generic starting point:

```text
Below 2 checkpoint sizes free: do not launch
Below 1 checkpoint size free: stop writes and investigate
Enough for outputs plus 20% headroom: normal launch
```

Percentage usage alone is misleading on large disks; use available bytes.

---

## 24. Logs

Recommended layout:

```text
logs/
├── train_<RUN_NAME>.log
├── eval_<RUN_NAME>.log
└── system_<DATE>.log
```

Best practices:

- set `PYTHONUNBUFFERED=1`;
- capture stdout and stderr;
- include timestamps when useful;
- avoid secrets in command lines;
- retain logs for runs cited in reports;
- compress old text logs rather than deleting them blindly.

Compression:

```bash
gzip logs/old_run.log
```

Read:

```bash
zless logs/old_run.log.gz
zgrep 'metric' logs/old_run.log.gz
```

---

## 25. Failure Recovery

### 25.1 SSH disconnected

Reconnect and inspect:

```bash
ssh <SSH_ALIAS>
tmux ls
ps -ef | grep '<RUN_NAME>' | grep -v grep
tail -100 logs/<RUN_NAME>.log
```

If the job was in tmux, it normally continues.

### 25.2 tmux session ended

Check:

```bash
ps -ef | grep '<RUN_NAME>' | grep -v grep
tail -150 logs/<RUN_NAME>.log
find <REMOTE_RUNS_ROOT>/<RUN_NAME> -maxdepth 2 -type f -ls
```

The job may have completed, failed, or spawned a surviving child process.

### 25.3 Out of memory

Record:

- GPU model;
- batch size;
- precision;
- input dimensions;
- number of processes;
- peak memory if available;
- exact traceback.

Potential remedies must be chosen deliberately:

- reduce per-device batch size;
- increase gradient accumulation;
- use mixed precision;
- reduce input size or sequence length;
- enable checkpointing;
- fix unintended duplicate model loading.

Do not silently change multiple variables and call the rerun comparable.

### 25.4 Disk full

Stop new writes, inspect large files, and identify safe cleanup candidates.
Partial checkpoints may be corrupt. Verify outputs after cleanup.

### 25.5 Orphan GPU process

Inspect ownership first:

```bash
nvidia-smi
ps -fp <PID>
```

Attempt graceful termination:

```bash
kill <PID>
```

Use force only when necessary:

```bash
kill -9 <PID>
```

Never kill another user’s process.

### 25.6 DDP or NCCL failure

Collect:

```bash
nvidia-smi
ps -ef | grep -E 'torchrun|python' | grep -v grep
tail -200 logs/<RUN_NAME>.log
```

Check:

- all ranks launched;
- GPU count matches `--nproc_per_node`;
- all ranks use the same code and config;
- no rank failed during model/data initialization;
- ports are not conflicting;
- output and cache paths are writable.

Do not restart repeatedly without understanding whether stale processes remain.

---

## 26. Job Schedulers

If the server uses Slurm, PBS, Kubernetes, or another scheduler, prefer it over
manual tmux for resource-managed workloads.

For Slurm:

```bash
sbatch job.slurm
squeue -u "$USER"
sacct -j <JOB_ID>
scancel <JOB_ID>
```

Document:

- partition;
- account;
- time limit;
- GPU request syntax;
- memory and CPU limits;
- log paths;
- module/environment activation.

tmux is still useful for monitoring but should not replace the scheduler.

---

## 27. Ports, Dashboards, and Jupyter

Use SSH tunneling rather than exposing services publicly.

Remote service bound to localhost:

```bash
jupyter lab --no-browser --ip 127.0.0.1 --port 8888
```

Local tunnel:

```bash
ssh -N -L 8888:127.0.0.1:8888 <SSH_ALIAS>
```

Then open:

```text
http://127.0.0.1:8888
```

For TensorBoard:

```bash
tensorboard --logdir <REMOTE_RUNS_ROOT> --host 127.0.0.1 --port 6006
ssh -N -L 6006:127.0.0.1:6006 <SSH_ALIAS>
```

Do not bind unauthenticated development services to `0.0.0.0` unless the network
and security implications are understood.

---

## 28. Security and Safety Rules

- Never commit secrets.
- Never paste private keys into an LLM conversation.
- Never run unreviewed `curl | bash` installers.
- Avoid `chmod -R 777`.
- Avoid running project commands with `sudo`.
- Never use `rm -rf` on a path containing unresolved variables.
- Quote paths and variables: `"$PATH_VARIABLE"`.
- Use `set -u` so missing variables fail rather than becoming empty strings.
- Inspect remote ownership before changing permissions.
- Do not disable SSH host-key checks.
- Do not expose Jupyter, TensorBoard, or debug servers publicly by default.
- Treat datasets and model weights as potentially license-restricted.

---

## 29. Commands an Agent Must Not Assume Are Safe

An LLM must request confirmation before:

- deleting remote files;
- using `rsync --delete`;
- overwriting a dirty remote worktree;
- killing processes not clearly associated with the project;
- installing or upgrading system packages;
- changing CUDA, drivers, or system Python;
- downloading large datasets or weights;
- syncing multi-gigabyte checkpoints;
- launching a costly job;
- changing the number of GPUs or effective batch size;
- force-pushing Git branches;
- resetting or cleaning a Git worktree.

---

## 30. Standard Agent Workflow

When Claude Code or another agent receives this file, it should:

1. Read **Project Configuration**.
2. Establish whether it has local shell access, remote shell access, or both.
3. Run `hostname`, `whoami`, and `pwd`.
4. Inspect local Git status.
5. Inspect remote Git status before syncing.
6. Determine which machine owns each relevant artifact.
7. Prefer local source edits and explicit synchronization.
8. Use a dry run for broad `rsync`.
9. Run the pre-launch health check before an expensive job.
10. Use tmux or the configured scheduler for long jobs.
11. Write a persistent log.
12. Verify outputs, errors, hashes, and metrics at completion.
13. Pull back only needed result summaries.
14. Report the exact commands, paths, commit, and artifacts involved.

The agent must not claim that a local change is active remotely until the sync or
remote pull has been verified.

---

## 31. Reporting Format

After a remote action, report:

```text
Action:
Machine:
Working directory:
Source commit:
Sync method:
Files transferred:
Command/session:
GPU allocation:
Log path:
Output path:
Current status:
Disk free:
Errors or warnings:
Next required action:
```

For completed jobs, add:

```text
Final metric(s):
Artifact size:
Artifact SHA-256:
Result files copied locally:
```

---

## 32. Quick Command Reference

### Connect

```bash
ssh <SSH_ALIAS>
```

### One-off status

```bash
ssh <SSH_ALIAS> '
  cd <REMOTE_PROJECT_ROOT> &&
  hostname &&
  git status --short &&
  df -h . &&
  nvidia-smi
'
```

### Dry-run source sync

```bash
rsync -avhn \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'data/' \
  --exclude 'runs/' \
  <LOCAL_PROJECT_ROOT>/ \
  <SSH_ALIAS>:<REMOTE_PROJECT_ROOT>/
```

### Start detached job

```bash
tmux new-session -d -s <SESSION> \
  "cd <REMOTE_PROJECT_ROOT> && bash scripts/launch_train.sh <RUN_NAME>"
```

### Monitor

```bash
tmux capture-pane -pt <SESSION> -S -100
tail -100 logs/<RUN_NAME>.log
nvidia-smi
df -h <REMOTE_PROJECT_ROOT>
```

### Pull metrics

```bash
rsync -avh \
  <SSH_ALIAS>:<REMOTE_RUNS_ROOT>/<RUN_NAME>/metrics.json \
  <LOCAL_PROJECT_ROOT>/results/<RUN_NAME>/
```

### Verify artifact

```bash
ssh <SSH_ALIAS> \
  'sha256sum <REMOTE_RUNS_ROOT>/<RUN_NAME>/<ARTIFACT>'
```

---

## 33. Project-Specific Addendum Template

Keep project-specific facts below this line. Do not modify the general operating
rules above unless the infrastructure itself changes.

```text
Project:

Local root:
Remote root:
Data root:
Runs root:

SSH alias:
Remote user/host:

Environment activation:

Required environment variables:

Primary launcher scripts:

Expected GPUs:
Expected disk use:

Important remote-only artifacts:

Directories never synced local -> remote:

Directories normally synced remote -> local:

Current active tmux sessions:

Current jobs:

Known machine-specific issues:

Recovery commands:

Last verified date:
Verified by:
```

Update this addendum whenever paths, environment setup, server hardware, sync
policy, or launch scripts change.

---

## 34. Deviation log — non-standard source transfers

Entries here record transfers that did **not** go through `make sync-apply`. The
point is that staleness must never be silent: a deviation that is written down,
hash-verified, and reflected in `.sync-state` is auditable; an unrecorded one is
the failure mode Lesson 14 exists to prevent. Do not read these entries as drift.

### 2026-07-27 — `scp-verified-tree-hash` at commit `f0f809e`

**What.** The Amendment K source change (`scripts/report_m4.py`,
`scripts/harvest_results.py`, `tests/test_report_m4.py`, `PREREGISTRATION.md`,
`CLAUDE.md`, `context.md`, `.gitignore`) was transferred by `scp` rather than
`make sync-apply`.

**Why.** `make sync-apply` failed six consecutive times with
`rsync: unexpected end of file` (exit 255), the SSH channel dropping mid-transfer.
Isolated by bisection: single-file `rsync` succeeds with and without `-z`, `scp`
succeeds, and plain `ssh` is responsive — but a full-tree traversal times out. The
box was carrying load ~9 from other tenants (`dockerd`, `itops`), so this is remote
contention during file-list traversal, not the SSH multiplexing and not our link.

**Why not `git pull` on the remote.** Checked first and rejected: the remote *is* a
clone, but pinned at `3e19f8b` with 163 files modified by weeks of `--delete`-free
rsync writing over the working tree, and it has no credentials for the origin
remote. Pulling would have conflicted with the rsync'd tree.

**How the guard was preserved.** The load-bearing part of `sync_source.sh --apply`
is not the transport — it is the refusal unless the remote `source_tree_hash`
matches the local one. That check was reproduced by hand and **initially failed**,
correctly: it caught a stale `kv_compression_tutorial.pdf` on the remote that no
longer existed locally (a Lesson 14 no-delete residue). That file turned out to be
a *third, byte-distinct build* present nowhere else, so it was copied down and
verified before removal, never deleted blind. After targeted cleanup both hashes
read `f06d2975…` and `.sync-state` was written with `sync_method=scp-verified-tree-hash`.

**Consequence for provenance.** `sync_method` is stamped into every result row's
runmeta (`src/kvcache/runmeta.py`), so any future run transferred this way is
labelled accordingly. Nothing validates the field against `rsync`, and
`check_remote.sh` gates only on `source_tree_hash`, so the preflight is unaffected.

**Standing rule.** An alternate transport is acceptable only when all three hold:
the tree hashes are compared and match, `.sync-state` records the true commit and
hash, and the deviation is logged here. Never write `.sync-state` from a transfer
whose hash was not verified — that is precisely the silent staleness the invariant
exists to prevent.

---

## 35. `make sync` restored to rsync — the hash guard was blocked by output in the source tree

### 2026-07-27 — deviation CLOSED; root cause was not (only) contention

**Outcome.** `make sync-apply` works again by the standard path.
`.sync-state` now reads `sync_method=rsync`, `source_commit=81c2642`,
`source_tree_hash=6dc9ac95…`, verified by the script's own local-vs-remote
comparison. The §34 `scp-verified-tree-hash` deviation is closed; do not
carry it forward as the normal path.

**Two distinct causes, and only one was the one §34 named.**

1. *Contention (real, transient).* The first apply of the day still died with
   §34's exact signature — `rsync: unexpected end of file`, exit 255, mid
   transfer, box at load ~10 from other tenants. A retry minutes later
   transferred all 220 files without complaint. So the transport fault is
   intermittent, not persistent: **retry before concluding the link is
   broken**, and never diagnose from the dry-run — `make sync` traverses but
   transfers nothing, so it succeeds while the apply fails. That asymmetry is
   what made an earlier read of "the link is healthy" wrong.

2. *The actual blocker: `amendk/` lived inside the hashed source tree.* With
   the transfer succeeding, the script still refused:
   `REFUSED: remote source hash does not match transferred local source`.
   Diagnosis was file-by-file, not inferred: all **206** local files were
   present on the remote with **byte-identical content** (zero content
   differences, zero local-only files), and the remote carried **39 extra
   files, all under `amendk/`** — the Amendment K working outputs
   (9 `summary.json`, 9 `table.md`, 9 `decomposition.svg`, the ADD set, and
   the three Class B artifacts). `kvcache/source_hash.py` ignores `results/`,
   `runs/`, `logs/`, `data/`, `kvcache-storage/` — but not `amendk/`, so
   remote-generated output entered the hash and could never match a local tree
   that had none of it.

**Fix.** `mv amendk results/amendk` on the remote, into the already-ignored
`results/`. Nothing was deleted; the whole directory was first pulled down and
compared — all 9 unit summaries and all 4 ADD summaries are byte-identical to
the copies already archived under `results/diag/amendment-k/`, so the only
remote-unique content was the three Class B artifacts (now archived locally at
`results/diag/amendment-k/classb/`) plus regenerable tables and SVGs. No code,
config, or cron entry referenced the `amendk` path.

**The generalisable rule.** *Never write generated output into the source tree
on either side.* The hash guard treats the source tree as the unit of identity,
so any artifact written inside it makes local and remote permanently
unequal and the guard fails closed — correctly, but for a reason that looks
like drift and invites someone to "fix" it by weakening the check. Analysis
output belongs in `results/` (ignored by both the hash and rsync) or under
`kvcache-storage/`. This is the same failure family as Lesson 14: rsync has no
`--delete`, so anything that lands in the tree stays there until someone
removes it deliberately.

**Note for the watcher.** The cron entries are still armed
(`nightly_autorun.py` every 10 min in the 01:45–10:45 window). Between the
refused apply and the fix, the remote briefly held the new source files with a
`.sync-state` still naming the old commit — the precise silent-staleness state
the invariant exists to prevent. It is now consistent. If an apply ever refuses
again, treat the remote as *unknown* until the hash is reconciled; do not
assume the transfer did not land.

---

## 36. Backup policy (2026-07-28)

The exposure this closes: 19 unpushed commits plus the only copy of every
artifact `paper_numbers.json` resolves to, on one disk. The GPU box is not a
backup — `sync_source.sh` deliberately transfers neither `.git` nor `results/`,
so the remote held source only, and its own clone was pinned at `3e19f8b` with
none of our history.

**Two layers, both verified:**

1. **Private GitHub mirror.** `backup` →
   `https://github.com/whyeventry0103/kv-caching-backup.git`, created private
   (`isPrivate: true`, verified via `gh repo view`). Carries full history *and*
   `results/diag`, which the public repo's workflow never needed to.
   **Push at every unit boundary:** `git push backup main`.
2. **Local bundle**, as the stopgap that needs no network:
   `git bundle create ~/kvcache-backups/kv-caching-$(date +%Y%m%d)-$(git rev-parse --short HEAD).bundle --all`
   Verify with `git bundle verify <file>` — it must say "records a complete
   history". Refresh whenever the mirror has been unreachable.

**`origin` (`whyeventry0103/kv-caching`, public) stays untouched until the
endgame.** Nothing is pushed there before the anonymization scrub, which still
has to remove the personal address from `context.md`'s narrative bullets and
keep `docs/strategy/` out of the artifact. The backup mirror is private
precisely so that constraint costs us nothing in the meantime.

**Do not** add `backup` to any automated hook or cron. It is a manual boundary
action, so that "what is backed up" always corresponds to a state someone chose.

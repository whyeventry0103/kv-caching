# scripts/ — rules local to this directory

Everything here is an orchestration shell around `src/kvcache`. If you find
yourself writing scoring, parsing, statistics, or metric logic in this
directory, stop and put it in the package, then import it here. History:
script-level scoring drifted from package logic once; commit 193e12f made the
package functions the single source of truth.

## Python runner scripts

- Every runner starts with the same bootstrap before any `kvcache` import:

  ```python
  PROJECT_ROOT = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(PROJECT_ROOT / "src"))
  ```

  This is why pyproject exempts `scripts/*.py` (and only them) from ruff
  `E402,I001`. Do not "fix" the import order here, and do not rely on this
  exemption anywhere else in the repo.
- Contract for any new runner (copy `build_bank.py`): argparse with
  `--config <yaml>` and a `--dry-run` mode that validates the config and
  prints the plan without touching models, datasets, or the network. Heavy
  execution additionally requires `--i-understand`; downloads require
  `--allow-model-download` / `--allow-dataset-download`.
- Register every new runner's `--dry-run` invocation in the Makefile `dryrun`
  target. Dry-run is the only execution these scripts get locally; a runner
  absent from that target is effectively untested.
- Long-running work must be resumable: `--resume` skips already-completed
  (benchmark, problem id, seed) keys and heals a partial trailing JSONL line
  left by an interrupted run.
- Build output rows with the package builders (`build_result_row`,
  `build_token_ids_record`, `append_result`, `append_token_ids`) — never
  hand-construct result dicts, or the schema tests and audits will diverge
  from reality.
- Keep logic in module-level functions with a `main()` entry guard: the test
  suite imports these scripts directly (`test_dryrun.py`,
  `test_build_bank_resume.py`, `test_nightly_autorun.py`).

## Shell scripts

- `set -euo pipefail` at the top, like every existing one.
- Anything that can run longer than about a minute on the remote goes through
  `launch_tmux.sh` (persistent log, survives SSH disconnect). `launch.sh` is
  for short jobs; direct foreground SSH is for dry-runs and probes only.
- `sync_source.sh` is dry-run by default with `--apply` to act. It must never
  gain `rsync --delete` and must keep excluding `.git`, environments, data,
  caches, runs, logs, results, and generated files. Preserve the multiplexed
  SSH connection pattern (ControlMaster/ControlPath) so a password is
  requested at most once per invocation.
- Launchers refuse without a verified integer `KVCACHE_PERSONAL_FREE_GB`.
  Never remove this check or give it a default: filesystem-wide `df` free
  space is meaningless on the shared server.

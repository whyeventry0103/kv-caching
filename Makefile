.PHONY: lint test dryrun audit parity integrate sync sync-apply setup check smoke bank grid replay metrics figures

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
RUFF := $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
BLACK := $(if $(wildcard .venv/bin/black),.venv/bin/black,black)
PYTEST := $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)

lint:
	$(RUFF) check .
	$(BLACK) --check .

test:
	$(PYTEST) -q

# Output-path gate. This repo's input-side gates are strong (pinned revisions,
# token-ID sidecars, Jaccard parity, an 800-row exact replication) and nothing
# ever checked that a number in the paper matched the artifact it came from --
# which is how a systemic estimand bug (Amendment K) reached the prose. Rebuild
# the catalogue from the artifacts, then fail if any literal has no source.
audit:
	$(PYTHON) scripts/build_paper_numbers.py --out paper_numbers.json
	$(PYTHON) scripts/audit_paper_numbers.py --strict

# Local/remote parity: commit, source tree hash, and per-file result artifact
# checksums. The source-tree guard alone was not enough -- the remote sat on an
# old commit with no results/ at all, which is invisible to a hash of the source.
parity:
	$(PYTHON) scripts/check_parity.py --strict

# Integrate a finished cell: harvest -> recatalogue -> gate -> figures -> PDF,
# printing a before/after diff of every number that moved. UNIT=<harvest unit>.
# It does not edit the .tex: docs/INTEGRATION_CHECKLIST.md holds the
# pre-registered read that says what the result means.
integrate:
	$(PYTHON) scripts/integrate_cell.py --unit $(UNIT)

dryrun:
	$(PYTHON) scripts/build_bank.py --config configs/experiments/smoke_fullcache.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/smoke_fullcache_4096.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/pilot_fullcache.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/pilot_fullcache_v2.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/pilot_fullcache_eager_check.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/pilot_fullcache_tokenized.yaml --dry-run
	$(PYTHON) scripts/probe_decoding.py --config configs/experiments/probe_decoding.yaml --dry-run
	$(PYTHON) scripts/probe_decoding.py --config configs/experiments/probe_decoding_random.yaml --dry-run
	$(PYTHON) scripts/probe_decoding.py --config configs/experiments/probe_decoding_h2o.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/pilot_compression.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_math500_powered.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_math500_powered.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/readout_math500_powered.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/readout_math500_fc_control.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/bank_math500_q15b.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/bank_math500_q14b.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_math500_diss256_q15b.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_math500_diss256_q14b.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/readout_math500_diss256_q15b.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/readout_math500_diss256_q14b.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/reverse_readout_diss256_q15b.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/reverse_readout_diss256_q14b.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_math500_diss512.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_math500_diss1024.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/readout_math500_diss512.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/readout_math500_diss1024.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/bank_math500_hottcap.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_math500_hottcap.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/bank_aime24_llama8b.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_aime24_rkv_published.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/bank_fullcache.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_compression.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_decomposition.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/pilot_replay_self_smoke.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/pilot_replay_self.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/pilot_replay_compression.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/pilot_readout.yaml --dry-run
	$(PYTHON) scripts/run_metrics.py --config configs/experiments/add_permutation.yaml --dry-run
	$(PYTHON) scripts/run_mechanism.py --config configs/experiments/mechanism_ladder.yaml --dry-run
	$(PYTHON) scripts/make_figures.py --config configs/experiments/negative_control.yaml --dry-run

sync:
	bash scripts/sync_source.sh

sync-apply:
	bash scripts/sync_source.sh --apply

setup:
	bash scripts/setup_remote.sh --i-understand

check:
	bash scripts/check_remote.sh --config configs/experiments/grid_compression.yaml --run-name preflight

smoke:
	bash scripts/launch_tmux.sh --i-understand smoke-fullcache bank configs/experiments/smoke_fullcache.yaml --allow-model-download --allow-dataset-download

bank:
	bash scripts/launch_tmux.sh --i-understand bank-fullcache bank configs/experiments/bank_fullcache.yaml

grid:
	bash scripts/launch.sh --i-understand grid configs/experiments/grid_compression.yaml

replay:
	bash scripts/launch.sh --i-understand replay configs/experiments/replay_decomposition.yaml

metrics:
	bash scripts/launch.sh --i-understand metrics configs/experiments/add_permutation.yaml

figures:
	bash scripts/launch.sh --i-understand figures configs/experiments/negative_control.yaml

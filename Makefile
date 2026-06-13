.PHONY: lint test dryrun sync sync-apply setup check smoke bank grid replay metrics figures

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
RUFF := $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
BLACK := $(if $(wildcard .venv/bin/black),.venv/bin/black,black)
PYTEST := $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,pytest)

lint:
	$(RUFF) check .
	$(BLACK) --check .

test:
	$(PYTEST) -q

dryrun:
	$(PYTHON) scripts/build_bank.py --config configs/experiments/smoke_fullcache.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/smoke_fullcache_4096.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/pilot_fullcache.yaml --dry-run
	$(PYTHON) scripts/build_bank.py --config configs/experiments/bank_fullcache.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_compression.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_decomposition.yaml --dry-run
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

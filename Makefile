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
	$(PYTHON) scripts/audit_paper_claims.py --strict

# The link the gates above do NOT cover: they trust results/diag. This re-runs
# every producing report against the raw rows on the REMOTE and diffs the
# artifacts field by field. It cannot run locally (no rows here) -- that is why
# it is documented rather than wired into `check`.
#   ssh kvcache-gpu 'cd ~/kv-caching && \
#     export REMOTE_DATA_ROOT=... HF_HOME=... HF_HUB_OFFLINE=1 && \
#     .venv/bin/python scripts/rederive_artifacts.py \
#       --runs-root <runs> --out /tmp/rederive.json'
.PHONY: rederive-help
rederive-help:
	@sed -n '/^# The link the gates above/,/^\.PHONY: rederive-help/p' Makefile

# Local/remote parity: commit, source tree hash, and per-file result artifact
# checksums. The source-tree guard alone was not enough -- the remote sat on an
# old commit with no results/ at all, which is invisible to a hash of the source.
parity:
	$(PYTHON) scripts/check_parity.py --strict

# Integrate a finished cell: harvest -> recatalogue -> gate -> figures -> PDF,
# printing a before/after diff of every number that moved. UNIT=<harvest unit>.
# It does not edit the .tex: docs/operations/INTEGRATION_CHECKLIST.md holds the
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
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_ext_smoke.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_ext_math500_diss256.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_ratio_matched_m3.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_sawtooth_i16_q1.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_sawtooth_i64_q1.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_sawtooth_i256_q1.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_onset_t0_q2.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_onset_t512_q2.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_onset_t2048_q2.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/restoration_lag1_q5.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/restoration_lag4_q5.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/restoration_lag16_q5.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/restoration_lag64_q5.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/restoration_lag256_q5.yaml --dry-run
	$(PYTHON) scripts/run_shape_analysis.py --config configs/experiments/shape_sawtooth_i16_q1.yaml --dry-run
	$(PYTHON) scripts/run_shape_analysis.py --config configs/experiments/shape_sawtooth_i64_q1.yaml --dry-run
	$(PYTHON) scripts/run_shape_analysis.py --config configs/experiments/shape_sawtooth_i256_q1.yaml --dry-run
	$(PYTHON) scripts/run_onset_surface.py --config configs/experiments/onset_surface_q2.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_math500_dacc_t0_r3.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_math500_dacc_t512_r3.yaml --dry-run
	$(PYTHON) scripts/run_grid.py --config configs/experiments/grid_math500_dacc_t2048_r3.yaml --dry-run
	$(PYTHON) scripts/dacc_gate_smoke.py --config configs/experiments/dacc_gate_smoke_r3.yaml --dry-run
	$(PYTHON) scripts/run_basis_fit.py --config configs/experiments/basis_fit_q6_r9.yaml --dry-run
	$(PYTHON) scripts/run_capture.py --config configs/experiments/capture_math500_smoke_r9.yaml --dry-run
	$(PYTHON) scripts/run_capture.py --config configs/experiments/capture_math500_b256_r9.yaml --dry-run
	$(PYTHON) scripts/run_capture.py --config configs/experiments/capture_math500_b2048_r9.yaml --dry-run
	$(PYTHON) scripts/run_capture_v2.py --config configs/experiments/capture_math500_smoke_r17.yaml --dry-run
	$(PYTHON) scripts/run_capture_v2.py --config configs/experiments/capture_math500_b2048_r17.yaml --dry-run
	$(PYTHON) scripts/run_basis_fit.py --config configs/experiments/basis_fit_q6_k256_r17.yaml --dry-run
	$(PYTHON) scripts/run_basis_fit.py --config configs/experiments/basis_fit_q6_llama8b_r17.yaml --dry-run
	$(PYTHON) scripts/run_capture_v2.py --config configs/experiments/capture_math500_llama8b_smoke_r17.yaml --dry-run
	$(PYTHON) scripts/run_capture_v2.py --config configs/experiments/capture_math500_llama8b_b256_r17.yaml --dry-run
	$(PYTHON) scripts/build_input_manifest.py --runs-root runs --reference-run fc-llama50 --model r1_distill_llama_8b --out /dev/null --dry-run
	$(PYTHON) scripts/report_capture.py --runs-root runs --run capture-b256 --out /dev/null --dry-run
	$(PYTHON) scripts/run_exact_lag.py --config configs/experiments/exact_lag_math500_r5.yaml --dry-run
	$(PYTHON) scripts/report_exact_lag.py --runs-root runs --run exact-lag-r5 --out /dev/null --dry-run
	$(PYTHON) scripts/run_fork_control.py --config configs/experiments/fork_control_random_r11.yaml --dry-run
	$(PYTHON) scripts/run_fork_control.py --config configs/experiments/fork_control_slm_r11.yaml --dry-run
	$(PYTHON) scripts/report_fork_control.py --runs-root runs --run fork-control-random --run fork-control-slm --out /dev/null --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_ext_math500_b512_r1.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/replay_ext_math500_b1024_r1.yaml --dry-run
	$(PYTHON) scripts/postprocess_meeting3.py --stage capture-b256 --dry-run
	$(PYTHON) scripts/postprocess_meeting3.py --stage capture-b2048 --dry-run
	$(PYTHON) scripts/postprocess_meeting3.py --stage dacc --dry-run
	$(PYTHON) scripts/postprocess_meeting3.py --stage exact-lag --dry-run
	$(PYTHON) scripts/postprocess_meeting3.py --stage fork-control --dry-run
	$(PYTHON) scripts/postprocess_meeting3.py --stage t1-tv --dry-run
	$(PYTHON) scripts/report_dacc.py --runs-root runs --arms 0=dacc-t0 512=dacc-t512 2048=dacc-t2048 --legacy 256=grid-diss256-tok 2048=grid-diss2048-tok --out /dev/null --dry-run
	$(PYTHON) scripts/report_onset_grid.py --runs-root runs --arms 0=replay-onset-t0 512=replay-onset-t512 2048=replay-onset-t2048 --out /dev/null --dry-run
	$(PYTHON) scripts/run_restore_v2.py --config configs/experiments/restore_v2_math500_r17.yaml --dry-run
	$(PYTHON) scripts/run_fork_precision.py --config configs/experiments/fork_precision_r17.yaml --dry-run
	$(PYTHON) scripts/pp_smoke.py --config configs/experiments/pp_smoke_s1.yaml --dry-run
	$(PYTHON) scripts/run_grid_pp.py --config configs/experiments/grid_math500_pp256_s1.yaml --dry-run
	$(PYTHON) scripts/run_grid_pp.py --config configs/experiments/grid_math500_pp1024_s1.yaml --dry-run
	$(PYTHON) scripts/run_replay_pp.py --config configs/experiments/readout_math500_pp256_s1.yaml --dry-run
	$(PYTHON) scripts/run_replay_pp.py --config configs/experiments/readout_math500_pp1024_s1.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/reverse_readout_pp256_s1.yaml --dry-run
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run eligibility
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run dacc-restricted
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run mdelta
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run omega-layers
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run bridge
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run collapse
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run auc-tv
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run restore
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run fork-precision
	$(PYTHON) scripts/report_r17.py --runs-root runs --dry-run capture-v2
	$(PYTHON) scripts/run_replay.py --config configs/experiments/restoration_smoke.yaml --dry-run
	$(PYTHON) scripts/run_replay.py --config configs/experiments/restoration_diss256.yaml --dry-run
	$(PYTHON) scripts/run_shape_analysis.py --config configs/experiments/shape_p2v2_restricted.yaml --dry-run
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

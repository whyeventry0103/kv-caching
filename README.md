<div align="center">

# The Trajectory Audit

### *Near-Lossless Under Readout, Collapse Under Free Generation:<br>A Trajectory Audit of KV-Cache Compression for Reasoning*

**Muhammad Ali Shah** · LUMS

![python](https://img.shields.io/badge/python-3.11-blue)
![tests](https://img.shields.io/badge/tests-236%20passing-brightgreen)
![status](https://img.shields.io/badge/status-working%20draft-orange)
![protocol](https://img.shields.io/badge/protocol-preregistered-purple)
![artifacts](https://img.shields.io/badge/artifacts-29%2F29%20re--derived-brightgreen)

</div>

---

> **The same compression method can be certified near-lossless and be collapsing in deployment — at the same time, honestly, on the same benchmark.** It depends on whether the evaluation lets the model write its own reasoning. This repo measures that gap, explains its mechanism, and ships the protocol for auditing it.

KV-cache eviction methods are certified *near-lossless* almost entirely on evidence that holds the generated text fixed (perplexity, teacher-forced accuracy, retrieval probes). Free-running reasoning benchmarks report severe degradation for the same methods. **Both are right — they measure different channels of the same loss.**

## The result in one figure

<p align="center"><img src="figures/fig1_phenomenon.png" width="85%"></p>

On MATH-500 with DeepSeek-R1-Distill-Qwen-7B at cache budget 256 (3.8% of the mean trace), a protocol that teacher-forces the reference reasoning trace and scores answer readout would certify **Random eviction at 0.765** — while its deployed, free-running accuracy is **0.422**. The difference is a **trajectory residual**: accuracy loss that held-trace evaluation *structurally cannot see*, because divergence is exactly zero at the moment of eviction and compounds over the trajectory that follows.

| Policy @256 | Free-run | Readout | Trajectory residual | Bracket (excl. 0) |
|---|---:|---:|---:|---:|
| Random | 0.422 | 0.765 | **−0.337** | [−0.337, −0.282] |
| StreamingLLM | 0.237 | 0.584 | **−0.344** | [−0.470, −0.344] |
| H2O-style | 0.247 | 0.597 | **−0.344** | [−0.478, −0.344] |
| R-KV (certified port) | 0.520 | 0.798 | **−0.268** | [−0.268, −0.251] |

*Matched full-cache reference **0.885** — the pinned bank restricted to each cell's own conditions. (The unrestricted 2000-row bank is 0.882 and is quoted in the setup only.) A residual of similar magnitude exists for every policy tested — including the best free-runner. All brackets and problem-clustered 95% CIs exclude zero, under every treatment of ambiguous rows.*

**Full results:** [`docs/results/RESULTS.md`](docs/results/RESULTS.md) — every cell, every control, every caveat.

## The trajectory audit (the deliverable)

A four-step, intervention-agnostic protocol for pricing what held-trace evaluation misses — applicable to cache quantization, speculative decoding, early exit, retrieval truncation, or any intervention applied *during* generation:

<p align="center"><img src="figures/fig0_protocol.png" width="85%"></p>

1. **Forward readout** — hold a reference trace fixed; measure the intervention's effect on answer readout.
2. **Free-run** — let the intervention shape the whole trajectory; measure the total effect.
3. **Reverse readout** — read the intervention's own traces out *without* it; forward + reverse bracket the trajectory share of the cost.
4. **Divergence profile** — check per-step divergence for compounding after the intervention's onset.

The audit ships with its **validity kit**, and every instrument certifies itself before any number is read:

| Control | What it proves | Result here |
|---|---|---:|
| Two-reference negative control | the distribution instrument doesn't false-positive | floor 0.068, 0% significant |
| Pre-intervention self-check | the replay harness adds zero divergence of its own | TV = 0.0000 exactly |
| Resampling calibration | answer-regeneration easiness can't manufacture the effect | +0.003 (132× below the residual) |
| Exact replication | generation is deterministic under the pinned seeds | 800/800 rows identical |
| Grader audit | grading error isn't driving the accuracies | 0 disagreements over 3600 rows |

## Why you can trust these numbers

This project is run under an unusually strict evidence discipline:

- **Preregistered protocol** with append-only, dated amendments — including a pre-committed confirm/complicate/refute read that came out **COMPLICATE** and is reported as such.
- **Every artifact re-derives from the raw generation rows.** `scripts/rederive_artifacts.py` replays each producing report against the JSONL on the GPU host and diffs it field by field: **29 of 29 identical**. It found one real defect — a compute figure whose inclusion list had silently drifted nine run directories behind.
- **Numbers are gated twice.** `make audit` rebuilds the catalogue from the artifacts and fails on any unsourced literal; a second gate pins 79 load-bearing numbers to the *specific quantity* each asserts, so a figure cannot pass by coincidentally matching an unrelated value.
- **Two-stage integrity gate** for the R-KV port: retained-set parity vs the authors' code (Jaccard 1.0), then a real-generation smoke — which caught a real bug the synthetic harness could not see.
- Truncated generations stay in every denominator, scored incorrect. Result rows are never patched — buggy-instrument data is regenerated wholesale.
- Boundaries are self-reported: the random-beats-recency ordering **reverses on the Llama base**, and the paper says so in its own threats section.

## Repository map

```
src/kvcache/     all computation and scientific logic (presses, replay, grading, stats)
scripts/         thin orchestration: runners, nightly queue, harvester, gates, figures
configs/         pinned models/datasets, one YAML per experiment, the run queue
tests/           CPU-only, no network, no weights
figures/         the paper's figures (scripts/make_paper_figures.py)
results/diag/    the report artifacts every number traces to
docs/            see docs/README.md for the index
kv_readout_vs_trajectory_paper.tex    the paper
paper_numbers.json / paper_claims.json  the number-provenance gate's inputs
```

| Document | What it is |
|---|---|
| [`docs/protocol/PREREGISTRATION.md`](docs/protocol/PREREGISTRATION.md) | frozen protocol; changes only via dated amendments |
| [`docs/results/RESULTS.md`](docs/results/RESULTS.md) | the full results record (generated from the artifacts) |
| [`docs/results/PAPER_FINDINGS.md`](docs/results/PAPER_FINDINGS.md) | results log + the binding framing rules |
| [`docs/operations/WORKFLOW.md`](docs/operations/WORKFLOW.md) | the local↔remote loop |
| [`docs/operations/remoteserver.md`](docs/operations/remoteserver.md) | authoritative ops and safety policy |

## Quickstart (local)

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
make lint     # ruff + black
make test     # pytest, CPU-only, a few seconds
make dryrun   # validates every runner × config without touching a model
make audit    # rebuild the number catalogue and gate the paper against it
```

The local machine is for source, configs, docs, and CPU tests **only**; models, datasets, caches, and all experiments run on a remote GPU host. Read [`docs/operations/remoteserver.md`](docs/operations/remoteserver.md) before any setup, synchronization, or launch. The loop is rsync-first ([`docs/operations/WORKFLOW.md`](docs/operations/WORKFLOW.md)): edit and test locally, `make sync` to preview / `make sync-apply` to apply a source-only rsync, run remotely under tmux (`scripts/launch_tmux.sh` — persistent logs, SSH-disconnect-safe), then commit locally.

Heavy commands refuse without an explicit `--i-understand`; first-time downloads require `--allow-model-download` / `--allow-dataset-download`; on the shared host, launchers also require a verified `KVCACHE_PERSONAL_FREE_GB` (filesystem-wide `df` is insufficient there). Scientific claims and analysis choices are frozen in [`docs/protocol/PREREGISTRATION.md`](docs/protocol/PREREGISTRATION.md) — changes happen only via dated amendments.

## Status

Working draft. All queued experimental cells have landed, including the scale axis (1.5B/14B), the cross-architecture Llama-8B check, AIME-24, and R-KV in its own published configuration. Total generation: 375 GPU-hours over 10,007 generations. arXiv preprint targeted for 2026.

## Citation

```bibtex
@article{shah2026trajectoryaudit,
  title  = {Near-Lossless Under Readout, Collapse Under Free Generation:
            A Trajectory Audit of KV-Cache Compression for Reasoning},
  author = {Shah, Muhammad Ali},
  year   = {2026},
  note   = {Working draft. Code: https://github.com/whyeventry0103/kv-caching}
}
```

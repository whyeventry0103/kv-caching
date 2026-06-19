# CONTEXT HANDOFF — kv-caching research project

**Purpose of this file:** a complete, self-contained handoff. An LLM with zero prior
context should be able to read this and resume exactly where we are. Last updated
2026-06-14. Repo: `/Users/apple/dev/kv-caching` (moved off `~/Desktop` 2026-06-14 to
escape a recurring macOS TCC lockout; `~/Desktop/kv-caching` is a stale backup),
branch `main`, GitHub `whyeventry0103/kv-caching`. Current HEAD: `61760a5`
(M4 complete + M5 ADD pipeline).

This document is authoritative for *narrative and decisions*. For *code*, the repo is
ground truth — always re-read files before acting (state moves fast; this file can
lag). `PREREGISTRATION.md` is authoritative for scientific protocol;
`remoteserver.md` is authoritative for ops.

---

## 0. TL;DR — where we are right now

- This is an **analysis/evaluation paper** about *why* KV-cache compression papers
  report near-lossless reasoning while the independent HOTT benchmark reports
  collapse. We are building the measurement harness, not a new compression method.
- **Roles:** the human (final-year CS undergrad, LUMS, strong ML) sets direction.
  An assistant named **Claude** (this reviewer) is the **technical decision-maker /
  reviewer**. **Codex** is the **executor** (writes agreed code, runs local
  validation, runs remote/GPU jobs). All heavy/remote work is gated.
- **Progress:** M0 ✓, M1/M2 (full-cache pilot harness) ✓ accepted, **M2.5 stack
  migration ✓ promoted** (transformers 4.52.4→**4.56.2**, kvpress 0.2.10→**0.5.3**,
  to get decoding-phase compression), **M3 compressed-generation code ✓ reviewed &
  approved**.
- **⚠ SUPERSEDED (double-BOS):** the paragraph below describes the FIRST
  compression pilot `pilot-compression-v1-r2` (51/80), which was generated with the
  **double-BOS bug** and is RETIRED to provenance. The CANONICAL single-BOS
  compression pilot is `pilot-compression-v2`, and the canonical decomposition is the
  **M4 result block further down** (free-run random-256 0.45 / SLM-256 0.20, etc.).
  Use the M4/M5 numbers, not v1-r2, for anything external.
- **Current state (M3 era, retired):** the **80-generation compression pilot completed** as
  `pilot-compression-v1-r2` and has been pulled locally. Audit: 80 schema-v2 rows,
  16 files, consistent stack/source/config hashes, 51/80 correct, 8/80 truncated,
  no logged errors, and no serialized tensor/cache artifacts. Cell results are in
  §6/§7. **M3 pilot REVIEWED & ACCEPTED by Claude (2026-06-14).** M4 design
  (`docs/M4_REPLAY_DESIGN.md`) is **REVIEWED & APPROVED by Claude (2026-06-14)** with
  rulings (below). **M4.0 CPU implementation REVIEWED & APPROVED by Claude
  (2026-06-14)** — verified alignment, enabled-flag compression scoping (step-counter
  gated via forward_hook early-return), onset off-by-one (correct), self-replay
  identity test, exact token-id threading + round-trip validation, readout estimator.
  86 tests pass. **GPU now AUTHORIZED for M4.1→M4.4 in order**, with a mandatory
  STOP-and-report at M4.2 (self-replay negative control) before any compressed replay.
  **M4.1 config must set `store_token_ids: true`; the new tokenized bank replaces
  v2-tf456 as the M4 reference** (v2-tf456 lacks sidecars). **Self-replay caveat:** the
  hard 1e-6 TV gate may be too tight — two CUDA forwards over identical tokens have a
  tiny nondeterminism floor (~1e-6–1e-5); make the tolerance configurable, measure the
  floor, confirm it is << τ (0.05). Minor: `compression_events` counts layer-level
  prune calls (×num_layers), not token-prune steps.

  **M4 rulings:** (1) token-ID sidecar approved (regen 20-row FC bank to capture IDs).
  (2) exact transient full-vocab TV primary, top-k bounds fallback-only. (3) one model
  / two caches, but scope compression via an explicit `press.enabled` flag toggled
  around each forward (NOT object-identity) + hard test that the cache is
  mutated-not-replaced. (4) **readout estimand defined** (accuracy-scale):
  teacher-force the reference reasoning up to the `</think>`/answer-marker boundary into
  the compressed cache, then free-generate the answer span under compression (sampled,
  temp 0.6, matched seed), score with is_equiv → `readout_accuracy`;
  `readout_change = readout_accuracy − fc_accuracy`; `residual = total_change −
  readout_change`. Per-step TV stays a SEPARATE divergence/TDP instrument — never
  subtracted from accuracy (Codex correctly blocked `accuracy − mean_TV` as
  dimensionally invalid). Disclose the on-policy bias (uses good reasoning → inflates
  residual; bracketed by reverse replay in M7). (5) replay self-control in M4;
  two-FC-bank ADD permutation control in M5. Plus: handle no-prune (high budget) →
  `onset_step=null`, TV≈0, not a bug.

  **M3 drill-down findings (Claude, per-problem, CPU):** regime separation confirmed
  (budget 2048 ≈ FC at 90–95%; budget 256 collapses to random 50% / SLM 20%). The
  **Random > StreamingLLM** gap at 256 is real and broad (Random ≥ SLM on 4/5
  problems), signed flip-mass random net −2.25 vs SLM −3.75, **no gains anywhere**
  (no >100% regime on small-budget MATH — expected). **Correction to the first
  report:** StreamingLLM-256's dominant failure is *answered-wrong* (13/20), not
  gibberish-truncation (only 3/20 truncated) — it confidently boxes wrong answers
  (trajectory derailment), the gibberish loop is secondary. Length explodes at 256
  (mean ~6.8–7.6k tokens vs FC ~1.3–6.1k) — the HOTT "low budget lengthens traces"
  effect, in-harness. Working hypothesis (to be TESTED by M4, not yet claimed): SLM
  deletes all mid-trace anchors (sinks+window) → derails; random preserves some by
  chance. Underpowered (n=4/problem); the 250-problem bank is the powered unit.

  **M4 design gotcha flagged:** replay needs the exact reference token-id sequence,
  but FC rows store only decoded text. Resolve before GPU: either store
  `trace_token_ids` (re-run the 20-row pilot FC bank; additive) or re-tokenize with a
  round-trip assertion. Lean: store token ids.

  **M4.1 correction discovered (2026-06-14):** the first tokenized-bank attempt,
  `pilot-fullcache-tokenized-v1`, stopped on its first sidecar validation because
  the rendered Qwen prompt already contained BOS while generation tokenized it with
  the default `add_special_tokens=True`, producing two BOS tokens. This affected all
  earlier M2/M3 pilot generations. Generation and prompt-length accounting now use
  `add_special_tokens=False`, with regression coverage. Earlier pilots remain
  provenance/harness diagnostics only. A fresh
  `pilot-fullcache-tokenized-v2` is the first corrected reference. It was launched
  in remote tmux on 2026-06-14 after a tokenizer-only remote assertion confirmed
  exactly one BOS and an exact prompt round trip; its first result row and token-ID
  sidecar both passed and were written successfully. **M4.1 completed and audited:
  20/20 correct, 0 truncations, mean 2,407.2 generated tokens, 20/20 sidecars
  independently validated, one BOS in every prompt, consistent source/config/version
  metadata, and no tensor artifacts.** The bank auditor was corrected to ignore the
  additive `token_ids/` namespace, with regression coverage. The authorized one-row
  M4.2 self-replay smoke (`pilot-replay-self-smoke-v1`) passed exactly: 1,004
  replay steps, min/max/mean/summed TV all 0.0, zero threshold crossings, no onset,
  and no compression events. The full 20-row M4.2 control
  (`pilot-replay-self-v1`) also **passed exactly**: 20/20 replay rows and profiles,
  48,144 total steps, global min/max/mean/summed TV all 0.0, zero nonzero steps,
  zero crossings at every preregistered threshold, no onset/TDP crossing in 20/20
  rows (operational control percentile 1.0), zero compression events, no errors,
  and no tensor artifacts. Scalar artifacts were pulled locally and the CPU-only
  report at `results/diag/pilot-replay-self-v1/` contains `summary.json`,
  `table.md`, and `self_replay_control.svg`. **Mandatory stop is active:** send
  this M4.2 report to Claude before any compressed replay. The 80-row M3
  compressed pilot must also be regenerated under this protocol before it can
  supply `total_change` for M4.4; mixing old double-BOS compressed rows with the
  corrected reference would violate exchangeability.

  **★ M4 COMPLETE + ACCEPTED by Claude (2026-06-14). Committed `67e1194`.**
  Codex hit its usage limit during the M4 report; Claude took over execution from
  here on (plans AND runs code/remote jobs). Single-BOS pilot, full-cache = 20/20.
  **THE FIRST REAL RESULT — a double dissociation at budget 256** (n=5 problems × 4
  seeds, underpowered, directional):

  | cell | free acc | readout acc | total Δ | readout Δ | residual | lost mass |
  |---|---|---|---|---|---|---|
  | random-2048 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
  | streamingllm-2048 | 0.90 | 1.00 | −0.10 | 0.00 | −0.10 | 0.10 |
  | random-256 | 0.45 | 0.90 | −0.55 | −0.10 | **−0.45** | 0.55 |
  | streamingllm-256 | 0.20 | 0.45 | −0.80 | **−0.55** | −0.25 | 0.80 |

  **Interpretation (framing-disciplined):** at 256, the two methods collapse for
  OPPOSITE reasons. StreamingLLM-256 is **readout-dominated** (−0.55 readout): its
  sinks+recent-window eviction deletes the MIDDLE of the reasoning where the
  answer-relevant work lives, so even reading off the *good* teacher-forced trace
  fails. Random-256 is **residual/trajectory-dominated** (−0.45 residual): uniform
  eviction preserves readout (0.90) but injects drift that compounds during free
  generation. This EXPLAINS the Random>StreamingLLM surprise mechanistically and is
  the decomposition's proof-of-value — accuracy alone (both "collapse at 256") cannot
  tell them apart. **Bias note:** residual is inflated by the off-policy bias
  (readout measured on good states), so SLM's readout-dominance is CONSERVATIVE
  (robust), while random's trajectory-dominance is bias-vulnerable and must be
  bounded by reverse replay (M7). No gains anywhere (no >100% on small-budget MATH).
  Length explodes at 256 (free-run mean ~8.2–8.4k tok vs FC ~2.4k); readout
  estimand short-circuits this (generates only the answer span) so it is not
  truncation-contaminated. High budget (2048) genuinely near-lossless (TV~0.002–0.004).
  Report: `results/diag/m4-pilot-v1/{summary.json,table.md,decomposition.svg}` via
  `scripts/report_m4.py`. Engine + report validated (89 tests pass).

  **★ M5 ADD pipeline BUILT + validated on pilot (2026-06-14). Committed `61760a5`.**
  `scripts/report_add.py` + test: per-problem observed ADD + permutation p, termination
  vs within-answered split, Miller-Madow entropy shift, BH across problems, mean signed
  flip-mass (net == total_change). Pilot result corroborates M4:
  streamingllm-256 ADD 0.80 (within-ans TV 0.80, 4/5 BH-significant even at n=4,
  entropy +1.22); random-256 ADD 0.55 (within-ans 0.53, 0/5 significant, entropy
  +0.98 = diffuse derailment); 2048 near-zero (random 0.00, slm 0.15). Report at
  `results/diag/m5-add-pilot-v1/`. **M5 negative control (zero-GPU smoke) PASSED**:
  FC-vs-FC (split seeds {11,47} vs {23,89}) → mean ADD 0.000, 0/5 significant — the
  ADD pipeline does not false-positive. (Weak control: pilot FC has no cross-seed
  answer variance, so the null is degenerate-but-correct; the POWERED two-FC-bank
  control comes FREE at M8 by splitting the n=8 bank — run it there, no separate GPU.)
  **M5 functionally complete for the pilot.**

  **★ EXTERNAL REVIEW + RESPONSES (2026-06-15). Committed `c502289`, `9fef4ef`;
  PREREGISTRATION Amendment D.** A sharp review praised the framing discipline, grader
  fix, double-BOS catch, and bitwise self-replay control, and flagged real risks. Actions:
  - **M4 dissociation is UNCONFIRMED, not a result.** Added problem-level bootstrap 95%
    CIs to `report_m4` — they are wide (SLM-256 readout 0.45 **CI[0.15,0.80]**;
    random-256 readout 0.90 **CI[0.80,1.00]**; the per-method readout/residual split is
    barely resolved at n=5). Pre-committed (Amendment D) to report the powered MATH-500
    result **even if it reverses** the dissociation. Do NOT let pilot directionals
    calcify; never assert the dissociation externally until 250×n=8 confirms it.
  - **Random>SLM mundane causes handled:** budget parity VERIFIED (both presses prune to
    256-after / 320-before identically at target_size=256 → not an accounting artifact);
    `press_seed(pid,seed)` adds per-(problem,seed) RNG so it can't be a lucky seed
    (wired into run_grid + run_replay; tested).
  - **One canonical FC reference pinned in code:** `report_m4` asserts all
    compression/readout rows reference `pilot-fullcache-tokenized-v2`'s (pid,seed) set.
    `pilot-fullcache-v2-tf456` (19/20, double-BOS) is RETIRED — do not use it.
  - **Readout estimand is math/span-shaped;** per-step TV is the general fallback for
    logic benchmarks (FOLIO/StrategyQA) where the answer isn't a span. Paper states which
    instrument carries the decomposition on which benchmark.
  - **R-KV port = HARD GATE (M6):** reproduce a specific published (model,benchmark,budget)
    number within noise, committed as a test, BEFORE any R-KV row enters analysis. Same
    for H2O. "If you can't reproduce their headline, you don't have R-KV."
  - **RSP demoted to bonus;** restoration-on-natural-traces is the PRIMARY mechanism
    instrument. **compression_events is layer-level** (never divide as a step rate).
    **Powered two-FC negative control MUST run at M8** (split the n=8 bank), not skipped.

  **★ M6 IN PROGRESS — H2O DONE + GPU-VERIFIED (2026-06-15). Committed in the M6 H2O
  commit.** kvpress 0.5.3 `ObservedAttentionPress` is the in-library H2O (docstring
  cites arXiv:2306.14048), a ScorerPress, requires eager attention. Implemented:
  `load_model_and_tokenizer(attn_implementation=...)`, registry `h2o`→EXECUTABLE +
  `press_needs_eager_attention`, run_grid/probe auto-load eager. Probe `h2o-probe-v1`
  passed: cache bounded 256/320 (`all_after_at_most_target`), coherent answer
  ((1,π/2) vs gold (3,π/2) — angle right, radius wrong under budget 256), no tensor
  artifacts, 810 tokens. **Three methods now executable: streamingllm, random, h2o.**
  H2O rides on kvpress (same library HOTT used), so its faithfulness is library-level;
  behavioral sanity on the pilot suffices (vs R-KV's hard published-number gate).
  **NEXT: R-KV port** (not in kvpress; our own port; the >100% method). HARD GATE per
  Amendment D: reproduce a specific published (model,benchmark,budget) number within
  noise, committed as a test, before ANY R-KV row enters analysis.

  **★ 2026-06-15 (second review actioned + scaling started):**
  - Second external review answered in full; commits include per-problem RNG +
    bootstrap CIs + pinned FC reference (`c502289`), Amendment D (`9fef4ef`), H2O
    (M6 commit), eager gate, readout-boundary rule, Amendment E. 93 tests.
  - **Eager-attention exchangeability gate PASSED:** full-cache eager 19/20 vs sdpa
    20/20, 19/20 per-problem agreement, one Bornoulli flip (927 s47), no systematic
    shift. Decision: **MIX** — sdpa for FC/StreamingLLM/Random (fast), eager for H2O;
    one pinned sdpa FC reference; verification committed as justification.
  - **POWERED MATH-500 FC BANK LAUNCHED (2026-06-15)** — the foundation. 250 problems
    × n=8 (16384 cap, sdpa, token-ID sidecars), split across both GPUs:
    `math500-fc-powered-g0` (seeds 11,23,47,89, GPU0) + `math500-fc-powered-g1`
    (seeds 131,197,263,347, GPU1), ~15-17h each. **Merge the two session dirs to the
    canonical n=8 reference when done** (different seed files, no collision).
  - **R-KV port (M6) — IMPLEMENTED + PARITY-GATED (commit `78659a6`).**
    `src/kvcache/presses/rkv.py::build_rkv_scorer` = kvpress `SnapKVPress` subclass:
    importance (observation-window attention, max-pool, window=8, kernel=7) minus
    `mix_lambda`(0.07)·cosine-redundancy; recent window force-kept; no eager needed
    (recomputes Q@K). `build_rkv_press` wraps it in DecodingPress (for the parity
    harness only). **`get_press("rkv")` REFUSES (PARITY-GATED)** so R-KV cannot enter
    any analysis until verified. Plan + exact parity procedure: `docs/RKV_PORT.md`.
    **PENDING (needs a free GPU — both busy with FC bank): the parity gate** — set up
    the authors' reference env, run their `r1_kv.py` (record_kept_token_indices) on
    Qwen-7B at two budgets, run our port on the same traces, require retained-set
    Jaccard≈1, commit artifacts; the one open detail to confirm is `_redundancy_score`
    vs R-KV's `cal_similarity` (threshold/retain_direction). On pass → move rkv to
    EXECUTABLE. Repo cloned at remote `~/scratch/R-KV` (HuggingFace/rkv is the ref).
  - **FC bank progress (2026-06-16): healthy ~37 tok/s** (median 61s/gen), g0 ~183/1000,
    g1 ~167/1000; ~a day to finish both halves. **Merge g0+g1 → canonical n=8 bank.**

  **★ POWERED FC REFERENCE BUILT (2026-06-17): `math500-fc-powered-merged`.** g0+g1
  merged → 2000 rows (250 problems × n=8), **full-cache acc 88.25%**, truncation 1.9%
  at 16K cap, versions consistent. 4 source_tree_hashes (from mid-run syncs) but the
  generation path is byte-identical across `ff4240b..HEAD` (verified by git diff) → rows
  exchangeable; `audit_bank --allow-mixed-provenance` accepts it. `merge_banks.py` did
  the union. This is THE canonical powered reference for all decomposition.
  **R-KV scoring parity-verified (Jaccard 1.0 vs authors' code at budgets
  {128,256,384,512}); press real-generation smoke still pending (GPU).**
  **NEXT: powered compression cells (the dissociation test at scale) — a multi-day GPU
  run because budget-256 free-runs ramble (~8k tokens). run_grid lacks --resume yet.**

  **★ DISSOCIATION GRID DONE (2026-06-19): `grid-math500-dissociation-v1` 800/800.**
  Powered free-run result (100 problems × n=4): FC 0.885 → StreamingLLM-256 **0.237**
  [0.172,0.302], Random-256 **0.422** [0.343,0.500]. **Random − StreamingLLM = +0.185,
  CI [0.130,0.242], 94/100 problems** → the surprising ordering (random eviction beats
  "principled" StreamingLLM at low budget) SURVIVES powering decisively. VERDICT: GO.
  Honest caveat: the grid confirms only the PHENOMENON; the paper's actual IDEA (the
  readout/trajectory decomposition) is UNTESTED until the readout run lands. Readout is
  therefore the single decisive next run.

  **★ TIMED/RESUMABLE REPLAY HARNESS BUILT (2026-06-19, synced):** `run_replay.py` gains
  `--resume` (heals partial trailing line + skips done (reference_key,press,budget)
  conditions, both readout & compression_replay modes) and `--seeds A,B` (per-GPU shard
  by seed → disjoint per-seed output files, two GPUs never collide). `launch.sh` gains
  `KVCACHE_MAX_HOURS=N` → wraps python in `timeout --kill-after=120s --signal=TERM`,
  prints "STOPPED by timer (resumable)"; rerun the identical command (same
  KVCACHE_RUN_NAME + --resume) to continue exactly. `write_replay_profile(overwrite=)`
  for orphan-profile safety on resume. Tested (heal+skip unit test, 96/96 suite). Configs:
  `readout_math500_diss256.yaml` (256 readout, 800 conds, DECISIVE) + `replay_math500_
  diss256.yaml` (256 TV profiles, 800 conds). User runs nightly 1am-8am on BOTH GPUs
  (seeds 11,23→GPU0 / 47,89→GPU1), run names readout-diss256-g{0,1}/replay-diss256-g{0,1};
  then I compute total−readout=residual (the decomposition that tests the idea).

  **★ ROADMAP TO PAPER (estimates, single GPU ~20GB; time is the only constraint):**
  Runs left (cells = 100 problems × n=4): (1) finish dissociation grid [~12h];
  (2) budget-2048 SLM+Random [~0.8d]; (3) R-KV press smoke [~0.5h]; (4) H2O cells
  256+2048 eager [~1.5d]; (5) R-KV cells 256+2048 [~1.5d]; (6) readout (teacher-force→
  answer span, all cells, cheap) [~0.7d]; (7) replay TV profiles, 256 cells [~1.5-2d];
  (8) self-replay + 2-FC-bank negative controls [~0.5d+CPU]; (9) AIME (R-KV >100% home
  turf — the contradiction) [needs dataset handle, ~1-2d]. **Compute subtotal ~8-10
  GPU-days** (budget-256 ramble is the time sink; GPU 1 ~halves it). Code left: **causal
  ladder M7** (restoration + TV-matched-noise + reverse replay — new engines, ~2-3d code
  +1-2 GPU-d); optional ADD at n=8 (doubles compression, ~2-3d). **Start writing:**
  Methods/Related-Work/Background/protocol are results-independent → draftable NOW; core
  dissociation table + Fig 2 after runs 1-6 (~1 week); Fig 4 (mechanism) after replay +
  one causal rung (~1.5 wks); Fig 1 (contradiction) after AIME. **~1 week to the core
  writable result; ~2-2.5 weeks to a strong submission.**

  **User decisions (2026-06-14):** (1) Claude launches/manages remote GPU jobs itself
  within the guards, reports after. (2) Scope = ports-first then MATH+AIME core
  (pre-registered path). Codex is retired; Claude is reviewer+executor.

  **Remote ops reference (verified 2026-06-14):** GPU idle, no tmux jobs. Launch via
  `ssh kvcache-gpu` then export the env block (KVCACHE_REMOTE=1,
  KVCACHE_PERSONAL_FREE_GB=70, REMOTE_{DATA,CACHE,RUNS}_ROOT=/home/sk10741/
  kvcache-storage/{data,cache,runs}, ENVIRONMENT_NAME=.venv, UV/PIP/HF/TORCH cache
  dirs under .../cache/*, CUDA_VISIBLE_DEVICES=0 or 1, GPU_MODEL, GPU_COUNT=1) and run
  `bash scripts/launch_tmux.sh --i-understand <session> <bank|grid|replay|...> <config>`.
  check_remote.sh validates `.sync-state` source hash → must `make sync-apply` after any
  local source/config change before launching. New configs must be synced to remote.

  **★ NEXT: M6 — port H2O + R-KV (the "science at scale" gate).** kvpress 0.5.3
  inventory checked: **H2O ≈ `ObservedAttentionPress`** (in-library attention
  heavy-hitter; HOTT used kvpress's attention press as H2O). Needs eager attention
  (`attn_implementation="eager"`) to expose attention weights — memory-OK because
  decode attention is [1×cache_len] (only prefill is O(n²), and MATH prompts are
  short); just slower. Verify it's a `ScorerPress` (DecodingPress-compatible), label
  "H2O (ObservedAttention)", sanity-check. **R-KV** is NOT in kvpress → real port from
  the official repo (redundancy-aware, reasoning-specific), must reproduce a source
  sanity number before entering the grid. Registry: move `h2o`→EXECUTABLE via
  ObservedAttention; keep `rkv` PORT_REQUIRED until ported. Then **M8 scale**:
  MATH-500 250×n=8 + AIME (R-KV's >100% home turf), all methods × budgets; powered
  negative control falls out of the n=8 bank. Other kvpress baselines available if
  wanted: SnapKV, TOVA, Knorm, PyramidKV, AdaKV, DuoAttention, ExpectedAttention.

  **M4.2 REVIEWED & ACCEPTED by Claude (2026-06-14):** self-replay control is
  bitwise-perfect (global max/sum TV = 0.0 over 48,144 steps, 0 nonzero steps, 0
  crossings, no onset) → the replay engine is validated. Double-BOS bug accepted as
  correctly caught (by the token-ID round-trip validation — vindicates the sidecar
  design) and fixed (`tokenize_rendered_prompt`, `add_special_tokens=False`, regression
  test). Corrected single-BOS bank `pilot-fullcache-tokenized-v2` (20/20) is the
  canonical M4 reference; all double-BOS pilots (incl. v2-tf456 19/20 and
  pilot-compression-v1-r2) retire to provenance. Note: the v2-tf456 927-s11 "sign
  error" flip was partly a double-BOS artifact (single-BOS → 20/20); does not change
  the M2.5 acceptance. **Claude RULING: rerun the M3 compressed pilot single-BOS
  (`pilot-compression-v2`) — required for exchangeability.** GPU AUTHORIZED now for:
  (1) M3 single-BOS rerun [also re-measures regime separation / Random>SLM],
  (2) M4.3 forward replay (80 cond, on corrected ref), (3) M4.4 readout (80 cond),
  (4) the scalar decomposition combining them. Engine validated → no mid-gate; send one
  combined report. Still gated: M5+, H2O/R-KV, RSP, KIVI, full grid.
- **Then:** review grid results → M4 (teacher-forced replay + self-replay negative
  control) → M5 (ADD) → M6 (port H2O, then R-KV — the "science starts" gate) → M7
  (mechanism interventions) → M8 (full pre-registered grid). **Do not skip ahead.**
- **Uncommitted right now:** the verified RandomPress device fix + random probe config
  + probe tweaks (see §7). Two untracked planning docs (`publication_strategy.md`,
  `research_landscape.md`) are intentionally never committed. `context.md` is also
  untracked and maintained as a handoff.

---

## 1. The project (science)

### 1.1 One-paragraph statement
The HOTT benchmark ("Hold Onto That Thought," arXiv:2512.12008, Dec 2025) showed
KV-cache compression methods marketed as near-lossless actually collapse on
long-generation reasoning, while method papers (R-KV 2505.24133, ThinKV 2510.01290)
report near-lossless or >100% accuracy. Nobody has explained *why*. This project
explains it. Core idea: compressing the cache **during generation** is an
intervention on the sampling process, so its effect decomposes into **readout
distortion** (how compression changes the next-token distribution holding the trace
fixed, measured by teacher-forced replay of a full-cache reference trace) and
**trajectory distortion** (the residual — how it changes what the model actually
writes, which compounds). Plus a suite of divergence instruments and a causal ladder.
Target: ICLR 2027 main (arXiv stake + workshop first); TMLR as the deadline-free
publication floor.

### 1.2 NON-NEGOTIABLE framing discipline (a reviewer kills the paper if we slip)
1. **Trajectory distortion is a DEFINITION BY CONSTRUCTION, never a measured causal
   effect.** Readout is measured on the reference trace under teacher forcing; total
   is measured under free-running compressed generation; these live on *different
   probability spaces*, so `trajectory := total − readout` is a definition, not an
   identity. Always describe the residual as *"accuracy change unexplained by
   on-reference readout distortion."* Never write "trajectory distortion causes the
   accuracy loss." Known bias: the reference trace visits *good* states, so
   on-reference readout likely *underestimates* per-step damage on the compressed
   model's own (worse) trajectory → inflates the trajectory share. Our headline is
   biased toward itself; we say so. Reverse replay + TV-matched noise control bound it.
2. **All instruments measure FIDELITY to the full-cache reference, NOT damage.** The
   field's most interesting phenomenon (>100% accuracy under compression) means
   deviation can be *improvement*. Every divergence quantity must be reported with
   its **signed** outcome consequence (gained vs lost flip-mass). Framing is
   "behavioral fidelity for deployment parity, with correctness as the quality
   arbiter." Never let divergence implicitly mean damage. (This is the
   PSNR-vs-perception blind spot.)

### 1.3 The metric suite (final post-review design)
- **ADD (Answer-Distribution Divergence):** TV distance between full-cache and
  compressed answer distributions over N seeded samples. Report **excess ADD via a
  per-problem PERMUTATION-TEST null** (pool N+N samples, permute labels, recompute TV
  → exact/MC null), NOT split-half. Coarsen answer support to {correct, top-m wrong,
  OTHER/invalid}. Entropy shift with Miller–Madow. pass@k via unbiased Codex
  (Chen 2021) estimator + paired bootstrap. **Sampled-inference only** — under greedy,
  ADD degenerates to a 0/1 flip indicator, so greedy runs report flip-rate, never
  ADD. BH-correct across the pre-registered problem family. **ADD is a LOWER BOUND on
  behavioral change** (only sees changes that reach the answer) — say so. Decompose
  ADD into **termination-mass shift** (truncation/OTHER mass moving — a length
  effect) vs **within-answered TV** (the reasoning channel), so truncation isn't
  read as reasoning change.
- **TDP + divergence profile:** per-step **total variation, NOT KL** (KL is unbounded
  and explodes on rare tokens; TV is bounded and = the probability optimally-coupled
  sampling differs, ties to speculative-decoding acceptance: `1 − acceptance ≈ TV`,
  Leviathan et al.). Threshold-free primaries — **both mean-TV (length-normalized)
  and summed-TV (total mass)** plus fraction-above-τ — with TDP (first crossing) as a
  localization aid only. τ is a **fixed interpretable level** swept over
  {0.05, 0.10, 0.15, 0.20} (NOT self-noise-calibrated; bitwise self-replay gives
  τ≈0). Onset-aligned (subtract first-compression-event step). Content-token variant
  over numerals/operators/answer-span (defuses "it's just paraphrase noise").
  Segment-aggregated by discourse boundaries.
- **Causal ladder:** evicted-anchor attribution + damage-latency; **restoration test**
  (re-insert evicted KV at a spike, recompute d_t — interventional); **branch
  recovery** with full-cache-fork AND random-fork controls; **reverse replay**
  (compressed free-run trace replayed under full cache — brackets the off-policy bias
  from the other side); **TV-matched noise control** (inject calibrated noise into a
  full-cache free-run matched to a method's TV profile — the on-policy causal test
  distinguishing structured damage from generic noise). Restoration + TV-matched
  noise are the strongest rungs; reverse replay + noise control are **mandatory** (an
  earlier deep review flagged the off-policy/covariate-shift problem as the framework's
  deepest risk).
- **Validity:** negative control (two independent full-cache banks through the whole
  suite → excess ADD ≈ 0, TDP% ≈ 1) — run as a **FIGURE**, not a sentence. If any
  instrument lights up, there's a bug. Exchangeability requires matched
  batching/kernel/transformers-version across conditions.

### 1.4 Known statistical landmines (all must be handled)
- Permutation power at N=8 is low per-problem; the powered unit is the **benchmark**
  via aggregation. Never claim per-problem significance on AIME.
- pass@k crossover hunting across ~150 cells is a multiple-comparisons trap.
  Pre-register the crossover test on R-KV-style methods at high budgets on math; use
  **paired bootstrap of the difference** (pass@1 and pass@k are correlated — naive CI
  overlap is invalid); BH the rest.
- TDP-as-minimum is an extreme-value stat with a length confound → threshold-free
  primaries are the fix; never compare raw TDP% across benchmarks of different lengths.

---

## 2. Roles, workflow, and gating

- **Human:** sets direction, makes strategic calls (e.g. stack pin changes), runs
  Codex, can run shell via `! <cmd>` in the Claude session.
- **Claude (reviewer / decision-maker):** reviews evidence, makes scientific/eng
  rulings, specifies exactly what Codex implements/runs next, authorizes gated work.
  Does NOT launch remote jobs unless explicitly asked. Has implemented CPU code
  directly when Codex was unavailable.
- **Codex (executor):** implements agreed changes, runs `make lint test dryrun`, syncs,
  runs remote/GPU jobs under the launcher, reports back.
- **Gating:** expensive/remote work requires preflight health check + `--i-understand`
  + `KVCACHE_REMOTE=1` + detached tmux. Experiment CLIs are double-guarded (refuse
  without `--i-understand` AND `KVCACHE_REMOTE=1`). The compression runner additionally
  refuses anything but the reviewed 80-row pilot.
- **Working style the human wants:** dense, rigorous, complete; pushback over
  sycophancy; verify version/library/dataset facts rather than asserting from memory
  (several past errors came from stale version assumptions).

### Sync = Policy B (rsync-first)
Edit local → `make lint test dryrun` → `make sync` (dry preview) → `make sync-apply`
→ commit after. Git is committed-history source of truth; rsync excludes
`.git data/ runs/ logs/ results/ *.safetensors *.npy` and the parallel venvs and
`.claude/`. The remote git worktree is dirty by design (rsync excludes `.git`).
`.sync-state` records a content hash of the transferred tree; remote preflight
recomputes it and refuses if files changed after sync.

---

## 3. Infrastructure (CURRENT)

### Local
- `/Users/apple/Desktop/kv-caching`, macOS, zsh. **Note:** `~/Desktop` is a
  TCC-protected folder; a sync that swaps the project directory inode can transiently
  break the sandboxed shell's CWD handle (EPERM "Operation not permitted"). It is
  **not** a real permission problem and self-heals; recovery levers: re-anchor the
  shell via `/tmp`, or run Bash with the sandbox bypass. A durable fix (not yet done)
  is moving the repo off `~/Desktop` and/or hardening `sync_source.sh` to update files
  in place rather than replacing the directory.

### Remote
- SSH alias `kvcache-gpu` → `sk10741@10.224.32.112` (hostname ADUAED16811WKLX).
  Project `/home/sk10741/kv-caching`. Storage under
  `/home/sk10741/kvcache-storage/{data,cache,runs}`.
- Ubuntu 22.04.5, **2× RTX 6000 Ada (~49 GB each, sm_89)**, driver 570.207 / CUDA 12.8,
  ~503 GB RAM. Env via **uv**, Python 3.11, `.venv`.
- **Storage discipline:** ~70 GB personal budget on a shared ~3.5 TB FS that runs near
  full (latest free ≈ 186 GB). Trace banks are **text JSONL only**; **KV caches are
  NEVER serialized to disk**; HF/uv caches live on storage and are pruned post-setup.
  `check_remote.sh` refuses if FS-wide free < ~30 GB.

### THE STACK (post-M2.5 promotion — canonical)
```
python 3.11, torch 2.7.1+cu126, transformers 4.56.2, tokenizers 0.22.2,
kvpress 0.5.3, datasets 2.21.0, accelerate 1.7.0, numpy 2.2.6, scipy 1.15.3, sympy 1.14.0
```
- **Why the change:** stock kvpress 0.2.10 (the original pin) compresses **prefill
  only** — it cannot enforce an absolute KV budget during long decoding, which is the
  entire regime of this project. `DecodingPress` (the decoding-phase, `target_size`
  token-budget API) first shipped in **kvpress 0.5.0**, which requires
  `transformers>=4.56,<5.3`. So decoding compression ⇒ kvpress 0.5.x ⇒ transformers
  ≥4.56. The human approved the controlled upgrade (validate-in-isolation first).
- Legacy env preserved on remote as `.venv-legacy-tf452`; legacy requirements retained.
- HF pinned revisions: R1-Distill-Qwen-7B `916b56a44061fd5cd7d6a8fb632557ed4f724f60`;
  Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c`; HuggingFaceH4/MATH-500
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.

---

## 4. Experimental design (current)

- **Primary model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. **Generalization:**
  `Qwen/Qwen3-4B` — this is a **cross-training-paradigm check within the Qwen family**
  (R1-Distill-Qwen = R1 distilled onto Qwen2.5), **NOT cross-architecture**. Saying
  "two families" / "cross-architecture" is a factual error caught repeatedly. The true
  cross-architecture check is R1-Distill-Llama-8B (restored only with a 2nd GPU).
  Qwen3-4B is **deferred from the pilot**: on transformers 4.56.2 its template renders
  valid ChatML but **no explicit `<think>` prefix** even with `enable_thinking: true`,
  pending model-card/template reconciliation.
- **Sampling:** temp 0.6, top-p 0.95 (R1-Distill standard). R1-Distill needs its chat
  template applied, **NO system prompt**. Verified template (byte-identical on 4.52.4
  and 4.56.2): `<｜begin▁of▁sentence｜><｜User｜>…<｜Assistant｜><think>`.
- **Benchmarks:** MATH-500 (PRIMARY ADD benchmark — 250 problems × n=8), AIME-24/25
  (tiered deep-cell design), GSM8K, FOLIO, StrategyQA, and a synthetic Recall-Span
  Probe (RSP). Currently only **MATH-500** (and RSP scaffold) are implemented;
  AIME/StrategyQA handles are still `TODO(human)` in `configs/benchmarks.yaml`.
- **Budgets:** {128, 256, 512, 1024, 2048} + random-eviction floor. **Always report
  the compression RATIO beside the absolute budget** (256 tokens is 1.6% of a 16K AIME
  trace but 25% of a 1K GSM8K trace).
- **Greedy:** one greedy run for flip-rate (not yet implemented; `generate_full_cache`
  currently hard-codes `do_sample=True`, so a greedy branch is still needed before the
  greedy ablation — not a blocker for current milestones).

### RSP status — STRUCTURALLY INVALID, needs redesign
Current `src/kvcache/rsp.py` builds an order-invariant "sum all visible values"
task: the linked-list traversal visits all items, so the answer is just the sum of
the on-screen numbers — there is **no recall span, no late-binding, no skippable
filler**. 12/12 correct at ~356 tokens confirms it's trivial. It must be redesigned
to the preregistered spec (k≥4 step-1 quantities; d∈{512,1024,2048,4096} tokens of
**required, non-skippable** filler; final question needs exactly one step-1 quantity
at a controlled distance; interleaved-reference fallback) + a 3-criterion validity
pilot, BEFORE any RSP-based claim. RSP is **decoupled** from MATH milestones (don't
let it gate M3/M4 on MATH).

---

## 5. Repository structure (current)

```
src/kvcache/
  determinism.py     set_determinism, log_versions (τ rationale lives here)
  runmeta.py         capture_runmeta (commit, dirty, hashes, host, GPU, versions)
  io_schema.py       FROZEN JSONL schema + validate_row. SCHEMA v2: adds
                     schema_version, hit_token_cap, has_answer_marker, finish_reason.
                     Validator is backward-compatible (rows w/o schema_version = v1,
                     skip v2 checks) and enforces truncated == hit_token_cap and not
                     marker, plus finish_reason ∈ {marker,length,eos} consistency.
  canonicalize.py    boxed/option extraction + sympify canonicalizer (DEMOTED to ADD
                     bucketing / sensitivity only — NOT the correctness oracle).
  grading.py         is_equiv(gold_raw, pred_raw) = vendored Hendrycks MATH
                     normalization. THE correctness authority. GRADER_VERSION.
  datasets.py        load_benchmark(math500 + rsp). MATH gold kept RAW (is_equiv
                     grades raw vs raw); canonical form in meta["gold_canon"].
  models.py          load_model_and_tokenizer (bf16, single GPU, KVCACHE_REMOTE
                     guarded, pinned-revision guarded), render_prompt (no system role).
  generation.py      finalize_generation (is_equiv scoring; truncation/finish_reason),
                     generate_full_cache, generate_with_press, generate_compressed,
                     has_answer_marker.
  presses/__init__.py  press registry: EXECUTABLE={streamingllm,random},
                     PREFILL_ONLY={snapkv,expected_attention}, PORT_REQUIRED={h2o,rkv},
                     DEFERRED={kivi2,triattention}. build_press(...) returns a
                     DecodingPress(base_press, target_size=budget, compression_interval,
                     hidden_states_buffer_size=0) or None for fullcache. Includes a
                     DeviceSafeRandomPress subclass (see §7).
  bank.py            build_result_row (press/budget/compression_ratio params; stamps
                     grader+canonicalizer versions for re-gradeability), append_result
                     (dedup on (bench,pid,seed) within a per-cell file).
  replay.py          forward teacher-forced TV profile, bounded top-k tail (BoundedTV).
  decomposition.py   Decomposition (total, on_reference_readout, residual,
                     reverse_readout_change, flip_mass_net, residual_bracket property).
                     Honest residual semantics enforced in docstrings.
  metrics/
    add.py           coarsen_answer (+ OTHER bucket), add_tv, add_decomposition
                     (termination-mass vs within-answered), flip_mass +
                     aggregate_flip_mass (signed gained/lost/net), excess_add_permutation,
                     entropy_miller_madow, pass_at_k, truncation_rate, greedy_flip_rate.
    tdp.py           mean_tv, total_tv, fraction_above, tdp, content_token_profile,
                     segment_aggregate. Fixed τ {0.05,0.10,0.15,0.20}, TV not KL.
    reverse_replay.py  CONTRACT stub (engine = M4). reverse_replay_score.
    noise_control.py   CONTRACT stub (engine = M7). calibration_error.
    attribution.py / restoration.py / branch_recovery.py  stubs (M7).
  stats.py           bootstrap, exact_permutation_test, benjamini_hochberg,
                     negative_control_ok.
scripts/
  build_bank.py        full-cache bank runner (guarded). --probe-template flag.
  run_grid.py          compressed-generation pilot runner (guarded; refuses != 80 rows).
  diagnose_math_failures.py  CPU MATH failure/canonicalization audit (re-scores rows
                       with is_equiv vs stored, classifies, counts grader disagreements).
  audit_bank.py        schema-validates a run dir, checks dup keys / version / hash
                       consistency, reports per-benchmark accuracy + truncation_rate.
  probe_decoding.py    DecodingPress smoke probe (cache bound + memory + 1 row).
  launch_tmux.sh / launch.sh / check_remote.sh / setup_remote.sh / sync_source.sh / ...
configs/
  models.yaml, benchmarks.yaml, presses.yaml, sampling.yaml, remote.yaml
  experiments/  pilot_fullcache.yaml, pilot_fullcache_v2.yaml (MATH-only, 16384 cap),
                pilot_compression.yaml (the 80-row grid), probe_decoding*.yaml,
                grid_compression.yaml (full grid — GATED), bank_fullcache.yaml,
                replay_decomposition.yaml, add_permutation.yaml, mechanism_ladder.yaml,
                negative_control.yaml, smoke_*.yaml
docs/  METRICS.md, REMOTE_SETUP.md, KVPRESS_API_PROBE.md, STACK_MIGRATION_M2_5.md,
       M4_REPLAY_DESIGN.md (design draft; no GPU authorization)
PREREGISTRATION.md  protocol authority (+ dated "Amendment A" with grading/claims/
                    off-policy/MCC/negative-control disciplines).
remoteserver.md     ops manual authority.
publication_strategy.md, research_landscape.md  UNTRACKED planning docs. publication_
                    strategy.md is partly SUPERSEDED (its §3c/§4d describe the retired
                    KL / self-noise-τ design; PREREGISTRATION wins). Do not commit.
```

---

## 6. Chronological narrative of this session (what happened & why)

The session-start snapshot was stale; the project was already well past "scaffold."
Milestones, in order:

1. **M0 (corrections before any run) — found already DONE & committed.** Truncated-row
   retention (retain/score-false/count/report rate), Qwen3 dropped from pilot, summed-TV
   beside mean-TV, press port/deferred marks, OTHER/invalid ADD bucket + tests.

2. **M1+M2 (full-cache vertical slice) — found already committed; accepted as
   infrastructure.** load_model_and_tokenizer, render_prompt, load_benchmark
   (MATH-500 + RSP), generate_full_cache with truncation, build_bank, schema-valid
   JSONL, audit tool.

3. **First pilots (full-cache).** A 4096-cap pilot (`pilot-fullcache-v1`) ran: 32 rows,
   MATH 13/20 (65%), 4 truncations, RSP 12/12. A smoke exposed a **canonicalization
   bug**: the sympify canonicalizer scored `(3, \dfrac{\pi}{2})` as `"2"` (false
   negative).

4. **Grader fix (the first big ruling).** The sympify canonical-string approach was
   demoted; **`grading.is_equiv` (Hendrycks)** became the correctness authority,
   applied to RAW gold vs RAW pred. A CPU **MATH audit** (`diagnose_math_failures.py`)
   on the 4096 pilot found: stored 65% → regraded **80%**, **3 grader false-negatives**
   (all `90°`≡`90` on precalculus/927), 0 false-positives, **4 token-cap truncations**,
   **0 genuine-wrong, 0 extraction failures**. Conclusion: scoring was the bug, and the
   4096 cap was the only real failure source.

5. **Schema v2 + 16384 re-baseline.** Added `hit_token_cap`/`has_answer_marker`/
   `finish_reason`/`schema_version` (additive, backward-compatible) to separate
   length/extraction/reasoning failures. Built **`pilot-fullcache-v2`** (MATH-only,
   16384 cap, transformers 4.52.4): **20/20 correct, 0 truncations, 0 grader
   disagreements.** The previously-truncated problems finished (e.g.
   intermediate_algebra/1994 up to 12,451 tokens). Cap confound resolved.

6. **kvpress decoding blocker (the second big finding).** A DecodingPress probe found
   **stock kvpress 0.2.10 is prefill-only** — `DecodingPress`/absolute decoding budget
   does not exist in it; `BasePress` skips compression once generation starts. So the
   core experiment (budget-256 *during decoding*) is impossible on 0.2.10. Verified:
   `DecodingPress` first shipped in **kvpress 0.5.0**, which needs **transformers
   ≥4.56**. HOTT itself used kvpress + a hand-rolled SnapKV-D and has no public repo.

7. **M2.5 stack migration (human chose "upgrade, validate first").** Phased:
   - Phase 0: isolated `.venv-next` with transformers **4.56.2** + kvpress **0.5.3**
     (torch/numpy/datasets/etc. held). Minimal, clean upgrade.
   - Phase 1: R1-Distill template **byte-identical** on 4.56.2. Qwen3 still no `<think>`
     → stays deferred.
   - Phase 2: re-baseline `pilot-fullcache-v2-tf456` = **19/20**. The one miss
     (precalculus/927 s11, 90→66) was verified to be a **coherent reasoning trace with
     a genuine sign error** ((2,−12,3) vs (2,−12,−3), which breaks the orthogonality
     that gives 90°), shorter trajectory (5345 vs 11374 tokens). **Ruling: Phase 2
     ACCEPTED.** The literal "20/20" gate was mis-specified — bitwise accuracy
     reproduction across a transformers numerics change is impossible by construction;
     19/20 vs 20/20 at n=20 is one Bernoulli flip (noise), and the real gates
     (template identity, 0 grader disagreements, 0 truncation regression, no systematic
     accuracy change) all passed. The flip is itself an in-vivo example of trajectory
     distortion under a numerical perturbation.
   - Phase 3: DecodingPress smoke (`decodingpress-smoke-v1`) PASS — StreamingLLM at
     `target_size=256`, post-prune max exactly 256, between-prune max 320 (=256+64),
     correct, non-truncated, **no tensor/cache artifacts written**.
   - Phase 4: **promoted** the stack to canonical; legacy preserved as
     `.venv-legacy-tf452`. `pilot-fullcache-v2-tf456` (19/20) is now the canonical
     pilot FC reference; the 4.52.4 banks retire to provenance.

8. **M3 compression code — reviewed & APPROVED.** `build_press` →
   `DecodingPress(base_press=StreamingLLMPress()/RandomPress(seed), target_size=budget,
   compression_interval=64, hidden_states_buffer_size=0)`; `generate_with_press` /
   `generate_compressed` reuse `finalize_generation` (identical is_equiv scoring);
   `run_grid.py` (double-guarded, hard 80-row check); `pilot_compression.yaml`
   (random+streamingllm × {256,2048} × 5 MATH × 4 seeds = 80). Verified both presses are
   `ScorerPress` subclasses with valid `seed`/`n_sink` signatures. `compression_ratio`
   = `min(1, budget/(prompt_tokens+gen_tokens))` (per-row realized ratio; 0.24474 =
   256/1046 in the smoke). Runner iterates `random` first → fails fast.

9. **RandomPress device fix (UNCOMMITTED).** The first authorized grid launch,
   `pilot-compression-v1`, failed before writing a row. kvpress 0.5.3's seeded
   `RandomPress` builds a CPU `torch.Generator()` then feeds it to CUDA `torch.rand` →
   device mismatch. Fixed via a lazy `DeviceSafeRandomPress` subclass that builds the
   generator on `keys.device` (preserves seeded behavior). Added
   `probe_decoding_random.yaml` and generalized the instrumented probe to exercise
   either approved press. Local gate: ruff/black clean, **67 tests**, all **13**
   aggregate dry-runs pass.

10. **RandomPress smoke PASS; corrected M3 grid launched.**
    `decodingpress-random-smoke-v1` completed one MATH generation at budget 256:
    schema-v2 row valid, correct, non-truncated, 1,176 generated tokens,
    `compression_ratio=0.2082994304312449`; 504 layer-level compression events;
    post-prune max exactly 256 and between-prune max 320; no
    `.pt/.pth/.safetensors/.npy/.npz` artifacts and no logged errors. This validates
    the device-safe adapter end-to-end. The corrected 80-row run was then launched as
    **`pilot-compression-v1-r2`** to preserve the failed first attempt for provenance.

11. **M3 compression pilot COMPLETE — strong regime separation, surprising baseline
    ordering.** The corrected run finished all 80 rows across 16 JSONL files. Corrected
    audit (after fixing `audit_bank.py`'s duplicate identity to include
    model/press/budget) reports 51/80 correct, 8/80 truncated, mean 5,007.7 generated
    tokens, consistent transformers 4.56.2 / kvpress 0.5.3 / grader / canonicalizer /
    source-tree / config hashes. No log errors and no
    `.pt/.pth/.safetensors/.npy/.npz` artifacts.
    - random-256: **10/20 (50%)**, 5/20 truncated, mean 7,636.9 tokens.
    - streamingllm-256: **4/20 (20%)**, 3/20 truncated, mean 6,789.9 tokens.
    - random-2048: **19/20 (95%)**, 0 truncated, exact correctness agreement with FC.
    - streamingllm-2048: **18/20 (90%)**, 0 truncated, one FC-correct→wrong flip.
    The canonical FC reference is 19/20. Thus high budget is near-full while budget
    256 collapses, supporting the intended regime separation. However, Random beating
    StreamingLLM at 256 contradicts the expected random-floor ordering; this is a real
    pilot observation requiring reviewer interpretation, not a result to suppress.
    A matched pair makes the low-budget failure concrete: algebra/2584 seed 89 is
    correct under FC (`14/3`, 2,219 tokens), while StreamingLLM-256 repeats fragments
    such as `ioneer`/`oupper` until 16,384 tokens, has no marker, and is retained as a
    truncated failure. This is trajectory collapse, not a grader/extraction error.

12. **M4.0 CPU implementation COMPLETE — awaiting Claude code review.** Added exact
    token-ID sidecars (`token_ids.py`) without changing the frozen result schema;
    generation now carries exact prompt/output IDs and the bank can write validated
    sidecars. Replaced the replay stub with a guarded pilot runner supporting
    self-replay, compressed replay, and accuracy-scale readout. The replay engine uses
    one model/two independent `DynamicCache` objects, tracks absolute positions
    separately from retained cache length, toggles compression explicitly via
    `press.enabled`, asserts caches are mutated-not-replaced, computes exact transient
    full-vocabulary TV, and writes scalar-only profiles. Compression onset accounts
    for kvpress pruning after the current attention output (first affected profile step
    is the following step); no-prune yields `onset_step=null`. Added teacher-force to
    `</think>`/answer-marker boundary then sampled answer continuation, scored against
    dataset gold with `is_equiv`. Added replay/readout configs, replay audit, sidecar
    validation, duplicate guards, cache-bound/pre-onset stop conditions, and
    `PREREGISTRATION.md` Amendment B defining the valid accuracy-scale readout
    decomposition. Local gate: ruff/black clean, **86 tests**, **18 dry-runs**, shell
    syntax clean. No sync, model load, GPU, or remote job was performed.

---

## 7. Exact current state

- **HEAD:** `0f56df4 Restore compression runner executable mode`.
- **Uncommitted (tracked) changes:** M3 fixes plus M4.0 across `Makefile`,
  `PREREGISTRATION.md`, `docs/METRICS.md`, `docs/KVPRESS_API_PROBE.md`,
  `scripts/{audit_bank,build_bank,launch,probe_decoding,run_replay}.py`,
  `src/kvcache/{generation,replay}.py`,
  `src/kvcache/metrics/tdp.py`, `src/kvcache/presses/__init__.py`, and related tests.
- **Untracked:** `configs/experiments/probe_decoding_random.yaml` (new random smoke),
  M4 pilot configs, `scripts/audit_replay.py`, `src/kvcache/token_ids.py`,
  M4 tests, `docs/M4_REPLAY_DESIGN.md`, `publication_strategy.md`,
  `research_landscape.md` (planning docs — never commit), and this `context.md`.
- **Local checks:** ruff clean, black clean, **86 tests pass**, all **18** aggregate
  dry-runs pass.
- **Banks/reports present (local `results/`):**
  - `diag/math-audit-v1` (4.52.4/4096: 65%→80% regrade, 4 truncations, 3 grader FNs)
  - `diag/math-audit-v2` (4.52.4/16384: 20/20)
  - `diag/math-audit-v2-tf456` (4.56.2/16384: 19/20)
  - `runs/pilot-fullcache-v2` (4.52.4 — retired)
  - `runs/pilot-fullcache-v2-tf456` (**CANONICAL FC reference**, 19/20)
  - `runs/decodingpress-smoke-v1` (Phase-3 StreamingLLM smoke, 1 row)
  - `runs/decodingpress-random-smoke-v1` (RandomPress smoke, PASS; pulled locally).
  - Remote-only: failed `pilot-compression-v1` attempt (0 rows; preserved log shows the
    upstream CPU-generator/CUDA mismatch).
  - `runs/pilot-compression-v1-r2` (**COMPLETE and pulled locally**): 80 rows / 16
    files; aggregate 63.75% accuracy, 10% truncation. Cell breakdown is in milestone
    11 above.
- **Audit tool correction (uncommitted):** `audit_bank.py` previously used
  `(bench,pid,seed)` as the duplicate identity, incorrectly rejecting valid rows from
  different press/budget cells. It now uses
  `(bench,pid,model,press,budget,seed)` with a regression test. Full local gate is
  ruff/black clean, **68 tests**.
- **kvpress DecodingPress semantics (verified, see docs/KVPRESS_API_PROBE.md):**
  `target_size` = exact KV tokens retained after each prune; compression every
  `compression_interval` decode steps; effective max cache ≈ `target_size +
  compression_interval`. We use `compression_interval=64`, `hidden_states_buffer_size=0`.

---

## 8. Immediate next steps (in order)

1. **Claude reviews the M4.0 code and tests.** Focus on token sidecar validation,
   enabled-flag press scoping, absolute cache positions, onset indexing, exact-TV
   scalar storage, readout boundary/sampling, and the guarded runner. No GPU action
   before this review.
2. **After approval:** sync and run M4.1 tokenized 20-row FC bank, then M4.2 one-row
   self-replay smoke and full 20-row self-replay. Stop and report unless max exact TV
   <=1e-6, no threshold crosses, and TDP percentile maps to 1.0. Only after that gate
   proceed to M4.3 replay and M4.4 readout.
3. Then M5 (ADD end-to-end) → **M6 (port H2O, then R-KV — the "science starts" gate)**
   → M7 (mechanism interventions: restoration, branch recovery, reverse replay,
   TV-matched noise) → M8 (full pre-registered grid; NOT before M0–M6 green +
   PREREGISTRATION committed).

**Still GATED / do not start without Claude's review:** the full grid
(`grid_compression.yaml`), the 250-problem MATH bank, M4+ engines, H2O/R-KV ports,
KIVI (separate quantized-cache wrapper, not a press), RSP-anything (needs redesign).

---

## 9. Open items / tech debt (non-blocking)

- **Grader comparability (before the reproduction figure, NOT before M3/M4):** verify
  which grader HOTT / kvpress-eval / MATH-500 use (`math_verify` vs Hendrycks). We use
  Hendrycks `is_equiv` for internal consistency; add `math_verify` as a parallel
  sensitivity for the "reproduce HOTT's collapse" Figure 1.
- **Greedy generation path** (`do_sample=False`, paired flip-rate) — needed for the
  greedy-vs-sampled ablation, not for M3/M4.
- **`generate_compressed` discards `prompt_len`** that `generate_with_press` computes,
  so `run_grid` re-tokenizes the prompt to recover it — harmless but redundant; thread
  it through.
- **RandomPress uses the same seed for all 5 problems** within a (budget, seed) cell —
  reproducible but mildly correlated; consider per-problem seeding before the full grid.
- **RSP redesign** to the preregistered recall-span spec + 3-criterion validity pilot.
- **`sync_source.sh` in-place hardening** so the directory-inode swap (which transiently
  broke local FS access) can't recur; or move the repo off `~/Desktop`.
- **AIME-24/25 and StrategyQA dataset handles** are still `TODO(human)` in
  `configs/benchmarks.yaml` (verify gating/licenses).

---

## 10. Landmines / gotchas for whoever continues

- **Never** call the residual a measured cause; **never** call divergence "damage"
  without signed flip-mass. (§1.2 — these two sentences are the whole load-bearing wall.)
- **Truncated rows are valid data**: retained, scored `correct=False`, counted in the
  denominator, with per-cell `truncation_rate`. Only structurally-malformed rows are
  rejected. Compression increases truncation; rejecting truncated rows would inflate
  compressed accuracy and deflate ADD.
- **Correctness uses `is_equiv` on RAW strings.** Do not route gold through the sympify
  canonicalizer (it caused a silent false-negative). `answer_canon` is bucketing only.
- **KV caches and full-vocab tensors must NEVER hit disk.** Banks are text JSONL.
- **Exchangeability:** FC and compressed rows must share the same stack
  (transformers/kvpress version, batching, kernels). That's why the whole bank was
  re-run on 4.56.2; do not mix stacks across conditions.
- **DecodingPress effective cache = target_size + compression_interval** between prunes
  — report the budget as nominal `target_size`, note the slack.
- **RandomPress on CUDA needs the device-safe generator** (kvpress 0.5.3 bug).
- **Local FS "Operation not permitted" is a transient sandbox/inode artifact**, not a
  permission you must grant — recover via `/tmp` re-anchor or sandbox bypass.
- **Qwen3 is deferred** (no `<think>` on 4.56.2); calling the generalization model
  "cross-architecture" is a factual error.

---

## 11. Quick-reference commands

```bash
# local gate
make lint test dryrun           # currently ruff+black clean, 68 tests, 13 dry-runs

# sync (Policy B)
make sync                       # dry preview
make sync-apply

# remote audit of a run dir (CPU, no GPU)
ssh kvcache-gpu 'cd /home/sk10741/kv-caching && \
  .venv/bin/python scripts/audit_bank.py --run-root <RUNS>/<session> --expected-rows N'

# MATH failure / canonicalization audit (CPU)
ssh kvcache-gpu 'cd /home/sk10741/kv-caching && KVCACHE_REMOTE=1 \
  REMOTE_DATA_ROOT=/home/sk10741/kvcache-storage/data \
  .venv/bin/python scripts/diagnose_math_failures.py \
    --rundir <RUNS>/<session> --gold-from-dataset --out <RUNS>/diagnostics/<name>'

# gated GPU job (tmux): stage ∈ {bank, grid, diagnostic}
ssh kvcache-gpu 'cd /home/sk10741/kv-caching && \
  bash scripts/launch_tmux.sh --i-understand <session> <stage> configs/experiments/<cfg>.yaml'

# pull small results back
rsync -avh kvcache-gpu:/home/sk10741/kvcache-storage/runs/<session>/ results/runs/<session>/

# completed M3 marker
ssh kvcache-gpu 'grep "Finished: pilot-compression-v1-r2" \
  /home/sk10741/kv-caching/logs/pilot-compression-v1-r2.log'
```

Reporting after remote actions follows `remoteserver.md` §31 (Action / Machine /
Commit / Sync / Files / Command / GPU / Log / Output / Status / Disk / Errors / Next).

---

*End of handoff. Re-read the live files before acting; this document records intent and
decisions, the repo records truth.*

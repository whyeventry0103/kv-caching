# KV Cache Research — Results

Readout distortion vs trajectory distortion in KV-cache compression for reasoning models. Every number here is generated from the artifact catalogue, not typed.

**29/29 artifacts re-derived from raw rows** · **79 numbers pinned to their quantity** · **Verified 2026-08-05**

---

> **How to read every number here**
>
> - **The comparator is the matched cell reference, not the bank.** The full-cache bank is 0.882 over 2000 rows and appears in the setup only. Every change is measured against that bank restricted to the cell's own conditions — 0.885 for the budget-256 MATH-500 cells (Amendment K).
> - **Truncated generations stay in every denominator, scored incorrect.** They are never dropped as malformed.
> - **The residual is a definition, not a measured cause.** It is 'accuracy change unexplained by on-reference readout' = total change minus readout change.
> - **Divergence is not damage.** Correctness is the only quality arbiter; every divergence quantity travels with its signed flip-mass.

## 1. Headline

At cache budget 256 on MATH-500 (DeepSeek-R1-Distill-Qwen-7B), the same compression measured two ways gives two different answers. The gap between the dots is the trajectory residual.

| Trajectory residual | Readout says | Free-running says | Matched reference |
|---|---|---|---|
| **≈ −0.34** | **0.765** | **0.422** | **0.885** |

Realized cache at budget 256 is **3.8%** of the mean trace (prompt + generation).

## 2. Budget 256 — the core decomposition

100 problems × 4 seeds = 400 conditions per cell.

| Policy | Free-run (95% CI) | Readout (95% CI) | Total Δ | Readout Δ | Residual (95% CI) | Trunc. |
|---|---|---|---|---|---|---|
| Random | 0.422 [0.345, 0.500] | 0.765 [0.686, 0.817] | -0.463 | -0.138 | **-0.337** [-0.404, -0.262] | 0.233 |
| StreamingLLM | 0.237 [0.175, 0.305] | 0.584 [0.495, 0.647] | -0.647 | -0.319 | **-0.344** [-0.405, -0.270] | 0.070 |
| H2O-style | 0.247 [0.180, 0.318] | 0.597 [0.502, 0.667] | -0.637 | -0.306 | **-0.344** [-0.403, -0.270] | 0.105 |
| R-KV | 0.520 [0.435, 0.610] | 0.798 [0.710, 0.850] | -0.365 | -0.105 | **-0.268** [-0.335, -0.195] | 0.135 |

- Readout share of the total loss: Random 29.8%, StreamingLLM 49.2%, H2O-style 48.0%, R-KV 28.7%.
- Residual spread across the three unrelated policies: **0.008**, against CI half-widths of roughly 0.07. The methods differ in the *readout* channel instead — readout-change spread **0.181**.

> **Pre-registered outcome: COMPLICATE**
>
> - The strong headline required Random to be readout-lossless (readout CI lower bound at least 0.75; actual 0.686) and StreamingLLM's readout share at least 0.5 (actual 49.2%). Neither held. The central residual survived robustly; the pilot's clean double dissociation did not.

## 3. Reverse-replay bracket (the off-policy defense)

Forward and reverse readout estimate the same residual from opposite directions. The bracket excludes zero for every policy, and all eight endpoints have problem-clustered CIs excluding zero.

| Policy | Reverse readout acc. | Reverse endpoint | Residual bracket |
|---|---|---|---|
| Random | 0.636 | -0.282 | [-0.337, -0.282] |
| StreamingLLM | 0.417 | -0.470 | [-0.470, -0.344] |
| H2O-style | 0.419 | -0.478 | [-0.478, -0.344] |
| R-KV | 0.662 | -0.251 | [-0.268, -0.251] |

> **Caveat that must travel with this**
>
> - Forward and reverse estimands disagree on the *method-level* split, so the method-level readout-vs-trajectory attribution is reported as estimand-dependent and exploratory. The *central* claim — a large residual for every policy — survives both directions.

## 4. Random beats the principled recency policy

- Mean paired difference **+0.185**, 95% CI **[0.130, 0.240]** (problem-clustered, pinned seed, 2000 resamples).
- Per-problem difference non-negative on **94/100** problems: **44 strict wins, 50 ties, 6 losses**.

> **Phrasing rule (binding)**
>
> - Say **'non-negative on 94/100'**, never 'beats on 94/100' — over half of the 94 are ties.
> - HOTT carries no random-eviction baseline, so this result is ours alone. HOTT must never be cited as corroborating it.
> - The ordering is architecture-dependent: it **reverses on Llama-8B** (section 12).

## 5. Dose response — budget 2048

The collapse is a budget effect. Same presses, same harness, budget 2048 (**56.6%** realized cache).

| Policy | Free-run (95% CI) | Readout (95% CI) | Total Δ | Readout Δ | Residual (95% CI) | Trunc. |
|---|---|---|---|---|---|---|
| Random | 0.850 [0.782, 0.912] | 0.901 [0.820, 0.943] | -0.035 | -0.003 | **-0.033** [-0.065, -0.010] | 0.030 |
| StreamingLLM | 0.807 [0.735, 0.870] | 0.903 [0.823, 0.943] | -0.078 | 0.000 | **-0.079** [-0.125, -0.042] | 0.028 |
| H2O-style | 0.823 [0.752, 0.887] | 0.898 [0.820, 0.938] | -0.062 | -0.005 | **-0.059** [-0.102, -0.020] | 0.085 |

- The residual shrinks by a factor of **4.4x to 10.2x** between budget 256 and budget 2048.
- At 2048 readout sits at or just above the matched reference, and the Random over StreamingLLM ordering persists.

## 6. Per-step TV divergence (mechanism: see the caveat)

Onset-aligned total variation between full-cache and compressed teacher-forced replay. This localizes where divergence *starts*, not where accuracy is lost.

**Caveat (Amendment M.3, 2026-08-15).** The RISE in these windows is a compression-ratio effect, not a horizon mechanism: every profile here holds the budget fixed while context grows, so the realized ratio worsens monotonically. Held at a fixed ratio the drift reverses (0.66-0.74 < 1). **No mechanistic reading of the compounding is licensed.** What DOES survive ratio-matching is the phase alignment to prune events (Amendment P.2-v2, 2026-08-27: Random phase-amplitude excess +0.252 [0.184, 0.271], peak phase 0). State the two together — within a prune cycle divergence is eviction-locked; across cycles the envelope tracks the ratio.

| Window | Random + StreamingLLM (800 profiles) | H2O-style (400) |
|---|---|---|
| Before first eviction | **0.0000** | **0.0000** |
| First 50 steps after onset | 0.0018 | 0.0008 |
| Later in the trajectory | 0.0884 | 0.0765 |
| Compounding factor | 48.6x | 98.7x |

> **The pre-eviction 0.0000 is an instrument check, not a finding**
>
> - Median onset step is 65. If the pre-eviction value is ever not about 0.0000, **stop** — that is a bug in the replay engine, not a result.

## 7. ADD — silent answer-distribution rewriting (budget 256)

Total variation between coarsened full-cache and compressed answer distributions, with a permutation null (10,000 resamples) and Benjamini-Hochberg correction across problems.

| Policy | ADD | Within-answered | Termination shift | Flip gained | Flip lost | Net flip-mass | Significant |
|---|---|---|---|---|---|---|---|
| Random | 0.559 | 0.504 | 0.224 | 0.004 | 0.465 | -0.461 | 42/100 |
| StreamingLLM | 0.751 | 0.749 | 0.081 | 0.001 | 0.647 | -0.646 | 72/100 |
| H2O-style | 0.715 | 0.713 | 0.101 | 0.000 | 0.636 | -0.636 | 62/100 |
| R-KV | 0.459 | 0.450 | 0.149 | 0.004 | 0.367 | -0.364 | 37/100 |

- The distortion runs overwhelmingly toward **lost correct answers** — net flip-mass is strongly negative for every policy.
- Policies distort through **different channels**: StreamingLLM almost entirely within-answered (it keeps answering, wrongly); Random carries the larger termination-mass shift (it rambles and fails to terminate), consistent with its **3.3x** higher truncation rate.
- Every cell sits **6.7x to 11.0x** above the negative-control noise floor (section 8).

## 8. Instrument validity controls

These exist so the instruments are defensible before any finding is read off them.

| Control | Result | What it rules out |
|---|---|---|
| Two-full-cache negative control | mean ADD **0.068**, median 0.000, **0.0%** significant | ADD false-positives from finite samples |
| Full-cache readout of full-cache traces | resampling effect **+0.003** | The readout protocol's own resampling explaining the residual |
| Residual vs resampling effect | Random **132x**, R-KV **105x** | The residual being a readout artifact |
| Grader audit, 3600 rows | **0** disagreements on regrade | Grader error driving the accuracy numbers |
| Exact replication, 800 rows | all rows identical (correctness, truncation, token counts) | Nondeterminism in generation |
| Pre-eviction TV | exactly 0.0000 | Replay-engine misalignment |

## 9. AIME-24 — second benchmark

30 problems × 4 seeds, budget 2048 (**19.5%** realized cache), own pinned reference bank **0.371** (mean generated length 10415 tokens under a 16384-token cap).

| Policy | Free-run (95% CI) | Readout (95% CI) | Total Δ | Readout Δ | Residual (95% CI) | Trunc. |
|---|---|---|---|---|---|---|
| Random | 0.283 [0.150, 0.425] | 0.462 [0.327, 0.661] | -0.075 | 0.000 | **-0.118** [-0.328, -0.051] | 0.308 |
| R-KV | 0.217 [0.100, 0.358] | 0.462 [0.327, 0.661] | -0.142 | 0.000 | **-0.194** [-0.402, -0.125] | 0.450 |
| StreamingLLM | 0.142 [0.042, 0.258] | 0.462 [0.327, 0.661] | -0.217 | 0.000 | **-0.280** [-0.491, -0.181] | 0.233 |

> **The sharpest readout result in the study**
>
> - All three presses read out at an *identical* **0.462**, and on the matched conditions the readout change is **exactly 0.000** for every press — zero gained, zero lost, per row. Free-running separates the same three policies by about 2x. Held-trace evaluation here does not merely misprice compression; it does not respond to it at all.

## 10. R-KV in its published configuration

R-KV publishes **105%** of full-KV accuracy at **16%** cache on R1-Distill-Llama-8B / AIME-24. We ran our parity-certified port in that exact configuration (Amendment N.2).

|  | Full cache | R-KV @2048 |
|---|---|---|
| Accuracy | **0.300** (36/120) | **0.308** (37/120) |
| Truncation @32768 | 0.075 | 0.208 |
| Accuracy, non-truncated | 0.324 | 0.389 |
| Mean tokens | 14799 | 18870 |

- Ratio **r = 1.028**; paired difference **+0.008**, problem-clustered 95% CI **[-0.075, 0.092]**. Realized cache **18.9%**.
- **Pre-registered branch: REPLICATES** (r at least 1.0 and the difference CI includes zero). The branch was selected by code, not judgment, so it could not be re-cut after the numbers were seen.

> **How to state this, exactly**
>
> - The margin is **one generation** (37 correct against 36 of 120) on a CI that spans zero, over 30 problems, against a bank that cleared its sanity band at exactly the floor.
> - **Defensible:** our certified port is not worse than full cache at about a fifth of the cache, in the published configuration.
> - **Not defensible:** 'we reproduced 105%'. **Also not defensible in the other direction:** calling it inconclusive because we would prefer it.
> - Two facts favour R-KV and are reported for that reason: it truncates nearly 3x as often (longer traces, every truncation scored incorrect) and leads anyway; among non-truncated generations it leads 0.389 to 0.324.

## 11. HOTT-configuration cell

Our own full-cache condition re-run under HOTT's 2048-token generation cap, changing nothing else (Amendment N.1).

|  | Our accuracy | Truncation | HOTT reports |
|---|---|---|---|
| Full cache | **0.515** (from 0.882) | 0.430 | 0.47 |
| StreamingLLM-128 | **0.100** | 0.475 | 0.03 |

The generation cap alone accounts for nearly all of the baseline gap, leaving **0.045**.

> **Mandatory caveat, and a standing landmine**
>
> - This matches HOTT's cap and **not** their greedy decoding, which our generation path does not support (sampling stays T=0.6, top-p 0.95). It measures how much of the gap the cap alone accounts for; any residual gap is not attributable to the cap.
> - Never write 'H2O collapses'. HOTT's H2O (with recency reservation) survives at 0.14; SnapKV at 0.38. Ours is 'H2O-style / ObservedAttentionPress (no recency reservation)'.

## 12. Cross-architecture check — Llama-8B

50 problems × 4 seeds, own pinned reference bank **0.805**. CIs correspondingly wider.

| Policy | Free-run (95% CI) | Readout (95% CI) | Total Δ | Readout Δ | Residual (95% CI) | Trunc. |
|---|---|---|---|---|---|---|
| Random | 0.000 [0.000, 0.000] | 0.021 [0.000, 0.045] | -0.805 | -0.805 | **-0.021** [-0.045, 0.000] | 0.805 |
| StreamingLLM | 0.235 [0.150, 0.320] | 0.508 [0.390, 0.595] | -0.570 | -0.318 | **-0.267** [-0.345, -0.175] | 0.010 |

> **This cell fits no pre-registered branch and is reported plainly**
>
> - The central result replicates on the sharpest case (StreamingLLM), but **Random collapses to 0.000** — the Random-beats-recency ordering *reverses* on this architecture. That is why the ordering is called architecture-dependent everywhere.

## 13. Scale axis

Qwen-1.5B (bank 0.800) and Qwen-14B (bank 0.880), 50 problems × 4 seeds each (Amendment J, complete).

| Model | Policy | Free-run (95% CI) | Readout (95% CI) | Total Δ | Readout Δ | Residual (95% CI) | Trunc. |
|---|---|---|---|---|---|---|---|
| Qwen-1.5B | R-KV | 0.355 [0.240, 0.465] | 0.688 [0.566, 0.806] | -0.445 | -0.146 | **-0.323** [-0.435, -0.225] | 0.210 |
| Qwen-1.5B | StreamingLLM | 0.220 [0.130, 0.315] | 0.495 [0.378, 0.617] | -0.580 | -0.339 | **-0.271** [-0.360, -0.191] | 0.150 |
| Qwen-14B | R-KV | 0.670 [0.535, 0.795] | 0.848 [0.750, 0.922] | -0.210 | -0.046 | **-0.168** [-0.273, -0.085] | 0.230 |
| Qwen-14B | StreamingLLM | 0.240 [0.150, 0.335] | 0.503 [0.395, 0.600] | -0.640 | -0.391 | **-0.259** [-0.335, -0.170] | 0.000 |

3 of the 4 cells replicate the central result. Read these as **exploratory at current scope** — subset scale, two presses per model.

## 14. Resolution collapse

How much of the policy separation visible under free-running survives held-trace readout. Spread = mean absolute pairwise difference across presses, paired on the problem.

| Regime | Free-run spread | Readout spread | Ratio (descriptive) |
|---|---|---|---|
| MATH-500 @256 | 0.1704 | 0.1328 | 0.779 |
| MATH-500 @512 | 0.2025 | 0.0608 | 0.300 |
| MATH-500 @1024 | 0.1125 | 0.0050 | 0.044 |
| MATH-500 @2048 | 0.0283 | 0.0033 | 0.118 |
| AIME-24 @2048 | 0.1012 | 0.0000 | 0.000 |

> **Status: post-hoc and exploratory**
>
> - The pattern was noticed after all cells were read. Its pre-registered confirmatory test at budgets 512 and 1024 was run and **refuted** the graded/monotone reading, which is why this is reported as endpoints rather than as a mechanism.
> - Press sets differ between regimes, so a matched-pair recomputation (Random + StreamingLLM only) is carried alongside: @256 free 0.1850 / readout 0.1792; @2048 free 0.0425 / readout 0.0025.

## 15. Compute

- **375 GPU-hours** of summed wall-clock across **10,007 generations**.
- Covers free-running and reference-bank generation only. Teacher-forced work (readout, reverse readout, replay) is excluded, as are superseded pilots and the two shards subsumed by the merged bank.
- The artifact lists every run directory with its hours and an inclusion flag, so any exclusion can be re-added and its effect seen.
- Corrected 2026-08-05 from a stale 280 h / 7,200 generations: the inclusion allowlist had drifted nine result-bearing run directories behind. Found by the artifact re-derivation harness, not by reading.

## 16. Framing discipline

These exist because each one has been got wrong at least once.

| Say this | Never this |
|---|---|
| 'accuracy change unexplained by on-reference readout' | 'measured trajectory damage' / any causal claim |
| 'non-negative on 94/100 (44 win / 50 tie / 6 loss)' | 'beats on 94/100' |
| 'H2O-style / ObservedAttentionPress (no recency reservation)' | 'H2O collapses' |
| 'divergence', with its signed flip-mass | 'damage to reasoning ability' |
| every absolute budget with its realized compression ratio | a bare budget number |
| 'our certified port at 7B' for R-KV numbers | 'R-KV scores X' — that is our port, not the authors' setup |
| 'exploratory at current scope' for generality claims | an unqualified generality claim |

- Truncated generations stay in every denominator, scored incorrect.
- A primary bound is a deterministic imputation range: **never** described as significance, never compared for width against a CI.
- Benjamini-Hochberg correction is mandatory for families of problem-level tests.
- Underwhelming pre-registered outcomes get reported plainly. The honesty trail is the moat.

## 17. Provenance

| Item | Value |
|---|---|
| Models | DeepSeek-R1-Distill-Qwen-7B (plus Llama-8B, Qwen-1.5B, Qwen-14B) |
| Benchmarks | MATH-500, AIME-24 |
| Sampling | T = 0.6, top-p 0.95 |
| Stack | transformers 4.56.2, kvpress 0.5.3, DecodingPress, compression_interval 64 |
| Bootstrap | problem-clustered, 2000 resamples, percentile CI |
| Permutation null | 10,000 resamples, Benjamini-Hochberg across problems |
| Grader | Hendrycks is_equiv, no eval/sympify on model text |
| Chain of custody | paper_numbers.json from results/diag/ from raw JSONL rows |

> **Regenerating this page**
>
> - `python scripts/export_results.py --format both` rewrites both the Markdown and the HTML from the catalogue. **Edit the script, not the output** — hand edits are overwritten on the next run, and a hand-typed number is what this pipeline exists to prevent.


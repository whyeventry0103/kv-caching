# Closest-work matrix and novelty gate N (execution plan §3.3, WP3 item 6)

Compiled 2026-10-05.

**Read depth for each source:**
- **Random Attention:** full HTML text and the official code (read directly with `gh api`).
- **Pitfalls:** full PDF text, keyword-searched directly. The summarizing fetch's account was wrong on two points (it invented teacher-forcing and "Gemma 4" experiments); the direct read is what is recorded here.
- **evidence2026:** raw HTML, with the tables grepped directly.
- **HOTT:** summarizing fetch of the full HTML, plus the earlier verified audit in PAPER_FINDINGS §1 (commit 12634e7). Entries marked † need a direct table read before they are quoted in the paper.

"Not found" means not found in the sections read. It does not mean never done. A second novelty pass is due at the submission freeze.

## 1. Matrix

| | Random Attention (Wang et al., arXiv:2609.03430v2, 2026-09-28) | evidence2026 (Ai, He, Guo, arXiv:2608.01631v1, 2026-08-03) | HOTT (Liu et al., arXiv:2512.12008v1, 2025-12-12) | Pitfalls (Chen, Geh, Grover, Van den Broeck, Israel; ACL 2026 long 1926; arXiv:2510.00231) | **This work** |
|---|---|---|---|---|---|
| Models | Qwen3-4B/14B/32B, Phi-4-reasoning | Qwen3-8B (primary), R1-Distill-Llama-8B, Qwen3-30B-A3B | Llama-3.1-8B-Instruct, R1-Distill-Llama-8B, R1-Distill-Qwen-7B/14B, Nemotron-Nano-8B | Llama3-8B, Qwen2.5-14B | R1-Distill-Qwen-7B (primary); 1.5B, 14B and Llama-8B checks; Qwen3-4B (gate FAIL, held) |
| Tasks | MATH500, GPQA-D, AIME25/26, HMMT, LiveCodeBench-v6 | AIME24/25/26, GPQA-D, MedCalc, RULER QA, IFEval | GSM8K, MATH-500, ReClor, DROP, StrategyQA, CSQA, FOLIO, OBQA | IFEval multi-instruction, leakage probes | MATH-500 (100 × 4 seeds); AIME-24 |
| Budget definition | Code: `--token_budget K --residual_length 64`. After each prune K entries in total: K−64 persistent slots (prompt pinned inside them, `pl = min(prompt_len, K−1)`) plus the 64 most recent, never scored. Text says "K plus r ≪ K buffer". MATH500 K=1024 | "256-token retained-cache budget"; the question cache is protected and only the trace is compressed in fixed-trace replay | "KV-cache size in tokens" during decoding; prompt handling † | Compression ratio of the query; query-only compression | Total resident entries after each prune, prompt included (S.1); relaxed to P+64 when P > B−64; no recency buffer |
| Prompt protection | Yes, `s_i = +∞` for prefill positions; Table 2 compares methods before/after protecting it | Yes in fixed-trace replay (question cache protected); end-to-end handling not stated † | Not as a protocol variable (StreamingLLM keeps 4 sinks; H2O has a recent window) † | Proposed fix: a keyword whitelist and fair eviction across instructions | Yes: matched protected vs unprotected arms at the same total budget (S.1) |
| Random selection | Fresh `torch.rand` at every eviction, per KV head and layer | — | No random baseline | — | Legacy kvpress RandomPress reseeds every prune, so it is a fixed slot pattern (a soft-recency sieve); a fresh-uniform family is prepared (WP4) |
| Free-running accuracy | Yes, the only protocol | Yes, a "deployment-relevance check" (Tables 2, 10, 11) | Yes, the only protocol (greedy) | Greedy free generation on the compressed query | Yes |
| Held-trace / teacher-forced readout | Not found (a planted-fact probe uses forcing for the synthetic fact only) | **Yes: fixed-trace replay is the primary diagnostic** (full-KV writes `<think>r</think>`, then the answer is regenerated from the compressed cache) | Not found | Not found (no teacher forcing, no perplexity; verified in the text) | Yes (forward readout) |
| Reverse replay (compressed trace, full-cache readout) | Not found | Not found | Not found | Not found | Yes |
| Matched-budget free-run minus readout | No | **No: free run at 2048, fixed trace at 256 (Table 10 caption)** | No | No | Yes: the residual, with problem-cluster CIs |
| Missing-boundary handling | Not found | Not found | Not found | — | Imputation bounds; common-support 2×2 |
| Reasoning models | Yes | Yes | Yes | No | Yes |
| What it demonstrates | Once the prompt is protected, selector differences mostly vanish; random with prompt protection matches the best selector at ~4× compression | Answers can be right while the rationale is wrong under compression (the answer–evidence gap) | Severe, method-specific degradation under free-running reasoning | Compression silently drops some instructions and leaks system prompts; eviction is biased by position and order | See §2 |

## 2. Novelty gate N (one page)

**The question.** Once prompt retention and cache accounting are matched, how do evaluation on a supplied reasoning trace and evaluation on self-generated reasoning differ? What does an explicit forward/reverse audit say about that difference?

**Why the closest papers do not already answer it:**

- **Random Attention** shows that prompt protection removes most of the *selector* gap in free-running accuracy. It never holds a trace fixed. So it cannot say how much of a policy's loss a held-trace evaluation sees, before or after protection.
  - What we add is not "the prompt matters", which is theirs and Pitfalls'. It is what protection does to the *gap between evaluation protocols*.
  - At B=256 protection improves both directions. For Random the forward improvement is larger (+0.239 against +0.161, difference +0.078 [+0.007, +0.149]); for StreamingLLM a difference was not detected (+0.271 against +0.278). Ledger A5.
  - After protection the two replay directions disagree significantly (interaction +0.130 / +0.127, A6).
- **evidence2026** has the forward instrument, but:
  - its free-running and fixed-trace numbers are at different budgets (2048 vs 256);
  - it never forms a matched-budget difference;
  - it has no reverse replay;
  - its estimand is answer–evidence faithfulness.
  Its data show the phenomenon exists even when the free run gets 8× the cache. That corroborates our question; it does not answer it.
- **HOTT** is free-running only. It ranks methods, and has no protocol comparison and no random baseline.
- **Pitfalls** studies instruction loss under query compression in non-reasoning models. Its "protect instructions / fair eviction" fixes and our prompt protection are two instances of the same lesson. It has no reasoning traces and no protocol audit.

**What is *not* novel and must not be claimed:**
- that random eviction is competitive (Random Attention, VaSE);
- that the prompt is fragile (Random Attention, Pitfalls);
- that free-running reasoning degrades under eviction (HOTT);
- that fixed-trace replay exists as an instrument (evidence2026);
- that conditioning on reference rather than self-generated trajectories biases evaluation (exposure bias / DAgger literature).

**One falsifying experiment.** Run the protected 2×2 at B=1024 (forward arm queued; the reverse arm is proposal P1, held) plus protected H2O-style and fresh-uniform random at B=256 (WP5, held).
- *Falsified:* our answer is falsified if, under matched protection, the forward residual and the reverse trajectory contrast both fall inside ±3 points with TOST support at B=1024 and for the fair selectors. Held-trace evaluation would then be practically adequate once the prompt is protected, and the contribution would shrink to a replication note.
- *Survives:* if a practically meaningful gap persists in either direction, or the directions keep disagreeing, the audit question survives.

**Verdict, 2026-10-05.**
- N is **provisionally passed** for a *measurement* contribution:
  - the matched-budget forward/reverse comparison under matched protection;
  - the replay-direction disagreement it exposes.
- N is **not** passed for:
  - a "prompt-loss explains the residual" story (Random Attention owns the prompt half);
  - a "random beats recency" story.
- It must be rechecked once:
  - B=1024 is read;
  - before the WP6 confirmation spend.

## 3. Evidence record for the evidence2026 budget claim (added after the review of fbf869f)

The independent reviewer could not retrieve arXiv:2608.01631. Evidence preserved here:
- **Version.** v1, submitted 2026-08-03. arXiv lists no other version as of 2026-10-05.
- **Source.** `https://arxiv.org/html/2608.01631v1`, fetched 2026-10-05. SHA-256 of the HTML: `800fc9b5e85d718aeb7beb9494d90e910a605358ac823e2ce57cbdb51bc8ddc1`. Identical to the unversioned URL fetched the same day.
- **Verbatim caption:** "Table 10: Qwen3-8B AIME26 end-to-end KV-compression results at budget 2048, compared with our fixed-trace baseline at KV budget 256."
- **Rows (verbatim numbers):**
  - AdaKV: Fixed-trace (256) 40.0, End-to-end (2048) 23.3.
  - SnapKV: Fixed-trace (256) 40.0, End-to-end (2048) 26.7.
  - HeadKV: 46.7 / 26.7.
  - StreamingLLM: 20.0 / 6.7.
  - Full-KV: 56.7.
- **Section 4 text:** "we additionally run end-to-end checks on AIME26 (Table 2)". Table 2 carries the same end-to-end numbers without a budget in its caption.

A finite matrix does not guarantee novelty. The second pass at the submission freeze still stands.

## 4. Meeting 5 update (2026-10-14): information-theoretic and protection precedents

**How these were read.** Every source below was read from its primary text on
2026-10-14:
- arXiv HTML for CapKV, K-VEC, InfoKV, Random Attention, evidence2026, Garcia
  and Prefix Sliding;
- the PMLR page and the GitHub README for CapKV;
- the PDFs for Xu et al., Hewitt et al. and McAllester & Stratos.

The quotes were checked against local text extracts. "Not found" means not
found in the text read.

### 4.1 Matrix

| Source | Quantity or design | Models / tasks | Relation to Meeting 5 |
|---|---|---|---|
| **CapKV**: Yang, Tang, Zhen, Lv, "Rethinking KV Cache Eviction via a Unified Information-Theoretic Objective", ICML 2026 (PMLR 306), arXiv:2604.25975v1 | `I(q; Y \| Z_C)`: information about the future *query* q in the attention output, under a linear-Gaussian surrogate ("adopted solely for analytical tractability"). Closed form `½ log det(I + Σ⁻¹ U K Λ Kᵀ Uᵀ)`. Leverage-score eviction. K-, U- and KU-capacity proxies "used solely as diagnostic indicators", validated by Spearman ρ ≈ 0.73–0.87 against accuracy | Qwen3-8B/14B, LongBench; Nemotron-7B on AIME25 (decoding) | An MI objective for KV eviction already exists, so "first information-theoretic view" is unavailable. Its MI is query-to-output capacity under a Gaussian surrogate, not task information about a randomized fact through the model's answer channel. Its proxies enter M5-D as labelled *surrogate capacity* comparators. No random baseline, prompt protection, planted fact or nuisance variable found |
| **K-VEC**: Roy, Zhai, Hajimirsadeghi, Samei, "Coverage-Driven KV Cache Eviction…", arXiv:2606.29563v1 | Token coverage. The bound `I(T;X) ≤ α log C` is *assumed* ("token contributions are additive… equally informative"); MI is not measured. Evidence is a coverage–F1 correlation | Llama-3.1-8B, Qwen2.5-7B; LongBench, NIAH (prefill) | Coverage is a competing explanation and enters as a predictor (M5-D), not as measured information |
| **InfoKV**: Kai, Xiao, Birch, Lin, "Information-Aware KV Cache Compression for Long Reasoning", arXiv:2606.26875v1 | "Forward Influence": deletion KL on a fixed continuation. Score = predictive entropy × layer representation distance, mixed with attention. No mutual information appears in the text | R1-Distill-Qwen-7B and Llama-8B; AIME24, IFEval, LiveCodeBench | Entropy-weighted importance in long reasoning is taken. It has no randomized target, finite channel or protection factorial |
| **Random Attention**: Wang et al., arXiv:2609.03430v2 | Matched prompt protection for four selectors plus random (§5.1; "separates the score from the protection"). Planted-fact probe (§5.2): a 4-digit value in a fixed 16-token box inside real MATH500 traces, teacher-forced through the decode path; the query is always kept; the fact span is pinned per (layer, head); graded recall by log-probabilities. Retrieval probe (§5.3) | Qwen3-4B/14B/32B, Phi-4-reasoning; probes on Qwen3-4B only | **Precedent for both our protected comparison and a decode-path planted fact with span pinning.** Not found: MI, nuisance variable, enumeration over a finite answer set, equal-budget nuisance-span control, forward/reverse readout |
| **evidence2026**: Ai, He, Guo, arXiv:2608.01631v1 | Fixed-trace replay with question protection; answer–evidence faithfulness; perturbation fidelity | Qwen3-8B etc.; AIME, GPQA-D | Precedent for the forward instrument (see §3). No MI, no protection factorial, no random baseline |
| **Garcia**, "Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction", arXiv:2605.18053v1 (independent researcher) | Seven policies (LRU, H2O, SnapKV, StreamingLLM-style, Ada-KV, QUEST, **Random**), each with and without boundary protection at the same global cap C: ⌈ρC⌉ prefix + suffix slots, ρ = 0.10. Without protection, quality collapses (F1 ≤ 0.064). Random+prot reaches F1 0.250 against 0.282–0.290. TOST with Δ = 0.02 and Holm | Ten models, 1.5B–27B; LongBench QA (N = 162), NIAH; not reasoning traces | **Precedent for the protection × policy cross with Random at a matched cap.** Its protection is positional (boundary slots), not the prompt as such and not a content span. Not found: MI, nuisance, span interventions, fixed-trace readout |
| **Prefix Sliding**: Muennighoff et al., arXiv:2608.26070v1 | Keeps the prefix (system + prompt) plus a recency window, as a kernel mask | Qwen3-1.7B (7B in the RL appendix); GPQA, MATH500, AIME25 | Precedent for prompt + recency protection in reasoning. No factorial, no random baseline, no information analysis |
| **Xu et al.**, "A Theory of Usable Information Under Computational Constraints", ICLR 2020, arXiv:2002.10689 | `I_V(X→Y) = H_V(Y\|∅) − H_V(Y\|X)`, with `H_V` an infimum over a predictive family; equals Shannon MI when V is all models; can violate data processing | — | Our 2 − cross-entropy *reader score* is a fixed-reader quantity in this spirit. It is not an infimum over a family, and it is labelled a declared-reader score |
| **Hewitt et al.**, "Conditional probing: measuring usable information beyond a baseline", EMNLP 2021 | `I_V(φ(X)→Y \| B) = H_V(Y\|B) − H_V(Y\|B, φ(X))` | — | The template for any optional probe beyond a baseline (protocol H4). Decodability by a probe is not use by the model |
| **McAllester & Stratos**, "Formal Limitations on the Measurement of Mutual Information", AISTATS 2020 (PMLR 108) | A distribution-free high-confidence lower bound on MI from N samples "cannot be larger than O(ln N)" (Thm 1.1: ≤ 2 ln N + 5) | — | Why MINE, kNN and critic-based estimates over cache activations are not the headline. Our channel is computed exactly from the model's output distribution for every one of the 16 inputs, is bounded by log₂ 4 = 2 bits, and its only sampling uncertainty is across contexts. The limitation targets a different estimation problem, and this is stated rather than assumed |

### 4.2 What this does to the Meeting 5 claim

**Not novel, and never to be claimed:**
- crossing prompt or boundary protection with eviction policies, Random
  included, at a matched budget (Garcia, Random Attention);
- that protection removes most selector differences (both);
- an information-theoretic objective or capacity for KV eviction (CapKV);
- coverage as an information motivation (K-VEC);
- entropy-informed importance for reasoning (InfoKV);
- planting a fact in a reasoning trace, feeding it through the decode path and
  pinning its span (Random Attention §5.2);
- fixed-trace replay (evidence2026).

Our M5-A protected comparison is therefore a **replication and extension**:
reasoning traces, the four historical policies, R-KV included, B = 256 with
capacity matching. It is not a new design.

**Candidate increment.** Each part is "not found" in every source above, which
is not proof of absence:
1. Exact information through a finite answer channel (4 symbols + OTHER) about
   a randomized target S, with an independent nuisance N enumerated and
   marginalized. It is reported with log loss, OTHER mass and accuracy, and
   the task-switch query reads the same cache.
2. An equal-budget contrast between protecting the target span and protecting
   an equal nuisance span, with a placebo span.
3. The forward/reverse readout (§2 above).
4. The link between them: does an earlier channel measurement predict later
   free-running accuracy beyond M/Δ/ω, occupancy and exposure (protocol H2)?

Without (4), items 1–2 are a characterization instrument, not an explanation.

**What the channel number is.** At a fixed context u and eviction seed R, the
readout draws Z from the cache alone, so S → C_π → Z and
I(S; Z | u, R) ≤ I(S; C_π | u, R). The output-channel MI is a **lower bound**
on the cache's information about S, never an estimate of it. The reader score
2 − CE is a Barber–Agakov-type lower bound for that one fixed reader.
Differences of lower bounds do not bound differences of cache MI (review §10.2).

**Alternative explanations to rule out before any mechanistic sentence:**
- prompt loss: already established by others;
- capacity differences: hence B_eff matching and the strict-fit cohort;
- recency and position: hence counterbalanced ages and the placebo span;
- coverage (K-VEC) and surrogate capacity (CapKV): entered as competing
  predictors;
- trace length and exposure;
- numerical artifacts: the full/full identity and the pre-eviction TV = 0
  controls.

**Verdict, 2026-10-14.**
- The gate passes for a **measurement contribution**: the forward/reverse
  audit, plus the target/nuisance finite channel as an *instrument*.
- It does **not** pass for "protection explains it" (precedented), "Random
  wins" (precedented), or any mechanism claim before G3.
- Recheck it after the M5-C calibration and at the submission freeze.

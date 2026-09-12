# arXiv scoop watch — weekly sweep log

**Cadence:** weekly, Mondays (decision 2026-07-03).
**Queries (publication_strategy §6e + sharpened):** "KV cache reasoning";
"compression evaluation reasoning"; "teacher forcing compression";
"KV cache compression readout trajectory decomposition"; "KV cache free-running evaluation".
**Foreclosure test:** does the paper separate trajectory from readout with an explicit
instrument (teacher-forced replay / accuracy-scale decomposition)? If NO → cite within
48h + add a differentiating sentence, continue. If YES → pivot per §6e (weight onto
ADD + failure-correlation; accelerate the arXiv stake).

---

## 2026-07-03 — first sweep (manual, Claude)

- **2603.01426 — "Understanding the Physics of KV Cache Compression through Attention
  Dynamics" (Mar 2026).** Closest adjacent work: an *analysis* paper about why compression
  degrades. Checked against the foreclosure test: synthetic long-INPUT probes
  (entity-tracking/coreference), attention-routing "lottery ticket" framing; NO
  accuracy-scale readout/trajectory decomposition, NO teacher-forced-vs-free-run
  instrument, not long-generation reasoning. **NOT foreclosing.** ACTION: cited in
  related work (third critical strand). DONE.
- **2606.03928 — "Value-Aware Stochastic KV Cache Eviction for Reasoning Models" (VaSE,
  Jun 2026).** Method paper: stochastic eviction + diversity promotion beats deterministic
  selection-based baselines on reasoning; also finds evicting large-magnitude value states
  causes repetitive-loop failures (consistent with our SLM gibberish-loop observations).
  **NOT foreclosing; CORROBORATES the Random>StreamingLLM phenomenon.** ACTION: cited in
  §4.1 as concurrent work. DONE.
- 2512.00504 G-KV (decoding-time eviction w/ global attention): method paper, no
  decomposition. No action.
- 2604.18002 Neural Garbage Collection; 2510.02312 KaVa: training-based compression /
  latent distillation; adjacent families only. No action.
- 2502.01941 "Semantic Integrity Matters" (benchmark + preservation): consider citing at
  camera-ready in the critical-evaluation strand. Low priority.

**VERDICT: no foreclosure as of 2026-07-03. The replay-decomposition instrument remains
unclaimed by anyone else. The arXiv stake is still the moat — keep drafting pace.**

---

## 2026-09-12 — sweep (Claude Code), and the verdict CHANGES

Three papers were uncited and are now in Related Work. One of them changes the
standing verdict, so the headline is stated first.

### VERDICT: PARTIAL FORECLOSURE. The instrument is no longer unclaimed.

**2608.01631 — "Does Accuracy Equal Evidence? Reasoning Faithfulness under KV
Cache Compression" (Ai, He, Guo, Aug 2026).** Applied the foreclosure test
honestly and it does **not** come out clean.

- Does it separate trajectory from readout with an explicit instrument? **It has
  the instrument.** It runs a "controlled fixed-trace replay protocol": the
  uncompressed model writes a trace once, each compression method replays that
  identical trace under its own compressed KV states and generates only the
  answer. That is structurally our forward readout.
- Does it have the numbers? **Yes, and they contain our residual.** Its Table 2
  (AIME26, Qwen3-8B, budget 256) reports end-to-end against fixed-trace accuracy
  for the same methods: AdaKV 0.233 vs 0.400, SnapKV 0.267 vs 0.400. Those
  differences, −0.167 and −0.133, are the quantity this project is about.
- Does it compute or name that difference? **No.** Its estimand is the
  *answer–evidence gap* — correctness against rationale validity — and the two
  accuracies live in different tables (2 and 3) serving a different claim. It
  never forms the difference, never gives it an interval, and never uses it to
  reconcile the two literatures.

So the earlier standing verdict, "the replay-decomposition instrument remains
unclaimed by anyone else," is **no longer true and is retracted.** What remains
unclaimed is narrower and should be stated that way from now on: the residual as
a *defined estimand* with problem-clustered intervals, the reverse-replay
bracket that corrects its off-policy bias, and the use of the decomposition to
explain a contradiction between two literatures.

**Action taken:** cited, with a differentiating paragraph in §2.2 that states the
overlap *before* the difference and quotes their numbers as containing our
residual. Also read as corroboration and said so — an independent group, on a
different model and benchmark, measured a large free-running-minus-fixed-trace
gap for eviction methods. Per the standing rule that an underwhelming or
uncomfortable finding is reported plainly, this is written as a narrowing of our
novelty claim, not as a footnote.

**Not pivoting per §6e.** The pivot trigger is a paper that performs the
decomposition, and this one does not. But the arXiv stake is now more urgent
than it was, and the novelty sentence in the intro should be re-read against this
before submission.

### Non-scoops, both cited

- **2604.10900 — CASK: Core-Aware Selective KV Compression for Reasoning Traces
  (Kim et al., Apr 2026).** Method paper: reframes reasoning-trace compression as
  behaviour-preserving structured consolidation rather than eviction, splitting
  the decode-time trace into a protected core and a mergeable scratch. No
  readout/trajectory separation. **NOT foreclosing.** Cited in §2.2 as evidence
  the field now treats the reasoning trace as internally structured.
- **2605.17613 — VeriCache: Turning Lossy KV Cache into Lossless LLM Inference
  (May 2026).** Drafts with the compressed cache and verifies against the full
  cache, recovering full-KV-identical output at most of the throughput. No
  decomposition; it *removes* the loss rather than measuring it. **NOT
  foreclosing, and corroborating**: its whole premise is that compressed-cache
  output is close step-by-step and diverges further the longer decoding runs,
  which is our per-step profile stated as an engineering constraint. Cited in
  §2.3, where it also justifies our choice of TV over KL/JSD.

**Standing note for the next sweep:** the foreclosure test as written is
binary and this sweep did not fit it. Recommend splitting it into "has the
instrument" and "computes the decomposition", because a paper can now do the
first without the second, and recording only "not foreclosing" would have lost
exactly the information that mattered.

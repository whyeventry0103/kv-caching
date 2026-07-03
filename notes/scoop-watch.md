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

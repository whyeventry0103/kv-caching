# COAUTHOR_REVIEW.md — living co-author review (Claude, joined 2026-07-03)

Standing review artifact for the submission push. Updated as work progresses; treat the
**Roadmap** (bottom) as the canonical prioritized TODO. Everything here is grounded in
repository evidence — file paths are cited throughout.

**Status line (update on every pass):** 2026-07-05 — RUN-ONLY STATE: 22-job queue now
carries every strengthening run (H2O full bracket, R-KV unlocked after the smoke gate
caught+we fixed a real press bug — Amendment I, Jaccard 0.805→1.0000; Llama-8B
cross-architecture trio; AIME-24 bank+cells with the >100% test). AIME loader +
Llama pin + marker-gated jobs + bench-aware progress all implemented and tested
(131 tests). Breadth criticism status: methods 2→4 queued, budgets 2, benchmarks 2
queued, architectures 2 queued — resolved BY THE QUEUE, not yet by data.
Prior: 2026-07-04 — remote back. ROADMAP #1 DONE:
bracket-endpoint bootstrap CIs implemented (report_m4 `unexplained_residual_ci95` /
`reverse_residual_ci95`), run on the powered data: all four endpoints exclude zero
(fwd [−0.402,−0.258]/[−0.403,−0.268]; rev [−0.383,−0.213]/[−0.553,−0.410] — reverse
endpoints separate with NON-overlapping CIs); paper §4.3/threats/abstract updated.
Roadmap #2 half-done: H2O-style grid 400/400 (free 0.2475 — see PAPER_FINDINGS §4h
LANDMINE: ObservedAttentionPress ≠ H2O, never claim HOTT contradiction; readout half
tonight). Fig 1 histogram panel landed (exposed a claim imprecision, fixed: 94/100 =
44 win + 50 tie + 6 loss, now stated as "non-negative on 94/100" everywhere). Table 3
estimator defined + reproduced exactly (per-trace 0.0884; pooled 0.1182). Scoop-sweep
cron installed on remote (Mon 09:20, scripts/scoop_sweep.py). "Predictive upgrade"
framing tempered per co-session pushback (see BEST_POSSIBLE_PAPER Phase 10/11).
Prior pass (07-03): blueprint; bracket bug fix + Amendment H; imputation bounds;
figures embedded; sweep clean.

---

## Phase 0 — The research journey (reconstructed from docs/project/context.md, docs/protocol/PREREGISTRATION.md, docs/)

1. **Origin (Jun 10–13):** thesis-selection memo (`research_landscape.md`) picks
   KV-compression-for-reasoning over domain generalization, targeting the live
   contradiction HOTT (2512.12008) created. `publication_strategy.md` scores 17 candidate
   gaps; picks G15 (trajectory/readout decomposition) + G16 (ADD/TDP) as the compound.
   Pre-registers hypotheses before any GPU run.
2. **Infrastructure epoch (Jun 13–14):** grader crisis — sympify canonicalizer produced a
   false negative; demoted to bucketing, Hendrycks `is_equiv` becomes the correctness
   authority (Amendment A). Schema v2 separates truncation/extraction/reasoning failures.
   Discovery that kvpress 0.2.10 is prefill-only forces the M2.5 stack migration
   (transformers 4.56.2 + kvpress 0.5.3 DecodingPress) — without it the entire project is
   impossible. Double-BOS bug caught by the token-ID sidecar round-trip (Amendment C);
   all prior pilots retired to provenance. Self-replay control: bitwise zero TV over
   48k steps — the replay instrument is validated before ever measuring compression.
3. **Pilot epoch (Jun 14–15):** M4 pilot (n=5) shows an apparent *double dissociation*
   (SLM=readout-dominated, Random=trajectory-dominated). External review forces
   Amendment D: the dissociation is explicitly UNCONFIRMED; powered run may overturn it
   (it later does). H2O identified as kvpress ObservedAttention; R-KV port hard-gated on
   retained-set parity (later passes at Jaccard 1.0).
4. **Powering epoch (Jun 15–23):** 250×8 FC bank (accuracy 0.882). Powered grid: Random
   0.422 vs SLM 0.237 (+0.185, 94/100). Pre-committed read (Amendment F) written BEFORE
   the readout run. Outcome: **COMPLICATE** — the pilot's clean dissociation dies; what
   survives is a large (~−0.34), method-invariant (forward) trajectory residual. The
   paper's actual thesis is born here, from an honest negative-ish result.
5. **Defense epoch (Jun 27–Jul 2):** reverse-replay bracket (residual survives
   off-policy correction), two-FC negative control (ADD floor 0.068), Fig-4 TV profiles
   (zero at eviction, compounding after). Amendment G commits the spine. RSP is cut
   (failed validity twice). >100% framing dropped (no gains observed).
6. **Correction epoch (Jul 3):** residual-bracket computation bug found in
   `report_m4.py`/`decomposition.py` (reverse endpoint was the on-policy readout change,
   not a residual estimate). Fixed; Amendment H; corrected brackets are FARTHER from zero
   (Random [−0.343,−0.246], SLM [−0.466,−0.347]) but change the method-level story
   (invariance is forward-estimand-specific). Figures generated; queue reordered to put
   H2O-256 (the method-invariance boundary test) next.

**Failed/killed ideas (evidence of discipline, usable in rebuttal):** RSP synthetic probe
(structurally invalid twice → cut); sympify grading (false negatives → demoted);
pilot double dissociation (didn't survive power → reframed); >100%/denoising framing
(no gains observed → dropped); KL-based divergence + self-noise τ (unbounded/circular →
TV + fixed τ sweep).

## Phase 1 — Independent scientific model (before re-reading the manuscript)

**Problem:** two literatures assign contradictory costs to KV-cache eviction on reasoning.
**Mechanism claim worth testing:** compression during generation is an intervention on
the *sampling process*; its cost decomposes into what it does to answer readout on a
fixed trace vs what it does to the trace the model writes.
**What would convince a skeptic:** (i) an accuracy-scale decomposition against one pinned
reference; (ii) an off-policy defense (the readout intervention conditions on good
states); (iii) instrument validity controls (self-replay zero, two-FC null); (iv) a
mechanism signature (divergence timing).
**What would falsify:** readout accuracy ≈ free-run accuracy (residual ≈ 0), or reverse
readout recovering to FC (loss was readout mis-assigned), or the negative control
lighting up. None occurred; all were genuinely at risk (prereg proves it).
**Actual contribution:** the *instrument* (forward/reverse readout pair + TV profile +
ADD with controls) and the first quantified, bracketed trajectory residual for KV
compression. The repo supports exactly this paper — no more, no less.

## Phase 2 — Repository audit (personally verified this session)

| Component | File | Verdict |
|---|---|---|
| Replay engine (TV, positions, press scoping) | `src/kvcache/replay.py` | ✅ correct; abs `cache_position` after pruning; exact live TV; cache-identity asserts |
| Nucleus sampler parity with HF warpers | `replay.py::sample_top_p` | ✅ same keep-rule (cum−p > top_p), temp-before-top_p |
| Boundary rule (last `</think>`/marker/undefined) | `replay.py` | ✅ matches paper Def. 2 |
| Decomposition + bracket | `src/kvcache/decomposition.py` | ✅ after Jul-3 fix (was WRONG; Amendment H) |
| Report (pinned ref, CI, flip mass) | `scripts/report_m4.py` | ✅ enforces one FC bank; problem-clustered bootstrap |
| ADD (coarsening, channels, permutation) | `src/kvcache/metrics/add.py` | ✅; within-answered TV is renormalized (paper wording fixed accordingly) |
| TDP/AUC | `src/kvcache/metrics/tdp.py` | ✅ |
| Stats (perm test, BH, bootstrap) | `src/kvcache/stats.py` | ✅ MC add-one correction; correct BH step-up |
| Grader | `src/kvcache/grading.py` | ✅ vendored Hendrycks, versioned, raw-vs-raw |
| Fullcache readout = true no-op press | `presses/__init__.py::build_replay_press` | ✅ `NoOpReplayPress` |
| Figure generator | `scripts/make_paper_figures.py` | ✅ constants match corrected evidence; ADD channels not stacked |
| Two-FC control | `scripts/report_two_fc_control.py` | ✅ |
| Tests | 120 passing | ✅ incl. regression tests for the bracket fix |

**Gap found & closed (Jul 3):** Table 3's windowed TV means had no committed script →
added `--print-windows` to `make_paper_figures.py` (runs on the remote profiles dir).

## Phase 3 — Claim traceability (paper → evidence)

| Claim | Evidence | Confidence | Residual reviewer risk |
|---|---|---|---|
| FC 0.882 (250×8) | `math500-fc-powered-merged` (remote), audit_bank | High | provenance para (orphaned commits) — disclosed |
| Phenomenon +0.185 CI [0.130,0.242], 94/100 | grid-diss256(-tok), report | High | none — HOTT-no-random caveat now stated |
| Forward decomposition row values | report_m4 (800 readout conds), Amendment F outcome | High | off-policy bias — bracketed |
| Brackets [−0.343,−0.246]/[−0.466,−0.347] | report_m4 (post-fix, DATA-mode Fig 2) | High | no endpoint CIs → **top-ranked next analysis**; imputation bounds now in §4.3 |
| TV windows 0/0.0018/0.0884; AUC 0.086/0.083 | 800 profiles; now reproducible via `--print-windows` | High | curve-window vs full-horizon mean — caption handles |
| ADD table + channels | report_add on grid vs FC bank | High | n=4 asymmetry — disclosed in Def. 5 |
| Two-FC null (0.068/0.976/0%) | `report_two_fc_control.py` output, Amendment G | High | — |
| R-KV parity Jaccard 1.0 | `docs/design/RKV_PORT.md`, parity artifacts, gated registry | High | smoke still pending — paper says so |
| Eager/sdpa exchangeability 19/20 vs 20/20 | pilot check (docs/project/context.md §, Amendment E) | Medium | n=20 only; fine for a threat-section control |

No claim in the manuscript lacks repository evidence. One claim direction reversed by
evidence during this review (bracket) — fixed at the source and disclosed via Amendment H.

## Phase 4 — Novelty audit

- **Genuinely novel:** the accuracy-scale forward/reverse readout decomposition applied
  to KV compression; the residual bracket; onset-aligned TV profile as the mechanism
  instrument; ADD with termination/within-answered channels + permutation null +
  signed flip-mass; the two-FC control-as-figure discipline.
- **Incremental/adjacent:** exposure-bias/teacher-forcing-vs-free-running lens is
  classical (Ranzato, Bengio, DAgger) — the paper correctly imports rather than claims it.
  Per-step distribution divergence exists in distillation/spec-decoding contexts; the
  application + onset alignment is new here.
- **Checked for foreclosure (sweep 2026-07-03, `notes/scoop-watch.md`):** physics/attention-
  dynamics (2603.01426) — synthetic long-input probes, no decomposition; VaSE (2606.03928)
  — corroborates the Random>SLM direction, no decomposition; G-KV, KaVa, NGC — method
  papers. **No one has the instrument.**
- **Positioning:** correct — measurement paper, generous to both literatures. Novelty is
  neither oversold (residual framed as definition) nor undersold.
- **Rating: 8/10 instrument novelty, 7/10 findings novelty.** The scope (1 model,
  1 benchmark, 1 budget final) is what caps the impact today; the queued envelope raises it.

## Phase 5 — The paper I would independently write from this repo

Nearly the one that exists. Title would emphasize the *evaluation-protocol* implication
(current title does). Two deltas I would make (both proposed below rather than imposed):
(1) lead the results with Fig 4's staircase earlier — it is the most memorable artifact
and the mechanism is the paper's beating heart; (2) elevate "the entire method gap is in
the readout channel" to a named finding — it is the most surprising quantitative fact
(a 0.185 accuracy gap fully explained by one channel) and currently lives inside a
paragraph.

## Phase 6 — Adversarial review (attacks attempted; outcome)

1. *"Replay isn't deployment."* Survives: instrument-not-deployment framing is explicit;
   decomposition stated as definition by construction.
2. *"Residual is off-policy-inflated."* Survives: DAgger caveat at the definition; reverse
   bracket; NOW also imputation bounds (added §4.3) covering the undefined-boundary
   selection effect — under any imputation of undefined rows the bracket excludes zero.
3. *"Undefined-boundary rows are informative missingness"* (rambling traces lack
   boundaries → Acc_rev selection-biased upward). **Was the strongest open attack**;
   neutralized by the imputation-bounds paragraph (worst-case: all undefined scored
   incorrect; best-case: all correct; zero excluded in every case).
4. *"No uncertainty on the bracket endpoints."* Partially survives as a limitation;
   fix ranked #1 in the experiment roadmap (problem-clustered bootstrap of endpoints —
   CPU-only, data exists).
5. *"Two methods, one budget, one model."* True; disclosed as the primary threat; the
   in-flight envelope (H2O-256 running now, 2048 grid complete) is the answer. Reviewers
   at ICLR will want at least H2O in the camera-ready — hence the queue priority.
6. *"Grader mismatch vs HOTT."* Survives: channel-structure-not-digits reconciliation is
   explicit; no digit merging anywhere.
7. *"ADD at n=4 is noisy."* Survives: benchmark-level aggregation, permutation null,
   two-FC floor an order of magnitude below signal, asymmetry note in Def. 5.
8. *"Random>SLM is a seed artifact / budget accounting."* Survives: per-(problem,seed)
   RNG; verified budget parity (256/320 both); 94/100 problems.
9. *"Fig 4 curve max (~0.07) contradicts 0.0884 mean."* Neutralized by caption note
   (±400-step window vs full horizon).
10. *"Prereg deviation (100×4 vs 250×8)."* Neutralized: disclosed in §3.3 with timing.

## Phase 7 — Simulated reviews (honest)

**R-A (efficient inference, ICLR):** S: protocol audit table; budget parity; same-stack
comparability with HOTT; actionable conclusion. W: only 2 methods final, wants H2O/SnapKV
cells; wants realized compression ratios beside absolute budgets. Score 6 (marginal
accept) now; 7-8 with H2O+2048 in.
**R-B (reasoning benchmarks):** S: 250×8 reference, truncation discipline, honest
COMPLICATE. W: one benchmark; AIME absent; wants per-difficulty breakdown (free re-bin —
see roadmap). Score 5-6 now; 6-7 with 2048+H2O, +1 if AIME lands.
**R-C (methodology):** S: prereg with dated amendments including a self-caught
computation bug (rare, credibility-building if presented as Amendment H); bracket;
controls-as-figures. W: no CI on bracket endpoints; method-split non-identifiability
paragraph is dense. Score 7-8.
**AC:** "Rigorous measurement paper; scope is the question. If the generality envelope
(4 methods × 2 budgets) lands, accept; if 2 methods × 1 budget, workshop/TMLR."

**Acceptance estimates:** current draft as-is: ICLR main ~20%, ACL/EMNLP ~25% (needs
reframe), TMLR ~55-65%, workshop ~60-70%. **After** H2O-256 + 2048 read + endpoint CIs +
venue template + author-filled bib: ICLR ~35-40%. With R-KV/AIME upside: ~40-45%. These
match `publication_strategy.md` §9's base-rate arithmetic; the ladder (arXiv → workshop →
ICLR → TMLR floor) remains the right play, and the mid-August arXiv stake is the
priority-protecting move.

## Phase 8 — Writing audit (top items; others applied directly this session)

1. Intro ¶1 is ~230 words, one paragraph carrying HOTT numbers + audit + thesis seed.
   *Proposal:* split after "…this paper's contribution." (mechanical; do at venue-format
   conversion, when the page budget forces edits anyway).
2. §4.3's qualifications sentence chain is 3 long sentences of hedging; correct but
   dense. *Proposal:* table-ize the two endpoints per method at camera-ready.
3. Thesis paragraph ≈ conclusion opening (deliberate bookend) — keep, but vary the
   second instance's wording slightly at final polish.
4. Fixed this session: abstract 450→~280 words; ADD additivity wording; hardcoded figure
   numbers; overfull equation; a-fortiori TV framing; self-contained captions.

## Phase 9 — Story audit

Primary: damage is real, large, readout-invisible; the gap is the trajectory residual;
divergence compounds. Secondary: Random>recency (readout-channel-explained); ADD failure
styles; instrument suite. Distraction risk: R-KV/>100% discussion — currently correctly
quarantined to audit-table + threats. **Story verdict: right paper, right framing.** The
one elevation worth making: name finding (ii) — "the entire method gap is in the readout
channel" — as its own bolded takeaway box/line in §4.2 at venue formatting.

## Phase 10 — Experiment ranking (impact ÷ cost; ↑ = do first)

1. **Bracket-endpoint CIs** (problem-clustered bootstrap over joint (free, readout,
   reverse) per problem). CPU-only, data exists on remote. Removes the paper's one
   stated statistical gap; may license the method-split claim. ~2h. **Critical.**
2. **H2O-256 grid+readout** (RUNNING). The method-invariance boundary test. **Critical.**
3. **readout-2048** + read of completed grid-2048 → dose-response/"silent flip" pole.
   High. (Queued.)
4. **Realized compression ratios** (mean prompt+gen tokens per cell → 256/total). Free
   from report fields (`mean_free_run_tokens`); prereg H2 requires it. **High, trivial.**
5. **Difficulty stratification** (MATH levels 1–5 re-bin of existing rows; G12 in
   strategy doc). Free; anticipates R-B. Medium.
6. **Fig 1 histogram panel** (rendered on remote, awaiting link). Low effort, cosmetic.
7. R-KV smoke → cells; AIME — behind the 2026-08-01 drop-dead; upside only.
8. TV-matched noise control (prereg-mandatory for the *causal* upgrade) — NOT required
   for this paper's claims as framed (residual stays a definition + bracket); keep as
   future work unless a reviewer demands it. Nice-to-have.

## Phase 11 — Figure audit

- Fig 1: clear in <10s. Pending: replace right text-box with per-problem paired-diff
  histogram (data on remote). Keep the box's three lines as annotation inside the hist.
- Fig 2: good; consider a light shaded band spanning each bracket on the x-axis at
  camera-ready. 30-second legible.
- Fig 3: correct non-stacked channels; legend outside plot is fine.
- Fig 4: the paper's best figure. At camera-ready: annotate one eviction step with an
  arrow ("eviction event (every 64 steps)").
- Fig 5: fine; DATA-mode histogram version preferred when link returns.

## Phase 12 — Repo release readiness (for the artifact link)

Done: pinned versions; one-command figures; tests; prereg with amendments; scoop log.
Before public: (a) fill bib author lists; (b) anonymize repo for submission (strategy
§1b note); (c) top-level README section "Reproducing the paper" mapping each
figure/table → command (mostly exists piecemeal; consolidate); (d) prune `.DS_Store`,
add to .gitignore; (e) decide whether `docs/project/context.md`/strategy docs ship in the public
artifact (recommend NO — internal; keep prereg, configs, code, figures).

## Phase 13 — Changes applied this session (see git log for diffs)

- Bracket bug: `decomposition.py`, `report_m4.py`, 2 test files, Amendment H,
  PAPER_FINDINGS §4c, docs/project/context.md, paper numbers (+ exact −0.466 reconciliation).
- Paper: abstract rewrite; figure embedding + self-contained captions; ADD-additivity
  wording; per-cell counts; prereg-deviation disclosure; a-fortiori TV framing;
  related-work breadth (KIVI, Pitfalls, Barriers, physics, VaSE); bibliography repairs;
  layout fixes (gather; footnotesize).
- NEW this pass: §4.3 imputation-bounds robustness paragraph; free-run CIs in Table 2;
  `--print-windows` provenance mode in `make_paper_figures.py`.
- Ops: scoop sweep run (clean) + log; queue reorder verified; decisions recorded
  (drop-dead 2026-08-01; weekly Monday sweep).

## Phase 14 — Final reflection: what I would still attack

1. The bracket has no endpoint CIs (roadmap #1 — do before arXiv).
2. Only teacher-forced evidence localizes the mechanism; a reviewer could ask for one
   *worked example* trace (FC vs compressed continuation from the same prefix) as a
   qualitative figure — cheap, humanizing, appendix. Nice-to-have.
3. The 0.882-vs-0.47 FC gap with HOTT is disclosed but a hostile reviewer may still ask
   why; one appendix sentence quantifying the decode-cap/sampling difference would
   close it fully. Nice-to-have.
4. Venue template conversion will force ~5 pages of cuts; plan which threats/definitions
   compress into the appendix BEFORE cutting under deadline pressure.

---

# ROADMAP (canonical prioritized TODO)

| # | Item | Class | Effort | Expected gain |
|---|---|---|---|---|
| 1 | Bracket-endpoint bootstrap CIs (CPU, remote rows) | **Critical** | ~2h | closes the stated statistical gap; R-C +1 |
| 2 | H2O-256 cells finish + read (running) | **Critical** | nights | generality envelope; R-A +1..2 |
| 3 | Fill bib author lists (6 `\todo`s) | **Critical** | 30m | submission-blocking |
| 4 | Venue template conversion + page-budget cuts | **High** | 1-2d | required for any venue |
| 5 | Realized compression ratios in Setup (prereg H2) | **High** | 30m | prereg consistency; R-A |
| 6 | readout-2048 read (dose-response pole) | **High** | nights | second budget pole |
| 7 | grid-2048 accuracy read (already 800/800 — report it) | **High** | 1h | free result |
| 8 | Difficulty-strata re-bin (free) | Medium | 2h | R-B anticipation |
| 9 | Fig 1 histogram swap (remote) | Medium | 10m | figure quality |
| 10 | Endpoint-CI → possibly upgrade method-split claim | Medium | after #1 | conditional |
| 11 | Worked-example trace figure (appendix) | Nice | 2h | persuasion |
| 12 | HOTT-FC-gap appendix sentence | Nice | 30m | closes a question |
| 13 | R-KV smoke → cells; AIME (drop-dead 2026-08-01) | Upside | nights | +5-10% acceptance if lands |
| 14 | Public-repo release checklist (Phase 12 a–e) | High (at submission) | 2h | artifact credibility |

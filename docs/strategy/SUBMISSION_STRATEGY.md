# SUBMISSION_STRATEGY.md — reviewer-cognition & submission-package pass (SAC, 2026-07-03)

No science in this document. Only: how reviewers will experience the paper, and the
smallest changes that make them experience it correctly. Companion docs:
COAUTHOR_REVIEW.md (audit + roadmap), BEST_POSSIBLE_PAPER.md (strategy blueprint).
Applied now: the optimized abstract (self-contained change). Everything else here is
paste-ready for the single venue-conversion rewrite, to avoid churning the draft twice.

---

## Phase 1 — Cognitive walkthrough (honest, against the current build)

- **0:30** — Title parsed: "readout vs free generation, KV compression, reasoning."
  Preliminary frame: *evaluation-gap paper, not a method paper.* Correct frame. Good.
- **1:00** — Abstract (pre-optimization) delivered the contradiction, then machinery,
  then numbers. The killer fact (same method: 0.765 vs 0.422) arrived mid-paragraph.
  Risk: a skimming reviewer files this as "another HOTT-style eval paper." → Fixed:
  abstract now puts the 0.765-vs-0.422 pair in sentence three (applied today).
- **3:00** — Contributions list: six bullets, each with numbers. Slight overload; the
  first-minute reviewer reads three. (Venue pass: merge bullets 5+6 into "mechanism +
  validity," keeping four bullets total.)
- **7:00** — Assumption forming: "the residual is probably an off-policy artifact."
  §3's Definition 3 states the bias *at the definition*, and §4.3 has bracket +
  imputation bounds. The defense exists; the reviewer must reach p.9 to see it. (Venue
  pass: one forward-pointer sentence in the intro: "the residual survives an off-policy
  bracket and every imputation of ambiguous rows (§4.3).")
- **15:00** — Excitement peaks at the staircase figure and the two-FC control. Both are
  in the back half. This is the paper's main attention-architecture flaw: delight is
  back-loaded (see Phase 8).
- **20:00** — Residual unease: "two methods, one budget, one model" (known; envelope in
  flight) and §4.3's qualification density (scheduled: table-ize at conversion).

## Phase 2 — Attention map

| Section | Attention | Remembered | Fix |
|---|---|---|---|
| Title/abstract | HIGH | the two-numbers gap (now) | done |
| Intro ¶1 | HIGH | HOTT contradiction, SLM 0.03 | keep; split at venue pass |
| Contributions | HIGH | first 3 bullets only | 6→4 bullets at venue pass |
| §2 audit table | MEDIUM-HIGH | "only R-KV free-runs" | keep — quiet killer |
| §3 estimands | LOW (skimmed) | symbols blur | notation diet at venue pass |
| §4.1 phenomenon | HIGH | random beats recency | fine as second act |
| §4.2 table + facts | HIGH | the ladder row; −0.34 | promote gap fact to §4 opener |
| §4.3 bracket | MEDIUM | "survives"; hedging blur | table-ize endpoints |
| §4.4 staircase | HIGH | the figure | annotate eviction arrow |
| §4.5 ADD | MEDIUM | two failure styles | fine |
| §4.6 control | MEDIUM-HIGH | 0.068 vs 0.75 | fine |
| Threats | MEDIUM | "honest" halo | keep as-is (asset) |
| Conclusion | HIGH | one-sentence implication | add recipe paragraph (blueprint P11) |

## Phase 3 — The three ideas that must stick (memory design)

1. **Certified near-lossless ≠ deploys near-lossless:** 0.765 held-trace vs 0.422
   free-running, same method, same budget, same problems.
2. **The damage compounds from ~zero:** exactly 0 before eviction, ~0.002 after,
   ~50× growth — even with the trace held fixed.
3. **The fix is a recipe:** measure both channels; forward + reverse readout bracket
   the invisible part.

Carrier check: idea 1 now in abstract-sentence-3, Fig (ladder), §4.2, conclusion ✓.
Idea 2 in abstract, staircase, conclusion ✓. Idea 3 is the weakest carrier — it appears
only in the conclusion's final clause. Venue pass: give it the named paragraph
(BEST_POSSIBLE_PAPER Phase 11) and one clause in the abstract's last sentence if words
allow. Three ideas, three artifacts (ladder, staircase, recipe) — that's the paper.

## Phase 4 — Figure psychology

- **Figure 1 should be the accuracy ladders (current Fig 2).** The current Fig 1
  (collapse bars) confirms what HOTT readers already believe; the ladder shows the thing
  only this paper knows. A reviewer who sees only one figure must see the gap. This is a
  communication decision, not a science change — schedule with venue conversion.
  Current Fig 1 (bars + paired-diff histogram once the remote panel lands) becomes Fig 2.
- **Staircase:** add one annotation arrow "eviction event (every 64 steps)" at a visible
  step; keep the title line inside the plot — it functions as a caption for skimmers.
- **Ladder:** at conversion, draw a light shaded band spanning [reverse, forward] on
  each panel so the "bracket" is visible geometry, not just an annotation box.
- **Do not merge figures.** Five figures, one idea each, is the right density. Captions
  are already self-contained (done earlier today) — this matters because figure-first
  reviewers read only captions.

## Phase 5 — Abstract (APPLIED to the tex today)

Rewritten so the minute-one reviewer gets: gap-fact (sentence 3) → both-literatures-
correct reconciliation (sentence 2) → instrument → bracket-with-robustness → mechanism →
control → one-sentence implication. ~230 words. See tex; the previous version survives
in git history.

## Phase 6 — Intro rewrite (paste-ready for venue conversion; do not apply twice)

Paragraph plan (replaces current ¶1–2; ¶3+ survive with trims):

> **¶1 (the gap, concretely).** At cache budget 256 on MATH-500, random KV eviction on
> DeepSeek-R1-Distill-Qwen-7B scores 0.765 when the full-cache reasoning trace is held
> fixed and only the answer is generated under compression — within sight of the 0.882
> full-cache reference. The same method, on the same problems, scores 0.422 when it
> writes its own reasoning. Nothing about the compression changed; only the evaluation
> protocol did. This paper is about that gap: where it comes from, how to measure it,
> and why it explains a live contradiction in the literature.
>
> **¶2 (the contradiction it explains).** Eviction methods are certified near-lossless
> on held-trace evidence (perplexity, few-shot accuracy, retrieval — audited in Table 1),
> while HOTT free-runs the same families on reasoning and reports severe method-specific
> degradation (StreamingLLM 0.03 vs full-cache 0.47 at budget 128). Both are internally
> careful. We show both are right — they measure different channels of one loss.
>
> **¶3 (the instrument, plain words).** [current ¶3, with the symbol soup deferred to §3
> and one added clause:] "…the residual survives a reverse-replay bracket and every
> imputation of ambiguous rows (§4.3)."
>
> **¶4 (thesis + what it is not).** [current Thesis + non-claims paragraphs, merged.]
>
> **¶5 (contributions, 4 bullets).** gap+decomposition / bracket-robustness /
> mechanism+validity / phenomenon+ADD.

## Phase 7 — Friction audit (smallest change per friction)

| Friction | Where | Smallest fix |
|---|---|---|
| "Isn't this just exposure bias?" | 1st pass | Intro ¶3 clause: "we import the classical teacher-forcing lens [refs] and make it *quantitative, on an accuracy scale, for inference-time cache interventions*" |
| "Off-policy artifact" | §3 | forward-pointer in intro (Phase 6 ¶3) |
| "Random eviction isn't a method" | §4.1 | clause: "a *floor control* the field's benchmarks lack — which is precisely why its ordering vs.\ StreamingLLM is diagnostic" |
| "Why budget 256?" | §3.1 | clause: "the collapse pole of HOTT's 128–512 range; 2048 is the near-lossless pole" |
| "What does 'Mostly' mean (H2O, Table 1)?" | Table 1 | footnote: "few-shot harness generates short completions; trace effectively fixed" |
| "tout court" | Def. 5 | plain English: "never described as behavioral change outright" |
| §4.3 hedge-chain density | §4.3 | endpoints mini-table at conversion |
| Symbol soup | §3 | notation diet (blueprint P8) |

## Phase 8 — Delight timing

Existing delight moments, in current order of appearance: budget-parity check (§3.1) →
pre-registration + COMPLICATE honesty (§3.3/§4.2) → bracket survives (§4.3) → imputation
bounds (§4.3) → **staircase** (§4.4) → zero-pre-eviction self-control (§4.4) → two-FC
control (§4.6). The two biggest (staircase, gap-row) arrive at minutes ~12 and ~8.
Fixes: gap-numbers now in abstract sentence 3 (done); ladder as Fig 1 (conversion);
mention "the replay harness contributes exactly zero divergence before the first
eviction" in the contributions' validity bullet — it is the cheapest "nice control"
signal and currently hides in §4.4.

## Phase 9 — Paper identity

The sentence people will use: **"the paper that showed KV-compression damage is
invisible to held-trace evaluation — certified near-lossless, deploys at half
accuracy."** Test: title ✓, abstract ✓ (post-rewrite), Fig-1-as-ladder (pending),
conclusion ✓. Identity is strong once the figure swap lands.

## Phase 10 — Pre-written rebuttal (answers that are already in the paper)

1. *"Scope: one model/benchmark/budget."* → "Scoped and disclosed as the primary threat
   (§5); the pre-registered generality envelope (H2O, budget 2048) was queued before
   any reviewer contact; [by rebuttal time: results attached]. The contribution is the
   instrument + the existence proof of a large invisible residual, which one
   well-powered setting establishes."
2. *"Just exposure bias / teacher forcing."* → "The lineage is cited and imported
   (§2.3); prior work concerns training-time model pathology and offers no accuracy-scale
   channel decomposition for an inference-time intervention, no reverse bracket, no
   compounding profile. None of the certified-near-lossless papers used that lens —
   which is the gap we quantify."
3. *"Residual is off-policy-inflated."* → "Stated at the definition (Def. 3); bounded
   by the reverse bracket, which excludes zero under every imputation of undefined rows
   (§4.3), and the residual is not truncation (−0.110/−0.277 under all-truncated-correct,
   §4.2)."
4. *"ADD with n=4 is underpowered."* → "Benchmark-level aggregation, per-problem
   permutation null, BH correction, and a measured noise floor (0.068) an order of
   magnitude below signal (0.56–0.75); asymmetric n only tightens the reference side
   (Def. 5)."
5. *"Your FC accuracy (0.882) disagrees with HOTT's (0.47)."* → "Different decode caps
   and sampling; we reconcile channel structure, never digits (§2.2), and use HOTT's own
   library for comparability."
6. *"No CIs on bracket endpoints."* → currently the one concession; by rebuttal time the
   problem-clustered bootstrap of endpoints (roadmap #1) should be in the appendix so
   this line is never written.
7. *"H2O kernel confound."* → eager-vs-sdpa exchangeability check (§5).
8. *"Boundary detection is heuristic."* → deterministic, logged, per-row kinds; undefined
   rows counted, excluded, and imputation-bounded — never silently dropped.

## Phase 11 — Submission package

- **Keywords:** KV cache compression; reasoning models; evaluation methodology;
  teacher forcing; exposure bias; total variation; preregistration.
- **Anonymization checklist:** repo URL + prereg link `\todo`s → anonymous.4open.science
  mirror; **scrub `scripts/nightly_report.py` (contains a personal email — found in this
  pass) or exclude ops scripts from the artifact**; exclude `context.md`,
  `publication_strategy.md`, `research_landscape.md`, `notes/`, `COAUTHOR_REVIEW.md`,
  `BEST_POSSIBLE_PAPER.md`, this file (internal docs); keep PREREGISTRATION.md, configs,
  src, scripts (analysis only), figures, tests. For the timestamp claim: the prereg's
  commit-dated history can be shown to chairs on request; the anonymized mirror carries
  the file with its amendment dates inline.
- **Reproducibility checklist mapping (fill at submission):** seeds → configs;
  versions → §3.1 + requirements; compute → §3.1 (~230 GPU-h); data → MATH-500 pinned
  revision; code → anonymized repo; statistical methods → §3.1; one-command figures →
  `make_paper_figures.py` (+ `--print-windows` for Table 3).
- **Supplementary zip:** appendix PDF + figures/ + the five per-figure commands + prereg.
- **Camera-ready blockers already tracked:** 6 bibliography author `\todo`s; venue
  template; Fig-1 histogram panel (remote).

## Phase 12 — Would I fight for it?

With the envelope in and the Fig-1 swap done: **yes, and I'd spend capital** — because
it is the rare submission where the controls are figures, the preregistration has a
dated amendment log including a self-caught correction, and the discussion can end with
a recipe other people will use. The single change that most increases the odds I fight:
**make the ladder Figure 1.** A discussion-phase AC skims figures; the ladder is the
argument.

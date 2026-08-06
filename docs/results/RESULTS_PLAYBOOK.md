# docs/results/RESULTS_PLAYBOOK.md — pre-written read-ins for every pending run

Written 2026-07-06 (Fable handoff). For each pending harvest unit: the digest to
expect, pre-drafted paper text with `{SLOT}` placeholders, the decision branches with
the interpretation already reasoned out, exact tex anchors (search strings), and the
figure/docs updates. The goal: when a unit lands, the session fills numbers, picks the
branch the numbers dictate, and applies edits — no fresh judgment required unless a
result falls outside every branch (then: report it plainly, add a dated
PAPER_FINDINGS note, and do NOT force a branch).

General rules for every unit:
- Numbers come from `harvest_results.py`'s digest / the unit's `summary.json`
  (copy it to `results/diag/`). Round to 3 decimals in text, keep CI brackets as-is.
- After tex edits: rebuild PDF ×2, check `^!`/Overfull, run gates bare, commit with a
  message naming the unit, `make sync-apply`.
- Update `docs/results/PAPER_FINDINGS.md` (new dated subsection) and `docs/project/context.md` (top bullet)
  every time — templates at the bottom.
- Figure regen (when a unit says so): edit constants in
  `scripts/make_paper_figures.py`, sync, regenerate on remote with the DATA args used
  in docs/project/context.md 2026-07-06 bullets, scp back, **view the PNGs**, rebuild PDF.
  **The complete figure method (palette validation, display convention, exact regen
  commands, per-figure anatomy, failure catalog) is `docs/operations/FIGURES_PLAYBOOK.md` —
  read it before touching any figure.**

---


# ═══ PENDING UNITS — the morning loop works from here ═══

## UNIT fc-aime24-acc then m4-aime-2048

> **CONSUMED** 2026-07-18 — Branch A (no >100% regime) + residual replication + press-independent readout, reported plainly; record in PAPER_FINDINGS §4r (fc bank: §4p, 2026-07-15). Kept under PENDING header order only to preserve file history; do not redo.

fc digest: {rows, accuracy, truncation, mean_tokens}. Sanity: accuracy should be
~0.4–0.7 (R1-Qwen-7B on AIME-24 with 16K cap); truncation may be sizable — it stays
in-denominator. Record in PAPER_FINDINGS ("AIME-24 FC bank: acc {A}, trunc {T},
mean {M} tokens; ratio 2048/mean = {RATIO}").

m4-aime-2048 (random/slm/rkv at 2048, --bench aime24). THE question: does any press
(especially R-KV) reach or exceed FC on AIME free-running (the >100% regime)?

**Branch A (no gains: all free < FC, gained flip-mass ≈ 0):** add to threats
">100%" paragraph: "On AIME-24 at budget 2048 ({RATIO_PCT} of the mean trace),
no press exceeded the reference (best {BEST} vs {FC_AIME}; gained flip-mass
{GAINED}) — including our certified R-KV port. At 7B, in our protocol, the >100%
regime did not appear; per Amendment E this is a finding about scale/protocol, not
a port failure (retained sets match the authors' code exactly)." Keep the paper's
framing unchanged (it never claimed the regime).

**Branch B (a press ≥ FC on AIME, gained mass > lost on ≥ some problems):** the
upside case. Add a short results paragraph (end of §4.2 or threats-promoted):
"On AIME-24, {PRESS} free-runs at {X} versus the {FC_AIME} reference" + decompose
it: if its readout ≤ FC while free ≥ FC, the residual is POSITIVE — the first
positive-trajectory cell; frame via the direction discipline (deviation can be
improvement; signed flip-mass carries it). Do NOT retitle or reframe the paper
around >100% (Amendment G stands); one measured paragraph + PAPER_FINDINGS entry.
Check per-problem BH-corrected significance before any strong wording (n=30 —
never per-problem claims; problem-clustered CI only).

**Either branch:** AIME numbers stay out of the abstract (scope stays MATH-500-led)
unless Branch B survives CIs, in which case add ONE clause at most.

## UNITS m4-q15b-256 and m4-q14b-256 (scale axis; Amendment J, added 2026-07-17)

> **CONSUMED 2026-07-24 — Amendment J COMPLETE, both units read.**
> m4-q15b-256 (1.5B, read 2026-07-20): both cells REPLICATE. m4-q14b-256
> (14B, read 2026-07-24, last job in the queue): SLM-14B REPLICATES;
> R-KV-14B's bracket spans zero (span tied to the +0.023 resampling floor)
> — "anything else" branch, reported verbatim, not forced. Scorecard 3/4
> replicate → scope-upgrade sentence fires: "one model family spanning
> three scales (1.5B/7B/14B), plus a Llama-base check." Recorded in
> PAPER_FINDINGS §4s + §4t. Kept under PENDING header order only to
> preserve file history; do not redo either half.

Two units, same shape as the llama unit but SLM + R-KV presses, own banks
(fc-q15b / fc-q14b; digest prints fc_accuracy). **Bank gates first** (soft bands,
Amendment J): 1.5B FC in [0.60, 0.85] (published 0.839), 14B FC in [0.82, 0.96]
(published 0.939). Outside band → STOP, investigate before reading cells.

Per cell, pre-committed branches (Amendment J wording is binding):
**REPLICATES** (fwd residual negative, CI and bracket exclude zero): add the
cell's numbers to the threats scale paragraph (create it on first read):
"On {MODEL} ({FC} reference), {PRESS} free-runs at {FREE} against readout {RO}:
residual {RES} (CI {CI}), bracket {BR} excluding zero." After BOTH models read,
if ≥3 of 4 cells replicate: upgrade threats scope sentence to "one model family
across three scales (1.5B/7B/14B) plus a Llama-base check". SLM-14B/1.5B also
extend the sharpest-case line. R-KV cells: keep certified-port labeling; compare
its free-run rank vs SLM (does best-free-runner persist across scale?).
**DEGENERATE** (free floors, trunc > 0.5): report plainly like Llama-Random;
decomposition uninformative; check traces (filler-loop vs garbage) before
writing anything.
**NON-REPLICATION** (anything else): verbatim in threats, no compensation.
R-KV self-check abort on new dims = STOP (instrument), not a result.
Figures: NO figure changes for these units (subset scale, threats-only) unless
≥3/4 replicate AND the scale story earns a small multiples panel — decide then,
per docs/operations/FIGURES_PLAYBOOK.md; never improvise into Fig 1/2.


# ═══ CONSUMED UNITS — record only; branches taken are marked; do not redo ═══

## UNIT m4-2048 (readout-diss2048 completes; grid already done)

> **CONSUMED** 2026-07-07 — Branch A (residual shrinks); record in PAPER_FINDINGS §4j.

**Expected shape:** free 0.850/0.807 (known; 0.8075 exact — display is `.3f` of the
artifact float per PAPER_FINDINGS §5, never hand-rounded up). Readout at 2048 should be ≥ free-run
(mild readout loss), residuals small. This is the dose-response endpoint: does the
trajectory residual shrink toward zero at the near-lossless budget?

**Branch A (expected): both residuals in [−0.10, 0.00], CIs overlapping 0 or small.**
Insert after the §4.2 "Why does random beat recency?" paragraph, as a new bold-led
paragraph:

> \textbf{Dose response: the residual shrinks with budget.} At budget 2048 the same
> decomposition yields readout accuracies of {RO_RAND} (95\% CI {RO_RAND_CI}) and
> {RO_SLM} ({RO_SLM_CI}) against free-running $0.850$ and $0.807$: trajectory
> residuals of {RES_RAND} (CI {RES_RAND_CI}) and {RES_SLM} ({RES_SLM_CI}), an order
> of magnitude below the $-0.34$ residuals at 256. The trajectory channel is a
> low-budget phenomenon: at 3.8\% of the trace it dominates the loss; at 56.6\% it
> {RESIDUAL_VERDICT: "is statistically indistinguishable from zero" if both CIs
> cover 0, else "shrinks to a small fraction of the reference gap"}.

Also: (i) Setup line "with their readout half \emph{in progress}" → "…are complete";
(ii) Threats generality: budget clause becomes "the budget-2048 decomposition is in";
(iii) abstract stays (256 is the headline; do NOT dilute it with 2048 numbers).

**Branch B (surprising): a residual at 2048 with CI excluding 0 beyond −0.10.**
Then the residual is NOT purely a low-budget phenomenon. Text:

> \textbf{Dose response.} Even at budget 2048 — where free-running accuracy is within
> $0.03$--$0.07$ of the reference — the decomposition attributes {RES_X} of the small
> remaining loss to the trajectory channel (CI {CI_X} excluding zero): the channel
> persists at high budget in proportion to the loss.
Keep claims modest; add one sentence to threats noting persistence. Do not claim
"invisible damage at near-lossless budgets" beyond what the CI supports.

**Either branch:** add a Table 2 companion line? NO — keep Table 2 at 256. Put the
2048 numbers in the paragraph only (and PAPER_FINDINGS §4i extension).

## UNIT m4-2048-h2o (grid + readout at 2048 for H2O-style)

> **CONSUMED** 2026-07-10 — behaves-alike branch; record in PAPER_FINDINGS §4k.

Fold into the same dose paragraph with one clause: "the H2O-style press behaves
alike ({FREE_H2O} free, {RO_H2O} readout, residual {RES_H2O})." If its free-run at
2048 is NOT near-lossless (< 0.75, i.e. unlike R/SLM), instead add: "the H2O-style
press remains depressed at 2048 ({FREE_H2O}) — consistent with the missing recency
reservation; we note it and defer interpretation" (+ PAPER_FINDINGS note; do not
speculate further). Remember the 8192-cap footnote applies to its free-run cells.

## UNIT replay-256-h2o (TV profiles for H2O-style)

> **CONSUMED** 2026-07-10 — Branch A; record in PAPER_FINDINGS §4l.

Digest gives mean_auc_tv + per-trace windows. Current paper sentence (§4.4, "Second,
\textbf{forward method-invariance corroborated}: AUC-TV is matched across methods
($0.086$ Random vs.\ $0.083$ StreamingLLM)…").

**Branch A (|auc − 0.085| ≤ 0.02):** extend to "($0.086$ / $0.083$ / {AUC_H2O})" and
change "across methods" to "across all three policies".
**Branch B (larger deviation):** keep the R/SLM sentence, append: "the H2O-style
press's AUC-TV is {AUC_H2O}; its divergence profile differs in level though not in
shape (onset-aligned compounding persists — median onset {ONSET})." Check
windows.txt: 'before' MUST be ~0.0000 (instrument validity); if not, STOP — that is
an instrument bug, not a result: investigate before any paper edit.

## UNIT m4-rkv-256 (grid + readout + reverse for R-KV)

> **CONSUMED** 2026-07-12 — Branch C (intermediate); record in PAPER_FINDINGS §4m.

Labeling (binding): "R-KV (parity- and smoke-verified port; Amendment I)". All
claims are about our port at 7B. Add a FOURTH row to Table 2 with a `$^{\ddagger}$`
footnote: "$^{\ddagger}$parity-verified against the authors' code (retained-set
Jaccard 1.0) and smoke-verified during decoding; 16384-token cap like Random/SLM."

**Branch A (R-KV collapses like the other concentrated policies: free ≤ 0.30,
residual ≈ −0.34±0.06, CI excluding 0):** method-invariance extends to FOUR policies
including the reasoning-specific one. Update: §4.2 fact (i) "three" → "four
mechanistically unrelated selection rules (uniform, positional, attention-score,
redundancy-aware)"; residual spread number; abstract "three" → "four" with the new
residual list; contribution bullet 3; threats counts. §4.3: fold its bracket into
the lists (check whether its reverse endpoint lands with the concentrated pair —
expected — and update the uniform-vs-concentrated sentence to name it).

**Branch B (R-KV does substantially better free-running: free ≥ 0.55):** the
reasoning-specific redundancy signal genuinely helps at 256. Then the decomposition's
value is DISCRIMINATIVE: report its readout/residual split; if its residual is small
where others are −0.34, write:

> R-KV is the outlier that proves the instrument discriminates: its redundancy-aware
> selection preserves the trajectory channel (residual {RES_RKV}, CI {CI_RKV})
> where uniform, positional, and attention-score selection all lose $\approx 0.34$.
Method-invariance language must then be WEAKENED everywhere to "invariant across
uniform/positional/attention-score selection; the redundancy-aware policy departs"
— abstract, bullets, §4.2, §4.4, threats. This is a GOOD outcome for the paper
(instrument sensitivity); do not spin it as a failure.

**Branch C (intermediate: free 0.30–0.55):** report the row + bracket plainly; place
its residual in the ordering; only claim what CIs support. No framing overhaul.

Figures: add R-KV to Fig 1 (bar, new hue — validate palette with 4th color
`#56B4E9` sky blue via dataviz validator; if it fails adjacency, table-only and say
so in the caption) and extend Fig 2 to a 2×2 grid ONLY if Branch A/B (a fourth panel
earns space when it either completes the invariance or breaks it).

## UNIT m4-llama-256 (cross-architecture replication)

> **CONSUMED** 2026-07-15 — fits no branch — reported plainly; record in PAPER_FINDINGS §4o.

Uses its OWN reference bank (fc-llama50; harvest digest prints its fc_accuracy —
also archive `harvest/m4-llama-256/summary.json`). 50 problems × n=4: CIs will be
~2× wider than the Qwen cells. Add ONE paragraph to threats-generality (not the main
results — subset scale):

**Branch A (replicates: both residuals negative with CIs excluding 0):**
> \emph{Cross-architecture check.} On DeepSeek-R1-Distill-Llama-8B (Llama base; 50
> problems $\times$ 4 seeds, its own pinned reference bank at {FC_LLAMA}), the
> pattern replicates: free-running {FREE_R}/{FREE_S} versus readout {RO_R}/{RO_S},
> residuals {RES_R} (CI {CI_R}) and {RES_S} ({CI_S}), brackets excluding zero. The
> readout-invisible trajectory residual is not an artifact of the Qwen lineage.
Also upgrade the scope sentence "one model" → "one primary model (with a
subset-scale Llama-base replication)". If Random>SLM ordering also replicates,
add the clause "and the uniform$>$concentrated free-run ordering persists
({FREE_R} vs {FREE_S})".

**Branch B (does not replicate: residual CIs cover 0 or sign flips):** report in
threats verbatim as a boundary: "the residual did not replicate at subset scale on
the Llama base ({numbers}); we flag architecture dependence as an open question."
Do NOT bury it, do NOT expand claims elsewhere to compensate; the 256 Qwen result
stands on its own preregistered feet.

## UNIT fc-readout-control (readout-fc-control completes; queued LAST, 2026-07-07)

> **CONSUMED** 2026-07-17 — Branch A (+0.023 measured); record in PAPER_FINDINGS §4q.

Full-cache readout of the FC bank itself = the pure answer-resampling effect of the
readout protocol (zero compression). Digest gives {rows, n_undefined,
readout_accuracy}. **Expected shape: readout_accuracy ≈ 0.88–0.92** (FC is 0.8825;
the 2048 readout cells bound resampling+near-lossless-compression at ~+0.02).

**Branch A (expected: |acc − 0.8825| ≤ 0.04):** replace the §4.2 dose-paragraph
sentence "The above-reference readout values also calibrate the readout protocol
itself: regenerating the answer span on reference traces is worth roughly $+0.02$
even with compression near-inactive" with the direct measurement: "A direct control
--- full-cache readout of the full-cache traces --- puts the answer-resampling
effect of the readout protocol at {ACC_MINUS_FC:+.3f}"; keep the rest of the
sentence (order-of-magnitude-below + direction clause). Add one line to
PAPER_FINDINGS §4f (instrument validity) with the numbers.

**Branch B (surprising: acc < 0.84 or > 0.93):** the readout protocol itself moves
accuracy by more than resampling noise should — STOP, do not edit the paper;
investigate (boundary detection on FC traces? seed collision with bank seeds?
n_undefined spike?) and treat as possible instrument bug per the standing rule.
Note: bank seeds 11/23/47/89 overlap the readout seeds by construction (matched-seed
convention, same as every readout run); that is protocol, not a bug.

## ALREADY HARVESTED (2026-07-06, in the paper — do not redo)

- add-2048: Random ADD 0.085 / SLM 0.146 (floor 0.068); §4.5 dose sentence.
- add-256-h2o: 0.715/0.713/+0.101/−0.636; Table 3 row + §4.5 + Fig 3 third group.

## Templates

PAPER_FINDINGS subsection:
```
### 4x. {UNIT} — read {DATE}
{one-paragraph digest with report-exact numbers, CI brackets, n's, truncation}
Branch taken: {A/B/C} because {one line}. In the paper: {sections touched}.
Archived: results/diag/{file}.json
```

docs/project/context.md top bullet:
```
**★ {UNIT} read ({DATE}):** {headline numbers}. Branch {X}. Paper: {sections}.
Figures {regenerated|unchanged}. Next pending: {name of next unit}.
```

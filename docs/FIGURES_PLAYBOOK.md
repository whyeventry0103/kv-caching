# FIGURES_PLAYBOOK.md — how to make and maintain this paper's figures

Written 2026-07-13 (Fable handoff, final session). This is the complete method
behind every figure in `kv_readout_vs_trajectory_paper.tex`, precise enough that a
successor session produces figures at the same standard without re-deriving any of
it. Read it top to bottom once; afterwards the checklist at the end is all you need
per batch.

**The one rule that outranks everything: LOOK AT THE RENDERED PNG before calling
any figure done.** Every figure batch in this repo's history shipped exactly one
defect that only eyes caught (catalog at the bottom). Code that runs is not a
figure that works.

---

## 1. Source of truth

`scripts/make_paper_figures.py` is the single generator. Structure:

- **Constants block** (top of file): `FC`, `C` (per-policy dict: free/readout/
  reverse/residual/bracket/CIs/add/within/term/flip/trunc), `C2048`, `PHEN`,
  `TVWIN`, `TWOFC`. Every constant is a verified, report-exact value with the
  provenance in a comment. When a new result lands, constants are updated ONLY
  from the harvest digest / archived `results/diag/*.json` — never from memory,
  never from the paper text.
- **Two render modes per figure.** SUMMARY: draws from constants, works anywhere.
  DATA: reads run artifacts (per-problem rows, report summaries, TV profiles) and
  is report-exact; degrades gracefully to SUMMARY when paths are missing. The
  committed `figures/` artifacts are DATA-mode wherever DATA exists (Fig 1
  histogram, Fig 2 cells, Fig 4 curve). `_cells_from_summary()` maps summary
  press names → display names (`rkv` → `R-KV` etc.); extend it when a press is
  added.

## 2. The display convention (binding)

Every 3-decimal number on any figure is `f"{v:.3f}"` of the artifact float — the
same string numpy/R produce from raw rows (PAPER_FINDINGS §5, pinned 2026-07-07).
NEVER hand-round a tie upward (99/400 = 0.2475 renders 0.247, not 0.248; 323/400 =
0.8075 renders 0.807). Tie-valued constants are stored at artifact precision
(0.2475, not 0.247) so SUMMARY and DATA modes render identically. Before writing
any new number into a figure or caption, run the value through `f"{v:.3f}"` in
Python and use exactly that string — do not do the rounding in your head.

## 3. Color (computable — so compute it)

Fixed assignment, never cycled, never re-ranked:
`Random=#0072B2 (blue) · StreamingLLM=#D55E00 (vermillion) · H2O-style=#CC79A7
(purple) · R-KV=#56B4E9 (sky) · grey=full-cache reference · GREEN=#009E73
reserved for the negative-control line only`. 2048 dose tints: `#9CC7E4/#F2B27F/
#E5BCD3` — tints are the SAME entity at a different dose, always direct-labeled,
never treated as new categories.

Any new hue must pass the dataviz validator BEFORE use — never eyeball CVD safety:

```bash
node <dataviz-skill>/scripts/validate_palette.js "#hex1,#hex2,..." --mode light
```

(The bundled skill path changes between harness versions — `find
/private/tmp/claude-501/bundled-skills -name validate_palette.js`.) Pass bar:
adjacent-pair CVD ΔE ≥ 12. Current validated worst pairs: sky↔purple 20.7,
sky↔blue-tint 15.7. A contrast WARN (<3:1 vs surface) is legal only because every
mark carries a direct value label — keep that invariant. If a 5th policy ever
lands, candidate hue is Okabe-Ito yellow `#F0E442`; validate it in the actual
adjacency order of the chart, and if it fails, the new method goes table-only and
the caption says so.

## 4. Design principles (what makes these figures land with reviewers)

1. **One message per figure, stated in the caption with exact numbers.** The
   caption makes the claim; the figure makes it undeniable.
2. **Encode findings in layout, not only in ink.** Fig 2's 2×2 grid puts the
   recoverable pair (Random, R-KV) on the top row and the concentrated pair
   below — the row split IS the §4.3 reverse-side finding. Fig 1's gap between
   bar groups IS the dose split (and doubles as CVD secondary encoding between
   sky and the blue tint). When a result has structure, ask what the layout can
   say before adding annotations.
3. **Annotations state the scientific reading, not the geometry.** "TV = 0.0000
   before first eviction (self-control)" — not "flat region". "eviction events
   every 64 steps" — not "steps in curve". Group headers carry the realized
   ratios (3.8% / 56.6%) so the figure argues by itself.
4. **Direct-label every mark** (value at 3 decimals). Legends only when ≥2 series
   share a panel. Text never wears a series color; identity comes from the mark.
5. **Grey is semantic**: the full-cache reference and only that (bars, dashed
   FC line in Fig 1, faint dotted FC vertical in Fig 2, termination bars in
   Fig 3 as the neutral channel).
6. **Never a dual axis; never stack non-additive quantities** (within-answered TV
   is renormalized — Fig 3 draws it BESIDE ADD, never stacked into it).
7. **Captions carry the discipline**: "non-negative on 94/100 (44 win/50 tie/6
   loss)", certified-port labeling for R-KV, † / ‡ footnote symbols matching
   Table 2. If a claim near a figure has a number, recompute it from the artifact
   at write time (the 0.459 ADD broke a stale "order of magnitude / 0.56–0.75"
   claim in FOUR places — found only by rechecking).

## 5. Regeneration — exact commands

Local (CPU, works always; DATA mode for Fig 2 via archived summaries):

```bash
.venv/bin/python scripts/make_paper_figures.py \
  --out-dir <scratch-dir> \
  --m4-summary results/diag/m4-powered-diss256-summary.json \
               results/diag/m4-h2o256-full-summary.json \
               results/diag/m4-rkv-256-summary.json
```

Remote (needed for Fig 1's paired-diff histogram and Fig 4's full TV curve —
sync the script FIRST with `make sync-apply`, or you regenerate with stale code):

```bash
ssh kvcache-gpu 'cd ~/kv-caching && .venv/bin/python scripts/make_paper_figures.py \
  --out-dir /tmp/figregen \
  --runs-root /home/sk10741/kvcache-storage/runs/grid-diss256-tok \
  --profiles-dir /home/sk10741/kvcache-storage/runs/replay-diss256/profiles'
scp kvcache-gpu:/tmp/figregen/fig{1_phenomenon,4_tvprofile}.{pdf,png} figures/
```

Then promote the locally-rendered DATA-mode fig2 (and any constants-only figs)
from scratch into `figures/`, NEVER regenerate straight into `figures/` — render
to scratch, view, then promote. Fig 5's optional `--twofc-report` lives at
`<runs>/two_fc_control.json` if you ever need its per-problem histogram.

Workflow per batch: edit script → `black` → render to scratch → **view every
changed PNG** → promote → rebuild PDF ×2 (`grep '^!'` and `Overfull`) → captions
synced → `make lint` + `make test` bare → commit (name the unit/decision) →
`make sync-apply`.

## 6. Per-figure anatomy (current state, 2026-07-13)

| File | Content | Mode of committed artifact | Regen trigger |
|---|---|---|---|
| `fig0_protocol` | 4-row estimand schematic; fill=cache condition, hatch=compressed, in-span text=token source | SUMMARY (no data) | only if estimands change |
| `fig1_phenomenon` | 8 bars (4 policies @256, 3 @2048 + FC), group headers with realized ratios, FC dashline; right: Random−SLM paired-diff histogram | DATA (remote, `--runs-root`) | any free-run number changes |
| `fig2_bracket` | 2×2 ladders; rows encode recoverable/concentrated; FC dotted vertical; residual+bracket+CIs footer per panel | DATA (local, 3 summaries) | any decomposition cell changes |
| `fig3_add` | 4 groups, within-answered + termination bars, ADD/flip boxes, noise-floor line | SUMMARY (constants) | any ADD cell changes |
| `fig4_tvprofile` | onset-aligned staircase; annotations: 64-step events + zero-before self-control | DATA (remote, `--profiles-dir`) | new replay profiles worth adding |
| `fig5_twofc` | negative control vs two exemplar cells | SUMMARY | control numbers change (they won't) |
| `fig6_aime` | AIME @2048 press-independence: separated free-run bars vs identical hatched readout bars, FC dashline | SUMMARY (constants `CAIME`) | AIME cells change (they won't) |

Pending-unit hooks (see RESULTS_PLAYBOOK.md): the Llama and AIME units are
TEXT-ONLY (no figure changes). If AIME Branch B (>100% regime) ever fires, that
gets a figure decision of its own — do not improvise it into Fig 1.
UPDATE 2026-07-18: the AIME read (Branch A) DID earn a threats-local figure —
`fig6_aime` — decided because the press-independence finding (identical 0.462
readout across all three presses vs 2× free-run separation) is inherently
visual and one bar chart states it faster than the sentence. Precedent: a
consumed unit may earn a NEW, section-local figure when its finding is visual;
main figures (1/2) still change only by their own rules. Hatch here = "the
readout protocol" (consistent with fig0's texture-not-hue role); subset scale
(n=30) stated in the caption.

## 7. Failure catalog — every defect a batch shipped, all caught by viewing

1. Fig 1 xlabel clipped by tight bbox → fixed with layout, not smaller text.
2. Fig 2 CI text struck through by a spine; annotation box over labels → moved to
   a data-coord footer under each panel (`ax.get_yaxis_transform()`).
3. Fig 3 three annotation boxes overlapping → two-line stacked boxes.
4. Protocol schematic: hatching struck through in-bar text → white rounded bbox
   behind every label; boundary line ran through its own label → line clipped to
   the bar region.
5. Hand-rounded ties (0.248 / 0.808) disagreeing between constants-mode and
   DATA-mode renders of the SAME quantity → display convention pinned (§2).
6. `"---"` in matplotlib text renders as three literal hyphens (matplotlib is not
   TeX) → use parentheses or a real dash character.
7. Stale claim orbit: R-KV's ADD 0.459 invalidated "order of magnitude below
   0.56–0.75" in abstract/bullet/§4.6/Fig-5 caption → when a figure gains data,
   grep the tex for every number-range near its topic.
8. `annotation_clip` is not a valid `ax.text` kwarg (use `clip_on=False`).
9. Histogram exposed that "94/100" was 44 win + 50 tie + 6 loss → figures don't
   just display claims, they AUDIT them; when a figure contradicts the prose,
   the prose changes.

## 8. Judgment calls already made (don't relitigate)

- Table 2 stays at budget 256; 2048 numbers live in prose + Fig 1 bars only.
- Fig 2 shows all four policies since 2026-07-12 (user-requested, supersedes the
  Branch-C figures-unchanged default that applied before the request).
- Fig 4 stays single-curve (R+SLM 800-profile mean); H2O's AUC lives in §4.4
  text. Overlaying three curves clutters the staircase, which is the point.
- Fig 5 keeps two exemplar bars; all four cells are Table 3's job.
- Within-group bar order everywhere = Table 2 order (Random, SLM, H2O, R-KV),
  except Fig 3 which is descending-ADD (SLM, H2O, Random, R-KV) — both
  deliberate; do not "harmonize" one into the other.

## 9. Constraints the Unit 9 redesign must inherit (recorded 2026-07-28)

**STATUS 2026-08-03: three of the four are FIXED; one remains open.** The
"wait for the redesign" advice below was written when the redesign looked
imminent. It stopped being right once these figures had to be shown to a
reader, so the defects were fixed in place. Marked inline.

These are open defects observed in the rendered PNGs, not preferences.

- **[FIXED 2026-08-03] `fig2_bracket`: value labels collide with the CI whiskers.** On every
  forward-readout rung the label is drawn at the marker while the whisker
  extends past it, so `0.765`, `0.584`, `0.597` and `0.798` all overlap their
  own interval tick. Visible at 100% zoom; it is the reason the figure reads as
  cluttered even before the density problem. The planned one-panel
  slope/dumbbell redesign must place labels clear of the interval geometry —
  offset above the mark, or outside the axis, but not at the mark.
  *Fixed:* the label now sits past `max(mark, ci_hi)`, so it clears the interval
  on all four panels and the whisker is fully visible for the first time.
- **[FIXED, already] `fig4_tvprofile` does not contain the numbers the text cites.** Its y-axis
  tops near 0.07 and its horizon stops at +400, while the text quotes a later
  window mean of 0.088 and a step-pooled 0.118. Extend the horizon or add an
  inset; as drawn, a reader who checks the figure against the sentence finds
  they disagree.
  *Resolved:* the figure is now the windowed 3-bar form and carries 0.0000 /
  0.0018 / 0.0884 directly, including the pre-eviction zero self-control. This
  entry was stale when re-checked on 2026-08-03.
- **[FIXED 2026-08-03] `fig1_phenomenon` shows only free-running accuracy**,
  which is what HOTT already showed. Adding readout as a paired lighter series
  is what makes it *our* figure.
  *Fixed:* readout is now a hatched paired series at both budgets (hatch = "the
  readout protocol", consistent with fig0/fig6). The figure now states the
  paper's thesis directly: the protocols disagree at 256 and agree at 2048.
  Two further defects were found only by viewing the first render -- the
  contrast box and the new legend each struck a budget-group header -- and were
  fixed with headroom; a third (the Acc_FC dashline striking Random@2048's
  "0.850") with a white backing box on every value label.
  *Also reclaimed:* in SUMMARY mode the second panel was `axis("off")` plus a
  three-line text box, spending ~38% of the figure on text. The second axis is
  now built only when there is a histogram to put in it. The .tex caption had
  described the DATA-mode histogram panel that the committed artifact never
  had; it now describes what is actually rendered.

- **[STILL OPEN] `fig2_bracket` is four panels carrying ~40 numbers** (see
  above). The label collision is fixed but the density is not: the one-panel
  slope/dumbbell redesign that moves the bracket numbers to a table has NOT
  been done.

**Standing constraint for any figure change from 2026-07-27 onward:** every
constant is loaded from `paper_numbers.json` via `_k()`. Do not reintroduce a
literal, and do not add a fallback — a missing key must raise. A figure drawn
from stale constants is worse than no figure, because it looks finished. This is
how the figures went on drawing pre-Amendment-K values while the summaries and
the `.tex` had both been regenerated.

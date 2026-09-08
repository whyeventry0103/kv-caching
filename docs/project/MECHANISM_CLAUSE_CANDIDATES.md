# Mechanism clause — candidate wordings for review

**Status: DECIDED 2026-09-08 — candidate A, sharpened. Propagated to all seven
marked sites plus three the markers missed.** Zero `TODO(M.3-RESTATEMENT)`
markers remain in `kv_readout_vs_trajectory_paper.tex`; each is replaced by a
`RESOLVED(M.3-RESTATEMENT)` comment naming its site number and what changed
there. The analysis below is kept as written, pre-decision, because the rejected
candidates and their attacks are the record of why A was chosen. **See "Decision"
at the foot of this file for what was actually done.**

## What is being replaced

Thesis, final clause (site 1, L74):

> ...The gap is the trajectory residual, and it arises from a divergence that is
> approximately zero at the moment of eviction and *compounds* over the
> generated trajectory.

Section title (site 3, L332): **"The mechanism: divergence is zero at eviction
and compounds"**

The other five sites are the contributions bullet (L93), the figure caption
(L362), the body's "three readings" paragraph (L374), the conclusion (L624), and
the Phase-3 windowing-defect paragraph (L588), which already carries the pairing
and needs no change under any candidate.

## The measured facts, and their licensing status

| fact | status |
|---|---|
| pre-eviction TV exactly `0.0000` | **instrument property**, not a finding — under teacher forcing with identical token IDs the two models see identical input until the first prune |
| `0.0018` first 50 post-onset steps, `0.0884` later | measured, fixed-budget arm |
| within a prune cycle, phase excess over sham: Random `+0.252` [0.184, 0.271] peak **0**; StreamingLLM `+0.076` [0.031, 0.099] peak **1** | measured, licensed as a mechanism (survives ratio-matching) |
| drift, fixed-budget arm `1.201` / `1.299`; ratio-matched arm `0.662` / `0.739` | measured (M.3, both presses FLATTENS) |
| *what produces the rise* | **not licensed.** Horizon and realized ratio are confounded by construction at fixed budget |
| *whether horizon matters at fixed cache size* | **never tested.** Holding the ratio fixed means letting the budget grow 32→626 |

---

## Candidate A — regime-anchored

**Thesis clause.**
> ...The gap is the trajectory residual: under the fixed cache budgets that
> deployments actually use, the divergence behind it is negligible where the
> eviction happens and large later in generation.

**Section title.** *Where the divergence is, and where readout evaluation is not*

**Propagation.** Clause swap at sites 1, 2, 7. Site 3 is a retitle. Site 5's
"three readings" paragraph keeps its numbers and loses the phrase
"slow-compounding drift"; one sentence. Site 4 (figure caption) needs more than a
swap — see the figure note below. Site 6 unchanged.

**Emphasizes / gives up.** Emphasizes that the finding is about *where you
measure*, which is the paper's actual contribution, and makes no claim the
evidence cannot carry. Gives up the causal story entirely: the reader is told
divergence is small early and large late and is never told why, so §4.4 stops
being an explanation and becomes a localisation.

**Strongest attack.** *"You describe a curve and decline to explain it. If the
growth tracks the compression ratio, your finding reduces to 'compressing harder
hurts more', which nobody disputes."* — **Partially succeeds.** It lands on §4.4,
not on the thesis: the paper's claim is the evaluation gap and the residual, both
untouched. But it correctly identifies that A leaves §4.4 with no thesis of its
own, and a reviewer who wanted a mechanism will read the retreat as one.

---

## Candidate B — mechanism-forward

**Thesis clause.**
> ...The gap is the trajectory residual, and the divergence behind it is locked
> to the eviction events themselves — near-zero between prunes and peaking at
> each prune step — rather than appearing as a single distortion at answer time.

**Section title.** *Eviction-locked divergence, and an envelope that tracks the
budget regime*

**Propagation.** Sites 1, 2, 7 swap. Site 3 retitles. Site 5 needs a **structural
rewrite**, not a swap: its "three readings" are ordered temporal-structure /
magnitude-across-policies / instrument-validity, and B promotes the phase result
into first position, which currently is not one of the three readings at all —
the phase analysis lives in the Phase-3 threats material, not in §4.4. Site 4
needs a re-plot. **Site 7 (conclusion) does not fit cleanly**: the conclusion
sentence is doing summary work about the *residual*, and inserting phase-locking
there makes it about a divergence-scale periodicity; it would need bespoke
wording.

**Emphasizes / gives up.** Emphasizes the one thing M.3 left standing as a
mechanism, so the paper still claims to explain something. Gives up the
0.0018→0.0884 contrast as the headline, which is the number that makes
short-horizon evaluation look inadequate.

**Strongest attack.** *"You lead with a phase statistic because it is the only
mechanism you have left. It is small, it is divergence-scale, and for
StreamingLLM the peak is at phase 1, not 0 — so 'locked to the eviction step' is
exact for one of two policies. Meanwhile the 34-point accuracy loss, which is
what the paper is about, now has no mechanism attached at all."* — **Succeeds.**
This is the strongest attack on any candidate. The substitution is visible, and
the StreamingLLM peak is a factual weakness at the thesis level: the clause as
written is true of Random and one step off for StreamingLLM, and a thesis
sentence cannot carry that caveat.

---

## Candidate C — explicitly two-level

**Thesis clause.**
> ...The gap is the trajectory residual, and the divergence behind it is
> two-level: within each prune cycle it is locked to the eviction event, while
> across cycles its envelope rises in the fixed-budget setting deployments use
> and does not when the compression ratio is held constant.

**Section title.** *Two levels of divergence: eviction-locked within cycles, and
an envelope that depends on the budget regime*

**Propagation.** Sites 1, 2, 3, 7 all take the pairing, so all four grow by a
clause. Site 5 needs the **most work** of any candidate: the "three readings"
become four, or the temporal-structure reading splits in two. Site 4 needs a
re-plot and the caption grows. Site 6 unchanged. This is the only candidate where
every marked site changes length rather than wording.

**Emphasizes / gives up.** Emphasizes the honest map of what survived a control,
which is the paper's stated moat, and is the only candidate that cannot be
misread as either a horizon claim or a claim that horizon is irrelevant. Gives up
compactness everywhere, and spends the thesis sentence — the one sentence a
skimming reader retains — on a structural distinction rather than on the result.

**Strongest attack.** *"This is the shape of a curve, presented as a structural
finding. What does 'two-level' predict, or explain, that the one-level story
didn't?"* — **Partially succeeds.** The honest answer is that the two levels have
different *licensing status*, not different predictive content, and that is a
statement about our evidence rather than about the model. That is a real
limitation of C. Its defense is that the distinction is exactly what the
ratio-matched control bought, and burying it would waste the control.

---

## The figure — a caption change is NOT sufficient

`fig4_tvprofile` plots mean per-step TV against **"Decoding step relative to
first eviction"**. At a fixed budget the realized compression ratio is a
deterministic function of that axis — ratio = budget / context, and context grows
with the step. **The figure therefore plots the confound as its x-axis**, and a
single rising curve against a horizon axis asserts the horizon reading visually
no matter what the caption says. Under A the caption would disclaim what the
picture shows; under B and C it would contradict it.

The fix is a re-plot, and the data for it already exists: overlay the
ratio-matched arm from `replay-ratio-matched-m3-v2` on the same axes. Two curves —
one rising, one flat-to-falling, same x-axis, same estimator — make the two-level
structure visible and remove the single-curve inference entirely. That is a
figure that *earns* candidate C, and it is the strongest argument for C.

Cost: the ratio-matched cell covers Random and StreamingLLM at 25 problems × 4
seeds, against 800 profiles for the fixed-budget curve, so the overlay is a
thinner line and the caption must say so.

---

## Recommendation

**Candidate A, with the §4.4 retitle**, because the paper's contribution is the
evaluation gap and the residual, and a thesis sentence that promises a mechanism
the evidence can no longer deliver costs more under review than one that promises
less. If the figure is re-plotted with the ratio-matched overlay, C becomes the
better choice and I would switch — but C without that figure asserts a two-level
structure the reader cannot see.

## Flagged while drafting

**1. A constraint is in tension with itself.** "Both findings travel together at
every site" cannot hold at the thesis site while the thesis stays one clause —
carrying both makes it a paragraph (candidate C is the proof). The resolution is
that the pairing rule binds *where a mechanism claim is made*: a thesis that
makes none owes no pairing. Under A the rule is vacuous at sites 1, 2 and 7 and
binding at 3, 4, 5 and 6. This should be stated explicitly in the rule rather
than left for the next editor to rediscover.

**2. StreamingLLM's peak is phase 1, not 0.** Any wording that says divergence is
locked *to the eviction step* is exact for Random and one step off for
StreamingLLM. B puts this in the thesis, where the caveat cannot fit. A and C
survive it because they say "locked to the eviction event" at cycle level rather
than naming the step.

**3. The clause may deserve cutting, not restating — and I would defend that.**
"Approximately zero at the moment of eviction" is an instrument property: under
teacher forcing the models see identical input until the first prune, so TV
*must* be zero. It is a validity control, and the paper already uses it as one
("the exactly-zero pre-eviction TV certifies the replay harness"). Stating it in
the thesis gives a control the standing of a discovery, and a sharp reviewer will
say so. Cutting the final clause entirely leaves "The gap is the trajectory
residual." — clauses 1 and 2 intact, no unlicensed attribution, nothing promised
that §4.4 must then deliver. Candidate A is one short step from this; the only
thing A buys over cutting is continuity with the abstract's existing shape.

**4. Nothing here should be decided from this file alone.** Each candidate's
propagation cost was assessed by reading the seven sites, but sites 5 and 7 carry
argumentative load that a clause swap can damage in ways only a full read will
show. Whichever is chosen, the next step is a full read of §4.4 and the
conclusion under the new clause, not a find-and-replace.

---

# Decision — 2026-09-08

**Chosen: candidate A (regime-anchored), with one substantive sharpening taken
from flag 3 above.** Propagated the same session; the paper no longer contains
an unlicensed horizon attribution.

## Why A, and why not the others

**C was ruled out by a blocked dependency, not by argument.** The recommendation
above makes C conditional on `fig4_tvprofile` being re-plotted with the
ratio-matched overlay, and this file's own verdict is that "C without that
figure asserts a two-level structure the reader cannot see." The re-plot could
not be done: `_onset_aligned_curve` needs `--profiles-dir`, and neither
`replay-ext-diss256` nor `replay-ratio-matched-m3-v2` exists locally — only
pilots and smokes are under `results/runs/`. The profiles are on the remote,
which was refusing connections on port 22 all session. So C's precondition is
unmet and C would have overclaimed. **If the overlay is ever plotted, revisit
this decision — C becomes correct at that point, and the site-4 comment in the
`.tex` carries the instructions.**

**B stands rejected on its own merits**, per the attack recorded above: it leads
with a phase statistic while the 34-point headline loss keeps no attached
mechanism, and its thesis wording is exact for Random and one step off for
StreamingLLM (peak phase 1, not 0). A thesis sentence cannot carry that caveat.

**Pure cutting (flag 3) was rejected, but its argument was accepted and is what
sharpened A.** Flag 3 is right that "approximately zero at the moment of
eviction" is an instrument property — under teacher forcing the two models see
identical token IDs until the first prune, so pre-eviction TV *must* be
`0.0000`, and `sec:mechanism` already spends that fact as the replay harness's
self-control. Putting it in the thesis gave a validity control the standing of a
discovery, and a sharp reviewer would say so.

The fix was not to cut the clause but to **move it onto the measured number**.
The thesis now cites `0.0018` — the mean per-step TV over the 50 steps after the
eviction, which is small *by measurement*, not by construction — instead of the
constructed `0.0000`. That single change does two things at once:

1. It restores the clause's status as a finding rather than a control.
2. It defeats A's strongest recorded attack ("you describe a curve and decline
   to explain it"). Anchored on the measured value, the claim stops being *why
   the curve grows* and becomes **where the standard protocol samples it** —
   readout-shaped evaluation measures at the point where the effect is smallest.
   That is an evaluation-protocol claim, which is the paper's actual
   contribution, and it is entirely untouched by M.3 delicensing the horizon
   reading of the rise.

## The clause, as it now stands (site 1)

> The gap is the trajectory residual: under the fixed cache budgets deployments
> actually use, the divergence behind it is small in the steps immediately
> following an eviction and far larger later in generation, so held-fixed-trace
> protocols sample it exactly where it is smallest.

Section title (site 3): **"Where the divergence is, and where readout evaluation
looks"**.

Note the clause carries no numeral. That is deliberate: landmine 11 (a prose
quantity is invisible to `audit --strict`) means any numeric phrasing in the
thesis would be an unchecked claim. The numbers live in `sec:mechanism` and
`tab:tv`, where the audit sees them.

## The seven marked sites, and three the markers missed

| site | location | what changed |
|---|---|---|
| 1 | thesis clause | replaced, per above |
| 2 | contributions bullet | retitled "The mechanism" → "Where the divergence lives"; `0.0000` relabelled inline as the self-control; "compounding to"/"slow-compounding drift" removed. All numbers unchanged. |
| 3 | `sec:mechanism` title | retitled |
| 4 | `fig:tvprofile` caption | rewritten to disclaim the horizon reading and name the ratio confound explicitly. **Partial — the rest is a plotting task.** |
| 5 | "three readings" | re-read whole, not swapped. The a-fortiori argument is kept (it survives M.3 — free-running adds a second divergence source whatever drives the rise); "slow-compounding drift" and "pre-compounding regime" removed; the licensed within-cycle result paired in per the §4w never-alone rule. |
| 6 | Phase-3 windowing paragraph | **visited and deliberately left unchanged** — it already pairs both findings and attributes the rise to nothing, which is exactly what A asks. Editing it would be churn. |
| 7 | conclusion | two edits, not one: the causal "because … compounds over the trajectory that follows" restated, **and** the list item "the TV profile supplies the compounding mechanism" four sentences later, which no marker pointed at but asserted the same delicensed cause. |

**The marker set was incomplete.** A sweep for the delicensed vocabulary after
propagating the seven found three more sites, one of them serious:

- **The abstract** said the profiles "show it grows with horizon" and that short-horizon
  evaluation "measures the pre-compounding regime" — the strongest unlicensed
  claim in the paper, in the most-read paragraph, and unmarked. Restated.
- **Summary of findings** said "near-zero immediate effect and strong compounding". Restated.
- **The trajectory-audit protocol** (step 4) told future users to "check the per-step
  divergence profile for compounding". It now tells them to hold the
  intervention's dose fixed before reading growth as a horizon effect — which
  generalizes our own M.3 correction into the protocol, where it belongs.

This is the concrete vindication of flag 4 above: a find-and-replace over the
markers would have shipped a paper whose abstract still made the claim its own
§4.4 had withdrawn.

## Verification

PDF rebuilt ×2: **0 LaTeX errors, 1 Overfull, 9 Underfull, 29 pages** — byte-for-byte
the same counts as a baseline build of `HEAD`'s `.tex` in a scratch directory,
so these edits introduced no new typesetting problems and the one Overfull
(lines 309–310) predates them. Gates bare: lint clean, 313 passed / 14 skipped,
dryrun clean, `audit --strict` clean on both the numbers and the claims gate.

## Still open after this

- The `fig4_tvprofile` overlay (blocked on the remote). Until it is plotted, the
  caption disclaims what the picture shows — correct, but not the end state.
- Page budget: unchanged at 29 against ICLR's 9.

---

# Revisit of the C condition — 2026-09-08, later the same day

The decision above set an explicit reopening condition: *"If the overlay is ever
plotted, revisit this decision — C becomes correct at that point."* The remote
came back the same session, the figure work was done, and the condition was
therefore evaluated rather than left to lapse.

**Outcome: the condition is NOT met. Candidate A stands.** Two findings, both
from doing the work rather than from re-reading this file.

## 1. The premise about fig4 was wrong

This file states that `fig4_tvprofile` "plots mean per-step TV against *decoding
step relative to first eviction*" and therefore "plots the confound as its
x-axis." **The committed figure does no such thing.** It is the three-window
SUMMARY bar chart — before eviction / first 50 after / later — because
`make_paper_figures.py` only draws the onset-aligned curve when `--profiles-dir`
is supplied, and the profiles are on the remote. Verified by reading
`figures/fig4_tvprofile.png` and by extracting the text of the embedded
`.pdf`: the axis label is "Window rel. to first eviction (onset step 65.0)" and
the three bars are 0.0000 / 0.0018 / 0.0884.

So the visual-assertion problem that motivated C **never applied to the paper as
printed**. It applied to a DATA-mode rendering that no one had looked at.

A separate, real defect came out of the same check: the caption described "a
compounding staircase whose steps align with eviction events" and said "the
curve is shown to +400 steps of onset" — a *curve the figure has never shown*.
That mismatch predates this whole restatement and is now fixed; the caption
describes the three windows it actually plots.

## 2. The proposed overlay would have been misleading, and was rejected on data

The plan was to overlay the ratio-matched arm on shared axes. Measured on the
paired traces (200 profiles, all 200 (press, seed, pid) keys shared between the
two runs), the two arms differ by roughly **two orders of magnitude in level**:
first-50 window mean TV is `0.0036` fixed against `0.1972` ratio-matched, and at
onset offset 0 it is `0.0008` against `0.3289`. The ratio-matched press spends
the early trace pinned at `min_budget` 32 against the fixed arm's 256, so it is
simply compressing much harder.

M.3's verdict is a **drift** statement *within* each arm — last-quarter mean over
first-quarter mean — and never a level comparison *between* them. A shared-axis
overlay would have invited exactly the level reading no measurement here
licenses, in a figure whose whole purpose is to stop an unlicensed reading.

## What was built instead

`fig:ratiomatched` (`figures/fig7_ratiomatched.pdf`): the paired above-floor
window M.3 actually uses, with each trace divided by its own first-quarter mean
before averaging, so 1.0 is "no drift" and the plotted quantity *is* the verdict.
Fixed arm rises, ratio-matched arm falls, both policies, levels normalized away
with the reason stated in the caption.

Data comes from `results/diag/amendment-p/m3_drift_curves.json`, emitted by
`report_m3_verdict.py --curves-out` — the same pairing and windowing code as the
verdict, so the figure and the number cannot drift apart. The same run
re-derived `m3_verdict.json` **byte-identically** (sha256 match against the
committed artifact), which doubles as a re-derivation check.

## Why this still does not earn C

C's case in this file rests on a specific figure: *"Two curves — one rising, one
flat-to-falling, same x-axis, same estimator — make the two-level structure
visible."* `fig:ratiomatched` is not that figure. It makes **one** of C's two
levels visible — the across-cycle envelope and its control. The within-cycle
eviction-locking, C's other level, is a phase statistic and is still not
plotted anywhere.

C's thesis sentence asks the reader to hold a two-level structure of which the
paper now shows one level. That is a weaker position than A, which claims only
what every figure in the paper supports. **A stands, and the reopening condition
is now closed rather than pending** — if anyone later plots the phase profile
alongside the envelope, reopen it again on that evidence.

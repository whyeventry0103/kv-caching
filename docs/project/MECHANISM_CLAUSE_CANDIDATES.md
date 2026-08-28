# Mechanism clause — candidate wordings for review

**Status: DRAFT FOR DECISION. Nothing here is in the paper.** All 7
`TODO(M.3-RESTATEMENT)` markers remain in `kv_readout_vs_trajectory_paper.tex`;
this file exists so the clause can be decided once, on paper, rather than
settled by whoever edits the `.tex` next.

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

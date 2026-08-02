# Proposed reframe — title and abstract, STAGED not applied

> ## WITHDRAWN 2026-08-02 — read this before anything below
>
> **The reframe is withdrawn.** B.9 gated it on Unit 6's resolution relationship
> being monotone with CIs supporting it. Budgets 512 and 1024 have since run, and
> the pre-registered prediction was **refuted**: the ratio falls 0.97 → 0.30 →
> 0.04 and then rises to 0.06 at 2048, non-monotone, with the 1024 value outside
> the pre-committed interval (PREREGISTRATION, Amendment L final result).
>
> Both proposed titles — "Blind Where It Certifies" and "Readout Accuracy Loses
> Resolution Exactly Where It Certifies Near-Losslessness" — assert the graded
> reading that the data does not support. **Keep the current title.** The
> contribution reorder that placed resolution collapse first is withdrawn for the
> same reason.
>
> What survives is the endpoint comparison (readout separates the policies at
> budget 256 and cannot at 1024/2048/AIME), which is now §4.1 "Held-trace
> resolution at four budgets", demoted from a primary contribution. Everything
> below is kept as the record of a proposal the data closed off.


**Status: not in the `.tex`.** Per the work order's B.9, the structural parts of the
reframe (the new §4.1 resolution section, the validity kit, the claim calibration,
the ρ_rev cancellation, the matched-schedule argument) have landed in the paper
normally. Title and abstract are the two taste-dependent, hard-to-revert choices,
so they are staged here for a decision instead.

**Gated on Unit 6, and the gate passed.** B.9 said not to reframe unless the
resolution relationship is monotone with CIs supporting it. It is: ratio 0.78 →
0.12 → 0.00 across MATH@256, MATH@2048, AIME@2048, with the two collapse cells'
spread differences excluding zero and the 256 cell (where readout works) spanning
zero. See PREREGISTRATION Amendment L and `tab:resolution`.

---

## 1. Title

**Current (in the `.tex`):**

> Near-Lossless Under Readout, Collapse Under Free Generation:
> A Trajectory Audit of KV-Cache Compression for Reasoning

**Proposed:**

> Blind Where It Certifies: Held-Trace Evaluation of KV-Cache Compression for Reasoning

**Conservative alternative:**

> Readout Accuracy Loses Resolution Exactly Where It Certifies Near-Losslessness

### Assessment

The proposed title is better than the current one on one axis and worse on another,
and the choice is genuinely yours.

*Better:* the current title states the phenomenon (two protocols disagree). The
proposed one states what we now know *about the instrument*, which is the more
transferable contribution and the one a reader can act on. "Blind where it
certifies" is also the sentence the resolution table actually supports.

*Worse:* "Blind Where It Certifies" is a rhetorical construction, and the finding it
names is the one part of the paper that is **post-hoc** (Amendment L). Leading with
the exploratory result while the pre-registered decomposition sits behind it is a
defensible framing choice but not obviously the honest ordering — a reviewer who
reads Amendment L may reasonably ask why the post-hoc finding got the title.

The conservative alternative avoids that problem and keeps the emphasis. It is
longer and flatter. My recommendation, stated once: **take the conservative
alternative** if the resolution result stays exploratory, and the proposed title
only if the 512/1024 confirmatory cells run and come back consistent (Amendment
M.5), at which point the finding is no longer post-hoc and has earned the position.

---

## 2. Abstract

**Current (in the `.tex`, after Unit 7's calibration):** ~310 words, 18 numbers.

**Proposed:** ≤180 words, ≤8 numbers, ordered contradiction → two-sided
measurement → the number → the resolution collapse → the recommendation.

> KV-cache eviction methods are certified near-lossless on evidence that holds the
> generated text fixed — perplexity on reference traces, few-shot accuracy,
> retrieval — while free-running reasoning benchmarks report severe degradation of
> the same methods. We show these protocols measure different quantities and price
> the difference. On MATH-500 with an R1-distilled 7B model at cache budget 256
> (3.8% of the mean trace), random eviction scores 0.765 when the full-cache
> reasoning trace is teacher-forced and only the answer is generated, but 0.422
> when the model writes its own trace, against a matched full-cache reference of
> 0.885. Replaying the compressed trajectory under a full cache reaches only 0.636:
> a clean reader cannot recover what the compressed model wrote. Measuring the gap
> from both directions leaves an interval excluding zero for every policy tested.
> We then show the certifying protocol's own limitation: its ability to separate
> policies degrades as the effect it measures shrinks, so it discriminates well
> where it reports damage and, in the near-lossless regime, cannot rank policies
> that free-running separates twofold. Report free-running reasoning accuracy.

Numbers used: 256, 3.8%, 0.765, 0.422, 0.885, 0.636 — six. Word count ~175.

### What was cut, and why it is a real cost

- **The TV/mechanism sentence.** It was doing genuine work (localizing when
  divergence arises) and its absence makes §4.4 arrive unannounced.
- **The ADD instrument and its negative control.** The two-full-cache floor is one
  of the paper's better methodological moves and now appears nowhere until §4.5.
- **"trajectory audit" as a named deliverable.** PAPER_FINDINGS §5 records this
  naming as a deliberate decision; dropping it from the abstract partly reverses
  that decision and should be a conscious call, not a casualty of the word budget.
- **The pre-registered COMPLICATE outcome.** Currently in the abstract; the
  proposed version omits it. Given that the honesty trail is treated as this
  paper's moat, I would argue for spending ~15 words to keep it.

**Recommendation:** the proposed abstract is better structured and materially
easier to read, but I would restore one clause naming the audit and one naming the
pre-registered outcome, landing near 200 words. Both cuts trade away something the
project has explicitly decided it values.

---

## 3. Contribution reorder (proposed; not applied)

1. Held-trace evaluation loses resolution exactly where it certifies (§4.1).
2. The readout/free-running gap, measured from both directions.
3. Compressed trajectories are not recoverable by a clean reader (reverse readout).
4. Temporal structure of the divergence.
5. Two failure styles converging across two independent instruments — the
   reverse-side recoverable/concentrated split and the ADD within-answered/
   termination split agree on the same taxonomy. **This is currently buried as
   three exploratory sentences and is a real result**: two instruments that share
   no machinery agreeing on one partition of the policies is evidence neither
   provides alone.
6. The audit and its validity kit, with transfer stated as an expectation.

Note contribution 1 is post-hoc and contribution 5 is exploratory. Putting two
non-pre-registered items at positions 1 and 5 is the main risk in this ordering,
and it is why the title recommendation above is the conservative one.

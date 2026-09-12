# ICLR 2027 body / appendix split — proposal

**Written 2026-09-12, after the venue conversion.** Proposal only; nothing is cut
yet. Deadlines: abstract **2026-09-18**, full paper **2026-09-25** (verified
against the ICLR 2027 author guidelines).

## The measurement, which changes the problem

| | pages |
|---|---|
| Before conversion (`article`, 11pt, 1in margins) | 29 |
| **After conversion (`iclr2027_conference`)** | **26** |
| Main text only (§1 to end of §6, excluding AI statement and references) | **~24** |
| Limit for the initial submission | **9** |

The old "29 against 9" figure was never like-for-like: it was measured under
`article` at 11pt with 1-inch margins, not under the venue's style. The real gap
is **~24 main-text pages against 9**, so roughly **15 pages must leave the body**.
References and appendices do not count, and ICLR allows unlimited appendix pages
after the bibliography, so this is a *relocation* problem and not a deletion
problem. The AI use statement is also exempt and stays where it is.

## Where the pages currently are

| Section | pages | span |
|---|---|---|
| §1 Introduction | 2 | 1–2 |
| §2 Related Work | 3 | 3–5 |
| §3 Estimands and Experimental Design | 3 | 6–8 |
| **§4 Results** | **9** | **9–17** |
| **§5 Threats to Validity** | **5** | **18–22** |
| §6 Conclusion | 2 | 23–24 |
| AI use statement (exempt) | 1 | 24–25 |
| References (exempt) | 1 | 25–26 |

§4 and §5 are 14 of the 24 pages, and they are where the relocation has to come
from. Cutting proportionally across all six sections would be the wrong move: it
would thin the introduction and related work, which are what a reviewer reads to
decide whether the contribution is real.

## Proposed split

**Body, 9 pages.** The argument in its shortest complete form: the contradiction,
the estimands, the headline decomposition, one figure per idea, and the honest
threats.

| Body element | budget |
|---|---|
| §1 Introduction, unchanged in substance | 1.5 |
| §2 Related Work, trimmed to the audit table + the nearest-neighbour paragraph | 1.5 |
| §3 Estimands, keeping the definitions and the off-policy caveat, moving the setup detail | 1.5 |
| §4.2 the powered phenomenon + §4.3 forward decomposition + §4.4 reverse bracket | 2.5 |
| §4.5 where the divergence is (with `fig:tvprofile`) | 0.75 |
| §5 Threats, compressed to the table plus the two invalidation paragraphs | 0.75 |
| §6 Conclusion + the trajectory-audit protocol | 0.5 |

**Appendix, everything below.** Ordered so a reviewer who follows one pointer
lands on the thing they doubted.

| Moves to appendix | from | pages |
|---|---|---|
| §4.1 held-trace resolution at four budgets | §4 | 1.5 |
| §4.6 ADD channel decomposition, full table + discussion | §4 | 1.5 |
| §4.7 negative control, full treatment (a one-line result stays in the body) | §4 | 1.0 |
| §4.8 follow-up probes: reinsertion, divergence–residual coupling, onset | §4 | 2.0 |
| §5 the full threats table, the R-KV port certification, the AIME cells | §5 | 3.0 |
| §5 the Phase-3 windowing-defect disclosure and both instrument invalidations | §5 | 1.5 |
| §3 setup detail: seeds, sampling, bank construction, grader | §3 | 1.5 |
| `fig:ratiomatched` and the M.3 control discussion | §5 | 1.0 |

That is ~13 pages relocated, which lands the body at roughly 10–11 and leaves the
final 1–2 pages to be recovered by tightening prose rather than by structural
cuts — deliberately, because a plan that reaches exactly 9 on paper always
overruns.

## Three things that must not be cut, whatever the arithmetic says

1. **The off-policy caveat in §3** stated up front, not in rebuttal. It is the
   first thing a competent reviewer will attack, and the paper's defence is that
   it raised the objection itself and bounded it.
2. **The residual-is-a-definition framing.** "Accuracy change unexplained by
   on-reference readout", never a measured causal effect. Shortening this into a
   causal claim to save a line would be the single most damaging edit available.
3. **The nearest-neighbour paragraph** (§2.2, arXiv:2608.01631). A reviewer who
   knows that paper and does not find it cited will assume we did not know about
   it. It is also where the honesty trail pays: we quote their numbers as already
   containing our residual.

## Sequencing against the deadline

The abstract is due **six days** from writing and the full paper **thirteen**. The
abstract deadline needs only title, abstract, and authors, none of which this
split touches, so the split is on the 25th's critical path and not the 18th's.
Meeting #3 cells compete for the same fortnight and are explicitly *not* on the
critical path: Amendment Q is pre-registered and queued, and nothing from it can
land in this submission.

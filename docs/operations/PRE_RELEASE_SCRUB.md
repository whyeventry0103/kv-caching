# Pre-release scrub — one deliberate step, run only if the repo is made public

**Status: NOT RUN. Nothing here is executed as routine maintenance.**

Both remotes are **private** (verified 2026-08-29 via `gh repo view`:
`whyeventry0103/kv-caching` and `-backup`, both `"visibility":"PRIVATE"`).
Everything below is scoped to *eventual public release* — an arXiv posting, a
double-blind submission, or flipping a remote to public. While the remotes stay
private, pushing is a backup operation and none of these items gate it.

Work through this in order and tick each item. Do not do them piecemeal: the
value is in running the whole list once, deliberately, against the state at that
moment rather than against this file's snapshot.

---

## 1. Author block — `kv_readout_vs_trajectory_paper.tex`

- **Line 42** carries the real name: `\author{Muhammad Ali Shah\\ ...}`.
- **Line 41** already records the requirement in a comment: *"venue conversion
  must restore 'Anonymous Authors' and re-anonymize the repo URL."*

For a double-blind submission, restore the anonymous author block. For a
non-anonymous arXiv posting the real name is correct and this item is a no-op —
decide which release this is before touching it.

## 2. Repository URL / anonymized mirror — same file

Find it with `grep -n 'github.com/whyeventry0103' kv_readout_vs_trajectory_paper.tex`
— do not trust a line number here, one already went stale (this item read "line
493" while the text sat at 634).

The `\paragraph{Reproducibility.}` line hard-codes
`https://github.com/whyeventry0103/kv-caching` and asserts the anonymized mirror
**twice**, not once — both must be resolved together:

1. the `\footnote{...}`: *"the double-blind submission carries an anonymized
   mirror"*;
2. the paragraph's closing sentence: *"The double-blind submission carries the
   same file through the anonymized mirror."*

If releasing double-blind, the mirror must actually exist and the URL must point
at it. Both sentences currently promise something that has not been created.
Verify before submission rather than at submission.

**Do not build the mirror reflexively.** This has two mutually exclusive correct
resolutions and they turn on a decision that has not been made: whether
double-blind anonymization is in scope for the ICLR submission copy at all. The
owner has stated they do not want the work anonymous in general; whether that
extends to the submission copy specifically is a separate, unconfirmed question.
Resolve the scope question first — then either build the mirror, or delete both
assertions.

## 3. `docs/strategy/` — four tracked, two gitignored

| file | state |
|---|---|
| `BEST_POSSIBLE_PAPER.md` | tracked, in both remotes |
| `COAUTHOR_REVIEW.md` | tracked, in both remotes |
| `PROPOSED_REFRAME.md` | tracked, in both remotes |
| `SUBMISSION_STRATEGY.md` | tracked, in both remotes |
| `publication_strategy.md` | untracked, gitignored |
| `research_landscape.md` | untracked, gitignored |

**Decided 2026-08-29: do NOT rewrite git history for these.** The four tracked
files are already in both remotes' history, so deleting them going forward would
not remove them from history; only a history rewrite would, and that is
disproportionate while the remotes are private.

At public-release time the options are, in increasing cost: (a) accept them as
part of the artifact — they are working notes, not embarrassing, and an honest
research trail is this project's stated moat; (b) remove them going forward and
accept that history retains them; (c) rewrite history, which invalidates every
published commit hash including those cited in the amendment log. The
inconsistency worth resolving either way is that two files of the same kind are
gitignored while four are not — that split was incidental, not decided.

## 4. `docs/project/context.md`

Contains machine paths, the remote username, hostnames, session narrative, and
dated internal decisions. None of it is secret, but it is not written for an
external reader. Decide whether it ships. If it does, it needs no edit; if it
does not, it must be excluded at the packaging step, not deleted (it is the
project's own record).

## 5. Corrected claim — read this before acting on `CLAUDE.md`

**`CLAUDE.md` line 98 states a personal email is embedded in
`scripts/nightly_report.py`. That is FALSE, verified 2026-08-26 and re-verified
2026-08-29.** The file contains **zero** address literals. The recipient and MX
host come from `KVCACHE_REPORT_TO` / `KVCACHE_REPORT_MX` (lines 41-42), with the
module docstring stating explicitly there is no compiled-in default and that it
refuses to send without them.

Do not act on that CLAUDE.md line. It is recorded here so the stale claim is not
re-derived by a future reader, chased, and "fixed" by editing a file that is
already correct.

**Corrected 2026-09-08.** `CLAUDE.md` no longer asserts the embedded email; it
now points at this checklist and records the retraction inline, so a reader who
encounters the old claim in git history can resolve it. This item is retained
(not deleted) because the retraction is the useful record, not the original
error.

---

## Not part of this scrub

- Result artifacts, `paper_numbers.json`, `PREREGISTRATION.md`, `PAPER_FINDINGS.md`
  — these are the artifact and are meant to ship.
- The `TODO(M.3-RESTATEMENT)` markers in the `.tex` — a pending scientific
  decision, not a release-hygiene item.

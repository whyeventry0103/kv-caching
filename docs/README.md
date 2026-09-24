# Documentation index

Five groups, by what you need them for. The authority order for facts is: the
repo's code > `protocol/` > `results/` > `operations/` > `project/`. For code,
disk is ground truth — the markdown can lag.

## protocol/ — what was committed to, before the data existed

| File | |
|---|---|
| [`PREREGISTRATION.md`](protocol/PREREGISTRATION.md) | The frozen scientific protocol and its append-only, dated amendment log (A–P, plus P.1–P.5, M.3 and P.2-v2, and **Amendment Q** — the Meeting #3 pre-registration of 2026-09-12 — with its Q.1 and Q.2 results and the Q.1 correction of 2026-09-24). Every deviation lives here as an amendment; nothing frozen changes without one. Start here to judge whether a result was predicted or discovered. |

## results/ — what was measured

| File | |
|---|---|
| [`RESULTS.md`](results/RESULTS.md) | **The results record for Meetings #1–#2.** Every cell, control and caveat through Amendment P. Generated from `paper_numbers.json`; regenerate with `python scripts/export_results.py --format both` rather than editing. **It does not yet cover Meeting #3** — the generator predates Amendment Q. For Meeting #3 results read `PAPER_FINDINGS.md` §5a (Q.1), §5b (Q.2) and §5c (Task 1 budget curve), and the Amendment Q entries in `PREREGISTRATION.md`. |
| [`RESULTS.html`](results/RESULTS.html) | The same content as a standalone designed page (dark/light, print-ready). |
| [`PAPER_FINDINGS.md`](results/PAPER_FINDINGS.md) | The dated results log, the §4-CURRENT headline table, and the **binding framing rules** (§5) — the exact phrasings claims must use. Gated: §4-CURRENT may contain no unsourced number. |
| [`RESULTS_PLAYBOOK.md`](results/RESULTS_PLAYBOOK.md) | Pre-reasoned read-ins for the Meeting #1–#2 units (all consumed), recording which observable numbers selected which interpretation. The Meeting #3 reads are pre-registered in Amendment Q instead. Written before the results landed, so a reader fills slots instead of exercising fresh judgement. |

## operations/ — how to run it

| File | |
|---|---|
| [`remoteserver.md`](operations/remoteserver.md) | **Authoritative ops and safety policy.** Read before any setup, sync, or launch. |
| [`WORKFLOW.md`](operations/WORKFLOW.md) | The local↔remote loop (rsync-first, "Policy B"). |
| [`REMOTE_SETUP.md`](operations/REMOTE_SETUP.md) | One-time remote host preparation. |
| [`complete_setup.md`](operations/complete_setup.md) | End-to-end project bootstrap. |
| [`INTEGRATION_CHECKLIST.md`](operations/INTEGRATION_CHECKLIST.md) | What each run (all landed — kept as procedure, not a live queue) touches and what to write when it lands — the pre-registered read per unit. |
| [`PRE_RELEASE_SCRUB.md`](operations/PRE_RELEASE_SCRUB.md) | The one deliberate step to run if and when the repo is made public — author block, repo URL, `docs/strategy/`, `context.md`. Items 1–2 resolved by the ICLR conversion (2026-09-12); items 3–5 **not run**. Both remotes are private. |
| [`FIGURES_PLAYBOOK.md`](operations/FIGURES_PLAYBOOK.md) | Figure method, conventions, and the open-defects list. |

## design/ — how it works and why

| File | |
|---|---|
| [`METRICS.md`](design/METRICS.md) | Definitions of every metric (ADD, TV/TDP, the decomposition estimands). |
| [`M4_REPLAY_DESIGN.md`](design/M4_REPLAY_DESIGN.md) | The replay/readout engine's design. |
| [`RKV_PORT.md`](design/RKV_PORT.md) | The R-KV port and its two-stage integrity gate. |
| [`KVPRESS_API_PROBE.md`](design/KVPRESS_API_PROBE.md) | What the kvpress decoding API actually offers, probed rather than assumed. |
| [`STACK_MIGRATION_M2_5.md`](design/STACK_MIGRATION_M2_5.md) | The pinned-stack migration and its go/no-go gates. |
| [`MEETING3_DESIGN.md`](design/MEETING3_DESIGN.md) | The Meeting #3 design: the three supervisor tasks (damage vs cache size; attention-map comparison; trajectory direction) and how each was reframed, with the corrected storage arithmetic and the redesigned Q.1 control. **Read with `context.md`'s top bullet**: of the three tasks only Task 1 has results; Tasks 2 and 3 were coded but never run. |

## project/ — narrative state

| File | |
|---|---|
| [`context.md`](project/context.md) | Dated ★-bullets, newest first: what landed, exact numbers, what is in flight, the next action. Written so a reader with zero prior context can resume. |
| [`MECHANISM_CLAUSE_CANDIDATES.md`](project/MECHANISM_CLAUSE_CANDIDATES.md) | Three candidate wordings for the thesis mechanism clause, with propagation cost and the strongest attack on each. **Decided 2026-09-08: candidate A, sharpened, and propagated into the paper**; the revisit condition for candidate C was evaluated and closed. |
| [`scoop-watch.md`](project/scoop-watch.md) | Related-work watch list. |

## strategy/ — internal

Positioning, venue, and review notes. **Not part of the published artifact** —
these are working documents about how to present the work, not evidence about
it. Exclude them from any submission bundle. Includes
[`ICLR2027_PAGE_BUDGET.md`](strategy/ICLR2027_PAGE_BUDGET.md), the proposed
body/appendix split against the 9-page limit (proposed, not executed).
`publication_strategy.md` and `research_landscape.md` are gitignored and exist
only on the author's machine.

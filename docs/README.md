# Documentation index

Five groups, by what you need them for. The authority order for facts is: the
repo's code > `protocol/` > `results/` > `operations/` > `project/`. For code,
disk is ground truth — the markdown can lag.

## protocol/ — what was committed to, before the data existed

| File | |
|---|---|
| [`PREREGISTRATION.md`](protocol/PREREGISTRATION.md) | The frozen scientific protocol and its append-only, dated amendment log (A–O). Every deviation lives here as an amendment; nothing frozen changes without one. Start here to judge whether a result was predicted or discovered. |

## results/ — what was measured

| File | |
|---|---|
| [`RESULTS.md`](results/RESULTS.md) | **The results record.** Every cell, control and caveat. Generated from `paper_numbers.json`; regenerate with `python scripts/export_results.py --format both` rather than editing. |
| [`RESULTS.html`](results/RESULTS.html) | The same content as a standalone designed page (dark/light, print-ready). |
| [`PAPER_FINDINGS.md`](results/PAPER_FINDINGS.md) | The dated results log, the §4-CURRENT headline table, and the **binding framing rules** (§5) — the exact phrasings claims must use. Gated: §4-CURRENT may contain no unsourced number. |
| [`RESULTS_PLAYBOOK.md`](results/RESULTS_PLAYBOOK.md) | Pre-reasoned read-ins: for each pending unit, which observable numbers select which interpretation. Written before the results landed, so a reader fills slots instead of exercising fresh judgement. |

## operations/ — how to run it

| File | |
|---|---|
| [`remoteserver.md`](operations/remoteserver.md) | **Authoritative ops and safety policy.** Read before any setup, sync, or launch. |
| [`WORKFLOW.md`](operations/WORKFLOW.md) | The local↔remote loop (rsync-first, "Policy B"). |
| [`REMOTE_SETUP.md`](operations/REMOTE_SETUP.md) | One-time remote host preparation. |
| [`complete_setup.md`](operations/complete_setup.md) | End-to-end project bootstrap. |
| [`INTEGRATION_CHECKLIST.md`](operations/INTEGRATION_CHECKLIST.md) | What each queued run touches and what to write when it lands — the pre-registered read per unit. |
| [`FIGURES_PLAYBOOK.md`](operations/FIGURES_PLAYBOOK.md) | Figure method, conventions, and the open-defects list. |

## design/ — how it works and why

| File | |
|---|---|
| [`METRICS.md`](design/METRICS.md) | Definitions of every metric (ADD, TV/TDP, the decomposition estimands). |
| [`M4_REPLAY_DESIGN.md`](design/M4_REPLAY_DESIGN.md) | The replay/readout engine's design. |
| [`RKV_PORT.md`](design/RKV_PORT.md) | The R-KV port and its two-stage integrity gate. |
| [`KVPRESS_API_PROBE.md`](design/KVPRESS_API_PROBE.md) | What the kvpress decoding API actually offers, probed rather than assumed. |
| [`STACK_MIGRATION_M2_5.md`](design/STACK_MIGRATION_M2_5.md) | The pinned-stack migration and its go/no-go gates. |

## project/ — narrative state

| File | |
|---|---|
| [`context.md`](project/context.md) | Dated ★-bullets, newest first: what landed, exact numbers, what is in flight, the next action. Written so a reader with zero prior context can resume. |
| [`scoop-watch.md`](project/scoop-watch.md) | Related-work watch list. |

## strategy/ — internal

Positioning, venue, and review notes. **Not part of the published artifact** —
these are working documents about how to present the work, not evidence about
it. Exclude them from any submission bundle.

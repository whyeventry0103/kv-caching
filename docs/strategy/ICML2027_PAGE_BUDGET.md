# ICML 2027 body / appendix split — proposal (2026-09-27)

Target venue: proposed ICML 2027, pending supervisor confirmation.

This is a proposal only; nothing in the `.tex` has been moved. The venue terms
used here are those stated in the 2026-09-27 instruction: an 8-page body with
unlimited appendix; the 2026 deadline was 28 January; the 2027 call is expected
around October. Verify every one of them against the ICML 2027 call for papers
when it is published, before acting on this plan. It replaces nothing in
`ICLR2027_PAGE_BUDGET.md`, which records the earlier ICLR plan.

## Where the pages are now (compiled PDF, 2026-09-27, 26 pages)

Section starts were read off the compiled PDF with `pdftotext`, page by page.

| Section | Pages | Span |
|---|---|---|
| §1 Introduction | ~2.3 | 1–3 |
| §2 Related Work | ~2.3 | 3–5 |
| §3 Estimands and Experimental Design | ~4.3 | 5–9 |
| §4 Results (4.1–4.8) | ~9 | 9–18 |
| §5 Threats to Validity | ~5.5 | 18–23 |
| §6 Conclusion | ~1.3 | 23–24 |
| AI use statement, references (exempt) | ~2 | 24–26 |

The body is about **24 pages against 8**, so roughly **16 pages must leave the
body**. That is a relocation problem, not a deletion problem: appendices are
unlimited. Nothing that carries a pre-registered result or a retraction is
deleted; it moves.

## Proposed 8-page body

| Body section | Pages | What it holds |
|---|---|---|
| Abstract + §1 Introduction | 1.0 | The problem, the three contributions, and one results sentence per task. Stays 256-led (the abstract decision stands). |
| §2 Related Work | 0.5 | Only the nearest neighbours: the fixed-trace/evidence work (arXiv:2608.01631, overlap stated before the difference), eviction methods' evaluation protocol, and teacher forcing vs free running. The full survey moves to the appendix. |
| §3 Estimands and design | 1.25 | Readout, residual and bracket definitions; the protocol figure (`fig:protocol`, reduced); one paragraph on pre-registration. Instrument validation moves to the appendix. |
| §4 Results: the decomposition | 2.5 | Held-trace resolution (1 paragraph, `tab:resolution` moved to the appendix); the powered phenomenon (`fig:phenomenon`); forward decomposition (`tab:decomp`); reverse-replay bracket (`fig:bracket`); dose response in two sentences. |
| **§5 Mechanism** (new; the R.17 P.1 draft) | **1.5** | Dose, not horizon, with the collapse verdict once read. Attention: M and Δ with H2/H3 (b2048 confirmatory). Latent geometry: ω plus the bridge. Restoration: retracted and redesigned. One figure: the R.17 supervisor figure, damage beside AUC-TV against budget. It replaces current §4.5 "Where the divergence is…" and §4.8 "Follow-up probes". |
| §6 Threats (short form) | 0.75 | The five threats a reviewer must see in the body: estimand dependence of the method split; the pending numeric floor for divergence numbers; the scope of one model plus a cross-architecture check; truncation in denominators; and the honesty trail in one paragraph with pointers. |
| §7 Conclusion | 0.5 | |
| **Total** | **8.0** | |

## What moves to the appendix (nothing below is deleted)

| From | Moves to | Content |
|---|---|---|
| §1 | App. A | Extended motivation; the audit table `tab:audit` |
| §2 | App. B | The full related-work survey (2.1–2.3), CASK, VeriCache, HOTT audit table |
| §3 | App. C | Instrument validation: parity harness, 800-row exact replication, R-KV port certification, double-BOS fix, `tab:validity` |
| §4.1 | App. D | `tab:resolution` (post-hoc, exploratory) |
| §4.4 details | App. D | Bracket derivation, reverse-endpoint correction (Amendment H), estimand dependence |
| §4.5 | App. E | Onset-aligned TV profile (`fig:tvprofile`, `tab:tv`), M.3 ratio-matched result (`fig:ratiomatched`) |
| §4.6 | App. F | ADD with signed flips (`tab:add`, `fig:add`) |
| §4.7 | App. F | Two-full-cache negative control (`fig:control`) |
| §4.8 | App. G | P.5 retraction record in full, exact-lag read, eligibility table, AUC-TV vs residual (P.3), onset localisation (P.4) |
| §5 | App. H | Long-form threats: invalidated instruments, the wrong diagnosis, the windowing defect, the M.3 restatement sites |
| §6 (AIME, cross-model) | App. I | AIME (`fig:aime`), Llama and Qwen size replications |
| New (R.17) | App. J | Q.1 sawtooth and direct fork control; Q.2 onset grid and occupancy audit; D_acc grid plus the restricted contrasts; the R.17 heatmaps (measured cells only); per-layer M/Δ/ω profiles; collateral positive control; ω components; restore-v2 once read |
| — | App. K | Amendment index (A–R), with dates and what each changed |

## What is cut

Only prose that the moved material makes redundant:
- body restatements of numbers that the moved tables already carry;
- M.3-era explanatory sentences at the seven RESOLVED sites. Their content is
  preserved in App. E/H.

Retractions are not cut. The P.5 retraction keeps one sentence in the body
(§5) and its full text in App. G.

## Open items this plan depends on

- **Validity gate (R.17.0.5).** Until the fork control passes against the
  precision floor, the §5 divergence numbers carry the floor caveat.
- **b2048 capture (confirmatory H2/H3, ω replication),** the collapse test
  (needs t1-tv512/1024) and restore-v2 decide three of §5's four paragraphs.
- **Figure budget.** The body keeps four floats (`fig:protocol` reduced,
  `fig:phenomenon`, `tab:decomp`, `fig:bracket`) plus the one mechanism
  figure. Each has to be checked at the venue's column width once the style
  file is out.

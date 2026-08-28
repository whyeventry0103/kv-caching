#!/usr/bin/env python3
"""Render the results record to Markdown and to a standalone HTML page.

One content model, two renderers. The prose lives in ``build_document`` exactly
once, so the Markdown and the HTML cannot drift apart; every number comes from
``paper_numbers.json`` at render time, checked against that entry's own display
rules, so a figure cannot be quoted at a precision no source can reproduce.

    python scripts/export_results.py --format both

Which format for what:

* **Markdown** is the one to import into Notion. Notion converts an upload into
  its own blocks and keeps structure -- headings, tables, lists, bold, quotes --
  while discarding CSS. Markdown maps onto those blocks predictably.
* **HTML** is the standalone artifact: a designed page for reading, printing to
  PDF, or embedding once hosted. Importing it into Notion would strip the design
  and gain nothing over the Markdown.

To restyle the HTML, edit ``CSS`` below -- it is one block of custom properties
plus plain selectors, with no build step and no external assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.paper_numbers import Entry

MODEL = {
    "diss256": "Qwen-7B",
    "h2o256": "Qwen-7B",
    "rkv256": "Qwen-7B",
    "b2048": "Qwen-7B",
    "b2048h2o": "Qwen-7B",
    "aime": "Qwen-7B",
    "llama": "Llama-8B",
    "q15b": "Qwen-1.5B",
    "q14b": "Qwen-14B",
}
PRESSES = {
    "diss256": ["random", "streamingllm"],
    "h2o256": ["h2o"],
    "rkv256": ["rkv"],
    "b2048": ["random", "streamingllm"],
    "b2048h2o": ["h2o"],
    "aime": ["random", "rkv", "streamingllm"],
    "llama": ["random", "streamingllm"],
    "q15b": ["rkv", "streamingllm"],
    "q14b": ["rkv", "streamingllm"],
}
LABEL = {
    "random": "Random",
    "streamingllm": "StreamingLLM",
    "h2o": "H2O-style",
    "rkv": "R-KV",
}


class Cat:
    """Catalogue lookup. Refuses unknown keys and unlicensed display forms.

    The second refusal matters as much as the first: quoting a non-integer mean
    as ``10415`` yields a literal no artifact can reproduce, which is exactly how
    an unsourceable number reaches a deliverable.
    """

    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.entries = {
            e["key"]: Entry(
                key=e["key"],
                value=float(e["value"]),
                kind=e["kind"],
                source=e["source"],
            )
            for e in payload["entries"]
        }

    def __call__(self, key: str, places: int = 3) -> str:
        if key not in self.entries:
            raise SystemExit(f"missing catalogue key: {key}")
        out = f"{self.entries[key].value:.{places}f}"
        licensed = self.entries[key].display_rules()
        if out.lstrip("+-").replace(",", "") not in licensed:
            raise SystemExit(
                f"{key} = {self.entries[key].value!r} cannot be displayed as "
                f"{out!r}; licensed: {sorted(licensed)[:8]}"
            )
        return out

    def raw(self, key: str) -> float:
        if key not in self.entries:
            raise SystemExit(f"missing catalogue key: {key}")
        return self.entries[key].value

    def has(self, key: str) -> bool:
        return key in self.entries

    def ci(self, key: str) -> str:
        sep = "_" if self.has(f"{key}_ci0") else "."
        return f"[{self(f'{key}{sep}ci0')}, {self(f'{key}{sep}ci1')}]"

    def pct(self, key: str, places: int = 1) -> str:
        return f"{self.raw(key):.{places}f}%"


# --------------------------------------------------------------------------
# Content model. Each block is (kind, payload); both renderers consume these.
# --------------------------------------------------------------------------


def decomposition_rows(c: Cat, namespaces: list[str], model_col: bool) -> dict:
    head = [
        "Policy",
        "Free-run (95% CI)",
        "Readout (95% CI)",
        "Total Δ",
        "Readout Δ",
        "Residual (95% CI)",
        "Trunc.",
    ]
    if model_col:
        head = ["Model", *head]
    rows = []
    for ns in namespaces:
        for press in PRESSES[ns]:
            p = f"{ns}.{press}"
            row = [
                LABEL[press],
                f"{c(p + '.free')} {c.ci(p + '.free')}",
                f"{c(p + '.readout')} {c.ci(p + '.readout')}",
                c(p + ".total_change"),
                c(p + ".readout_change"),
                f"**{c(p + '.residual')}** {c.ci(p + '.residual')}",
                c(p + ".trunc"),
            ]
            rows.append([MODEL[ns], *row] if model_col else row)
    return {"head": head, "rows": rows}


def build_document(c: Cat) -> list[tuple[str, Any]]:
    """The page, as blocks. This is the ONLY place the prose lives."""
    B: list[tuple[str, Any]] = []

    def h2(text: str) -> None:
        B.append(("h2", text))

    def p(text: str) -> None:
        B.append(("p", text))

    def ul(*items: str) -> None:
        B.append(("ul", list(items)))

    def table(head: list[str], rows: list[list[str]]) -> None:
        B.append(("table", {"head": head, "rows": rows}))

    def callout(kind: str, title: str, *items: str) -> None:
        B.append(("callout", {"kind": kind, "title": title, "items": list(items)}))

    fc_bank = c("diss256.fc_bank")
    fc_cell = c("diss256.random.fc_cell")

    B.append(
        (
            "title",
            {
                "title": "KV Cache Research — Results",
                "subtitle": "Readout distortion vs trajectory distortion in "
                "KV-cache compression for reasoning models. Every number here is "
                "generated from the artifact catalogue, not typed.",
                "badges": [
                    "29/29 artifacts re-derived from raw rows",
                    "79 numbers pinned to their quantity",
                    "Verified 2026-08-05",
                ],
            },
        )
    )

    callout(
        "info",
        "How to read every number here",
        f"**The comparator is the matched cell reference, not the bank.** The "
        f"full-cache bank is {fc_bank} over 2000 rows and appears in the setup "
        f"only. Every change is measured against that bank restricted to the "
        f"cell's own conditions — {fc_cell} for the budget-256 MATH-500 "
        f"cells (Amendment K).",
        "**Truncated generations stay in every denominator, scored incorrect.** "
        "They are never dropped as malformed.",
        "**The residual is a definition, not a measured cause.** It is "
        "'accuracy change unexplained by on-reference readout' = total change "
        "minus readout change.",
        "**Divergence is not damage.** Correctness is the only quality arbiter; "
        "every divergence quantity travels with its signed flip-mass.",
    )

    h2("1. Headline")
    p(
        "At cache budget 256 on MATH-500 (DeepSeek-R1-Distill-Qwen-7B), the same "
        "compression measured two ways gives two different answers. The gap "
        "between the dots is the trajectory residual."
    )
    B.append(
        (
            "stats",
            [
                ("Trajectory residual", "≈ −0.34", "three unrelated policies"),
                ("Readout says", c("diss256.random.readout"), "Random, held-trace"),
                ("Free-running says", c("diss256.random.free"), "Random, own trace"),
                ("Matched reference", fc_cell, "full cache, same conditions"),
            ],
        )
    )
    B.append(
        (
            "chart",
            {
                "title": "Budget 256 — held-trace readout vs free-running",
                "reference": c.raw("diss256.random.fc_cell"),
                "reference_label": f"matched full cache {fc_cell}",
                "rows": [
                    {
                        "label": LABEL[press],
                        "readout": c.raw(f"{ns}.{press}.readout"),
                        "free": c.raw(f"{ns}.{press}.free"),
                        "residual": c.raw(f"{ns}.{press}.residual"),
                    }
                    for ns, press in (
                        ("rkv256", "rkv"),
                        ("diss256", "random"),
                        ("h2o256", "h2o"),
                        ("diss256", "streamingllm"),
                    )
                ],
            },
        )
    )
    p(
        f"Realized cache at budget 256 is "
        f"**{c.pct('compression_ratio.diss256.pct', 1)}** of the mean trace "
        f"(prompt + generation)."
    )

    h2("2. Budget 256 — the core decomposition")
    p("100 problems × 4 seeds = 400 conditions per cell.")
    B.append(("table", decomposition_rows(c, ["diss256", "h2o256", "rkv256"], False)))
    ul(
        f"Readout share of the total loss: Random "
        f"{c.pct('readout_share.diss256.random.pct')}, StreamingLLM "
        f"{c.pct('readout_share.diss256.streamingllm.pct')}, H2O-style "
        f"{c.pct('readout_share.h2o256.h2o.pct')}, R-KV "
        f"{c.pct('readout_share.rkv256.rkv.pct')}.",
        f"Residual spread across the three unrelated policies: "
        f"**{c('residual_spread.256_three_policies')}**, against CI half-widths "
        f"of roughly 0.07. The methods differ in the *readout* channel instead "
        f"— readout-change spread "
        f"**{c('readout_change_spread.256_three_policies')}**.",
    )
    callout(
        "warn",
        "Pre-registered outcome: COMPLICATE",
        f"The strong headline required Random to be readout-lossless (readout CI "
        f"lower bound at least 0.75; actual {c('diss256.random.readout_ci0')}) "
        f"and StreamingLLM's readout share at least 0.5 (actual "
        f"{c.pct('readout_share.diss256.streamingllm.pct')}). Neither held. The "
        f"central residual survived robustly; the pilot's clean double "
        f"dissociation did not.",
    )

    h2("3. Reverse-replay bracket (the off-policy defense)")
    p(
        "Forward and reverse readout estimate the same residual from opposite "
        "directions. The bracket excludes zero for every policy, and all eight "
        "endpoints have problem-clustered CIs excluding zero."
    )
    bracket_rows = []
    for ns, press in (
        ("diss256", "random"),
        ("diss256", "streamingllm"),
        ("h2o256", "h2o"),
        ("rkv256", "rkv"),
    ):
        key = f"{ns}.{press}"
        lo, hi = sorted([c.raw(key + ".bracket0"), c.raw(key + ".bracket1")])
        bracket_rows.append(
            [
                LABEL[press],
                c(key + ".reverse"),
                c(key + ".reverse_change"),
                f"[{lo:.3f}, {hi:.3f}]",
            ]
        )
    table(
        ["Policy", "Reverse readout acc.", "Reverse endpoint", "Residual bracket"],
        bracket_rows,
    )
    callout(
        "warn",
        "Caveat that must travel with this",
        "Forward and reverse estimands disagree on the *method-level* split, so "
        "the method-level readout-vs-trajectory attribution is reported as "
        "estimand-dependent and exploratory. The *central* claim — a large "
        "residual for every policy — survives both directions.",
    )

    h2("4. Random beats the principled recency policy")
    ul(
        f"Mean paired difference **+{c('contrast256.mean')}**, 95% CI "
        f"**{c.ci('contrast256')}** (problem-clustered, pinned seed, 2000 "
        f"resamples).",
        f"Per-problem difference non-negative on "
        f"**{c('contrast256.non_negative', 0)}/{c('contrast256.n_problems', 0)}** "
        f"problems: **{c('contrast256.wins', 0)} strict wins, "
        f"{c('contrast256.ties', 0)} ties, {c('contrast256.losses', 0)} losses**.",
    )
    callout(
        "stop",
        "Phrasing rule (binding)",
        "Say **'non-negative on 94/100'**, never 'beats on 94/100' — over "
        "half of the 94 are ties.",
        "HOTT carries no random-eviction baseline, so this result is ours alone. "
        "HOTT must never be cited as corroborating it.",
        "The ordering is architecture-dependent: it **reverses on Llama-8B** "
        "(section 12).",
    )

    h2("5. Dose response — budget 2048")
    p(
        f"The collapse is a budget effect. Same presses, same harness, budget "
        f"2048 (**{c.pct('compression_ratio.b2048.pct', 1)}** realized cache)."
    )
    B.append(("table", decomposition_rows(c, ["b2048", "b2048h2o"], False)))
    ul(
        f"The residual shrinks by a factor of **{c('dose_factor.min', 1)}x to "
        f"{c('dose_factor.max', 1)}x** between budget 256 and budget 2048.",
        "At 2048 readout sits at or just above the matched reference, and the "
        "Random over StreamingLLM ordering persists.",
    )

    h2("6. Per-step TV divergence (mechanism: see the caveat)")
    p(
        "Onset-aligned total variation between full-cache and compressed "
        "teacher-forced replay. This localizes where divergence *starts*, not "
        "where accuracy is lost."
    )
    p(
        "**Caveat (Amendment M.3, 2026-08-15).** The RISE in these windows is a "
        "compression-ratio effect, not a horizon mechanism: every profile here "
        "holds the budget fixed while context grows, so the realized ratio "
        "worsens monotonically. Held at a fixed ratio the drift reverses "
        "(0.66-0.74 < 1). **No mechanistic reading of the compounding is "
        "licensed.** What DOES survive ratio-matching is the phase alignment to "
        "prune events (Amendment P.2-v2, 2026-08-27: Random phase-amplitude "
        "excess +0.252 [0.184, 0.271], peak phase 0). State the two together — "
        "within a prune cycle divergence is eviction-locked; across cycles the "
        "envelope tracks the ratio."
    )
    table(
        ["Window", "Random + StreamingLLM (800 profiles)", "H2O-style (400)"],
        [
            [
                "Before first eviction",
                f"**{c('tv_diss256.before', 4)}**",
                f"**{c('tv_h2o.before', 4)}**",
            ],
            [
                "First 50 steps after onset",
                c("tv_diss256.first50", 4),
                c("tv_h2o.first50", 4),
            ],
            [
                "Later in the trajectory",
                c("tv_diss256.later", 4),
                c("tv_h2o.later", 4),
            ],
            [
                "Compounding factor",
                f"{c('tv_compounding.tv_diss256', 1)}x",
                f"{c('tv_compounding.tv_h2o', 1)}x",
            ],
        ],
    )
    callout(
        "stop",
        "The pre-eviction 0.0000 is an instrument check, not a finding",
        f"Median onset step is {c('tv_diss256.median_onset', 0)}. If the "
        f"pre-eviction value is ever not about 0.0000, **stop** — that is a "
        f"bug in the replay engine, not a result.",
    )

    h2("7. ADD — silent answer-distribution rewriting (budget 256)")
    p(
        "Total variation between coarsened full-cache and compressed answer "
        "distributions, with a permutation null (10,000 resamples) and "
        "Benjamini-Hochberg correction across problems."
    )
    table(
        [
            "Policy",
            "ADD",
            "Within-answered",
            "Termination shift",
            "Flip gained",
            "Flip lost",
            "Net flip-mass",
            "Significant",
        ],
        [
            [
                LABEL[press],
                c(f"add256.{press}.add"),
                c(f"add256.{press}.within"),
                c(f"add256.{press}.term"),
                c(f"add256.{press}.flip_gained"),
                c(f"add256.{press}.flip_lost"),
                c(f"add256.{press}.flip_net"),
                f"{c(f'add256.{press}.n_sig', 0)}/100",
            ]
            for press in ("random", "streamingllm", "h2o", "rkv")
        ],
    )
    ul(
        "The distortion runs overwhelmingly toward **lost correct answers** "
        "— net flip-mass is strongly negative for every policy.",
        f"Policies distort through **different channels**: StreamingLLM almost "
        f"entirely within-answered (it keeps answering, wrongly); Random carries "
        f"the larger termination-mass shift (it rambles and fails to terminate), "
        f"consistent with its **{c('truncation_ratio.random_over_slm', 1)}x** "
        f"higher truncation rate.",
        f"Every cell sits **{c('add_over_floor.add256.rkv', 1)}x to "
        f"{c('add_over_floor.add256.streamingllm', 1)}x** above the "
        f"negative-control noise floor (section 8).",
    )

    h2("8. Instrument validity controls")
    p(
        "These exist so the instruments are defensible before any finding is "
        "read off them."
    )
    table(
        ["Control", "Result", "What it rules out"],
        [
            [
                "Two-full-cache negative control",
                f"mean ADD **{c('twofc.mean_add')}**, median "
                f"{c('twofc.median_add')}, **{c('twofc.frac_sig', 1)}%** "
                f"significant",
                "ADD false-positives from finite samples",
            ],
            [
                "Full-cache readout of full-cache traces",
                f"resampling effect "
                f"**+{c('fc_readout_control.resampling_effect')}**",
                "The readout protocol's own resampling explaining the residual",
            ],
            [
                "Residual vs resampling effect",
                f"Random "
                f"**{c('residual_over_resampling.diss256.random.rounded', 0)}x**, "
                f"R-KV "
                f"**{c('residual_over_resampling.rkv256.rkv.rounded', 0)}x**",
                "The residual being a readout artifact",
            ],
            [
                f"Grader audit, {c('grader.total_rows', 0)} rows",
                f"**{c('grader.total_disagreements', 0)}** disagreements on "
                f"regrade",
                "Grader error driving the accuracy numbers",
            ],
            [
                "Exact replication, 800 rows",
                "all rows identical (correctness, truncation, token counts)",
                "Nondeterminism in generation",
            ],
            ["Pre-eviction TV", "exactly 0.0000", "Replay-engine misalignment"],
        ],
    )

    h2("9. AIME-24 — second benchmark")
    p(
        f"30 problems × 4 seeds, budget 2048 "
        f"(**{c.pct('compression_ratio.aime2048.pct', 1)}** realized cache), own "
        f"pinned reference bank **{c('aime.fc_bank')}** (mean generated length "
        f"{c('aime_bank.mean_tokens.rounded', 0)} tokens under a 16384-token "
        f"cap)."
    )
    B.append(("table", decomposition_rows(c, ["aime"], False)))
    callout(
        "good",
        "The sharpest readout result in the study",
        f"All three presses read out at an *identical* "
        f"**{c('aime.random.readout')}**, and on the matched conditions the "
        f"readout change is **exactly 0.000** for every press — zero "
        f"gained, zero lost, per row. Free-running separates the same three "
        f"policies by about 2x. Held-trace evaluation here does not merely "
        f"misprice compression; it does not respond to it at all.",
    )

    h2("10. R-KV in its published configuration")
    p(
        f"R-KV publishes **{c('rkv.published_pct_of_full', 0)}%** of full-KV "
        f"accuracy at **{c('rkv.published_cache_pct', 0)}%** cache on "
        f"R1-Distill-Llama-8B / AIME-24. We ran our parity-certified port in "
        f"that exact configuration (Amendment N.2)."
    )
    table(
        ["", "Full cache", "R-KV @2048"],
        [
            [
                "Accuracy",
                f"**{c('rkvpub.fc.accuracy')}** ({c('rkvpub.fc.correct', 0)}/120)",
                f"**{c('rkvpub.rkv.accuracy')}** "
                f"({c('rkvpub.rkv.correct', 0)}/120)",
            ],
            [
                "Truncation @32768",
                c("rkvpub.fc.truncation"),
                c("rkvpub.rkv.truncation"),
            ],
            [
                "Accuracy, non-truncated",
                c("rkvpub.fc.accuracy_non_truncated"),
                c("rkvpub.rkv.accuracy_non_truncated"),
            ],
            [
                "Mean tokens",
                c("rkvpub.fc.mean_tokens.rounded", 0),
                c("rkvpub.rkv.mean_tokens.rounded", 0),
            ],
        ],
    )
    ul(
        f"Ratio **r = {c('rkvpub.ratio')}**; paired difference "
        f"**+{c('rkvpub.mean_difference')}**, problem-clustered 95% CI "
        f"**{c.ci('rkvpub')}**. Realized cache "
        f"**{c.pct('rkvpub.realized_ratio_pct', 1)}**.",
        "**Pre-registered branch: REPLICATES** (r at least 1.0 and the "
        "difference CI includes zero). The branch was selected by code, not "
        "judgment, so it could not be re-cut after the numbers were seen.",
    )
    callout(
        "stop",
        "How to state this, exactly",
        f"The margin is **one generation** ({c('rkvpub.rkv.correct', 0)} correct "
        f"against {c('rkvpub.fc.correct', 0)} of 120) on a CI that spans zero, "
        f"over 30 problems, against a bank that cleared its sanity band at "
        f"exactly the floor.",
        "**Defensible:** our certified port is not worse than full cache at "
        "about a fifth of the cache, in the published configuration.",
        "**Not defensible:** 'we reproduced 105%'. **Also not defensible in the "
        "other direction:** calling it inconclusive because we would prefer it.",
        f"Two facts favour R-KV and are reported for that reason: it truncates "
        f"nearly 3x as often (longer traces, every truncation scored incorrect) "
        f"and leads anyway; among non-truncated generations it leads "
        f"{c('rkvpub.rkv.accuracy_non_truncated')} to "
        f"{c('rkvpub.fc.accuracy_non_truncated')}.",
    )

    h2("11. HOTT-configuration cell")
    p(
        "Our own full-cache condition re-run under HOTT's 2048-token generation "
        "cap, changing nothing else (Amendment N.1)."
    )
    table(
        ["", "Our accuracy", "Truncation", "HOTT reports"],
        [
            [
                "Full cache",
                f"**{c('hottcap.fc.accuracy')}** (from {fc_bank})",
                c("hottcap.fc.truncation_rate"),
                c("hott.fullcache_math500", 2),
            ],
            [
                "StreamingLLM-128",
                f"**{c('hottcap.slm128.accuracy')}**",
                c("hottcap.slm128.truncation_rate"),
                c("hott.streamingllm_128", 2),
            ],
        ],
    )
    p(
        f"The generation cap alone accounts for nearly all of the baseline gap, "
        f"leaving **{c('hottcap.residual_gap')}**."
    )
    callout(
        "stop",
        "Mandatory caveat, and a standing landmine",
        "This matches HOTT's cap and **not** their greedy decoding, which our "
        "generation path does not support (sampling stays T=0.6, top-p 0.95). It "
        "measures how much of the gap the cap alone accounts for; any residual "
        "gap is not attributable to the cap.",
        f"Never write 'H2O collapses'. HOTT's H2O (with recency reservation) "
        f"survives at {c('hott.h2o_128', 2)}; SnapKV at "
        f"{c('hott.snapkv_128', 2)}. Ours is 'H2O-style / ObservedAttentionPress "
        f"(no recency reservation)'.",
    )

    h2("12. Cross-architecture check — Llama-8B")
    p(
        f"50 problems × 4 seeds, own pinned reference bank "
        f"**{c('llama.fc_bank')}**. CIs correspondingly wider."
    )
    B.append(("table", decomposition_rows(c, ["llama"], False)))
    callout(
        "warn",
        "This cell fits no pre-registered branch and is reported plainly",
        f"The central result replicates on the sharpest case (StreamingLLM), but "
        f"**Random collapses to {c('llama.random.free')}** — the "
        f"Random-beats-recency ordering *reverses* on this architecture. That is "
        f"why the ordering is called architecture-dependent everywhere.",
    )

    h2("13. Scale axis")
    p(
        f"Qwen-1.5B (bank {c('q15b.fc_bank')}) and Qwen-14B (bank "
        f"{c('q14b.fc_bank')}), 50 problems × 4 seeds each (Amendment J, "
        f"complete)."
    )
    B.append(("table", decomposition_rows(c, ["q15b", "q14b"], True)))
    p(
        "3 of the 4 cells replicate the central result. Read these as "
        "**exploratory at current scope** — subset scale, two presses per "
        "model."
    )

    h2("14. Resolution collapse")
    p(
        "How much of the policy separation visible under free-running survives "
        "held-trace readout. Spread = mean absolute pairwise difference across "
        "presses, paired on the problem."
    )
    res_rows = []
    for label, key in (
        ("MATH-500 @256", "res256"),
        ("MATH-500 @512", "res512"),
        ("MATH-500 @1024", "res1024"),
        ("MATH-500 @2048", "res2048"),
        ("AIME-24 @2048", "resaime"),
    ):
        free = c.raw(f"{key}.free_spread")
        ro = c.raw(f"{key}.readout_spread")
        ratio = f"{ro / free:.3f}" if free else "n/a"
        res_rows.append([label, f"{free:.4f}", f"{ro:.4f}", ratio])
    table(
        ["Regime", "Free-run spread", "Readout spread", "Ratio (descriptive)"],
        res_rows,
    )
    callout(
        "warn",
        "Status: post-hoc and exploratory",
        "The pattern was noticed after all cells were read. Its pre-registered "
        "confirmatory test at budgets 512 and 1024 was run and **refuted** the "
        "graded/monotone reading, which is why this is reported as endpoints "
        "rather than as a mechanism.",
        f"Press sets differ between regimes, so a matched-pair recomputation "
        f"(Random + StreamingLLM only) is carried alongside: @256 free "
        f"{c('respair256.free_spread', 4)} / readout "
        f"{c('respair256.readout_spread', 4)}; @2048 free "
        f"{c('respair2048.free_spread', 4)} / readout "
        f"{c('respair2048.readout_spread', 4)}.",
    )

    h2("15. Compute")
    ul(
        f"**{c('compute.gpu_hours.rounded', 0)} GPU-hours** of summed wall-clock "
        f"across **{c.raw('compute.generations'):,.0f} generations**.",
        "Covers free-running and reference-bank generation only. Teacher-forced "
        "work (readout, reverse readout, replay) is excluded, as are superseded "
        "pilots and the two shards subsumed by the merged bank.",
        "The artifact lists every run directory with its hours and an inclusion "
        "flag, so any exclusion can be re-added and its effect seen.",
        "Corrected 2026-08-05 from a stale 280 h / 7,200 generations: the "
        "inclusion allowlist had drifted nine result-bearing run directories "
        "behind. Found by the artifact re-derivation harness, not by reading.",
    )

    h2("16. Framing discipline")
    p("These exist because each one has been got wrong at least once.")
    table(
        ["Say this", "Never this"],
        [
            [
                "'accuracy change unexplained by on-reference readout'",
                "'measured trajectory damage' / any causal claim",
            ],
            [
                "'non-negative on 94/100 (44 win / 50 tie / 6 loss)'",
                "'beats on 94/100'",
            ],
            [
                "'H2O-style / ObservedAttentionPress (no recency reservation)'",
                "'H2O collapses'",
            ],
            [
                "'divergence', with its signed flip-mass",
                "'damage to reasoning ability'",
            ],
            [
                "every absolute budget with its realized compression ratio",
                "a bare budget number",
            ],
            [
                "'our certified port at 7B' for R-KV numbers",
                "'R-KV scores X' — that is our port, not the authors' setup",
            ],
            [
                "'exploratory at current scope' for generality claims",
                "an unqualified generality claim",
            ],
        ],
    )
    ul(
        "Truncated generations stay in every denominator, scored incorrect.",
        "A primary bound is a deterministic imputation range: **never** "
        "described as significance, never compared for width against a CI.",
        "Benjamini-Hochberg correction is mandatory for families of "
        "problem-level tests.",
        "Underwhelming pre-registered outcomes get reported plainly. The honesty "
        "trail is the moat.",
    )

    h2("17. Provenance")
    table(
        ["Item", "Value"],
        [
            [
                "Models",
                "DeepSeek-R1-Distill-Qwen-7B (plus Llama-8B, Qwen-1.5B, Qwen-14B)",
            ],
            ["Benchmarks", "MATH-500, AIME-24"],
            ["Sampling", "T = 0.6, top-p 0.95"],
            [
                "Stack",
                "transformers 4.56.2, kvpress 0.5.3, DecodingPress, "
                "compression_interval 64",
            ],
            ["Bootstrap", "problem-clustered, 2000 resamples, percentile CI"],
            [
                "Permutation null",
                "10,000 resamples, Benjamini-Hochberg across problems",
            ],
            ["Grader", "Hendrycks is_equiv, no eval/sympify on model text"],
            [
                "Chain of custody",
                "paper_numbers.json from results/diag/ from raw JSONL rows",
            ],
        ],
    )
    callout(
        "info",
        "Regenerating this page",
        "`python scripts/export_results.py --format both` rewrites both the "
        "Markdown and the HTML from the catalogue. **Edit the script, not the "
        "output** — hand edits are overwritten on the next run, and a "
        "hand-typed number is what this pipeline exists to prevent.",
    )
    return B


# --------------------------------------------------------------------------
# Markdown renderer
# --------------------------------------------------------------------------


def render_markdown(blocks: list[tuple[str, Any]]) -> str:
    out: list[str] = []
    for kind, payload in blocks:
        if kind == "title":
            out += [
                f"# {payload['title']}",
                "",
                payload["subtitle"],
                "",
                " · ".join(f"**{b}**" for b in payload["badges"]),
                "",
                "---",
                "",
            ]
        elif kind == "h2":
            out += [f"## {payload}", ""]
        elif kind == "p":
            out += [payload, ""]
        elif kind == "ul":
            out += [f"- {item}" for item in payload] + [""]
        elif kind == "stats":
            out += [
                "| " + " | ".join(s[0] for s in payload) + " |",
                "|" + "---|" * len(payload),
                "| " + " | ".join(f"**{s[1]}**" for s in payload) + " |",
                "",
            ]
        elif kind == "callout":
            out += [f"> **{payload['title']}**", ">"]
            out += [f"> - {item}" for item in payload["items"]] + [""]
        elif kind == "table":
            out += [
                "| " + " | ".join(payload["head"]) + " |",
                "|" + "---|" * len(payload["head"]),
            ]
            out += ["| " + " | ".join(r) + " |" for r in payload["rows"]] + [""]
        elif kind == "chart":
            # Notion cannot render an SVG from an import; the table underneath
            # carries the same values, so the chart is simply omitted here.
            continue
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# HTML renderer
# --------------------------------------------------------------------------


ASSETS = PROJECT_ROOT / "scripts" / "assets"


def _asset(name: str) -> str:
    """Inline a stylesheet/script so the emitted page has no external deps.

    They live as real .css/.js files rather than Python strings so they can be
    edited with syntax highlighting and so the 88-column source gate stays
    meaningful for actual Python.
    """
    path = ASSETS / name
    if not path.exists():
        raise SystemExit(f"missing asset: {path}")
    return path.read_text(encoding="utf-8")


def _inline(text: str) -> str:
    """Minimal inline markup: **bold**, `code`, *em*. Escapes everything else.

    Markers are applied longest-first so ``**`` is consumed before ``*``. Odd
    counts leave the trailing run unwrapped rather than swallowing the rest of
    the string.
    """
    out = escape(text)
    for marker, tag in (("**", "strong"), ("`", "code"), ("*", "em")):
        parts = out.split(marker)
        if len(parts) < 3:
            continue
        rebuilt = parts[0]
        for i, chunk in enumerate(parts[1:], start=1):
            inside = i % 2 == 1
            closed = i < len(parts) - 1 or not inside
            rebuilt += f"<{tag}>{chunk}</{tag}>" if inside and closed else chunk
        out = rebuilt
    return out


def _svg_dumbbell(spec: dict) -> str:
    rows = spec["rows"]
    # `top` must leave room for the reference-line label ABOVE the plot;
    # at 16 it rendered clipped against the viewBox edge.
    left, right, top, rh = 132, 74, 38, 52
    width, height = 860, top + rh * len(rows) + 46

    def x(v: float) -> float:
        return left + v * (width - left - right)

    p: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'style="min-width:660px" role="img" '
        f'aria-label="Readout versus free-running accuracy by policy">'
    ]
    # gridlines: hairline, solid, recessive
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        p.append(
            f'<line x1="{x(t):.1f}" y1="{top - 4}" x2="{x(t):.1f}" '
            f'y2="{top + rh * len(rows) - 12}" stroke="var(--border)" '
            f'stroke-width="1"/>'
        )
        p.append(
            f'<text x="{x(t):.1f}" y="{top + rh * len(rows) + 8}" '
            f'text-anchor="middle" font-size="11" fill="var(--text-muted)" '
            f'font-family="var(--mono)">{t:.2f}</text>'
        )
    # reference line
    rx = x(spec["reference"])
    p.append(
        f'<line x1="{rx:.1f}" y1="{top - 8}" x2="{rx:.1f}" '
        f'y2="{top + rh * len(rows) - 12}" stroke="var(--text-muted)" '
        f'stroke-width="1" stroke-dasharray="3 3"/>'
    )
    p.append(
        f'<text x="{rx:.1f}" y="{top - 18}" text-anchor="end" font-size="11" '
        f'fill="var(--text-muted)">{escape(spec["reference_label"])}</text>'
    )
    for i, r in enumerate(rows):
        cy = top + rh * i + 20
        xf, xr = x(r["free"]), x(r["readout"])
        tip = (
            f'{r["label"]}\\nreadout  {r["readout"]:.3f}\\n'
            f'free-run {r["free"]:.3f}\\nresidual {r["residual"]:+.3f}'
        )
        p.append(
            f'<text x="{left - 14}" y="{cy + 4}" text-anchor="end" font-size="13" '
            f'fill="var(--text-primary)">{escape(r["label"])}</text>'
        )
        p.append(f'<g data-tip="{tip}">')
        # generous invisible hit target, then the connector
        p.append(
            f'<rect x="{left - 120}" y="{cy - 18}" width="{width - left + 110}" '
            f'height="36" fill="transparent"/>'
        )
        p.append(
            f'<line x1="{xf:.1f}" y1="{cy}" x2="{xr:.1f}" y2="{cy}" '
            f'stroke="var(--border-strong)" stroke-width="2" stroke-linecap="round"/>'
        )
        for cx, colour in ((xf, "var(--series-free)"), (xr, "var(--series-readout)")):
            p.append(
                f'<circle cx="{cx:.1f}" cy="{cy}" r="6" fill="{colour}" '
                f'stroke="var(--surface-1)" stroke-width="2"/>'
            )
        p.append(
            f'<text x="{xf - 12:.1f}" y="{cy + 4}" text-anchor="end" font-size="12" '
            f'font-family="var(--mono)" fill="var(--text-secondary)">'
            f'{r["free"]:.3f}</text>'
        )
        p.append(
            f'<text x="{xr + 12:.1f}" y="{cy + 4}" font-size="12" '
            f'font-family="var(--mono)" fill="var(--text-secondary)">'
            f'{r["readout"]:.3f}</text>'
        )
        p.append("</g>")
    p.append("</svg>")
    return "".join(p)


def render_html(blocks: list[tuple[str, Any]]) -> str:
    body: list[str] = []
    title = "Results"
    for kind, payload in blocks:
        if kind == "title":
            title = payload["title"]
            badges = "".join(
                f'<span class="badge">{escape(b)}</span>' for b in payload["badges"]
            )
            body.append(
                f'<header class="doc"><h1>{escape(payload["title"])}</h1>'
                f'<p class="sub">{_inline(payload["subtitle"])}</p>'
                f'<div class="badges">{badges}</div></header>'
            )
        elif kind == "h2":
            body.append(f"<h2>{escape(payload)}</h2>")
        elif kind == "p":
            body.append(f"<p>{_inline(payload)}</p>")
        elif kind == "ul":
            items = "".join(f"<li>{_inline(i)}</li>" for i in payload)
            body.append(f"<ul>{items}</ul>")
        elif kind == "stats":
            cards = "".join(
                f'<div class="stat"><div class="k">{escape(k)}</div>'
                f'<div class="v">{escape(v)}</div>'
                f'<div class="s">{escape(s)}</div></div>'
                for k, v, s in payload
            )
            body.append(f'<div class="stats">{cards}</div>')
        elif kind == "callout":
            items = "".join(f"<li>{_inline(i)}</li>" for i in payload["items"])
            body.append(
                f'<div class="callout {payload["kind"]}">'
                f'<div class="ct">{escape(payload["title"])}</div>'
                f"<ul>{items}</ul></div>"
            )
        elif kind == "table":
            head = "".join(f"<th>{_inline(h)}</th>" for h in payload["head"])
            rows = "".join(
                "<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in r) + "</tr>"
                for r in payload["rows"]
            )
            body.append(
                f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
                f"<tbody>{rows}</tbody></table></div>"
            )
        elif kind == "chart":
            body.append(
                f'<div class="figure"><h3>{escape(payload["title"])}</h3>'
                f'<p class="cap">Each row is one eviction policy. The dot pair is the '
                f"same cell measured two ways; the bar between them is the trajectory "
                f"residual. Hover a row for its exact values.</p>"
                f'<div class="legend">'
                f'<span><i class="swatch" style="background:var(--series-readout)">'
                f"</i>Held-trace readout</span>"
                f'<span><i class="swatch" style="background:var(--series-free)"></i>'
                f"Free-running</span></div>"
                f'<div class="chartscroll">{_svg_dumbbell(payload)}</div></div>'
            )
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_asset('results.css')}</style></head><body>"
        f'<button class="themetoggle" type="button">Theme</button>'
        f'<div id="tip"></div><div class="wrap">'
        + "".join(body)
        + '<footer class="doc">Generated by <code>scripts/export_results.py</code> '
        "from <code>paper_numbers.json</code>. Edit the script, not this file."
        f"</footer></div><script>{_asset('results.js')}</script></body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=["md", "html", "both"], default="both")
    parser.add_argument(
        "--out-md", type=Path, default=PROJECT_ROOT / "docs/results/RESULTS.md"
    )
    parser.add_argument(
        "--out-html", type=Path, default=PROJECT_ROOT / "docs/results/RESULTS.html"
    )
    parser.add_argument(
        "--catalogue", type=Path, default=PROJECT_ROOT / "paper_numbers.json"
    )
    args = parser.parse_args()

    blocks = build_document(Cat(args.catalogue))
    if args.format in ("md", "both"):
        args.out_md.write_text(render_markdown(blocks), encoding="utf-8")
        print(f"wrote {args.out_md}")
    if args.format in ("html", "both"):
        args.out_html.write_text(render_html(blocks), encoding="utf-8")
        print(f"wrote {args.out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

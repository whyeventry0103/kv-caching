"""Property tests for the number gate: invariants, not instances.

This module has now had six distinct defect classes -- the bibliography label
width, the zero-place display inflation, its *replacement* which still let a
fractional value claim its integer rounding, four extractor false-positive
classes, coincidental key attribution, and missing exact-match precedence.

Every one was found by accident or by a later session reading the code, never by
its own tests, because the tests all encoded defects somebody had already
noticed. Regression tests cannot find the seventh. These assert the properties
the gate is supposed to have, over a generated space of values and documents, so
a violation surfaces without anyone having to think of it first.

`hypothesis` is deliberately not used: the CPU test gate is dependency-light on
purpose and adding a library to a pinned stack for this is not worth it. The
sampling here is explicit and seeded, so failures are reproducible.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from kvcache.paper_numbers import Entry, classify, extract_literals

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260729

# A value space spanning the shapes the catalogue actually holds: sub-unit
# accuracies and residuals, both signs, exact ties at the .3f boundary, ratios
# above 1, percentages, integer counts, and large token counts.
VALUE_SPACE = [
    0.0,
    0.885,
    -0.3367,
    0.4225,
    -0.0026,
    0.2475,
    0.8075,
    1.0,
    -1.0,
    6.7,
    10.153846,
    17.397,
    29.7848,
    49.2475,
    44.0,
    100.0,
    105.0,
    279.8658,
    6725.0,
    16384.0,
]


def _entries(values, prefix="k"):
    return [
        Entry(key=f"{prefix}{i}", value=v, kind="primary", source="artifact#/x")
        for i, v in enumerate(values)
    ]


def test_every_sourced_match_is_reproducible_from_a_named_rule():
    """No rule, no match. A display form that no declared rule produces must
    never lend provenance to anything -- that is exactly how the bibliography
    label and both zero-place defects passed."""
    catalogue = _entries(VALUE_SPACE)
    rules = {e.key: e.display_rules() for e in catalogue}
    text = " ".join(form for e in catalogue for form in sorted(e.display_rules()))
    literals = extract_literals(text, "t", strip=False)
    classify(literals, catalogue)
    for lit in literals:
        if lit.classification != "sourced":
            continue
        assert lit.matched_keys, f"{lit.raw} sourced with no key"
        for key in lit.matched_keys:
            raw = lit.raw.replace("{,}", "")
            assert raw in rules[key], (
                f"{lit.raw!r} was sourced to {key} but no declared display rule "
                f"of that entry produces it (rules: {sorted(rules[key])})"
            )


def test_no_entry_claims_a_form_its_rules_do_not_produce():
    """The inverse direction: the match index and the declared rules must be the
    same set, so one cannot drift from the other."""
    for value in VALUE_SPACE:
        entry = Entry(key="k", value=value, kind="primary", source="x")
        assert entry.displays() == set(entry.display_rules())


def test_exact_match_always_outranks_a_rounding_match():
    """Sampled over pairs where one entry equals the literal exactly and another
    only rounds onto it."""
    rng = random.Random(SEED)
    for _ in range(200):
        exact = round(rng.uniform(-1, 1), 4)
        # A neighbour that shares a rounded form but is not equal.
        neighbour = exact + rng.choice([1e-5, -1e-5, 4e-5, -4e-5])
        if neighbour == exact:
            continue
        catalogue = [
            Entry(key="exact", value=exact, kind="primary", source="x"),
            Entry(key="near", value=neighbour, kind="primary", source="x"),
        ]
        literal_text = f"{exact:.4f}"
        literals = extract_literals(literal_text, "t", strip=False)
        classify(literals, catalogue)
        if literals[0].classification != "sourced":
            continue
        assert "exact" in literals[0].matched_keys, (
            f"{literal_text} did not resolve to the exactly-equal entry; "
            f"got {literals[0].matched_keys}"
        )
        if "near" in literals[0].matched_keys:
            assert (
                catalogue[1].value == literals[0].value
            ), "a non-equal entry survived alongside an exact match"


def test_adding_an_unrelated_entry_never_changes_an_existing_classification():
    """THE property that would have caught coincidental attribution directly.

    Cataloguing a new, unrelated number must not silently re-attribute a literal
    that was already resolved -- and must never turn an unsourced literal into a
    sourced one by coincidence of rounding.
    """
    rng = random.Random(SEED)
    base = _entries(VALUE_SPACE, prefix="base")
    text = " ".join(f"{v:.3f}" for v in VALUE_SPACE) + " 0.777 0.999"
    before = extract_literals(text, "t", strip=False)
    classify(before, base)
    baseline = {(lit.line, lit.raw): lit.classification for lit in before}

    for trial in range(150):
        extra = Entry(
            key=f"extra{trial}",
            value=round(rng.uniform(-50, 50), 6),
            kind="primary",
            source="x",
        )
        after = extract_literals(text, "t", strip=False)
        classify(after, base + [extra])
        for lit in after:
            was = baseline[(lit.line, lit.raw)]
            if was == "sourced":
                assert (
                    lit.classification == "sourced"
                ), f"adding {extra.value} un-sourced {lit.raw!r}"
            elif was == "unsourced":
                # It may only become sourced if the new entry genuinely displays
                # as this literal under a declared rule.
                if lit.classification != "unsourced":
                    assert lit.raw.replace("{,}", "") in extra.display_rules(), (
                        f"adding {extra.value} made {lit.raw!r} sourced without "
                        "a declared display rule producing it"
                    )


def test_excluded_constructs_are_never_classified_at_all():
    """A literal inside an excluded construct must not reach classification --
    not merely be classified harmlessly. Four of the six defect classes were
    exactly this."""
    cases = [
        ("arXiv:2505.24133", "arXiv id"),
        ("2505.24133", "bare arXiv id"),
        ("Qwen-1.5B", "model scale"),
        ("14B", "model scale"),
        ("4M tokens", "token scale"),
        ("PG19", "identifier suffix"),
        ("§4.1", "section reference"),
        ("2026-07-17", "ISO date"),
        (
            "<!-- AUDIT:ARCHIVAL-BEGIN -->\n-0.343\n<!-- AUDIT:ARCHIVAL-END -->",
            "archival span",
        ),
    ]
    for text, label in cases:
        literals = extract_literals(text, "t", strip=False)
        assert literals == [], f"{label}: expected no literals, got {literals}"


def test_bibliography_body_is_excluded_in_the_latex_path():
    tex = (
        "\\begin{document}\nresidual $-0.343$\n"
        "\\begin{thebibliography}{99}\n"
        "\\bibitem{x} ICLR, 2024. arXiv:2309.17453.\n"
        "\\end{thebibliography}\n\\end{document}\n"
    )
    values = {lit.value for lit in extract_literals(tex, "t")}
    assert values == {-0.343}


def test_the_real_catalogue_has_no_two_entries_sharing_an_exact_value_and_rule():
    """A live-catalogue property: if two entries produce the same display AND
    both equal the literal exactly, the gate cannot say which one a sentence
    means. Not fatal -- but it must be visible, because that ambiguity is what
    the semantic prose audit has to resolve by reading."""
    path = PROJECT_ROOT / "paper_numbers.json"
    if not path.exists():  # pragma: no cover - catalogue always present in CI
        return
    entries = [
        Entry(key=e["key"], value=float(e["value"]), kind=e["kind"], source=e["source"])
        for e in json.loads(path.read_text(encoding="utf-8"))["entries"]
    ]
    by_exact: dict[str, list[str]] = {}
    for entry in entries:
        by_exact.setdefault(f"{entry.value!r}", []).append(entry.key)
    collisions = {v: k for v, k in by_exact.items() if len(k) > 1}
    # Recorded rather than forbidden: derived entries legitimately coincide with
    # their inputs (a bracket endpoint equals the residual it was built from).
    # The assertion is that the count does not silently grow.
    #
    # 120 -> 127 on 2026-09-23, examined rather than waved through. Cataloguing
    # the budget-512 and budget-1024 cells (Meeting-#3 Task 1) added 14 new
    # collision groups: 3 are within-cell identities by construction (e.g.
    # b512.random.primary_residual0 == b512.random.total_change -- the kind this
    # comment already calls legitimate) and 11 are coincidences between
    # unrelated quantities on the coarse k/400, k/392 accuracy lattice (e.g.
    # b1024.streamingllm.free == q15b.rkv.readout == 0.6875). None of those keys
    # is cited in the .tex, so none makes a sentence ambiguous; the claims gate
    # pins any that are cited later. Set to the exact new count, with no slack,
    # so the next growth trips this again and gets the same examination.
    #
    # 127 -> 129 on 2026-09-27 (R.17), examined. Cataloguing the exact-lag
    # eligibility profile added 2 groups, both small-integer coincidences
    # between unrelated counts: m3.el.eligible.n_problems == 8 ==
    # grader.tok.xc_only_correct, and m3.el.ineligible.n_problems == 17 ==
    # auc_tv_spread_pct.rounded. The 8 is cited in the .tex and is pinned to its
    # own key by a claim; the 17 is not cited.
    #
    # 129 -> 130 on 2026-09-27 (R.17 mechanism draft), examined: one new group,
    # m3.bridge.q_min == residual_contrast.rkv_vs_streamingllm.ci1 == 0.165, a
    # coincidence between unrelated quantities; the draft's 0.165 is pinned to
    # the bridge key by a claim. (m3.dacc.onset.random.unrestricted.cilo joined
    # the existing 0.03 group; that group is not new.)
    #
    # 130 -> 139 on 2026-10-05 (S.1 and protocol-audit entries), examined. Two
    # identities: s1...streamingllm_pp.residual_base.cilo == diss256.streamingllm
    # .residual_ci0 (-0.405; same rows, estimand and bootstrap) and
    # audit.matched.random_minus_h2o.metric.mean == m3.h2.M.spread (same
    # contrast on the same 25 problems). Seven coincidences on the k/100 and
    # k/400 accuracy lattices (0.19, 0.31, 0.14, 0.22, 0.315, 0.4675, -0.0783);
    # the abstract's 0.078 is pinned to its S.1 key by a claim. (-0.125, -0.27,
    # 0.0 and 0.898 gained members; those groups are not new.)
    #
    # 139 -> 164 on 2026-10-15 (S.1 budget-1024 forward reads, its decision read
    # and the realized-compression artifact), examined: 25 new groups.
    # Identities by construction (14): the decision read reproduces S.1's
    # residual mean/CI exactly (as at b256); the unprotected arm equals the
    # earlier b1024 cell reads (b1024.*.residual_ci*, free_ci0: same rows and
    # estimand); and the never-compressed share/count is equal across the four
    # b1024 cells because a trace that stays under 1024 tokens is never evicted,
    # so its generation is the full-cache one under every policy. Coincidences
    # (11) on the k/400 and k/392 accuracy lattices or small counts: -0.0025,
    # -0.005, 0.0125, 0.0175, 0.0375, 0.04, 0.05, 0.1075, 0.16, 0.2, 0.8875 (and
    # 31 / 0.0775 gained unrelated members). No new key is cited in the .tex.
    #
    # 164 -> 172 on 2026-10-16 (P1 reverse read at b1024 and the evidence-bound
    # nominal-compression read), examined: 8 new groups. Identities by
    # construction (4): Holm over two members gives both the same adjusted p
    # (0.6396 = 2 x the smaller p); StreamingLLM's interaction equals its
    # readout change on the compressed trace because its reference-trace
    # readout change is exactly 0; the b256 effective-budget medians equal the
    # nominal ones (the 12 relaxed conditions per policy are not at the
    # median). Small-count coincidences (4): 97 (Random df = StreamingLLM
    # problems), 382 (Random common support = p4.tvjsd_shared_support), 96
    # (StreamingLLM df = p2v2.rotation_n_traces), 99 (both reverse supports).
    # 0.0 / 31 / 0.0775 gained members. No new key is cited in the .tex.
    #
    # 172 -> 178 on 2026-10-17 (M5-A strict-cohort read), examined: 6 new
    # groups. Identities by construction (3): in a two-member Holm family the
    # larger p is its own adjusted value whenever it is >= twice the smaller
    # (registered p, primary and sensitivity TOST p for Random - StreamingLLM).
    # Coincidences (3) on the k/388 and k/194 lattices or counts: -0.00515
    # (= -1/194, also the b1024 StreamingLLM registered mean), 388 (the
    # strict-cohort condition count, also a b1024 reverse support) and -0.25.
    # The TOST interval is registered once (it does not depend on the margin).
    # No new key is cited in the .tex.
    #
    # 178 -> 179 on 2026-10-17 (protection-improvement derived keys),
    # examined: 1 coincidence on the k/388 lattice -- Random's gain from
    # protection (96/388) equals the unprotected Random - StreamingLLM CI's
    # upper bound; neither is derived from the other.
    #
    # 179 -> 192 on 2026-10-20 (Qwen3-4B S.4 read, registered like q15b/q14b),
    # examined: 13 new groups. Identities by construction (9), the same ones
    # every M4 unit carries: full-bank = per-cell full-cache accuracy; each
    # bracket endpoint = the forward residual or the reverse change (Random,
    # StreamingLLM); flip_net = total_change (both); primary bracket = primary
    # reverse bounds (Random lo, hi); one readout reference shared by both
    # presses. Coincidences on the k/50, k/200 and k/400 lattices (4): -0.30
    # (also AIME Random primary residual bound), 0.12 (also the matched
    # Random - StreamingLLM accuracy), 0.735 (also a b2048 CI bound), -0.205
    # (also b512 Random flip_net). No new key is cited in the .tex.
    #
    # 192 -> 235 on 2026-11-15 (M5-B GSM8K replication read, 400 conditions
    # per arm), examined: 43 new groups. The readout-side "free" gap equals
    # the side contrast by construction, so it is NOT registered. Of the
    # rest:
    # - Identities by construction (6): Holm over two members gives both
    #   primary TOST p the same adjusted value (0.7513). The sensitivity
    #   margin 0.05 is twice |Random - StreamingLLM| = 0.025, so that TOST p
    #   equals the registered two-sided p (0.1141). Overflow-reached rates
    #   and realized-ratio p90 coincide across arms that share the same
    #   problems, prompts and budget (0.9875, 0.9925, 0.7665, 0.7711).
    # - Coincidences on the k/400 lattice (37): 10 within the read (e.g.
    #   H2O-style's protected readout change equals its loss because its
    #   residual is exactly 0) and 27 with older keys (e.g. 0.8675, 0.11,
    #   0.125). 32 existing groups also gained M5-B members.
    # No new key is cited in the .tex.
    #
    # 235 -> 258 on 2026-11-21 (M5-B BBH replication read), examined: 23 new
    # groups. 398/400 readout keys are defined, so the readout-side "free" gap
    # differs from the side contrast and IS registered for BBH (plus the
    # matched-key count).
    # - Identities by construction (4): both primary and both sensitivity
    #   TOST p share one Holm-adjusted value (0.3714, 0.1084); the matched-key
    #   count is 398 on both sides (same full-cache reference); and protected
    #   Random - R-KV free == readout because that residual gap is exactly 0.
    # - Coincidences (19): 5 within the read on the k/398 lattice (e.g. the
    #   unprotected Random - H2O-style free gap equals the Random -
    #   StreamingLLM residual gap, 51/398) and 14 with older keys on the k/400
    #   lattice (e.g. 0.8725 BBH full cache = GSM8K protected StreamingLLM;
    #   0.45, 0.68, 0.18). 40 existing groups also gained BBH members.
    # No new key is cited in the .tex.
    assert len(collisions) <= 258, (
        f"{len(collisions)} exact-value collisions in the catalogue; if this "
        "grew, a new derived quantity is duplicating an existing one"
    )


def test_identity_bearing_artifact_basenames_are_unique_under_results_diag():
    """No two artifacts under results/diag/ may share an identity-bearing name.

    Nine `m4-*-summary.json` files existed at BOTH results/diag/ and
    results/diag/amendment-k/ with different values -- the top-level copies
    measured every change against the unrestricted 0.8825 bank, the amendment-k
    copies against each change's own population per Amendment K. The
    divergence was not cosmetic: H2O-2048's `readout_change` was **+0.0155**
    top-level against **-0.0051** canonical, opposite signs for the same named
    quantity. build_paper_numbers.py and rederive_artifacts.py both read the
    amendment-k copies, so the paper was always correct; but a bare filename
    resolved to two different files, which is how a wrong number gets cited by
    someone reading a citation rather than the loader.

    ROLE names are exempt and that exemption is deliberate, not a workaround:
    `summary.json`, `records.json` and `table.md` describe a file's ROLE inside
    its own unit directory (`<unit>/summary.json`) and are never referenced by
    bare basename -- the directory disambiguates. Identity names like
    `m4-rkv-256-summary.json` name a specific artifact and ARE cited bare, so
    they must resolve to exactly one file. Adding a fourth role name has to be
    a conscious edit here, which is the point.
    """
    import collections

    role_names = {"summary.json", "records.json", "table.md"}
    diag = Path(__file__).resolve().parents[1] / "results" / "diag"
    assert diag.is_dir(), f"missing {diag}"

    by_base = collections.defaultdict(list)
    for path in diag.rglob("*"):
        if path.is_file():
            by_base[path.name].append(path.relative_to(diag).as_posix())

    collisions = {
        name: sorted(paths)
        for name, paths in by_base.items()
        if len(paths) > 1 and name not in role_names
    }
    assert not collisions, (
        "identity-bearing basenames must resolve to one file under "
        f"results/diag/, found {len(collisions)}: "
        + "; ".join(f"{n} -> {p}" for n, p in sorted(collisions.items()))
    )

    # A role name is only legitimate when each copy sits in its own directory.
    for name in role_names:
        dirs = [str(Path(p).parent) for p in by_base.get(name, [])]
        assert len(dirs) == len(set(dirs)), f"{name} duplicated within one directory"

#!/usr/bin/env python3
"""Build paper_numbers.json: every number in the paper, mapped to its source.

Thin orchestration only -- the catalogue types and matching logic live in
``kvcache.paper_numbers``. Run after any report regeneration; the audit script
then gates the .tex against the result.

    python scripts/build_paper_numbers.py --out paper_numbers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.paper_numbers import Entry

DIAG = PROJECT_ROOT / "results" / "diag"
K = DIAG / "amendment-k"

# Unit -> the post-Amendment-K summary that is now authoritative for it.
UNITS = {
    "diss256": K / "m4-powered-diss256-summary.json",
    "h2o256": K / "m4-h2o256-full-summary.json",
    "rkv256": K / "m4-rkv-256-summary.json",
    "b2048": K / "m4-2048-summary.json",
    "b2048h2o": K / "m4-2048-h2o-summary.json",
    "aime": K / "m4-aime-2048-summary.json",
    "llama": K / "m4-llama-256-summary.json",
    "q15b": K / "m4-q15b-256-summary.json",
    "q14b": K / "m4-q14b-256-summary.json",
}
ADD = K / "add"
CLASSB = K / "classb"


def _load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing source artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _cells(path: Path) -> dict[str, dict[str, Any]]:
    return {c["press"]: c for c in _load(path)["cells"]}


def build_catalogue() -> list[Entry]:
    entries: list[Entry] = []

    def add(key, value, source, description, kind="primary", inputs=()):
        entries.append(
            Entry(
                key=key,
                value=float(value),
                kind=kind,
                source=source,
                description=description,
                inputs=tuple(inputs),
            )
        )

    # ---- primary: decomposition cells -------------------------------------
    for unit, path in UNITS.items():
        summary = _load(path)
        rel = path.relative_to(PROJECT_ROOT)
        add(
            f"{unit}.fc_bank",
            summary["full_cache"]["accuracy"],
            f"{rel}#/full_cache/accuracy",
            "full-cache bank accuracy (provenance; NOT the cell comparator)",
        )
        for press, cell in _cells(path).items():
            stem = f"{unit}.{press}"
            add(
                f"{stem}.fc_cell",
                cell["full_accuracy"],
                f"{rel}#/cells/{press}/full_accuracy",
                "Amendment K comparator: Acc_FC on this cell's conditions",
            )
            add(
                f"{stem}.free",
                cell["compressed_accuracy"],
                f"{rel}#/cells/{press}/compressed_accuracy",
                "free-running accuracy",
            )
            add(
                f"{stem}.readout",
                cell["readout_accuracy"],
                f"{rel}#/cells/{press}/readout_accuracy",
                "on-reference readout accuracy (complete-case)",
            )
            add(
                f"{stem}.residual",
                cell["unexplained_residual"],
                f"{rel}#/cells/{press}/unexplained_residual",
                "trajectory residual (complete-case)",
            )
            add(
                f"{stem}.total_change",
                cell["total_change"],
                f"{rel}#/cells/{press}/total_change",
                "total accuracy change",
            )
            add(
                f"{stem}.readout_change",
                cell["readout_change"],
                f"{rel}#/cells/{press}/readout_change",
                "readout change vs the matched reference",
            )
            add(
                f"{stem}.trunc",
                cell["truncation_rate"],
                f"{rel}#/cells/{press}/truncation_rate",
                "truncation rate (in denominator, scored incorrect)",
            )
            for name, field in (
                ("free_ci", "compressed_accuracy_ci95"),
                ("readout_ci", "readout_accuracy_ci95"),
            ):
                for i, bound in enumerate(cell.get(field) or []):
                    add(
                        f"{stem}.{name}{i}",
                        bound,
                        f"{rel}#/cells/{press}/{field}/{i}",
                        "problem-clustered 95% accuracy CI",
                    )
            for i, bound in enumerate(cell["unexplained_residual_ci95"]):
                add(
                    f"{stem}.residual_ci{i}",
                    bound,
                    f"{rel}#/cells/{press}/unexplained_residual_ci95/{i}",
                    "problem-clustered 95% CI on the forward residual",
                )
            if cell.get("reverse_readout_accuracy") is not None:
                add(
                    f"{stem}.reverse",
                    cell["reverse_readout_accuracy"],
                    f"{rel}#/cells/{press}/reverse_readout_accuracy",
                    "reverse-readout accuracy",
                )
                add(
                    f"{stem}.reverse_change",
                    cell["reverse_readout_change"],
                    f"{rel}#/cells/{press}/reverse_readout_change",
                    "reverse bracket endpoint",
                )
                for i, bound in enumerate(cell["residual_bracket"]):
                    add(
                        f"{stem}.bracket{i}",
                        bound,
                        f"{rel}#/cells/{press}/residual_bracket/{i}",
                        "residual bracket endpoint (complete-case)",
                    )
                for i, bound in enumerate(cell["reverse_residual_ci95"] or []):
                    add(
                        f"{stem}.reverse_ci{i}",
                        bound,
                        f"{rel}#/cells/{press}/reverse_residual_ci95/{i}",
                        "problem-clustered 95% CI on the reverse endpoint",
                    )
            pb = cell.get("primary_bounds") or {}
            for i, bound in enumerate(pb.get("residual") or []):
                add(
                    f"{stem}.primary_residual{i}",
                    bound,
                    f"{rel}#/cells/{press}/primary_bounds/residual/{i}",
                    "Amendment K layer-1 residual bound (width = n_undef/N)",
                )
            for i, bound in enumerate(pb.get("residual_bracket") or []):
                add(
                    f"{stem}.primary_bracket{i}",
                    bound,
                    f"{rel}#/cells/{press}/primary_bounds/residual_bracket/{i}",
                    "Amendment K layer-1 bracket bound",
                )
            flip = cell.get("signed_flip_mass") or {}
            for name in ("gained", "lost", "net"):
                if name in flip:
                    add(
                        f"{stem}.flip_{name}",
                        flip[name],
                        f"{rel}#/cells/{press}/signed_flip_mass/{name}",
                        f"signed flip mass ({name})",
                    )
            if cell.get("mean_auc_tv") is not None:
                add(
                    f"{stem}.auc_tv",
                    cell["mean_auc_tv"],
                    f"{rel}#/cells/{press}/mean_auc_tv",
                    "mean AUC-TV over the cell's replay profiles",
                )
            if cell.get("mean_free_run_tokens") is not None:
                add(
                    f"{stem}.mean_tokens",
                    cell["mean_free_run_tokens"],
                    f"{rel}#/cells/{press}/mean_free_run_tokens",
                    "mean generated length (free-running)",
                )

    # ---- primary: ADD ------------------------------------------------------
    for name in ("add-256-randslm", "add-256-h2o", "add-256-rkv", "add-2048"):
        path = ADD / f"{name}.json"
        rel = path.relative_to(PROJECT_ROOT)
        for press, cell in _cells(path).items():
            stem = f"add{cell['budget']}.{press}"
            for field_name, key, desc in (
                ("mean_add", "add", "ADD (answer-distribution TV)"),
                ("mean_within_answered_tv", "within", "within-answered TV channel"),
                ("mean_termination_shift", "term", "termination-mass shift channel"),
            ):
                add(
                    f"{stem}.{key}",
                    cell[field_name],
                    f"{rel}#/cells/{press}/{field_name}",
                    desc,
                )
            add(
                f"{stem}.n_sig",
                cell["n_significant_bh_0p05"],
                f"{rel}#/cells/{press}/n_significant_bh_0p05",
                "problems significant after Benjamini-Hochberg",
            )

    # ---- primary: controls -------------------------------------------------
    twofc = _load(ADD / "twofc.json")
    rel = (ADD / "twofc.json").relative_to(PROJECT_ROOT)
    add(
        "twofc.mean_add",
        twofc["mean_add"],
        f"{rel}#/mean_add",
        "two-FC ADD noise floor",
    )
    add(
        "twofc.median_add",
        twofc["median_add"],
        f"{rel}#/median_add",
        "two-FC median ADD",
    )
    add(
        "twofc.mean_perm_p",
        twofc["mean_perm_p"],
        f"{rel}#/mean_perm_p",
        "two-FC mean permutation p",
    )
    add(
        "twofc.frac_sig",
        twofc["frac_p_below_05"],
        f"{rel}#/frac_p_below_05",
        "two-FC fraction significant",
    )

    ctrl = _load(DIAG / "fc-readout-control.json")
    rel = (DIAG / "fc-readout-control.json").relative_to(PROJECT_ROOT)
    add(
        "fc_readout_control.accuracy",
        ctrl["readout_accuracy"],
        f"{rel}#/readout_accuracy",
        "full-cache readout of full-cache traces",
    )

    bank = _load(DIAG / "fc-aime24-bank.json")
    rel = (DIAG / "fc-aime24-bank.json").relative_to(PROJECT_ROOT)
    for field_name, key in (
        ("accuracy", "accuracy"),
        ("truncation", "trunc"),
        ("mean_tokens", "mean_tokens"),
    ):
        add(
            f"aime_bank.{key}",
            bank[field_name],
            f"{rel}#/{field_name}",
            f"AIME-24 reference bank {key}",
        )

    auc = _load(DIAG / "replay-256-h2o-auc.json")
    rel = (DIAG / "replay-256-h2o-auc.json").relative_to(PROJECT_ROOT)
    add(
        "h2o.auc_tv", auc["mean_auc_tv"], f"{rel}#/mean_auc_tv", "H2O-style mean AUC-TV"
    )
    add(
        "h2o.median_onset",
        auc["median_onset"],
        f"{rel}#/median_onset",
        "H2O-style median onset step",
    )

    # ---- primary: Class B (mechanism + phenomenon headline numbers) --------
    # These are quoted in the abstract and in contribution bullets and had no
    # machine-readable source at all until now -- they were computed once, by
    # hand or by a --print flag, and transcribed.
    for label, fname in (
        ("tv_diss256", "tv_windows_diss256.json"),
        ("tv_h2o", "tv_windows_h2o.json"),
    ):
        path = CLASSB / fname
        rel = path.relative_to(PROJECT_ROOT)
        payload = _load(path)
        add(
            f"{label}.median_onset",
            payload["median_onset"],
            f"{rel}#/median_onset",
            "median onset step (one step after the first prune)",
        )
        add(
            f"{label}.n_profiles",
            payload["n_profiles"],
            f"{rel}#/n_profiles",
            "replay profiles contributing to this cell",
        )
        for window in ("before", "first50", "later"):
            add(
                f"{label}.{window}",
                payload["windows"][window]["per_profile"],
                f"{rel}#/windows/{window}/per_profile",
                "windowed mean per-step TV (per-trace estimator; the paper's)",
            )
            add(
                f"{label}.{window}_pooled",
                payload["windows"][window]["pooled"],
                f"{rel}#/windows/{window}/pooled",
                "windowed mean per-step TV (step-pooled estimator)",
            )

    contrast = _load(CLASSB / "policy_contrast_256.json")
    rel = (CLASSB / "policy_contrast_256.json").relative_to(PROJECT_ROOT)
    add(
        "contrast256.mean",
        contrast["mean_difference"],
        f"{rel}#/mean_difference",
        "Random minus StreamingLLM free-run accuracy, paired on the problem",
    )
    for i, bound in enumerate(contrast["ci95"]):
        add(
            f"contrast256.ci{i}",
            bound,
            f"{rel}#/ci95/{i}",
            "problem-clustered 95% CI (seed and n_resamples pinned in artifact)",
        )
    for name, desc in (
        ("wins", "problems where Random strictly beats StreamingLLM"),
        ("ties", "problems where the two policies score identically"),
        ("losses", "problems where StreamingLLM strictly beats Random"),
        ("non_negative", "wins + ties -- the honest '94/100' phrasing"),
        ("n_problems", "problems in the paired contrast"),
    ):
        add(f"contrast256.{name}", contrast[name], f"{rel}#/{name}", desc)

    # ---- derived ----------------------------------------------------------
    # These are the quantities a manual pass loses: they read as prose, and a
    # regenerated primary silently invalidates them.
    by_key = {e.key: e.value for e in entries}

    def derive(key, value, formula, inputs, description):
        add(key, value, formula, description, kind="derived", inputs=inputs)

    # Dose-response factor: residual at 256 over residual at 2048, per press.
    dose = {}
    for press, u256, u2048 in (
        ("random", "diss256", "b2048"),
        ("streamingllm", "diss256", "b2048"),
        ("h2o", "h2o256", "b2048h2o"),
    ):
        a, b = f"{u256}.{press}.residual", f"{u2048}.{press}.residual"
        ratio = by_key[a] / by_key[b]
        dose[press] = ratio
        derive(
            f"dose_factor.{press}",
            ratio,
            f"{a} / {b}",
            (a, b),
            "how many times larger the 256 residual is than the 2048 residual",
        )
    derive(
        "dose_factor.min",
        min(dose.values()),
        "min over presses of dose_factor.*",
        tuple(f"dose_factor.{p}" for p in dose),
        "low end of the dose-response factor range quoted in the text",
    )
    derive(
        "dose_factor.max",
        max(dose.values()),
        "max over presses of dose_factor.*",
        tuple(f"dose_factor.{p}" for p in dose),
        "high end of the dose-response factor range quoted in the text",
    )

    # Readout share: how much of the total change the readout channel explains.
    for unit, press in (
        ("diss256", "random"),
        ("diss256", "streamingllm"),
        ("h2o256", "h2o"),
        ("rkv256", "rkv"),
    ):
        rc, tc = f"{unit}.{press}.readout_change", f"{unit}.{press}.total_change"
        derive(
            f"readout_share.{unit}.{press}",
            by_key[rc] / by_key[tc],
            f"{rc} / {tc}",
            (rc, tc),
            "fraction of the total accuracy change attributable to readout",
        )

    # The abstract's "compounds roughly 50x": the later-window mean over the
    # first-50-steps mean. Derived, so regenerating either window recomputes it
    # instead of leaving the "50x" claim stranded.
    for label in ("tv_diss256", "tv_h2o"):
        derive(
            f"tv_compounding.{label}",
            by_key[f"{label}.later"] / by_key[f"{label}.first50"],
            f"{label}.later / {label}.first50",
            (f"{label}.later", f"{label}.first50"),
            "growth factor from the first 50 post-eviction steps to later steps",
        )

    # Cross-policy residual spread at budget 256 (superseded by a paired test,
    # but must not sit stale while that lands).
    spread_keys = (
        "diss256.random.residual",
        "diss256.streamingllm.residual",
        "h2o256.h2o.residual",
    )
    vals = [by_key[k] for k in spread_keys]
    derive(
        "residual_spread.256_three_policies",
        max(vals) - min(vals),
        "max - min over the three budget-256 residuals",
        spread_keys,
        "cross-policy residual spread (exploratory; see paired test)",
    )

    # Resampling effect: the control against its own matched reference.
    a = "fc_readout_control.accuracy"
    b = "diss256.random.fc_cell"
    derive(
        "resampling_effect.unmatched",
        by_key[a] - by_key["diss256.fc_bank"],
        f"{a} - diss256.fc_bank",
        (a, "diss256.fc_bank"),
        "RETIRED unmatched form (+0.023); kept so the stale value is recognised",
    )

    # ADD headroom over the noise floor.
    for stem in ("add256.random", "add256.streamingllm", "add256.h2o", "add256.rkv"):
        k = f"{stem}.add"
        if k in by_key:
            derive(
                f"add_over_floor.{stem}",
                by_key[k] / by_key["twofc.mean_add"],
                f"{k} / twofc.mean_add",
                (k, "twofc.mean_add"),
                "ADD as a multiple of the two-FC noise floor",
            )

    # Realized compression ratios. The paper's stated definition is
    # budget / mean(prompt + generation), POOLED across the Random/SLM pair.
    # The prompt term (~66 tokens) had no source artifact until Class C, so an
    # earlier revision of this file recorded that the published figures "cannot
    # be fully reproduced here" and catalogued only a generation-only variant.
    # That note was wrong about the *paper*: with the prompt term restored from
    # the token-ID sidecars the published values reproduce exactly (6725.00 and
    # 3618.20 pooled trace tokens -> 3.807% and 56.603%). The gap was
    # provenance, not correctness. Both forms are kept: the prompt-inclusive
    # one is the paper's, the generation-only one shows what the prompt term is
    # worth, which is what makes the definition checkable.
    lengths = _load(CLASSB / "prompt_lengths.json")
    rel_len = (CLASSB / "prompt_lengths.json").relative_to(PROJECT_ROOT)
    add(
        "prompt.mean_tokens",
        lengths["pooled"]["mean_prompt_tokens"],
        f"{rel_len}#/pooled/mean_prompt_tokens",
        "mean rendered-prompt length over every compressed cell (token-ID sidecars)",
    )
    add(
        "prompt.median_tokens",
        lengths["pooled"]["median_prompt_tokens"],
        f"{rel_len}#/pooled/median_prompt_tokens",
        "median rendered-prompt length",
    )
    for cell, payload in sorted(lengths["realized"].items()):
        for field_name, key, desc in (
            ("mean_trace_tokens", "trace_tokens", "mean prompt+generation length"),
            ("ratio_pct", "ratio_pct", "realized compression ratio (percent)"),
        ):
            add(
                f"realized.{cell}.{key}",
                payload[field_name],
                f"{rel_len}#/realized/{cell}/{field_name}",
                desc,
            )
    for unit, budget, cells in (
        ("diss256", 256, ("random-256", "streamingllm-256")),
        ("b2048", 2048, ("random-2048", "streamingllm-2048")),
    ):
        ks = tuple(f"realized.{c}.trace_tokens" for c in cells)
        by_key.update({e.key: e.value for e in entries})
        pooled = sum(by_key[k] for k in ks) / len(ks)
        derive(
            f"mean_trace_pooled.{unit}",
            pooled,
            "mean over the Random/SLM pair of mean_trace_tokens (prompt+generation)",
            ks,
            "pooled mean TRACE length -- the paper's compression-ratio denominator",
        )
        derive(
            f"compression_ratio.{unit}",
            budget / pooled,
            f"{budget} / mean_trace_pooled.{unit}",
            (f"mean_trace_pooled.{unit}",),
            "realized ratio as the paper defines it (prompt + generation)",
        )
        gk = (f"{unit}.random.mean_tokens", f"{unit}.streamingllm.mean_tokens")
        pooled_gen = sum(by_key[k] for k in gk) / len(gk)
        derive(
            f"mean_tokens_pooled.{unit}",
            pooled_gen,
            "mean over the Random/SLM pair of mean_free_run_tokens",
            gk,
            "pooled mean GENERATED length (excludes prompt; contrast only)",
        )
        derive(
            f"compression_ratio_gen_only.{unit}",
            budget / pooled_gen,
            f"{budget} / mean_tokens_pooled.{unit}",
            (f"mean_tokens_pooled.{unit}",),
            "realized ratio against generation only -- what the prompt term is worth",
        )

    # ---- primary: Class C (compute) ---------------------------------------
    compute = _load(CLASSB / "compute.json")
    rel_c = (CLASSB / "compute.json").relative_to(PROJECT_ROOT)
    add(
        "compute.gpu_hours",
        compute["gpu_hours"],
        f"{rel_c}#/gpu_hours",
        "summed generation wall-clock under the artifact's stated inclusion rule",
    )
    add(
        "compute.generations",
        compute["generations"],
        f"{rel_c}#/generations",
        "generations counted in the compute total",
    )

    # ---- external: Class A (numbers quoted from other papers) -------------
    # These are not ours to regenerate, so "sourced" means something different:
    # a citation key plus a locator precise enough that a reviewer can check the
    # transcription without reading the whole paper. Where the repo's verified
    # audit (PAPER_FINDINGS.md, "verified against the papers 2026-07") records a
    # table, the table travels with the entry. Where it does not, the locator is
    # UNVERIFIED and main() prints it -- an unverified locator is a number we
    # believe but cannot point at, which is exactly the state worth surfacing
    # rather than rounding up to "sourced".
    def external(key, value, cite, locator, description):
        entries.append(
            Entry(
                key=key,
                value=float(value),
                kind="external",
                source=f"\\cite{{{cite}}} @ {locator}",
                description=description,
            )
        )

    for key, value, cite, locator, desc in (
        (
            "hott.fullcache_math500",
            0.47,
            "hott2025",
            "Table 2 (MATH-500, R1-Distill-Qwen-7B, full cache)",
            "HOTT's full-cache MATH-500 reference -- not comparable digit-for-digit",
        ),
        (
            "hott.streamingllm_128",
            0.03,
            "hott2025",
            "Table 2 (MATH-500, R1-Distill-Qwen-7B, budget 128)",
            "HOTT StreamingLLM at budget 128 -- the sharpest external contradiction",
        ),
        (
            "hott.h2o_128",
            0.14,
            "hott2025",
            "Table 2 (MATH-500, R1-Distill-Qwen-7B, budget 128)",
            "HOTT H2O at budget 128 (WITH recency reservation -- it survives)",
        ),
        (
            "hott.snapkv_128",
            0.38,
            "hott2025",
            "Table 2 (MATH-500, R1-Distill-Qwen-7B, budget 128)",
            "HOTT SnapKV-D at budget 128",
        ),
        (
            "rkv.published_pct_of_full",
            105.0,
            "rkv2025",
            "UNVERIFIED (headline; R1-Distill-Llama-8B, AIME-24)",
            "R-KV's published >100% headline -- our port runs a different scale",
        ),
        (
            "rkv.published_cache_pct",
            16.0,
            "rkv2025",
            "UNVERIFIED (headline; R1-Distill-Llama-8B, AIME-24)",
            "cache fraction at R-KV's published headline",
        ),
        (
            "deepseek.qwen7b_math500",
            0.928,
            "deepseek2025r1",
            "UNVERIFIED (distilled-model MATH-500 pass@1, R1-Distill-Qwen-7B)",
            "published 7B MATH-500 pass@1; our 0.882 bank is anchored against it",
        ),
        (
            "deepseek.qwen14b_math500",
            0.939,
            "deepseek2025r1",
            "UNVERIFIED (distilled-model MATH-500 pass@1, R1-Distill-Qwen-14B)",
            "published 14B MATH-500 pass@1; the 14B bank gate band was set from it",
        ),
    ):
        external(key, value, cite, locator, desc)

    # Percentage forms, catalogued explicitly rather than generated, so that a
    # value quoted as "29.8\\%" is sourced without every entry also matching
    # unrelated two-digit numbers elsewhere in the text.
    for e in list(entries):
        if e.kind == "derived" and (
            e.key.startswith("readout_share.") or e.key.startswith("compression_ratio")
        ):
            derive(
                f"{e.key}.pct",
                e.value * 100.0,
                f"100 * {e.key}",
                (e.key,),
                "percentage form of the preceding ratio",
            )

    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "paper_numbers.json")
    args = parser.parse_args()

    entries = build_catalogue()
    payload = {
        "note": (
            "Generated by scripts/build_paper_numbers.py. Every number in the "
            "paper must map to an entry here. Derived entries record their "
            "formula and inputs so a regenerated primary invalidates them."
        ),
        "convention": "amendment-k-cell-population",
        "n_entries": len(entries),
        "entries": [
            {
                "key": e.key,
                "value": e.value,
                "display": f"{e.value:.3f}",
                "kind": e.kind,
                "source": e.source,
                "description": e.description,
                **({"inputs": list(e.inputs)} if e.inputs else {}),
            }
            for e in sorted(entries, key=lambda e: e.key)
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    kinds: dict[str, int] = {}
    for e in entries:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    breakdown = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
    print(f"wrote {args.out} -- {len(entries)} entries ({breakdown})")

    # An external number whose locator is UNVERIFIED is traceable to a paper but
    # not to a place in it. That is weaker than every other entry in the
    # catalogue and must not be silently absorbed into the "sourced" count.
    unverified = sorted(
        e.key for e in entries if e.kind == "external" and "UNVERIFIED" in e.source
    )
    if unverified:
        print(
            f"NOTE: {len(unverified)} external entries carry an UNVERIFIED "
            "locator (cited paper known, place in it not recorded):"
        )
        for key in unverified:
            print(f"  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    "b512": K / "m4-512-summary.json",
    "b1024": K / "m4-1024-summary.json",
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
            # The reference each change is ACTUALLY measured against. Amendment K
            # fixes a separate matched reference per change, because readout is
            # defined only where the reference trace reaches an answer boundary
            # and reverse readout only where the compressed trace does. The
            # paper named a single AccFC for all of them, which is false for two
            # of the three -- these entries give the real ones a source so the
            # prose can state them (Amendment P.3-adjacent documentation fix).
            add(
                f"{stem}.readout_ref",
                cell["readout_accuracy"] - cell["readout_change"],
                f"{rel}#/cells/{press}/readout_accuracy - .../readout_change",
                "Acc_FC on the readout-defined conditions (the Delta_ro reference)",
                kind="derived",
                inputs=(f"{stem}.readout", f"{stem}.readout_change"),
            )
            if cell.get("reverse_readout_accuracy") is not None:
                add(
                    f"{stem}.reverse_ref",
                    cell["reverse_readout_accuracy"] - cell["reverse_readout_change"],
                    f"{rel}#/cells/{press}/reverse_readout_accuracy - "
                    f".../reverse_readout_change",
                    "Acc_FC on the reverse-defined conditions (the reverse "
                    "endpoint's reference)",
                    kind="derived",
                    inputs=(f"{stem}.reverse", f"{stem}.reverse_change"),
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
            # The reverse endpoint under both imputations, quoted directly in
            # the selection-effect argument: index 0 scores every undefined row
            # incorrect, index 1 scores every one correct.
            for i, bound in enumerate(pb.get("reverse_change") or []):
                add(
                    f"{stem}.primary_reverse{i}",
                    bound,
                    f"{rel}#/cells/{press}/primary_bounds/reverse_change/{i}",
                    "reverse endpoint with undefined rows scored "
                    + ("incorrect" if i == 0 else "correct"),
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
            # Signed flip mass is what licenses reading divergence as loss
            # rather than mere change, so it is quoted throughout the ADD
            # section -- and it was catalogued only from the decomposition
            # summaries, never from the ADD artifacts the section actually
            # cites. Same quantity, different producer; both must be sourced.
            flip_mass = cell.get("signed_flip_mass") or {}
            for name in ("gained", "lost", "net"):
                if name in flip_mass:
                    add(
                        f"{stem}.flip_{name}",
                        flip_mass[name],
                        f"{rel}#/cells/{press}/signed_flip_mass/{name}",
                        f"signed flip mass ({name}) as measured by the ADD run",
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
    # Amendment K reached this control late: its published +0.023 differenced a
    # 392-row readout accuracy against the 2000-row bank. On the matched
    # population the answer-resampling effect is +0.003 -- an order of magnitude
    # smaller, and the calibration sentence it supports gets stronger, not weaker.
    add(
        "fc_readout_control.fc_matched",
        ctrl["fc_accuracy_matched"],
        f"{rel}#/fc_accuracy_matched",
        "Acc_FC restricted to the conditions where FC readout is defined",
    )
    add(
        "fc_readout_control.resampling_effect",
        ctrl["resampling_effect_matched"],
        f"{rel}#/resampling_effect_matched",
        "pure answer-resampling effect of the readout protocol, matched population",
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

    # HOTT-configuration cell (Amendment N.1): how much of the baseline gap
    # HOTT's 2048-token generation cap accounts for on its own. Greedy decoding
    # is NOT matched, so the residual gap is not attributable to the cap.
    _hc_path = CLASSB / "hottcap.json"
    _hc_rel = _hc_path.relative_to(PROJECT_ROOT)
    _hc = _load(_hc_path)
    for _cell, _short in (("fullcache", "fc"), ("streamingllm128", "slm128")):
        for _field, _desc in (
            ("accuracy", "accuracy under HOTT's 2048-token generation cap"),
            ("truncation_rate", "truncation rate under that cap"),
        ):
            add(
                f"hottcap.{_short}.{_field}",
                _hc["cells"][_cell][_field],
                f"{_hc_rel}#/cells/{_cell}/{_field}",
                _desc,
            )

    # R-KV in its published configuration (Amendment N.2): R1-Distill-Llama-8B
    # on AIME-24 at budget 2048. The ratio and its CI are the pre-registered
    # read; the accuracies and truncation rates are catalogued alongside because
    # the ratio is a difference of two SMALL counts (37 vs 36 of 120) and must
    # never be quoted without them.
    _rp_path = CLASSB / "rkv_published_cap32768.json"
    _rp_rel = _rp_path.relative_to(PROJECT_ROOT)
    _rp = _load(_rp_path)
    add(
        "rkvpub.ratio",
        _rp["accuracy_ratio_r"],
        f"{_rp_rel}#/accuracy_ratio_r",
        "R-KV accuracy / full-cache accuracy in the published configuration",
    )
    add(
        "rkvpub.mean_difference",
        _rp["contrast_cell_minus_fullcache"]["mean_difference"],
        f"{_rp_rel}#/contrast_cell_minus_fullcache/mean_difference",
        "R-KV minus full-cache accuracy, paired on the problem",
    )
    for _i, _end in enumerate(("ci0", "ci1")):
        add(
            f"rkvpub.{_end}",
            _rp["contrast_cell_minus_fullcache"]["ci95"][_i],
            f"{_rp_rel}#/contrast_cell_minus_fullcache/ci95/{_i}",
            "problem-clustered 95% CI on that difference",
        )
    for _side, _blk in (("rkv", "cell"), ("fc", "full_cache")):
        for _field, _desc in (
            ("correct", "correct generations out of 120"),
            ("accuracy", "accuracy in the published configuration"),
            ("truncation", "truncation rate at the 32768-token cap"),
            ("accuracy_non_truncated", "accuracy among non-truncated generations"),
            ("mean_tokens", "mean generated tokens"),
        ):
            add(
                f"rkvpub.{_side}.{_field}",
                _rp[_blk][_field],
                f"{_rp_rel}#/{_blk}/{_field}",
                _desc,
            )
    add(
        "rkvpub.realized_ratio",
        _rp["cell"]["realized_compression_ratio"],
        f"{_rp_rel}#/cell/realized_compression_ratio",
        "mean realized cache fraction at budget 2048 on AIME-24",
    )
    add(
        "rkvpub.realized_ratio_pct",
        100.0 * _rp["cell"]["realized_compression_ratio"],
        f"{_rp_rel}#/cell/realized_compression_ratio (x100)",
        "the same fraction as a percentage, which is how the paper quotes it",
    )
    add(
        "rkvpub.gain_pct",
        100.0 * (_rp["accuracy_ratio_r"] - 1.0),
        f"100 * ({_rp_rel}#/accuracy_ratio_r - 1)",
        "nominal percent gain over full cache; the paper quotes it only to deny "
        "that the cell establishes it",
    )

    # Resolution collapse (Amendment L, POST-HOC). One artifact per regime.
    for label, fname in (
        ("res256", "resolution_math500_256.json"),
        ("res512", "resolution_math500_512.json"),
        ("res1024", "resolution_math500_1024.json"),
        ("res2048", "resolution_math500_2048.json"),
        ("resaime", "resolution_aime24_2048.json"),
        # Matched-press-pair recomputation of 256 and 2048 (Random + SLM only),
        # so the budget comparison holds the press SET fixed. Run AFTER the 512
        # result was seen, as a comparability check; disclosed as such in
        # Amendment L's result section.
        ("respair256", "resolution_pair_256.json"),
        ("respair2048", "resolution_pair_2048.json"),
    ):
        path = CLASSB / fname
        rel = path.relative_to(PROJECT_ROOT)
        payload = _load(path)
        for field_name, key, desc in (
            ("free_run_spread", "free_spread", "mean abs pairwise free-run difference"),
            (
                "readout_spread",
                "readout_spread",
                "mean abs pairwise readout difference",
            ),
            ("spread_difference", "difference", "free-run spread minus readout spread"),
            (
                "rank_agreement",
                "rank_agreement",
                "fraction of press pairs ordered alike",
            ),
        ):
            add(
                f"{label}.{key}",
                payload[field_name],
                f"{rel}#/{field_name}",
                desc,
            )
        for i, bound in enumerate(payload["spread_difference_ci95"]):
            add(
                f"{label}.difference_ci{i}",
                bound,
                f"{rel}#/spread_difference_ci95/{i}",
                "problem-clustered 95% CI on the spread difference",
            )
        # Percentage form of the descriptive ratio, catalogued explicitly.
        add(
            f"{label}.ratio",
            payload["spread_ratio_descriptive"],
            f"{rel}#/spread_ratio_descriptive",
            "readout spread as a fraction of free-run spread (descriptive, no CI)",
        )
        add(
            f"{label}.ratio_pct",
            100.0 * payload["spread_ratio_descriptive"],
            f"{rel}#/spread_ratio_descriptive",
            "readout spread as a percent of free-run spread (descriptive, no CI)",
        )

    # Paired cross-policy residual contrast. This supersedes the residual-spread
    # argument: a spread is a paired quantity and was being set against unpaired
    # marginal CI half-widths, which is not an admissible comparison in either
    # direction.
    rescon = _load(CLASSB / "residual_contrast_256.json")
    rel = (CLASSB / "residual_contrast_256.json").relative_to(PROJECT_ROOT)
    add(
        "residual_contrast.mapd",
        rescon["mean_abs_pairwise_difference"],
        f"{rel}#/mean_abs_pairwise_difference",
        "mean absolute pairwise residual difference, paired on the problem",
    )
    for i, bound in enumerate(rescon["mean_abs_pairwise_difference_ci95"]):
        add(
            f"residual_contrast.mapd_ci{i}",
            bound,
            f"{rel}#/mean_abs_pairwise_difference_ci95/{i}",
            "problem-clustered 95% CI on the mean absolute pairwise difference",
        )
    add(
        "residual_contrast.n_pairs_excluding_zero",
        rescon["n_pairs_excluding_zero"],
        f"{rel}#/n_pairs_excluding_zero",
        "how many of the six paired contrasts exclude zero",
    )
    for name, pair in sorted(rescon["pairs"].items()):
        add(
            f"residual_contrast.{name}",
            pair["mean_difference"],
            f"{rel}#/pairs/{name}/mean_difference",
            "paired per-problem residual difference",
        )
        for i, bound in enumerate(pair["ci95"]):
            add(
                f"residual_contrast.{name}.ci{i}",
                bound,
                f"{rel}#/pairs/{name}/ci95/{i}",
                "problem-clustered 95% CI on the paired difference",
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

    # Truncation-adjusted residual: rescore every truncated free-run as CORRECT.
    # An extreme bound by construction -- it scores as successes the very
    # termination failures ADD measures as a compression-induced channel -- so
    # it is the conservative end of a range, not an estimate.
    for unit, press in (
        ("diss256", "random"),
        ("diss256", "streamingllm"),
        ("h2o256", "h2o"),
        ("rkv256", "rkv"),
    ):
        s = f"{unit}.{press}"
        free_adj = by_key[f"{s}.free"] + by_key[f"{s}.trunc"]
        derive(
            f"residual_trunc_adjusted.{unit}.{press}",
            (free_adj - by_key[f"{s}.fc_cell"]) - by_key[f"{s}.readout_change"],
            f"({s}.free + {s}.trunc - {s}.fc_cell) - {s}.readout_change",
            (f"{s}.free", f"{s}.trunc", f"{s}.fc_cell", f"{s}.readout_change"),
            "residual with every truncated free-run rescored correct (extreme bound)",
        )
    # The residual restated in accuracy points ("roughly 34 accuracy points").
    # A percentage form, so it is catalogued explicitly rather than generated.
    for unit, press in (
        ("diss256", "random"),
        ("diss256", "streamingllm"),
        ("h2o256", "h2o"),
    ):
        rk = f"{unit}.{press}.residual"
        derive(
            f"residual_points.{unit}.{press}",
            abs(by_key[rk]) * 100.0,
            f"100 * |{rk}|",
            (rk,),
            "trajectory residual expressed in accuracy points",
        )
    # AUC-TV spread across policies, as a percent of the smallest. Quoted to
    # make the point that "similar magnitude" on the TV scale is a loose
    # statement, not a matched one.
    _auc = ("diss256.random.auc_tv", "diss256.streamingllm.auc_tv", "h2o.auc_tv")
    _av = [by_key[k] for k in _auc]
    derive(
        "auc_tv_spread_pct",
        100.0 * (max(_av) - min(_av)) / min(_av),
        "100 * (max - min) / min over the three AUC-TV values",
        _auc,
        "cross-policy AUC-TV spread as a percent of the smallest",
    )
    derive(
        "truncation_ratio.random_over_slm",
        by_key["diss256.random.trunc"] / by_key["diss256.streamingllm.trunc"],
        "diss256.random.trunc / diss256.streamingllm.trunc",
        ("diss256.random.trunc", "diss256.streamingllm.trunc"),
        "how much more often Random fails to terminate than StreamingLLM",
    )
    # The (ii) claim: method differences appear in readout, not in the residual.
    # Both spans are catalogued so the contrast cannot go stale one-sidedly.
    ro_keys = (
        "diss256.random.readout_change",
        "diss256.streamingllm.readout_change",
        "h2o256.h2o.readout_change",
    )
    ro_vals = [by_key[k] for k in ro_keys]
    derive(
        "readout_change_spread.256_three_policies",
        max(ro_vals) - min(ro_vals),
        "max - min over the three budget-256 readout changes",
        ro_keys,
        "cross-policy readout-change spread (contrast with the residual spread)",
    )
    derive(
        "readout_change_gap.random_slm",
        by_key["diss256.random.readout_change"]
        - by_key["diss256.streamingllm.readout_change"],
        "diss256.random.readout_change - diss256.streamingllm.readout_change",
        ("diss256.random.readout_change", "diss256.streamingllm.readout_change"),
        "how much of the +0.185 free-run gap the readout channel accounts for",
    )
    derive(
        "residual_gap.random_slm",
        by_key["diss256.random.residual"] - by_key["diss256.streamingllm.residual"],
        "diss256.random.residual - diss256.streamingllm.residual",
        ("diss256.random.residual", "diss256.streamingllm.residual"),
        "how much of the +0.185 free-run gap the residual accounts for",
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

    # Resampling effect: the control against its own matched reference. The
    # primary value is now a PRIMARY entry read straight off the artifact
    # (fc_readout_control.resampling_effect, +0.003); the unmatched form stays
    # only so the stale +0.023 is recognised rather than silently unmatched.
    a = "fc_readout_control.accuracy"
    derive(
        "resampling_effect.unmatched",
        by_key[a] - by_key["diss256.fc_bank"],
        f"{a} - diss256.fc_bank",
        (a, "diss256.fc_bank"),
        "RETIRED unmatched form (+0.023); kept so the stale value is recognised",
    )
    # How far the residuals sit above the readout protocol's own noise. The
    # published text says "an order of magnitude"; on the matched control it is
    # two, which is the direction that helps us -- so it is stated with the
    # correction's timing disclosed rather than quietly upgraded.
    for unit, press in (("diss256", "random"), ("rkv256", "rkv")):
        rk = f"{unit}.{press}.residual"
        derive(
            f"residual_over_resampling.{unit}.{press}",
            abs(by_key[rk]) / by_key["fc_readout_control.resampling_effect"],
            f"|{rk}| / fc_readout_control.resampling_effect",
            (rk, "fc_readout_control.resampling_effect"),
            "residual as a multiple of the pure answer-resampling effect",
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

    # AIME's realized ratio was stated against a generation-only denominator
    # while the MATH-500 ratios use prompt + generation -- the same quantity
    # computed two ways in one paper. Restated on the paper's own definition.
    # The AIME denominator remains the REFERENCE bank's trace (that is what the
    # sentence describes) where MATH's is the compressed cells'; that difference
    # is deliberate and is now stated in the text rather than left implicit.
    aime_len = _load(CLASSB / "prompt_lengths_aime.json")
    rel_al = (CLASSB / "prompt_lengths_aime.json").relative_to(PROJECT_ROOT)
    add(
        "aime_prompt.mean_tokens",
        aime_len["pooled"]["mean_prompt_tokens"],
        f"{rel_al}#/pooled/mean_prompt_tokens",
        "mean rendered-prompt length on AIME-24 (token-ID sidecars)",
    )
    by_key.update({e.key: e.value for e in entries})
    derive(
        "aime_mean_trace_tokens",
        by_key["aime_bank.mean_tokens"] + by_key["aime_prompt.mean_tokens"],
        "aime_bank.mean_tokens + aime_prompt.mean_tokens",
        ("aime_bank.mean_tokens", "aime_prompt.mean_tokens"),
        "mean AIME reference trace, prompt + generation",
    )
    derive(
        "compression_ratio.aime2048",
        2048 / (by_key["aime_bank.mean_tokens"] + by_key["aime_prompt.mean_tokens"]),
        "2048 / aime_mean_trace_tokens",
        ("aime_mean_trace_tokens",),
        "realized AIME ratio on the paper's prompt+generation definition",
    )

    # ---- primary: grader audit (instrument validity) -----------------------
    grader_dir = K / "grader"
    grader_total_rows = 0
    grader_total_disagreements = 0
    for path in sorted(grader_dir.glob("*.json")):
        rel = path.relative_to(PROJECT_ROOT)
        payload = _load(path)
        stem = f"grader.{path.stem.replace('grid-diss256-', '')}"
        for field_name, key in (
            ("rows", "rows"),
            ("stored_accuracy", "stored_accuracy"),
            ("regraded_accuracy", "regraded_accuracy"),
            ("grader_disagreements", "disagreements"),
            ("crosscheck_comparable", "xc_comparable"),
            ("crosscheck_disagreements", "xc_disagreements"),
            ("crosscheck_only_correct", "xc_only_correct"),
        ):
            add(
                f"{stem}.{key}",
                payload[field_name],
                f"{rel}#/{field_name}",
                "grader audit",
            )
        # The rate that matters: how often the INDEPENDENT normalization calls a
        # row correct where is_equiv does not, over rows where both produce a
        # comparable answer. This is the under-crediting direction -- the one
        # that would inflate apparent compression damage.
        add(
            f"{stem}.xc_only_pct",
            100.0
            * payload["crosscheck_only_correct"]
            / payload["crosscheck_comparable"],
            f"100 * {rel}#/crosscheck_only_correct / #/crosscheck_comparable",
            "under-crediting rate over comparable rows (percent)",
            kind="derived",
        )
        add(
            f"{stem}.xc_worst_shift",
            payload["crosscheck_only_correct"] / payload["rows"],
            f"{rel}#/crosscheck_only_correct / #/rows",
            "worst-case accuracy shift if every disagreement were a grader miss",
            kind="derived",
        )
        grader_total_rows += int(payload["rows"])
        grader_total_disagreements += int(payload["grader_disagreements"])
    # Worst-case grader bias in each accuracy CHANGE: the treated arm's
    # under-crediting rate minus the control arm's. This is the quantity the
    # threat is actually about -- a rate that is high in both arms biases
    # nothing. Quoted in section 5; it had no entry, and was passing the gate
    # only because the old sign-blind lookup matched it to a positive value.
    by_key.update({e.key: e.value for e in entries})
    _control = "grader.math500-fc-powered-merged.xc_worst_shift"
    if _control in by_key:
        for cell in ("tok", "h2o", "rkv"):
            treated = f"grader.{cell}.xc_worst_shift"
            if treated in by_key:
                derive(
                    f"grader.{cell}.xc_bias_vs_control",
                    by_key[treated] - by_key[_control],
                    f"{treated} - {_control}",
                    (treated, _control),
                    "worst-case bias in this cell's accuracy change",
                )

    if grader_total_rows:
        gsrc = f"sum over {grader_dir.relative_to(PROJECT_ROOT)}/*.json"
        add(
            "grader.total_rows",
            grader_total_rows,
            f"{gsrc} #/rows",
            "rows re-graded across every budget-256 compressed cell",
        )
        add(
            "grader.total_disagreements",
            grader_total_disagreements,
            f"{gsrc} #/grader_disagreements",
            "grader disagreements across every budget-256 compressed cell",
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
    # audit (docs/results/PAPER_FINDINGS.md, "verified against the papers
    # 2026-07") records a
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
            'Abstract, verbatim; Sec. 4 "Ratio Budget" para (R1-Llama-8B)',
            "R-KV's published >100% headline. VERIFIED 2026-07-28 against the "
            'arXiv text: "Given 16% KV cache budget, our method even surpasses '
            'the FullKV baseline, reaching 105% of its accuracy", in the '
            "R1-Llama-8B paragraph. The DATASET is not named in that sentence; "
            "AIME-24 is inferred because 16% sits above its stated 10% lossless "
            "point and below MATH-500's 34%. Model confirmed, dataset inferred.",
        ),
        (
            "rkv.published_cache_pct",
            16.0,
            "rkv2025",
            'Abstract, verbatim; Sec. 4 "Ratio Budget" para (R1-Llama-8B)',
            "cache fraction at R-KV's published headline. VERIFIED 2026-07-28: "
            "the source says 16%, not ~16%.",
        ),
        (
            "deepseek.qwen7b_math500",
            0.928,
            "deepseek2025r1",
            "Table 5 (distilled models on reasoning benchmarks), MATH-500 pass@1",
            "published 7B MATH-500 pass@1; our 0.882 bank is anchored against it. "
            "VERIFIED 2026-07-28: row DeepSeek-R1-Distill-Qwen-7B reads 92.8.",
        ),
        (
            "deepseek.qwen14b_math500",
            0.939,
            "deepseek2025r1",
            "Table 5 (distilled models on reasoning benchmarks), MATH-500 pass@1",
            "published 14B MATH-500 pass@1; the 14B bank gate band was set from "
            "it. VERIFIED 2026-07-28: row DeepSeek-R1-Distill-Qwen-14B reads 93.9.",
        ),
        (
            "evidence2026.adakv_gap",
            -0.167,
            "evidence2026",
            "Table 10 (AIME26, Qwen3-8B): end-to-end 23.3% at budget 2048 minus "
            "fixed-trace 40.0% at budget 256",
            "free-running minus fixed-trace for AdaKV in the nearest neighbour's "
            "table, at UNMATCHED budgets (2048 vs 256), so NOT our residual. "
            "DERIVED by us. CORRECTED 2026-10-05: the 2026-09-12 check missed "
            "Table 10's caption ('end-to-end ... at budget 2048, compared with "
            "our fixed-trace baseline at KV budget 256').",
        ),
        (
            "evidence2026.snapkv_gap",
            -0.133,
            "evidence2026",
            "Table 10 (AIME26, Qwen3-8B): end-to-end 26.7% at budget 2048 minus "
            "fixed-trace 40.0% at budget 256",
            "the same unmatched-budget difference for SnapKV (2048 vs 256). "
            "DERIVED by us. CORRECTED 2026-10-05 (see the AdaKV entry).",
        ),
    ):
        external(key, value, cite, locator, desc)

    # Whole-number display forms, catalogued explicitly for exactly the
    # quantities the prose quotes rounded. Entry.displays() no longer generates
    # a zero-place form for a fractional value -- a ratio of 6.7 must not claim
    # every bare "7" -- so a legitimate rounding like "280 GPU-hours" needs an
    # entry of its own. Listing them is the point: each one is a decision that
    # this quantity is quoted whole somewhere, not a blanket licence.
    by_key.update({e.key: e.value for e in entries})
    for key, desc in (
        ("compute.gpu_hours", "GPU-hours, quoted whole in the setup"),
        ("mean_trace_pooled.diss256", "pooled trace length, quoted whole"),
        ("mean_trace_pooled.b2048", "pooled trace length, quoted whole"),
        ("aime_mean_trace_tokens", "AIME trace length, quoted whole"),
        ("aime_bank.mean_tokens", "AIME bank generation length, quoted whole"),
        ("rkvpub.rkv.mean_tokens", "published-config R-KV trace, quoted whole"),
        ("rkvpub.fc.mean_tokens", "published-config full-cache trace, quoted whole"),
        ("auc_tv_spread_pct", "AUC-TV spread, quoted whole"),
        ("residual_points.diss256.random", "residual in accuracy points, whole"),
        ("residual_over_resampling.diss256.random", "residual/resampling, whole"),
        ("residual_over_resampling.rkv256.rkv", "residual/resampling, whole"),
    ):
        if key in by_key:
            derive(
                f"{key}.rounded",
                round(by_key[key]),
                f"round({key})",
                (key,),
                desc,
            )

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

    # Amendment N.1: what the generation cap leaves unexplained.
    #
    # KNOWN GATE LIMITATION, recorded here because the green tick is misleading.
    # The paper's 0.045 (0.515 - 0.47) is sourced, but the audit attributes it to
    # `llama.random.readout_ci1`, an unrelated Llama CI bound that happens to
    # equal 0.045 exactly. This derived entry does NOT change that: exact-match
    # precedence prefers the entry whose stored float is exactly 0.045 over this
    # one's 0.04500000000000004. So the number is real and its arithmetic is
    # checked here, but the audit's provenance trail for it points at the wrong
    # quantity.
    #
    # This is the residual weakness of value-based attribution: two distinct
    # quantities that share a value are indistinguishable without context. It is
    # the same family as the sign-blind-attribution defect (fixed) and the
    # coincidence-matching risk that made display rules data. A real fix needs
    # context-aware matching (surrounding tokens), which is not built.
    #
    # Placed after the external entries because hott.fullcache_math500 is added
    # below the derived section.
    _bk = {e.key: e.value for e in entries}
    add(
        "hottcap.residual_gap",
        _bk["hottcap.fc.accuracy"] - _bk["hott.fullcache_math500"],
        "hottcap.fc.accuracy - hott.fullcache_math500",
        "baseline gap left after HOTT's generation cap is matched",
        kind="derived",
        inputs=("hottcap.fc.accuracy", "hott.fullcache_math500"),
    )

    # ---- Meeting #2 / Amendment P-series: the 2026-08 cells ---------------
    # These artifacts were read after the original campaign closed. Cataloguing
    # them from disk is what lets the paper cite them at all: audit --strict
    # refuses any result-precision literal it cannot trace back to a file, and
    # before this block every P-series number in the .tex would have been
    # unsourced. Two coincidental value collisions in the existing catalogue
    # (-0.0013 against an unrelated flip-mass, 0.0384 against a compression
    # ratio) are exactly why "sourced" is checked per key, not per value.
    P = DIAG / "amendment-p"

    # Task 2 (Amendment P.5): restoration arms, corrected cell.
    _p5 = _load(P / "restoration_diss256_v2.json")
    _p5rel = (P / "restoration_diss256_v2.json").relative_to(PROJECT_ROOT)
    for _cell, _c in _p5["cells"].items():
        _press = _cell.split("-")[0]
        add(
            f"p5.{_press}.baseline_divergence",
            _c["baseline_mean_divergence"],
            f"{_p5rel} :: cells.{_cell}.baseline_mean_divergence",
            f"no-intervention mean per-step TV over the read window ({_press})",
        )
        for _arm, _v in _c["arms"].items():
            add(
                f"p5.{_press}.{_arm}.effect",
                _v["effect_vs_no_intervention"],
                f"{_p5rel} :: cells.{_cell}.arms.{_arm}.effect_vs_no_intervention",
                f"{_arm} arm minus no-intervention over the read window ({_press}); "
                "positive would mean divergence reduced",
            )
            for _i, _b in enumerate(_v["effect_ci95"]):
                add(
                    f"p5.{_press}.{_arm}.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_p5rel} :: cells.{_cell}.arms.{_arm}.effect_ci95[{_i}]",
                    f"{_arm} arm effect CI bound ({_press})",
                )

    # Specificity-control magnitude, derived so the paper can state it without
    # a verbal range. The earlier prose said "five to six times", which excludes
    # Random's actual 6.49 -- a characterisation the number gate cannot catch
    # because it was words, not a literal.
    for _cell, _c in _p5["cells"].items():
        _press = _cell.split("-")[0]
        add(
            f"p5.{_press}.random_over_baseline",
            abs(_c["arms"]["random"]["effect_vs_no_intervention"])
            / _c["baseline_mean_divergence"],
            f"|{_p5rel} :: cells.{_cell}.arms.random.effect| / "
            f"cells.{_cell}.baseline_mean_divergence",
            f"random-reinsertion effect as a multiple of baseline "
            f"divergence ({_press})",
            kind="derived",
            inputs=(f"p5.{_press}.random.effect", f"p5.{_press}.baseline_divergence"),
        )

    # Task 3 (Amendment P.3): AUC-TV against the subtraction residual.
    _p3 = _load(P / "auc_resid_tv.json")
    _p3rel = (P / "auc_resid_tv.json").relative_to(PROJECT_ROOT)
    add(
        "p3.pooled.rho",
        _p3["pooled"]["spearman_rho"],
        f"{_p3rel} :: pooled.spearman_rho",
        "Spearman rank correlation, AUC-TV vs residual, pooled across budgets",
    )
    for _i, _b in enumerate(_p3["pooled"]["rho_ci95"]):
        add(
            f"p3.pooled.ci{'lo' if _i == 0 else 'hi'}",
            _b,
            f"{_p3rel} :: pooled.rho_ci95[{_i}]",
            "pooled rank-correlation CI bound",
        )
    for _bud, _w in _p3["within_budget"].items():
        add(
            f"p3.b{_bud}.rho",
            _w["spearman_rho"],
            f"{_p3rel} :: within_budget.{_bud}.spearman_rho",
            f"rank correlation within budget {_bud}",
        )
        for _i, _b in enumerate(_w["rho_ci95"]):
            add(
                f"p3.b{_bud}.ci{'lo' if _i == 0 else 'hi'}",
                _b,
                f"{_p3rel} :: within_budget.{_bud}.rho_ci95[{_i}]",
                f"within-budget-{_bud} rank-correlation CI bound",
            )

    # Task 1 / Amendment P.2: phase-amplitude excess over the shuffled sham.
    for _tag, _relpath, _presses in (
        ("prior", CLASSB / "shape" / "diss256.json", ("streamingllm",)),
        ("ext", P / "shape_ext_b256.json", ("streamingllm",)),
        ("restricted", P / "shape_m3v2_RESTRICTED.json", ("random", "streamingllm")),
        (
            "misaligned",
            P / "shape_m3v2_RESTRICTED_INVALID_phase-rotation.json",
            ("random", "streamingllm"),
        ),
    ):
        _d = _load(_relpath)
        _r = _relpath.relative_to(PROJECT_ROOT)
        for _press in _presses:
            _s = _d["per_policy"][_press]
            add(
                f"phase.{_tag}.{_press}.excess",
                _s["phase_excess"]["excess"],
                f"{_r} :: per_policy.{_press}.phase_excess.excess",
                f"phase-amplitude excess over sham ({_press}, {_tag} window)",
            )
            for _i, _b in enumerate(_s["phase_excess"]["excess_ci95"]):
                add(
                    f"phase.{_tag}.{_press}.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_r} :: per_policy.{_press}.phase_excess.excess_ci95[{_i}]",
                    f"phase-excess CI bound ({_press}, {_tag} window)",
                )

    # M.3 drift, both arms. Uncatalogued until 2026-09-08 because the paper
    # stated the verdict in words; fig:ratiomatched now prints the numbers, and
    # a printed number must be sourced.
    _m3 = _load(P / "m3_verdict.json")
    _m3rel = (P / "m3_verdict.json").relative_to(PROJECT_ROOT)
    for _press in ("random", "streamingllm"):
        _c = _m3["per_press"][_press]
        for _arm in ("fixed", "matched"):
            _stat = _c[f"drift_{_arm}"]
            add(
                f"m3.{_press}.{_arm}.drift",
                _stat["mean"],
                f"{_m3rel} :: per_press.{_press}.drift_{_arm}.mean",
                f"drift (last-quarter/first-quarter mean), {_arm} arm ({_press})",
            )
            for _i, _b in enumerate(_stat["ci95"]):
                add(
                    f"m3.{_press}.{_arm}.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_m3rel} :: per_press.{_press}.drift_{_arm}.ci95[{_i}]",
                    f"drift CI bound, {_arm} arm ({_press})",
                )
    _m3cov = _load(P / "m3v2_coverage.json")
    _m3covrel = (P / "m3v2_coverage.json").relative_to(PROJECT_ROOT)
    add(
        "m3.max_target",
        _m3cov["max_target"],
        f"{_m3covrel} :: max_target",
        "largest per-layer cache target the ratio-matched press reached",
    )
    _m3c = _load(P / "m3_drift_curves.json")
    _m3crel = (P / "m3_drift_curves.json").relative_to(PROJECT_ROOT)
    add(
        "m3.paired_traces_per_press",
        _m3c["per_press"]["random"]["fixed"]["n_traces"],
        f"{_m3crel} :: per_press.random.fixed.n_traces",
        "paired traces per policy on the M.3 above-floor window",
    )

    # Amendment Q.1 sawtooth control. One artifact per arm because
    # phase_amplitude_excess refuses profiles that mix compression intervals.
    Q = DIAG / "amendment-q"
    for _iv in (16, 64, 256):
        _qp = Q / f"shape_sawtooth_i{_iv}.json"
        if not _qp.exists():
            continue
        _qd = _load(_qp)
        _qr = _qp.relative_to(PROJECT_ROOT)
        for _press in ("random", "streamingllm"):
            _ps = _qd["per_policy"][_press]["phase_excess"]
            add(
                f"q1.i{_iv}.{_press}.excess",
                _ps["excess"],
                f"{_qr} :: per_policy.{_press}.phase_excess.excess",
                f"phase-amplitude excess over sham ({_press}, interval {_iv})",
            )
            for _i, _b in enumerate(_ps["excess_ci95"]):
                add(
                    f"q1.i{_iv}.{_press}.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_qr} :: per_policy.{_press}.phase_excess.excess_ci95[{_i}]",
                    f"Q.1 phase-excess CI bound ({_press}, interval {_iv})",
                )

    # Task 4 onset localisation, and the two supporting diagnostics that had no
    # artifact until they were re-derived from the stored profiles on
    # 2026-08-30. Cataloguing them is what lets the paper state them at all.
    _o = _load(P / "onset_ext.json")
    _orel = (P / "onset_ext.json").relative_to(PROJECT_ROOT)
    _b256 = {k: v for k, v in _o["per_cell"].items() if k.endswith("-256")}
    add(
        "p4.crossing_median_tv_jsd",
        _b256["random-256"]["tv"]["median_steps_after_onset"],
        f"{_orel} :: per_cell.random-256.tv.median_steps_after_onset",
        "median first-crossing, TV and JSD, budget 256 (identical by construction)",
    )
    _rd = [c["rank_displacement"]["median_steps_after_onset"] for c in _b256.values()]
    add(
        "p4.rank_displacement_median_min",
        min(_rd),
        f"{_orel} :: min over per_cell.*-256.rank_displacement",
        "earliest rank-displacement median crossing across budget-256 cells",
    )
    add(
        "p4.rank_displacement_median_max",
        max(_rd),
        f"{_orel} :: max over per_cell.*-256.rank_displacement",
        "latest rank-displacement median crossing across budget-256 cells",
    )

    _dg = _load(P / "p2v2_supporting_diagnostics.json")
    _dgrel = (P / "p2v2_supporting_diagnostics.json").relative_to(PROJECT_ROOT)
    add(
        "p2v2.rotation_constant",
        _dg["phase_rotation"]["random"]["constant_shift"],
        f"{_dgrel} :: phase_rotation.random.constant_shift",
        "phase-bucket rotation introduced by the pre-fix onset re-basing",
    )
    add(
        "p2v2.rotation_n_traces",
        _dg["phase_rotation"]["random"]["n_traces"],
        f"{_dgrel} :: phase_rotation.random.n_traces",
        "Random traces over which the rotation was constant",
    )
    add(
        "p4.tvjsd_profiles_checked",
        _dg["tv_jsd_support"]["profiles_checked"],
        f"{_dgrel} :: tv_jsd_support.profiles_checked",
        "budget-256 profiles compared for shared TV/JSD non-zero support",
    )
    add(
        "p4.tvjsd_shared_support",
        _dg["tv_jsd_support"]["profiles_checked"]
        - _dg["tv_jsd_support"]["profiles_where_tv_jsd_nonzero_patterns_differ"],
        f"{_dgrel} :: profiles_checked - profiles_where_..._differ",
        "profiles on which TV and JSD share a non-zero support",
        kind="derived",
        inputs=("p4.tvjsd_profiles_checked",),
    )
    add(
        "p4.tvjsd_differing",
        _dg["tv_jsd_support"]["profiles_where_tv_jsd_nonzero_patterns_differ"],
        f"{_dgrel} :: tv_jsd_support.profiles_where_tv_jsd_nonzero_patterns_differ",
        "profiles whose TV and JSD non-zero supports differ somewhere",
    )

    # Meeting #3 Q.5 exact-lag restoration (R.10), identity-repaired, and who
    # it describes (R.17 addendum A). These replace the retracted P.5 claims.
    M3 = DIAG / "meeting3"
    _el = M3 / "exact_lag.json"
    if _el.exists():
        _eld, _elr = _load(_el), _el.relative_to(PROJECT_ROOT)
        for _press, _pv in _eld["policies"].items():
            for _lag, _lv in _pv["per_lag"].items():
                for _arm, _av in (_lv.get("arms") or {}).items():
                    add(
                        f"m3.el.{_press}.lag{_lag}.{_arm}.effect",
                        _av["effect_vs_no_intervention"],
                        f"{_elr} :: policies.{_press}.per_lag.{_lag}.arms.{_arm}"
                        ".effect_vs_no_intervention",
                        f"exact-lag {_arm} arm, none minus arm ({_press}, lag {_lag})",
                    )
                    for _i, _b in enumerate(_av["effect_ci95"]):
                        add(
                            f"m3.el.{_press}.lag{_lag}.{_arm}."
                            f"ci{'lo' if _i == 0 else 'hi'}",
                            _b,
                            f"{_elr} :: policies.{_press}.per_lag.{_lag}.arms."
                            f"{_arm}.effect_ci95[{_i}]",
                            f"exact-lag {_arm} CI bound ({_press}, lag {_lag})",
                        )
    _eg = M3 / "r17" / "exact_lag_eligibility.json"
    if _eg.exists():
        _egd, _egr = _load(_eg), _eg.relative_to(PROJECT_ROOT)
        for _s in ("eligible", "ineligible"):
            _v = _egd[_s]
            add(
                f"m3.el.{_s}.fc_accuracy",
                _v["fc_accuracy"]["mean"],
                f"{_egr} :: {_s}.fc_accuracy.mean",
                f"full-cache accuracy of the {_s} restoration traces",
            )
            for _i, _bd in enumerate(_v["fc_accuracy"]["ci95"]):
                add(
                    f"m3.el.{_s}.fc_accuracy.ci{'lo' if _i == 0 else 'hi'}",
                    _bd,
                    f"{_egr} :: {_s}.fc_accuracy.ci95[{_i}]",
                    f"full-cache accuracy CI bound, {_s} restoration traces",
                )
            add(
                f"m3.el.{_s}.n_problems",
                _v["n_problems"],
                f"{_egr} :: {_s}.n_problems",
                f"{_s} problems in the exact-lag read",
            )
            add(
                f"m3.el.{_s}.n_traces",
                _v["n_traces"],
                f"{_egr} :: {_s}.n_traces",
                f"{_s} traces in the exact-lag read",
            )
            add(
                f"m3.el.{_s}.realized_ratio_pct",
                100 * _v["realized_ratio"],
                f"100 * {_egr} :: {_s}.realized_ratio",
                f"realized cache ratio (%) on the {_s} traces at budget 256",
                kind="derived",
            )

    # R.17 mechanism-section literals (capture-b256 FINAL read and the R.17
    # reads on disk). Every value is taken from its artifact, never retyped.
    _cap = M3 / "capture_b256.json"
    if _cap.exists():
        _cd, _cr = _load(_cap), _cap.relative_to(PROJECT_ROOT)
        for _f in ("M", "Delta"):
            for _press, _s in _cd["descriptive"][_f].items():
                add(
                    f"m3.cap256.{_f}.{_press}.mean",
                    _s["mean"],
                    f"{_cr} :: descriptive.{_f}.{_press}.mean",
                    f"capture b256 {_f}, {_press}, equal-problem mean",
                )
                for _i, _b in enumerate(_s["ci95"]):
                    add(
                        f"m3.cap256.{_f}.{_press}.ci{'lo' if _i == 0 else 'hi'}",
                        _b,
                        f"{_cr} :: descriptive.{_f}.{_press}.ci95[{_i}]",
                        f"capture b256 {_f} CI bound, {_press}",
                    )
        _q6 = _cd["q6_registered_contrast"]
        add(
            "m3.cap256.q6.mean",
            _q6["mean"],
            f"{_cr} :: q6_registered_contrast.mean",
            "omega(Random) - omega(StreamingLLM), b256",
        )
        for _i, _b in enumerate(_q6["ci95"]):
            add(
                f"m3.cap256.q6.ci{'lo' if _i == 0 else 'hi'}",
                _b,
                f"{_cr} :: q6_registered_contrast.ci95[{_i}]",
                "omega contrast CI bound, b256",
            )
    R17 = M3 / "r17"
    _md = R17 / "mdelta.json"
    if _md.exists():
        _mdd, _mdr = _load(_md), _md.relative_to(PROJECT_ROOT)
        add(
            "m3.cap256.realized_ratio_pct",
            100 * _mdd["realized_ratio"]["ratio"],
            f"100 * {_mdr} :: realized_ratio.ratio",
            "capture b256 realized ratio (%), budget / (mean prompt + mean trace)",
            kind="derived",
        )
        add(
            "m3.h3.mean",
            _mdd["H3"]["mean"],
            f"{_mdr} :: H3.mean",
            "M(H2O-style) - M(Random), b256 (exploratory)",
        )
        for _i, _b in enumerate(_mdd["H3"]["ci95"]):
            add(
                f"m3.h3.ci{'lo' if _i == 0 else 'hi'}",
                _b,
                f"{_mdr} :: H3.ci95[{_i}]",
                "H3 CI bound",
            )
    _md2 = R17 / "mdelta_capture-b2048.json"
    if _md2.exists():
        _m2d, _m2r = _load(_md2), _md2.relative_to(PROJECT_ROOT)
        add(
            "m3.cap2048.n_no_event_per_policy",
            _m2d["excluded_no_measured_event"]["random"],
            f"{_m2r} :: excluded_no_measured_event.random",
            "b2048 traces per policy with no eviction event (unavailable, R.4)",
        )
    if _md2.exists():
        add(
            "m3.h3_2048.mean",
            _m2d["H3"]["mean"],
            f"{_m2r} :: H3.mean",
            "M(H2O-style) - M(Random), b2048 (confirmatory, R.17.1.5)",
        )
        for _i, _b in enumerate(_m2d["H3"]["ci95"]):
            add(
                f"m3.h3_2048.ci{'lo' if _i == 0 else 'hi'}",
                _b,
                f"{_m2r} :: H3.ci95[{_i}]",
                "H3 b2048 CI bound",
            )
        add(
            "m3.cap2048.realized_ratio_pct",
            100 * _m2d["realized_ratio"]["ratio"],
            f"100 * {_m2r} :: realized_ratio.ratio",
            "budget 2048 as a percentage of mean prompt + mean trace tokens",
            kind="derived",
        )
        for _f in ("M", "Delta"):
            add(
                f"m3.h2_2048.{_f}.spread",
                _m2d["H2"][_f]["spread"],
                f"{_m2r} :: H2.{_f}.spread",
                f"H2 b2048 spread of {_f} across the four policies",
            )
    _cb2 = M3 / "capture_b2048.json"
    if _cb2.exists():
        _cbd, _cbr = _load(_cb2), _cb2.relative_to(PROJECT_ROOT)
        _q6 = _cbd["q6_registered_contrast"]
        add(
            "m3.cap2048.omega_contrast.mean",
            _q6["mean"],
            f"{_cbr} :: q6_registered_contrast.mean",
            "omega(Random) - omega(StreamingLLM), b2048 replication (R.17.1.5)",
        )
        for _i, _b in enumerate(_q6["ci95"]):
            add(
                f"m3.cap2048.omega_contrast.ci{'lo' if _i == 0 else 'hi'}",
                _b,
                f"{_cbr} :: q6_registered_contrast.ci95[{_i}]",
                "omega contrast b2048 CI bound",
            )
        add(
            "m3.cap2048.n_problems",
            _q6["n_problems"],
            f"{_cbr} :: q6_registered_contrast.n_problems",
            "problems with a measured event at b2048",
        )
        add(
            "m3.cap2048.n_pairs",
            _q6["n_pairs"],
            f"{_cbr} :: q6_registered_contrast.n_pairs",
            "paired traces at b2048",
        )
    _ex = R17 / "capture_v2_extras_capture-b2048.json"
    if _ex.exists():
        _exd, _exr = _load(_ex), _ex.relative_to(PROJECT_ROOT)
        for _k, _v in _exd["omega_by_k_random_minus_slm"]["per_k"].items():
            add(
                f"m3.omega_k.{_k}.mean",
                _v["contrast"]["mean"],
                f"{_exr} :: omega_by_k_random_minus_slm.per_k.{_k}.contrast.mean",
                f"omega contrast at nested k={_k}, b2048 (R.17.1.7)",
            )
            for _i, _b in enumerate(_v["contrast"]["ci95"]):
                add(
                    f"m3.omega_k.{_k}.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_exr} :: omega_by_k_random_minus_slm.per_k.{_k}"
                    f".contrast.ci95[{_i}]",
                    f"omega contrast CI bound at k={_k}",
                )
    _co = R17 / "collapse.json"
    if _co.exists():
        _cod, _cor = _load(_co), _co.relative_to(PROJECT_ROOT)
        for _press in ("random", "streamingllm"):
            _pp = _cod["per_policy"][_press]["pooled"]
            add(
                f"m3.collapse.{_press}.delta_r2",
                _pp["delta_r2_budget_after_r"],
                f"{_cor} :: per_policy.{_press}.pooled.delta_r2_budget_after_r",
                f"collapse test: R^2 gain of budget after log r, {_press} (R.17.1.1)",
            )
            for _i, _b in enumerate(_pp["delta_r2_ci95"]):
                add(
                    f"m3.collapse.{_press}.delta_r2.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_cor} :: per_policy.{_press}.pooled.delta_r2_ci95[{_i}]",
                    f"collapse test delta R^2 CI bound, {_press}",
                )
            add(
                f"m3.collapse.{_press}.n_problems",
                _cod["per_policy"][_press]["n_problems"],
                f"{_cor} :: per_policy.{_press}.n_problems",
                f"collapse test panel problems, {_press}",
            )
    _rv = R17 / "restore_v2.json"
    if _rv.exists():
        _rvd, _rvr = _load(_rv), _rv.relative_to(PROJECT_ROOT)
        for _press in ("random", "streamingllm"):
            _fs = _rvd["per_policy"][_press]["full_restore"]["R_fs"]
            add(
                f"m3.restore2.{_press}.R_fs",
                _fs["ratio"],
                f"{_rvr} :: per_policy.{_press}.full_restore.R_fs.ratio",
                f"restore-v2 full self-restoration ratio R_fs, {_press} (R.17.2.3)",
            )
            for _i, _b in enumerate(_fs["ci95"]):
                add(
                    f"m3.restore2.{_press}.R_fs.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_rvr} :: per_policy.{_press}.full_restore.R_fs.ci95[{_i}]",
                    f"R_fs CI bound, {_press}",
                )
        _el = _rvd["per_policy"]["random"]["eligibility"]["eligible"]
        add(
            "m3.restore2.eligible.n_problems",
            _el["n_problems"],
            f"{_rvr} :: per_policy.random.eligibility.eligible.n_problems",
            "restore-v2 eligible problems",
        )
        add(
            "m3.restore2.eligible.fc_accuracy",
            _el["fc_accuracy"]["mean"],
            f"{_rvr} :: per_policy.random.eligibility.eligible.fc_accuracy.mean",
            "restore-v2 eligible subset full-cache accuracy",
        )
    _cl = M3 / "capture_llama8b_b256.json"
    if _cl.exists():
        _cld, _clr = _load(_cl), _cl.relative_to(PROJECT_ROOT)
        _q6l = _cld["q6_registered_contrast"]
        add(
            "m3.llama.omega_contrast.mean",
            _q6l["mean"],
            f"{_clr} :: q6_registered_contrast.mean",
            "omega(Random) - omega(StreamingLLM) on Llama-8B, b256 (R.17.3; REVERSED)",
        )
        for _i, _b in enumerate(_q6l["ci95"]):
            add(
                f"m3.llama.omega_contrast.ci{'lo' if _i == 0 else 'hi'}",
                _b,
                f"{_clr} :: q6_registered_contrast.ci95[{_i}]",
                "Llama omega contrast CI bound",
            )
        for _p in ("random", "streamingllm", "h2o", "rkv"):
            for _f in ("M", "Delta", "omega"):
                add(
                    f"m3.llama.{_f}.{_p}.mean",
                    _cld["descriptive"][_f][_p]["mean"],
                    f"{_clr} :: descriptive.{_f}.{_p}.mean",
                    f"Llama-8B b256 {_f} for {_p} (descriptive)",
                )
    _mdl = R17 / "mdelta_capture-llama8b-b256.json"
    if _mdl.exists():
        _mdld, _mdlr = _load(_mdl), _mdl.relative_to(PROJECT_ROOT)
        add(
            "m3.llama.h3.mean",
            _mdld["H3"]["mean"],
            f"{_mdlr} :: H3.mean",
            "H3 on Llama-8B b256 (confirmatory)",
        )
        for _i, _b in enumerate(_mdld["H3"]["ci95"]):
            add(
                f"m3.llama.h3.ci{'lo' if _i == 0 else 'hi'}",
                _b,
                f"{_mdlr} :: H3.ci95[{_i}]",
                "H3 Llama CI bound",
            )
        add(
            "m3.llama.realized_ratio_pct",
            100 * _mdld["realized_ratio"]["ratio"],
            f"100 * {_mdlr} :: realized_ratio.ratio",
            "budget 256 as a percentage of mean prompt + trace tokens on Llama-8B",
            kind="derived",
        )
    _dr = R17 / "dacc_restricted.json"
    if _dr.exists():
        _drd, _drr = _load(_dr), _dr.relative_to(PROJECT_ROOT)
        for _press in ("random", "streamingllm"):
            _c = _drd["cells"][f"{_press}-256"]["t0_0_vs_2048"]
            add(
                f"m3.dacc.onset.{_press}.unrestricted",
                _c["unrestricted"]["mean"],
                f"{_drr} :: cells.{_press}-256.t0_0_vs_2048.unrestricted.mean",
                f"D_acc onset contrast t0 0 vs 2048, B256, {_press}, all pairs",
            )
            add(
                f"m3.dacc.onset.{_press}.restricted",
                _c["restricted"]["mean"],
                f"{_drr} :: cells.{_press}-256.t0_0_vs_2048.restricted.mean",
                f"same, restricted to reference > 2304 tokens, {_press}",
            )
            for _i, _b in enumerate(_c["restricted"]["ci95"]):
                add(
                    f"m3.dacc.onset.{_press}.restricted.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_drr} :: cells.{_press}-256.t0_0_vs_2048.restricted.ci95[{_i}]",
                    f"restricted onset contrast CI bound, {_press}",
                )
    if _dr.exists():
        for _press in ("random", "streamingllm"):
            _c = _drd["cells"][f"{_press}-256"]["t0_0_vs_2048"]["unrestricted"]
            for _i, _b in enumerate(_c["ci95"]):
                _lh = "lo" if _i == 0 else "hi"
                add(
                    f"m3.dacc.onset.{_press}.unrestricted.ci{_lh}",
                    _b,
                    f"{_drr} :: cells.{_press}-256.t0_0_vs_2048"
                    f".unrestricted.ci95[{_i}]",
                    f"unrestricted onset contrast CI bound, {_press}",
                )
        _rf = _drd["retained_fc_accuracy"]
        add(
            "m3.dacc.retained_fc_accuracy",
            _rf["mean"],
            f"{_drr} :: retained_fc_accuracy.mean",
            "full-cache accuracy of the pairs with reference > 2304 tokens",
        )
        for _i, _b in enumerate(_rf["ci95"]):
            add(
                f"m3.dacc.retained_fc_accuracy.ci{'lo' if _i == 0 else 'hi'}",
                _b,
                f"{_drr} :: retained_fc_accuracy.ci95[{_i}]",
                "retained-pair full-cache accuracy CI bound",
            )
    if _md.exists():
        for _f in ("M", "Delta"):
            for _q in ("spread", "median_half_width"):
                add(
                    f"m3.h2.{_f}.{_q}",
                    _mdd["H2"][_f][_q],
                    f"{_mdr} :: H2.{_f}.{_q}",
                    f"H2 {_q} of {_f} across the four policies, b256",
                )
    _at = R17 / "auc_tv.json"
    if _at.exists():
        _atd, _atr = _load(_at), _at.relative_to(PROJECT_ROOT)
        for _cell, _v in _atd["cells"].items():
            add(
                f"m3.auc.{_cell}.mean",
                _v["mean"],
                f"{_atr} :: cells.{_cell}.mean",
                f"AUC-TV (per-trace mean TV), {_cell}",
            )
            for _i, _b in enumerate(_v["ci95"]):
                add(
                    f"m3.auc.{_cell}.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_atr} :: cells.{_cell}.ci95[{_i}]",
                    f"AUC-TV CI bound, {_cell}",
                )
    _fp = R17 / "fork_precision.json"
    if _fp.exists() and _load(_fp).get("status") == "READ":
        _fpd, _fpr = _load(_fp), _fp.relative_to(PROJECT_ROOT)
        for _iv, _fl in _fpd["floor"].items():
            add(
                f"m3.fp.floor.i{_iv}.p95",
                _fl["p95"],
                f"{_fpr} :: floor.{_iv}.p95",
                f"precision floor: p95 window-max TV, windows of {_iv}",
            )
        for _arm, _v in _fpd["gate"]["arms"].items():
            for _q in ("ratio", "fork_p95_max_tv"):
                add(
                    f"m3.fp.{_arm}.{_q}",
                    _v[_q],
                    f"{_fpr} :: gate.arms.{_arm}.{_q}",
                    f"R.17.0.5 gate {_q}, {_arm}",
                )
        for _q in ("mean", "median", "p95", "p99", "max"):
            add(
                f"m3.fp.floor_all_steps.{_q}",
                _fpd["floor_all_steps_tv"][_q],
                f"{_fpr} :: floor_all_steps_tv.{_q}",
                f"bf16-vs-fp32 full-cache per-step TV, {_q}",
            )
        for _t, _v in _fpd["diagnosis"].items():
            for _q in ("max_tv", "max_abs_logit", "median_argmax_rank_in_b"):
                add(
                    f"m3.fp.diag.{_t}.{_q}",
                    _v[_q],
                    f"{_fpr} :: diagnosis.{_t}.{_q}",
                    f"first-fork diagnosis {_q}, {_t}",
                )
    _ol = R17 / "omega_layers.json"
    if _ol.exists():
        _old, _olr = _load(_ol), _ol.relative_to(PROJECT_ROOT)
        _lc = _old["components"]["log_perp_over_par"]
        add(
            "m3.omega.log_perp_over_par.mean",
            _lc["mean"],
            f"{_olr} :: components.log_perp_over_par.mean",
            "Random - SLM contrast of log(|d_perp| / |d_par|)",
        )
        for _i, _b in enumerate(_lc["ci95"]):
            add(
                f"m3.omega.log_perp_over_par.ci{'lo' if _i == 0 else 'hi'}",
                _b,
                f"{_olr} :: components.log_perp_over_par.ci95[{_i}]",
                "log(perp/par) contrast CI bound",
            )
    _br = R17 / "bridge.json"
    if _br.exists():
        _brd, _brr = _load(_br), _br.relative_to(PROJECT_ROOT)
        _au = [t["auroc"] for t in _brd["tests"]]
        add(
            "m3.bridge.auroc_min",
            min(_au),
            f"min over {_brr} :: tests[].auroc",
            "smallest of the six bridge AUROCs",
            kind="derived",
        )
        add(
            "m3.bridge.auroc_max",
            max(_au),
            f"max over {_brr} :: tests[].auroc",
            "largest of the six bridge AUROCs",
            kind="derived",
        )
        add(
            "m3.bridge.q_min",
            min(t["q_bh"] for t in _brd["tests"]),
            f"min over {_brr} :: tests[].q_bh",
            "smallest Benjamini-Hochberg q of the six bridge tests",
            kind="derived",
        )

    # Amendment S.1 (2026-10-04) and the execution-plan protocol audit
    # (2026-10-05): protected residuals and the paired direction contrasts.
    _s1 = DIAG / "amendment-s" / "protection_b256.json"
    if _s1.exists():
        _s1d, _s1r = _load(_s1), _s1.relative_to(PROJECT_ROOT)
        for _pp, _blk in _s1d["per_press"].items():
            _pr = _blk["primary"]
            for _f in ("residual_pp", "residual_base", "delta_residual"):
                add(
                    f"s1.b256.{_pp}.{_f}.mean",
                    _pr[_f]["mean"],
                    f"{_s1r} :: per_press.{_pp}.primary.{_f}.mean",
                    f"S.1 b256 {_f} for {_pp} (equal-problem)",
                )
                for _i, _b in enumerate(_pr[_f]["ci95"]):
                    add(
                        f"s1.b256.{_pp}.{_f}.ci{'lo' if _i == 0 else 'hi'}",
                        _b,
                        f"{_s1r} :: per_press.{_pp}.primary.{_f}.ci95[{_i}]",
                        f"S.1 b256 {_f} CI bound, {_pp}",
                    )
    # 2026-10-15: S.1 at budget 1024 (forward) and the realized compression of
    # every S.1 grid cell. Separate keys; the b256 entries above are unchanged.
    _s14 = DIAG / "amendment-s" / "protection_b1024.json"
    if _s14.exists():
        _s14d, _s14r = _load(_s14), _s14.relative_to(PROJECT_ROOT)
        for _pp, _blk in _s14d["per_press"].items():
            _pr = _blk["primary"]
            for _f in (
                "residual_pp",
                "residual_base",
                "delta_residual",
                "delta_free",
                "delta_readout",
                "free_accuracy_pp",
                "free_accuracy_base",
                "readout_accuracy_pp",
                "readout_accuracy_base",
            ):
                add(
                    f"s1.b1024.{_pp}.{_f}.mean",
                    _pr[_f]["mean"],
                    f"{_s14r} :: per_press.{_pp}.primary.{_f}.mean",
                    f"S.1 b1024 {_f} for {_pp} (equal-problem)",
                )
                for _i, _b in enumerate(_pr[_f]["ci95"]):
                    add(
                        f"s1.b1024.{_pp}.{_f}.ci{'lo' if _i == 0 else 'hi'}",
                        _b,
                        f"{_s14r} :: per_press.{_pp}.primary.{_f}.ci95[{_i}]",
                        f"S.1 b1024 {_f} CI bound, {_pp}",
                    )
            add(
                f"s1.b1024.{_pp}.fc_accuracy_on_cell",
                _pr["fc_accuracy_on_cell"],
                f"{_s14r} :: per_press.{_pp}.primary.fc_accuracy_on_cell",
                f"S.1 b1024 full-cache accuracy on the cell, {_pp}",
            )
            for _arm in ("flip_mass_base", "flip_mass_pp"):
                for _q in ("harmful", "beneficial"):
                    add(
                        f"s1.b1024.{_pp}.{_arm}.{_q}",
                        _blk[_arm][_q],
                        f"{_s14r} :: per_press.{_pp}.{_arm}.{_q}",
                        f"S.1 b1024 {_q} flip mass ({_arm}), {_pp}",
                    )
    _rc = DIAG / "amendment-s" / "realized_compression.json"
    # nominal-budget ratios (review of 5616630, R4): ratio-1 counts are a lower
    # bound on never-compressed conditions; only a FINAL, evidence-bound read
    if _rc.exists() and _load(_rc).get("status") == "FINAL":
        _rcd, _rcr = _load(_rc), _rc.relative_to(PROJECT_ROOT)
        for _bud, _rcells in _rcd["cells"].items():
            for _pp, _c in _rcells.items():
                for _f in ("mean", "median", "n_ratio_one", "ratio_one_share"):
                    add(
                        f"s1.realized.{_bud}.{_pp}.{_f}",
                        _c[_f],
                        f"{_rcr} :: cells.{_bud}.{_pp}.{_f}",
                        f"nominal-budget ratio {_f}, {_bud} {_pp}",
                    )
                if "effective" in _c:
                    for _f in ("mean", "median"):
                        add(
                            f"s1.realized.{_bud}.{_pp}.effective.{_f}",
                            _c["effective"][_f],
                            f"{_rcr} :: cells.{_bud}.{_pp}.effective.{_f}",
                            f"effective-budget ratio {_f} (S.1 relaxation), "
                            f"{_bud} {_pp}",
                        )
    _dc4 = DIAG / "paper_audit" / "decision_b1024.json"
    if _dc4.exists():
        _dc4d, _dc4p = _load(_dc4), _dc4.relative_to(PROJECT_ROOT)
        for _pp, _blk in _dc4d["per_press"].items():
            for _arm in ("practical_protected", "practical_unprotected"):
                for _q in ("R_loss", "F_loss", "residual"):
                    add(
                        f"audit.decision.b1024.{_pp}.{_arm}.{_q}",
                        _blk[_arm][_q]["mean"],
                        f"{_dc4p} :: per_press.{_pp}.{_arm}.{_q}.mean",
                        f"{_q} ({_arm}), {_pp}, b1024 (SENSITIVITY margins)",
                    )
                    for _i, _b in enumerate(_blk[_arm][_q]["ci95"]):
                        add(
                            f"audit.decision.b1024.{_pp}.{_arm}.{_q}"
                            f".ci{'lo' if _i == 0 else 'hi'}",
                            _b,
                            f"{_dc4p} :: per_press.{_pp}.{_arm}.{_q}.ci95[{_i}]",
                            f"{_q} CI bound ({_arm}), {_pp}, b1024",
                        )
    _rv = DIAG / "paper_audit" / "reverse_b256.json"
    if _rv.exists():
        _rvd, _rvp = _load(_rv), _rv.relative_to(PROJECT_ROOT)
        for _pp, _blk in _rvd["protection_by_direction"].items():
            for _c in ("residual_forward", "trajectory_under_full_readout"):
                add(
                    f"audit.b256.{_pp}.{_c}.shrinkage",
                    _blk[_c]["ratio"],
                    f"{_rvp} :: protection_by_direction.{_pp}.{_c}.ratio",
                    f"protection shrinkage ratio of {_c}, {_pp}, four-arm support",
                )
        for _pp, _blk in _rvd["protection_by_direction"].items():
            for _c in ("residual_forward", "trajectory_under_full_readout"):
                for _i, _b in enumerate(_blk[_c]["ci95"]):
                    _end = "lo" if _i == 0 else "hi"
                    add(
                        f"audit.b256.{_pp}.{_c}.shrinkage.ci{_end}",
                        _b,
                        f"{_rvp} :: protection_by_direction.{_pp}.{_c}.ci95[{_i}]",
                        f"protection shrinkage CI bound, {_c}, {_pp}",
                    )
            _di = _blk["difference_of_improvements"]
            add(
                f"audit.b256.{_pp}.difference_of_improvements.mean",
                _di["mean"],
                f"{_rvp} :: protection_by_direction.{_pp}"
                ".difference_of_improvements.mean",
                f"forward minus reverse protection improvement, {_pp}",
            )
            for _i, _b in enumerate(_di["ci95"]):
                _end = "lo" if _i == 0 else "hi"
                add(
                    f"audit.b256.{_pp}.difference_of_improvements.ci{_end}",
                    _b,
                    f"{_rvp} :: protection_by_direction.{_pp}"
                    f".difference_of_improvements.ci95[{_i}]",
                    f"difference of protection improvements CI bound, {_pp}",
                )
            for _c in ("residual_forward", "trajectory_under_full_readout"):
                add(
                    f"audit.b256.{_pp}.{_c}.improvement",
                    _blk[_c]["absolute_change"]["mean"],
                    f"{_rvp} :: protection_by_direction.{_pp}.{_c}"
                    ".absolute_change.mean",
                    f"absolute protection improvement, {_c}, {_pp}",
                )
            add(
                f"audit.b256.{_pp}.four_arm.n",
                _blk["n_conditions"],
                f"{_rvp} :: protection_by_direction.{_pp}.n_conditions",
                f"four-arm paired support, {_pp}",
            )
        for _pp, _blk in _rvd["per_policy"].items():
            _cs = _blk["common_support"]
            add(
                f"audit.b256.{_pp}.common.n",
                _cs["n_conditions"],
                f"{_rvp} :: per_policy.{_pp}.common_support.n_conditions",
                f"2x2 common support size, {_pp}",
            )
            for _c in (
                "residual_forward",
                "trajectory_under_full_readout",
                "interaction",
            ):
                add(
                    f"audit.b256.{_pp}.common.{_c}.mean",
                    _cs[_c]["mean"],
                    f"{_rvp} :: per_policy.{_pp}.common_support.{_c}.mean",
                    f"2x2 {_c} on the common support, {_pp}",
                )
                for _i, _b in enumerate(_cs[_c]["ci95"]):
                    add(
                        f"audit.b256.{_pp}.common.{_c}.ci{'lo' if _i == 0 else 'hi'}",
                        _b,
                        f"{_rvp} :: per_policy.{_pp}.common_support.{_c}.ci95[{_i}]",
                        f"2x2 {_c} CI bound, {_pp}",
                    )
    # P1: the protected B=1024 reverse read (2026-10-16), its REGISTERED
    # problem-level t test + Holm, and the 2x2 on the common support
    _rv4 = DIAG / "paper_audit" / "reverse_b1024.json"
    if _rv4.exists():
        _rv4d, _rv4p = _load(_rv4), _rv4.relative_to(PROJECT_ROOT)
        for _pp, _m in _rv4d["registered_inference"]["members"].items():
            for _f in ("mean", "t", "df", "p", "p_holm", "n_problems", "n_conditions"):
                add(
                    f"audit.b1024.{_pp}.registered.{_f}",
                    _m[_f],
                    f"{_rv4p} :: registered_inference.members.{_pp}.{_f}",
                    f"registered B=1024 interaction t test {_f}, {_pp}",
                )
        for _pp, _blk in _rv4d["per_policy"].items():
            _cs = _blk["common_support"]
            for _c in (
                "residual_forward",
                "trajectory_under_full_readout",
                "readout_on_compressed_trace",
                "readout_on_reference_trace",
                "interaction",
            ):
                add(
                    f"audit.b1024.{_pp}.common.{_c}.mean",
                    _cs[_c]["mean"],
                    f"{_rv4p} :: per_policy.{_pp}.common_support.{_c}.mean",
                    f"B=1024 2x2 {_c} on the common support, {_pp}",
                )
                for _i, _b in enumerate(_cs[_c]["ci95"]):
                    add(
                        f"audit.b1024.{_pp}.common.{_c}.ci{'lo' if _i == 0 else 'hi'}",
                        _b,
                        f"{_rv4p} :: per_policy.{_pp}.common_support.{_c}.ci95[{_i}]",
                        f"B=1024 2x2 {_c} CI bound, {_pp}",
                    )
            for _i, _b in enumerate(_blk["bounds_full_population"]["interaction"]):
                _end = "lo" if _i == 0 else "hi"
                add(
                    f"audit.b1024.{_pp}.bounds.interaction.{_end}",
                    _b,
                    f"{_rv4p} :: per_policy.{_pp}.bounds_full_population"
                    f".interaction[{_i}]",
                    f"B=1024 worst-case interaction bound (identification), {_pp}",
                )
            _rs = _blk["reverse_support"]
            _t = _rs["trajectory_under_full_readout"]
            add(
                f"audit.b1024.{_pp}.reverse.trajectory.mean",
                _t["mean"],
                f"{_rv4p} :: per_policy.{_pp}.reverse_support"
                ".trajectory_under_full_readout.mean",
                f"B=1024 trajectory contrast under full readout (own support), {_pp}",
            )
            for _i, _b in enumerate(_t["ci95"]):
                _end = "lo" if _i == 0 else "hi"
                add(
                    f"audit.b1024.{_pp}.reverse.trajectory.ci{_end}",
                    _b,
                    f"{_rv4p} :: per_policy.{_pp}.reverse_support"
                    f".trajectory_under_full_readout.ci95[{_i}]",
                    f"B=1024 reverse trajectory contrast CI bound, {_pp}",
                )
            add(
                f"audit.b1024.{_pp}.reverse.n_problems",
                _rs["n_problems"],
                f"{_rv4p} :: per_policy.{_pp}.reverse_support.n_problems",
                f"B=1024 reverse-readout support problems, {_pp}",
            )
            add(
                f"audit.b1024.{_pp}.reverse.n",
                _rs["n_conditions"],
                f"{_rv4p} :: per_policy.{_pp}.reverse_support.n_conditions",
                f"B=1024 reverse-readout support, {_pp}",
            )
    _dc = DIAG / "paper_audit" / "decision_b256.json"
    if _dc.exists():
        _dcd, _dcp = _load(_dc), _dc.relative_to(PROJECT_ROOT)
        for _pp, _blk in _dcd["per_press"].items():
            _sh = _blk["shrinkage"]
            add(
                f"audit.decision.{_pp}.shrinkage",
                _sh["ratio"],
                f"{_dcp} :: per_press.{_pp}.shrinkage.ratio",
                f"S.1 forward shrinkage ratio, {_pp}",
            )
            for _i, _b in enumerate(_sh["ci95"]):
                add(
                    f"audit.decision.{_pp}.shrinkage.ci{'lo' if _i == 0 else 'hi'}",
                    _b,
                    f"{_dcp} :: per_press.{_pp}.shrinkage.ci95[{_i}]",
                    f"S.1 forward shrinkage CI bound, {_pp}",
                )
            for _arm in ("practical_protected", "practical_unprotected"):
                for _q in ("R_loss", "F_loss"):
                    add(
                        f"audit.decision.{_pp}.{_arm}.{_q}",
                        _blk[_arm][_q]["mean"],
                        f"{_dcp} :: per_press.{_pp}.{_arm}.{_q}.mean",
                        f"{_q} ({_arm}), {_pp}",
                    )
    _mk = DIAG / "paper_audit" / "matched_b256.json"
    if _mk.exists():
        _mkd, _mkp = _load(_mk), _mk.relative_to(PROJECT_ROOT)
        for _pair, _blk in _mkd["contrasts_M"].items():
            _tag = _pair.replace(" - ", "_minus_")
            for _f in ("accuracy", "metric"):
                add(
                    f"audit.matched.{_tag}.{_f}.mean",
                    _blk[_f]["mean"],
                    f"{_mkp} :: contrasts_M[{_pair!r}].{_f}.mean",
                    f"matched-key {_f} contrast, {_pair}",
                )
                for _ci in ("ci95", "ci_family"):
                    for _i, _b in enumerate(_blk[_f][_ci]):
                        _end = "lo" if _i == 0 else "hi"
                        add(
                            f"audit.matched.{_tag}.{_f}.{_ci}{_end}",
                            _b,
                            f"{_mkp} :: contrasts_M[{_pair!r}].{_f}.{_ci}[{_i}]",
                            f"matched-key {_f} {_ci} bound, {_pair}",
                        )

    _fc = _load(P / "m3_INVALID_floor_collapse.json")
    add(
        "m3v1.compression_events",
        _fc["n_compression_events"],
        f"{(P / 'm3_INVALID_floor_collapse.json').relative_to(PROJECT_ROOT)} "
        ":: n_compression_events",
        "compression events in the invalidated M.3 v1 cell, all pinned to the floor",
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

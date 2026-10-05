#!/usr/bin/env python3
"""Amendment S reads over rows already on disk (registered 2026-09-30; readers
repaired 2026-10-02 after the independent review of f69dce3).

Thin orchestration: load rows, pass them through ``kvcache.read_validation``,
call ``kvcache.metrics.s_reads``, write one JSON plus a per-condition TSV per
read under ``results/diag/amendment-s/``. Runs where the rows live (the GPU
host) and needs no GPU.

Status contract. Every output carries ``status``: FINAL (complete coverage
against the frozen design, every control passed, nothing malformed), PARTIAL
(incomplete but clean; NO scientific label), or INVALID (duplicates,
unexpected or malformed rows, a failed control). Exit 0 on FINAL; on PARTIAL
exit 3 unless ``--partial`` was given; exit 2 on INVALID. A previous output of
the same name is retired (renamed) before any write, so a stale FINAL can never
outlive a failed or changed-input read. ``consequence`` accepts only FINAL
protection reads whose recorded input hash still matches the rows on disk.

    python scripts/report_s.py --runs-root <runs> protection --budget 256
    python scripts/report_s.py --runs-root <runs> protection --budget 1024
    python scripts/report_s.py --runs-root <runs> consequence
    python scripts/report_s.py --runs-root <runs> origin [--partial]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from report_m4 import _jsonl, _run_key

from kvcache.coverage import COMPLETE, equal_problem_bootstrap
from kvcache.metrics import s_reads
from kvcache.presses.prompt_protect import PP_BASES, PP_SUFFIX, effective_budget
from kvcache.provenance import write_json_atomic
from kvcache.read_checks import validate_rows
from kvcache.read_validation import (
    FINAL,
    INVALID,
    PARTIAL,
    dependency_fingerprint,
    design_keys,
    exit_code,
    final_status,
    legacy_provenance_summary,
    provenance_v1_summary,
    restoration_row_problems,
    retire_stale,
    tree_files,
    validate_population,
)

OUT = PROJECT_ROOT / "results" / "diag" / "amendment-s"
FC_RUN = "math500-fc-powered-merged"
BENCH = "math500"
CELLS = {
    256: {
        "grid_pp": "grid-pp256",
        "grid_pp_config": "configs/experiments/grid_math500_pp256_s1.yaml",
        "readout_pp": ["readout-pp256"],
        "grid_base": "grid-diss256-tok",
        "grid_base_config": "configs/experiments/grid_math500_dissociation.yaml",
        "readout_base": ["readout-diss256-g0", "readout-diss256-g1"],
    },
    1024: {
        "grid_pp": "grid-pp1024",
        "grid_pp_config": "configs/experiments/grid_math500_pp1024_s1.yaml",
        "readout_pp": ["readout-pp1024"],
        "grid_base": "grid-diss1024",
        "grid_base_config": "configs/experiments/grid_math500_diss1024.yaml",
        "readout_base": ["readout-diss1024"],
    },
}
ORIGIN_CONFIG = "configs/experiments/restore_v3_math500_s2.yaml"
RESTORE2_CONFIG = "configs/experiments/restore_v2_math500_r17.yaml"
# Everything the reads import or read besides rows: a change to any of these
# invalidates a stored read (review of 49c572c, issue 2).
ANALYSIS_CODE = (
    "scripts/report_s.py",
    "scripts/report_m4.py",
    "src/kvcache/metrics/s_reads.py",
    "src/kvcache/read_validation.py",
    "src/kvcache/read_checks.py",
    "src/kvcache/coverage.py",
    "src/kvcache/presses/prompt_protect.py",
    "src/kvcache/provenance.py",
    "src/kvcache/gates.py",
)
FROZEN_SPECS = (
    "configs/meeting3/grid100_pids.json",
    "configs/meeting3/q2_subset_pids.json",
    "configs/meeting3/input_manifest.json",
    "configs/models.yaml",
)


def _project_files(paths: tuple[str, ...] | list[str]) -> list[Path]:
    return [PROJECT_ROOT / p for p in paths]


def protection_dependencies(runs: Path, budget: int) -> dict[str, Any]:
    """Every input the protection read consumes, one fingerprint (shared with
    ``consequence``, which recomputes it before accepting a stored read)."""
    cell = CELLS[budget]
    return dependency_fingerprint(
        {
            "grid_pp_rows": (runs, tree_files(runs / cell["grid_pp"], BENCH)),
            "grid_pp_sidecars": (
                runs,
                tree_files(runs / cell["grid_pp"], "token_ids", token_ids=None),
            ),
            "grid_base_rows": (runs, tree_files(runs / cell["grid_base"], BENCH)),
            "grid_base_sidecars": (
                runs,
                tree_files(runs / cell["grid_base"], "token_ids", token_ids=None),
            ),
            "readout_pp_rows": (
                runs,
                [
                    f
                    for r in cell["readout_pp"]
                    for f in tree_files(runs / r, "readout")
                ],
            ),
            "readout_base_rows": (
                runs,
                [
                    f
                    for r in cell["readout_base"]
                    for f in tree_files(runs / r, "readout")
                ],
            ),
            "full_cache_bank": (runs, tree_files(runs / FC_RUN, BENCH)),
            "specs": (
                PROJECT_ROOT,
                _project_files(
                    [cell["grid_pp_config"], cell["grid_base_config"], *FROZEN_SPECS]
                ),
            ),
            "analysis_code": (PROJECT_ROOT, _project_files(ANALYSIS_CODE)),
        }
    )


def origin_dependencies(runs: Path) -> dict[str, Any]:
    return dependency_fingerprint(
        {
            "restore_v3_rows": (runs, tree_files(runs / "restore-v3", BENCH)),
            "restore_v2_rows": (runs, tree_files(runs / "restore-v2", BENCH)),
            "full_cache_bank": (runs, tree_files(runs / FC_RUN, BENCH)),
            "specs": (
                PROJECT_ROOT,
                _project_files([ORIGIN_CONFIG, RESTORE2_CONFIG, *FROZEN_SPECS]),
            ),
            "analysis_code": (PROJECT_ROOT, _project_files(ANALYSIS_CODE)),
        }
    )


def _cfg(path: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((PROJECT_ROOT / path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ loading
def _row_files(root: Path, sub: str) -> list[Path]:
    return [
        p for p in sorted((root / sub).rglob("*.jsonl")) if "token_ids" not in p.parts
    ]


def _rows(root: Path, sub: str) -> list[dict[str, Any]]:
    return [r for p in _row_files(root, sub) for r in _jsonl(p)]


def _grid_key(r: dict[str, Any]) -> tuple:
    return (
        str(r["press"]),
        str(r["budget"]),
        str(r["pid"]),
        int(r["decoding"]["seed"]),
    )


def _readout_key(r: dict[str, Any]) -> tuple:
    ref = r["reference_key"]
    return (
        str(r["condition"]["press"]),
        str(r["condition"]["budget"]),
        str(ref[1]),
        int(ref[5]),
    )


def _origin_key(r: dict[str, Any]) -> tuple:
    return (str(r["press"]), str(r["budget"]), str(r["pid"]), int(r["seed"]))


def _prompt_lengths(root: Path) -> dict[str, set[int]]:
    """pid -> set of prompt lengths seen in the run's token sidecars."""
    out: dict[str, set[int]] = defaultdict(set)
    for path in sorted(root.rglob("*.jsonl")):
        if "token_ids" not in path.parts:
            continue
        for rec in _jsonl(path):
            out[str(rec["pid"])].add(len(rec["prompt_token_ids"]))
    return out


def _write(name: str, payload: dict[str, Any], argv: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    retired = retire_stale(path)
    payload = {
        "amendment": "S",
        "reproduce": "python scripts/report_s.py " + " ".join(argv),
        "retired_previous_output": str(retired.name) if retired else None,
        **payload,
    }
    write_json_atomic(path, payload)
    print(f"wrote {path} [{payload['status']}]")
    return path


def _write_tsv(name: str, header: list[str], rows: list[list[Any]]) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.conditions.tsv"
    retire_stale(path)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)
    return {
        "sufficient_statistics_tsv": path.name,
        "sufficient_statistics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


# ------------------------------------------------------------------ S.1
def protection(runs: Path, a: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """S.1 for one budget: per press, the residual and the paired contrasts."""
    cell = CELLS[a.budget]
    budget = str(a.budget)
    pp_design, pp_keys = design_keys(cell["grid_pp_config"], root=PROJECT_ROOT)
    _, base_keys = design_keys(cell["grid_base_config"], root=PROJECT_ROOT)
    if {k[2:] for k in pp_keys} != {k[2:] for k in base_keys}:
        raise SystemExit("REFUSED: protected and unprotected designs differ")
    if set(pp_design["budgets"]) != {budget}:
        raise SystemExit("REFUSED: config budget does not match --budget")
    pids, seeds = pp_design["pids"], pp_design["seeds"]

    # rows, every one of them, keyed before anything else
    grid_pp = _rows(runs / cell["grid_pp"], BENCH)
    grid_base = [
        r for r in _rows(runs / cell["grid_base"], BENCH) if str(r["budget"]) == budget
    ]
    ro_pp = [r for p in cell["readout_pp"] for r in _rows(runs / p, "readout")]
    ro_base = [
        r
        for p in cell["readout_base"]
        for r in _rows(runs / p, "readout")
        if str(r["condition"]["budget"]) == budget
    ]
    fc_rows = _rows(runs / FC_RUN, BENCH)
    # Review of fbf869f (R1): values are checked BEFORE any bool() conversion
    # ("False" and NaN are truthy), and readout flags must agree with values.
    grid_required = ("pid", "press", "budget", "decoding.seed", "trace")
    ro_required = ("reference_key", "condition")
    gkw = {"required": grid_required, "bool_fields": ("correct",)}
    rkw = {"required": ro_required, "readout": True}
    pop = {
        "grid_pp": validate_rows(grid_pp, key=_grid_key, expected=pp_keys, **gkw),
        "grid_base": validate_rows(grid_base, key=_grid_key, expected=base_keys, **gkw),
        "readout_pp": validate_rows(ro_pp, key=_readout_key, expected=pp_keys, **rkw),
        "readout_base": validate_rows(
            ro_base, key=_readout_key, expected=base_keys, **rkw
        ),
    }

    # controls; the bank is an allowed superset, its relevant rows are checked
    fc_keys = {(p, s) for p in pids for s in seeds}
    fc_rel = [r for r in fc_rows if _run_key(r) in fc_keys]
    pop["full_cache"] = validate_rows(
        fc_rel,
        key=_run_key,
        expected=fc_keys,
        required=("pid", "decoding.seed"),
        bool_fields=("correct",),
    )
    fc = {_run_key(r): r["correct"] for r in fc_rel}
    controls: dict[str, dict[str, Any]] = {
        "reference_covers_design": {
            "passed": fc_keys <= set(fc),
            "n_missing": len(fc_keys - set(fc)),
        }
    }
    lp_pp, lp_base = _prompt_lengths(runs / cell["grid_pp"]), _prompt_lengths(
        runs / cell["grid_base"]
    )
    missing_meta = [p for p in pids if p not in lp_pp or p not in lp_base]
    ambiguous = [
        p
        for p in pids
        if p in lp_pp and p in lp_base and len(lp_pp[p] | lp_base[p]) != 1
    ]
    prompt_len = {
        p: next(iter(lp_pp[p]))
        for p in pids
        if p not in missing_meta and p not in ambiguous
    }
    controls["prompt_metadata"] = {
        "passed": not missing_meta and not ambiguous,
        "n_pids_with_length": len(prompt_len),
        "missing": missing_meta[:20],
        "ambiguous": ambiguous[:20],
        "source": "token-ID sidecars of both grids; lengths must exist and agree",
    }
    for name, rows in (
        ("grid_pp", grid_pp),
        ("grid_base", grid_base),
        ("readout_pp", ro_pp),
        ("readout_base", ro_base),
    ):
        controls[f"provenance_{name}"] = legacy_provenance_summary(rows)
    status = final_status(
        {"status": _combine(pop)},
        controls,
    )
    relaxed = sorted(
        p for p, n in prompt_len.items() if effective_budget(a.budget, n)[1]
    )
    deps = protection_dependencies(runs, a.budget)
    out: dict[str, Any] = {
        "read": "S.1 prompt-protected decomposition",
        "status": status,
        "budget": a.budget,
        "inputs": {k: v for k, v in cell.items()},
        "dependencies": deps,
        "design": {
            "n_problems": len(pids),
            "seeds": seeds,
            "presses": pp_design["presses"],
        },
        "population": pop,
        "controls": controls,
        "budget_relaxed_problems": relaxed,
        "per_press": {},
    }
    if status == INVALID:
        return out, status

    free_pp_all = {_grid_key(r): bool(r["correct"]) for r in grid_pp}
    free_base_all = {_grid_key(r): bool(r["correct"]) for r in grid_base}
    ro_pp_all = {_readout_key(r): r for r in ro_pp}
    ro_base_all = {_readout_key(r): r for r in ro_base}
    tsv_rows: list[list[Any]] = []
    for base in PP_BASES:
        pp = base + PP_SUFFIX
        keys = [(p, s) for p in pids for s in seeds]
        fp = {
            k: free_pp_all[(pp, budget, *k)]
            for k in keys
            if (pp, budget, *k) in free_pp_all
        }
        fb = {
            k: free_base_all[(base, budget, *k)]
            for k in keys
            if (base, budget, *k) in free_base_all
        }
        rp = {
            k: bool(ro_pp_all[(pp, budget, *k)]["readout_correct"])
            for k in keys
            if (pp, budget, *k) in ro_pp_all
            and ro_pp_all[(pp, budget, *k)].get("readout_defined")
        }
        rb = {
            k: bool(ro_base_all[(base, budget, *k)]["readout_correct"])
            for k in keys
            if (base, budget, *k) in ro_base_all
            and ro_base_all[(base, budget, *k)].get("readout_defined")
        }
        shared = set(fp) & set(fb)
        both_ro = set(rp) & set(rb) & shared
        if status == PARTIAL:
            fp = {k: v for k, v in fp.items() if k in shared}
            fb = {k: v for k, v in fb.items() if k in shared}
            rp = {k: v for k, v in rp.items() if k in both_ro}
            rb = {k: v for k, v in rb.items() if k in both_ro}
        kw = {"n_resamples": a.n_resamples, "seed": a.seed, "reference": fc}
        block: dict[str, Any] = {
            "n_expected": len(keys),
            "n_free_pairs": len(fp),
            "n_readout_pairs": len(rp),
            "readout_defined_pp_only": sorted(set(rp) - set(rb))[:10],
            "readout_defined_base_only": sorted(set(rb) - set(rp))[:10],
        }
        if not fp or not rp:
            block["note"] = "no shared conditions yet"
        else:
            primary = s_reads.protection_contrasts(fp, fb, rp, rb, **kw)
            sens = s_reads.protection_contrasts(
                fp, fb, rp, rb, exclude_pids=relaxed, **kw
            )
            if status != FINAL:
                for b_ in (primary, sens):
                    b_["residual_label"] = b_["protection_effect"] = "PARTIAL: no label"
            block.update(
                primary=primary,
                sensitivity_without_budget_relaxed=sens,
                flip_mass_pp=s_reads.flip_masses(fc, fp),
                flip_mass_base=s_reads.flip_masses(fc, fb),
            )
        out["per_press"][pp] = block
        for k in keys:
            tsv_rows.append(
                [
                    pp,
                    budget,
                    k[0],
                    k[1],
                    fp.get(k, ""),
                    fb.get(k, ""),
                    rp.get(k, ""),
                    rb.get(k, ""),
                    int(fc.get(k, "")) if k in fc else "",
                    prompt_len.get(k[0], ""),
                    k[0] in relaxed,
                ]
            )
    out.update(
        _write_tsv(
            f"protection_b{a.budget}",
            [
                "press",
                "budget",
                "pid",
                "seed",
                "free_pp",
                "free_base",
                "readout_pp",
                "readout_base",
                "fc",
                "prompt_len",
                "budget_relaxed",
            ],
            tsv_rows,
        )
    )
    return out, status


def _combine(pop: dict[str, dict[str, Any]]) -> str:
    statuses = {b["status"] for b in pop.values()}
    if INVALID in statuses:
        return INVALID
    if PARTIAL in statuses:
        return PARTIAL
    return COMPLETE


# ------------------------------------------------------------------ consequence
def consequence(runs: Path, a: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """S.1 paper-level branch, from two FINAL, still-current protection reads."""
    labels: dict[str, dict[int, str]] = {}
    checks = {}
    for budget in sorted(CELLS):
        path = OUT / f"protection_b{budget}.json"
        if not path.exists():
            raise SystemExit(f"REFUSED: {path.name} has not been read")
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["status"] != FINAL:
            raise SystemExit(f"REFUSED: {path.name} is {d['status']}, not FINAL")
        current = protection_dependencies(runs, budget)
        stored = (d.get("dependencies") or {}).get("sha256")
        if current["sha256"] != stored:
            changed = sorted(
                g
                for g, v in current["groups"].items()
                if v != ((d.get("dependencies") or {}).get("groups") or {}).get(g)
            )
            raise SystemExit(
                f"REFUSED: inputs behind {path.name} changed since it was read "
                f"({', '.join(changed) or 'fingerprint absent'}); re-run the "
                "protection read"
            )
        checks[str(budget)] = {"dependencies_sha256": stored, "status": d["status"]}
        for press, block in d["per_press"].items():
            labels.setdefault(press, {})[budget] = block["primary"]["residual_label"]
    return {
        "read": "S.1 consequence for the paper",
        "status": FINAL,
        "inputs": checks,
        "residual_labels": labels,
        "consequence": s_reads.paper_consequence(labels),
    }, FINAL


# ------------------------------------------------------------------ S.2
def origin(runs: Path, a: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """S.2: coverage, provenance, gates and determinism first; ratios only on FINAL."""
    design, keys = design_keys(ORIGIN_CONFIG, root=PROJECT_ROOT)
    rows = _rows(runs / "restore-v3", BENCH)
    v2_rows = _rows(runs / "restore-v2", BENCH)
    fc_rows = _rows(runs / FC_RUN, BENCH)
    if any(not isinstance(r.get("correct"), bool) for r in fc_rows):
        raise SystemExit("REFUSED: a full-cache row's correct is not a boolean")
    fc = {_run_key(r): r["correct"] for r in fc_rows}
    arms = list(s_reads.ORIGIN_ARMS)
    cfg3, cfg2 = _cfg(ORIGIN_CONFIG), _cfg(RESTORE2_CONFIG)
    pop = validate_population(
        rows,
        key=_origin_key,
        expected=keys,
        required=("pid", "seed", "press", "budget", "status", "provenance"),
    )
    # EVERY row: a status the producer emits, a legitimate exclusion given the
    # row's own trace length / cohort and the config, and for READ rows passed
    # gates and in-domain TV (read_validation.restoration_row_problems)
    for r in rows:
        probs = restoration_row_problems(
            r,
            family="restore_v3",
            trigger=int(cfg3["trigger_step"]),
            interval=int(cfg3["compression_interval"]),
            window=int(cfg3["window"]),
        )
        if probs:
            k = repr(_origin_key(r)) if "press" in r else repr(r.get("pid"))
            pop["malformed"][k] = pop["malformed"].get(k, []) + probs
            pop["status"] = INVALID
    controls = {
        "provenance_restore_v3": provenance_v1_summary(
            rows,
            kind="restore_v3",
            config=ORIGIN_CONFIG,
            runs_root=runs,
            key=lambda r: (r["pid"], int(r["seed"])),
            root=PROJECT_ROOT,
        ),
        "provenance_restore_v2": provenance_v1_summary(
            v2_rows,
            kind="restore_v2",
            config=RESTORE2_CONFIG,
            runs_root=runs,
            key=lambda r: (r["pid"], int(r["seed"])),
            root=PROJECT_ROOT,
        ),
    }
    v2_cov = validate_population(
        v2_rows,
        key=_origin_key,
        expected=design_keys(RESTORE2_CONFIG, root=PROJECT_ROOT)[1],
        required=("pid", "seed", "press", "status"),
    )
    v2_bad: dict[str, list[str]] = {}
    for r in v2_rows:
        probs = restoration_row_problems(
            r,
            family="restore_v2",
            trigger=int(cfg2["trigger_step"]),
            interval=int(cfg2["compression_interval"]),
            window=int(cfg2["window"]),
        )
        if probs:
            v2_bad[repr(_origin_key(r))] = probs
    controls["restore_v2_coverage"] = {
        **v2_cov,
        "passed": v2_cov["status"] == COMPLETE,
    }
    controls["restore_v2_rows"] = {
        "passed": not v2_bad,
        "n_failing": len(v2_bad),
        "failing": dict(list(v2_bad.items())[:20]),
        "checked": "status, legitimate exclusion, full_restore and lag gates, "
        "TV domain, window fit (the same validator restore-v2 is read with)",
    }
    per_policy: dict[str, Any] = {}
    tsv_rows: list[list[Any]] = []
    for press in design["presses"]:
        rs = [r for r in rows if r["press"] == press]
        read = [r for r in rs if r["status"] == "READ"]
        rkeys = [(r["pid"], int(r["seed"])) for r in read]
        v2 = {
            (r["pid"], int(r["seed"])): {
                "full": (r.get("full_restore") or {}).get("window_mean_tv"),
                "none": (r.get("full_restore") or {}).get("none_window_mean_tv"),
            }
            for r in v2_rows
            if r["press"] == press
            and r["status"] == "READ"
            and (r.get("full_restore") or {}).get("eligible")
        }
        v3 = {
            (r["pid"], int(r["seed"])): {
                "full": r["arms"]["full"]["window_mean_tv"],
                "none": r["none_window_mean_tv"],
            }
            for r in read
            if all(arm in (r.get("arms") or {}) for arm in arms)
        }
        det = s_reads.determinism_control(v3, v2)
        det["eligible_sets_equal"] = set(v3) == set(v2)
        # the eligible-set relationship is part of the control: a complete read
        # must have exactly restore-v2's eligible set; an incomplete one a subset
        if pop["status"] == COMPLETE:
            det["eligible_set_rule"] = "complete read: v3 READ set == v2 eligible set"
            det["passed"] = bool(det["passed"]) and det["eligible_sets_equal"]
        else:
            # determinism_control already fails on a v3 key absent from v2
            det["eligible_set_rule"] = "incomplete read: v3 READ set within v2's"
        det["only_in_v3"] = [list(k) for k in sorted(set(v3) - set(v2))][:20]
        det["only_in_v2"] = [list(k) for k in sorted(set(v2) - set(v3))][:20]
        controls[f"determinism_{press}"] = det
        per_policy[press] = {
            "status_counts": {
                s: sum(1 for r in rs if r["status"] == s)
                for s in sorted({r["status"] for r in rs})
            },
            "eligible": {
                "n_traces": len(rkeys),
                "n_problems": len({k[0] for k in rkeys}),
                "fc_accuracy": (
                    equal_problem_bootstrap(
                        {k: float(fc[k]) for k in rkeys if k in fc},
                        n_resamples=a.n_resamples,
                        seed=a.seed,
                    )
                    if rkeys
                    else None
                ),
            },
            "pad_max": max(
                (
                    r["arms"]["prompt"]["gate"].get("pad_max", 0)
                    for r in read
                    if "arms" in r
                ),
                default=None,
            ),
            "n_prompt_return": sorted({r.get("n_prompt_return") for r in read}),
        }
        for r in rs:
            arms_blk = r.get("arms") or {}
            tsv_rows.append(
                [
                    press,
                    r["budget"],
                    r["pid"],
                    r["seed"],
                    r["status"],
                    r.get("cohort_step"),
                    r.get("n_prompt_return"),
                    (arms_blk.get("prompt") or {}).get("gate", {}).get("pad_max"),
                    r.get("none_window_mean_tv"),
                    *[(arms_blk.get(arm) or {}).get("window_mean_tv") for arm in arms],
                ]
            )
    status = final_status(pop, controls)
    out: dict[str, Any] = {
        "read": "S.2 restoration by origin",
        "status": status,
        "inputs": {
            "restore_v3": "restore-v3",
            "restore_v2": "restore-v2",
            "config": ORIGIN_CONFIG,
        },
        "dependencies": origin_dependencies(runs),
        "design": {k: v for k, v in design.items() if k != "pids"}
        | {"n_problems": len(design["pids"])},
        "population": pop,
        "controls": controls,
        "per_policy": per_policy,
    }
    if status == FINAL:
        for press in design["presses"]:
            read = [r for r in rows if r["press"] == press and r["status"] == "READ"]
            tv = {
                arm: {
                    (r["pid"], int(r["seed"])): r["arms"][arm]["window_mean_tv"]
                    for r in read
                }
                for arm in arms
            }
            per_policy[press]["ratios"] = s_reads.origin_ratios(
                tv, n_resamples=a.n_resamples, seed=a.seed
            )
    else:
        for press in per_policy:
            per_policy[press]["ratios"] = f"WITHHELD: read is {status}"
    out.update(
        _write_tsv(
            "origin",
            [
                "press",
                "budget",
                "pid",
                "seed",
                "status",
                "cohort_step",
                "n_prompt_return",
                "pad_max",
                "none_tv",
                *[f"{arm}_tv" for arm in arms],
            ],
            tsv_rows,
        )
    )
    return out, status


READS = {"protection": protection, "consequence": consequence, "origin": origin}


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--budget", type=int, choices=sorted(CELLS), default=256)
    ap.add_argument(
        "--partial",
        action="store_true",
        help="accept a PARTIAL read (exit 0, no label)",
    )
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
        help="where outputs go (tests redirect it; production uses the default)",
    )
    ap.add_argument("read", choices=sorted(READS))
    a = ap.parse_args()
    OUT = a.out_dir
    if a.dry_run:
        print(json.dumps({"command": "report_s", "read": a.read, "out": str(OUT)}))
        return 0
    name = f"protection_b{a.budget}" if a.read == "protection" else a.read
    # Retire the previous output BEFORE anything can fail, so a crash, a parse
    # error or a refusal never leaves an old FINAL standing as current.
    retire_stale(OUT / f"{name}.json")
    retire_stale(OUT / f"{name}.conditions.tsv")
    try:
        payload, status = READS[a.read](a.runs_root, a)
    except (SystemExit, Exception) as exc:  # noqa: BLE001 -- evidence, then exit 2
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        msg = str(exc.code) if isinstance(exc, SystemExit) else repr(exc)
        payload, status = {"read": a.read, "status": INVALID, "error": msg}, INVALID
        print(f"INVALID: {msg}", file=sys.stderr)
    _write(name, payload, sys.argv[1:])
    code = exit_code(status, partial_ok=a.partial)
    if code:
        print(f"{status}: exit {code}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

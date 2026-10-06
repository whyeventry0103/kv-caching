#!/usr/bin/env python3
"""Protocol-audit reads for the paper (execution plan WP1/WP3, 2026-10-05).

Thin orchestration over ``kvcache.metrics.protocol_audit`` and the validation
boundary in ``kvcache.read_validation``. Two reads:

``reverse``  -- the operational 2 x 2 (FF, FC, CC, CF) per policy at one budget,
               for the prompt-protected cells and their unprotected
               counterparts: free run on the whole population, each contrast on
               its own support, the common support, and worst-case bounds.
               Rows live on the GPU host; run it there.
``matched``  -- paired free-run accuracy and lost-mass (M, Delta) contrasts
               between policies on the 25 capture problems at B=256, from one
               joint draw, behind the R.17 capture validation boundary.
``decision`` -- practical margins (1/3/5 pts) and the shrinkage-ratio interval
               from the committed S.1 sufficient statistics. CPU-only, local;
               refuses unless the stored S.1 read is FINAL and its TSV hash
               matches.

Neither read relabels a registered analysis: the 2 x 2 is a new, separately
named protocol comparison, and the practical margins were proposed after the
B=256 result was seen (they are a sensitivity for that result).

Status contract as ``report_s``: FINAL / PARTIAL / INVALID, previous output
retired before analysis, exit 0 / 3 (0 with --partial) / 2.

    python scripts/report_protocol_audit.py --runs-root <runs> reverse --budget 256
    python scripts/report_protocol_audit.py --runs-root <runs> matched
    python scripts/report_protocol_audit.py decision
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import report_s as rs

from kvcache.coverage import equal_problem_bootstrap, equal_problem_mean
from kvcache.metrics import protocol_audit as pa
from kvcache.presses.prompt_protect import PP_BASES, PP_SUFFIX
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
    retire_stale,
    tree_files,
)

OUT = PROJECT_ROOT / "results" / "diag" / "paper_audit"
S_OUT = PROJECT_ROOT / "results" / "diag" / "amendment-s"
REVERSE = {
    256: {
        "reverse_pp": "reverse-readout-pp256",
        "reverse_base": "reverse-readout-diss256",
    },
    # Proposal P1 (HELD): only the PROTECTED reverse arm exists at 1024, so the
    # read gives the protected 2 x 2 and withholds the protected-vs-unprotected
    # comparison instead of inventing an arm.
    1024: {"reverse_pp": "reverse-readout-pp1024", "reverse_base": None},
}
REVERSE_CONFIGS = {
    256: (
        "configs/experiments/reverse_readout_pp256_s1.yaml",
        "configs/experiments/reverse_readout_diss256.yaml",
    ),
    1024: ("configs/experiments/reverse_readout_pp1024_s1x.yaml",),
}
ANALYSIS_CODE = (
    "scripts/report_protocol_audit.py",
    "src/kvcache/metrics/protocol_audit.py",
    *rs.ANALYSIS_CODE,
)


def _reverse_key(r: dict[str, Any]) -> tuple:
    """(source policy, budget, pid, seed) of the compressed trace read out."""
    ref = r["reference_key"]
    return (str(ref[3]), str(ref[4]), str(ref[1]), int(ref[5]))


def _fc_key(r: dict[str, Any]) -> tuple[str, int]:
    return (str(r["pid"]), int(r["decoding"]["seed"]))


def reverse_dependencies(runs: Path, budget: int) -> dict[str, Any]:
    cell, rev = rs.CELLS[budget], REVERSE[budget]
    groups = {
        "protection_inputs": (
            runs,
            [
                *tree_files(runs / cell["grid_pp"], rs.BENCH),
                *tree_files(runs / cell["grid_base"], rs.BENCH),
                *[
                    f
                    for r in cell["readout_pp"]
                    for f in tree_files(runs / r, "readout")
                ],
                *[
                    f
                    for r in cell["readout_base"]
                    for f in tree_files(runs / r, "readout")
                ],
                *tree_files(runs / rs.FC_RUN, rs.BENCH),
            ],
        ),
        "reverse_rows": (
            runs,
            [
                *tree_files(runs / rev["reverse_pp"], "readout"),
                *(
                    tree_files(runs / rev["reverse_base"], "readout")
                    if rev["reverse_base"]
                    else []
                ),
            ],
        ),
        "specs": (
            PROJECT_ROOT,
            rs._project_files(
                [
                    cell["grid_pp_config"],
                    cell["grid_base_config"],
                    *REVERSE_CONFIGS[budget],
                    *rs.FROZEN_SPECS,
                ]
            ),
        ),
        "analysis_code": (PROJECT_ROOT, rs._project_files(ANALYSIS_CODE)),
    }
    return dependency_fingerprint(groups)


def _write(name: str, payload: dict[str, Any], argv: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    retired = retire_stale(path)
    payload = {
        "reproduce": "python scripts/report_protocol_audit.py " + " ".join(argv),
        "retired_previous_output": retired.name if retired else None,
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


# ------------------------------------------------------------------ reverse
def reverse(runs: Path, a: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if a.budget not in REVERSE:
        raise SystemExit(f"REFUSED: no reverse readout registered at {a.budget}")
    cell, rev = rs.CELLS[a.budget], REVERSE[a.budget]
    budget = str(a.budget)
    pp_design, pp_keys = design_keys(cell["grid_pp_config"], root=PROJECT_ROOT)
    _, base_keys = design_keys(cell["grid_base_config"], root=PROJECT_ROOT)
    if {k[2:] for k in pp_keys} != {k[2:] for k in base_keys}:
        raise SystemExit("REFUSED: protected and unprotected designs differ")
    pids, seeds = pp_design["pids"], pp_design["seeds"]

    grid_pp = rs._rows(runs / cell["grid_pp"], rs.BENCH)
    grid_base = [
        r
        for r in rs._rows(runs / cell["grid_base"], rs.BENCH)
        if str(r["budget"]) == budget
    ]
    ro_pp = [r for p in cell["readout_pp"] for r in rs._rows(runs / p, "readout")]
    ro_base = [
        r
        for p in cell["readout_base"]
        for r in rs._rows(runs / p, "readout")
        if str(r["condition"]["budget"]) == budget
    ]
    rev_pp = rs._rows(runs / rev["reverse_pp"], "readout")
    has_base = rev["reverse_base"] is not None
    if not has_base:
        grid_base, ro_base = [], []
    rev_base = rs._rows(runs / rev["reverse_base"], "readout") if has_base else []
    fc_rows = rs._rows(runs / rs.FC_RUN, rs.BENCH)

    grid_req = ("pid", "press", "budget", "decoding.seed", "trace")
    ro_req = ("reference_key", "condition")
    gkw = {"required": grid_req, "bool_fields": ("correct",)}
    rkw = {"required": ro_req, "readout": True}
    pop = {
        "grid_pp": validate_rows(grid_pp, key=rs._grid_key, expected=pp_keys, **gkw),
        "grid_base": validate_rows(
            grid_base, key=rs._grid_key, expected=base_keys, **gkw
        ),
        "readout_pp": validate_rows(
            ro_pp, key=rs._readout_key, expected=pp_keys, **rkw
        ),
        "readout_base": validate_rows(
            ro_base, key=rs._readout_key, expected=base_keys, **rkw
        ),
        "reverse_pp": validate_rows(rev_pp, key=_reverse_key, expected=pp_keys, **rkw),
        "reverse_base": validate_rows(
            rev_base, key=_reverse_key, expected=base_keys, **rkw
        ),
    }
    if not has_base:
        for name in ("grid_base", "readout_base", "reverse_base"):
            del pop[name]
    for name, rows, keyf in (
        ("reverse_pp", rev_pp, _reverse_key),
        ("reverse_base", rev_base, _reverse_key),
    ):
        bad = {
            repr(keyf(r)): ["reverse readout not under the full cache"]
            for r in rows
            if (r.get("condition") or {}).get("press") != "fullcache"
        }
        if bad:
            pop[name]["status"] = INVALID
            pop[name]["malformed"].update(bad)

    # The bank is an allowed SUPERSET of the design; the relevant rows must
    # still be unique, boolean and complete.
    design = {(p, s) for p in pids for s in seeds}
    fc_rel = [r for r in fc_rows if _fc_key(r) in design]
    pop["full_cache"] = validate_rows(
        fc_rel,
        key=_fc_key,
        expected=design,
        required=("pid", "decoding.seed"),
        bool_fields=("correct",),
    )
    fc = {_fc_key(r): r["correct"] for r in fc_rel}
    controls: dict[str, dict[str, Any]] = {
        "reference_covers_design": {
            "passed": design <= set(fc),
            "n_missing": len(design - set(fc)),
        }
    }
    for name, rows in (
        ("grid_pp", grid_pp),
        ("grid_base", grid_base),
        ("readout_pp", ro_pp),
        ("readout_base", ro_base),
        ("reverse_pp", rev_pp),
        ("reverse_base", rev_base),
    ):
        if name in pop:
            controls[f"provenance_{name}"] = legacy_provenance_summary(rows)
    status = final_status({"status": rs._combine(pop)}, controls)
    out: dict[str, Any] = {
        "read": "protocol audit: operational 2 x 2 (forward and reverse readout)",
        "registration": "NEW ANALYSIS 2026-10-05 (execution plan WP1); not an "
        "Amendment S read and relabels nothing",
        "status": status,
        "budget": a.budget,
        "inputs": {**cell, **rev},
        "dependencies": reverse_dependencies(runs, a.budget),
        "population": pop,
        "controls": controls,
        "per_policy": {},
    }
    if status != FINAL:
        out["per_policy"] = f"WITHHELD: read is {status}"
        return out, status

    def free(rows: list[dict[str, Any]], press: str) -> dict[tuple, bool]:
        return {k[2:]: r["correct"] for r in rows if (k := rs._grid_key(r))[0] == press}

    def readout(rows: list[dict[str, Any]], keyf, press: str) -> dict[tuple, bool]:
        return {
            k[2:]: r["readout_correct"]
            for r in rows
            if (k := keyf(r))[0] == press and r.get("readout_defined")
        }

    tsv: list[list[Any]] = []
    population = sorted(design)
    out["protection_by_direction"] = {}
    for base in PP_BASES:
        cells: dict[str, dict[str, dict[tuple, bool]]] = {}
        arms = [(base + PP_SUFFIX, grid_pp, ro_pp, rev_pp)]
        if has_base:
            arms.append((base, grid_base, ro_base, rev_base))
        for press, grid, ro, rv in arms:
            cc = free(grid, press)
            fcm = readout(ro, rs._readout_key, press)
            cfm = readout(rv, _reverse_key, press)
            ffm = {k: fc[k] for k in population}
            block = pa.two_by_two(
                ff=ffm,
                fc=fcm,
                cc=cc,
                cf=cfm,
                population=population,
                n_resamples=a.n_resamples,
                seed=a.seed,
            )
            block["practical"] = pa.practical_reading(
                ff=ffm, cc=cc, fc=fcm, n_resamples=a.n_resamples, seed=a.seed
            )
            out["per_policy"][press] = block
            cells[press] = {"ff": ffm, "fc": fcm, "cc": cc, "cf": cfm}
            tsv.extend(
                [
                    press,
                    budget,
                    k[0],
                    k[1],
                    int(ffm[k]),
                    int(cc[k]),
                    int(fcm[k]) if k in fcm else "",
                    int(cfm[k]) if k in cfm else "",
                ]
                for k in population
            )
        out["protection_by_direction"][base + PP_SUFFIX] = (
            pa.protection_by_direction(
                cells[base + PP_SUFFIX],
                cells[base],
                n_resamples=a.n_resamples,
                seed=a.seed,
            )
            if has_base
            else "NOT AVAILABLE: no unprotected reverse arm at this budget "
            "(not registered); the protected 2 x 2 is reported alone"
        )
    out.update(
        _write_tsv(
            f"reverse_b{a.budget}",
            ["press", "budget", "pid", "seed", "ff", "cc", "fc", "cf"],
            tsv,
        )
    )
    return out, status


# ------------------------------------------------------------------ decision
def _load_s1(budget: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    path = S_OUT / f"protection_b{budget}.json"
    if not path.exists():
        raise SystemExit(f"REFUSED: {path.name} has not been read")
    d = json.loads(path.read_text(encoding="utf-8"))
    if d["status"] != FINAL:
        raise SystemExit(f"REFUSED: {path.name} is {d['status']}, not FINAL")
    tsv = S_OUT / d["sufficient_statistics_tsv"]
    if (
        hashlib.sha256(tsv.read_bytes()).hexdigest()
        != d["sufficient_statistics_sha256"]
    ):
        raise SystemExit(f"REFUSED: {tsv.name} does not match its recorded hash")
    with tsv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return d, rows


def _tsv_bool(v: str) -> bool | None:
    return {"True": True, "False": False, "1": True, "0": False, "": None}[v]


def decision(runs: Path | None, a: argparse.Namespace) -> tuple[dict[str, Any], str]:
    budget = a.budget
    d, rows = _load_s1(budget)
    out: dict[str, Any] = {
        "read": "practical margins and shrinkage ratio over the S.1 conditions",
        "registration": "SENSITIVITY 2026-10-05 (execution plan Section 4.3/9.4); "
        "margins proposed after the B=256 result was seen",
        "status": FINAL,
        "budget": budget,
        "inputs": {
            "s1_read": f"protection_b{budget}.json",
            "s1_dependencies_sha256": d["dependencies"]["sha256"],
            "s1_tsv_sha256": d["sufficient_statistics_sha256"],
        },
        "per_press": {},
    }
    for pp in sorted({r["press"] for r in rows}):
        sub = [r for r in rows if r["press"] == pp]
        key = [(r["pid"], int(r["seed"])) for r in sub]
        ff = {k: _tsv_bool(r["fc"]) for k, r in zip(key, sub, strict=True)}
        if any(v is None for v in ff.values()):
            raise SystemExit("REFUSED: a condition has no full-cache reference")
        arms = {}
        for col in ("free_pp", "free_base", "readout_pp", "readout_base"):
            arms[col] = {
                k: v
                for k, r in zip(key, sub, strict=True)
                if (v := _tsv_bool(r[col])) is not None
            }
        ro = set(arms["readout_pp"])
        res_pp = {k: float(arms["free_pp"][k]) - arms["readout_pp"][k] for k in ro}
        res_b = {k: float(arms["free_base"][k]) - arms["readout_base"][k] for k in ro}
        kw = {"n_resamples": a.n_resamples, "seed": a.seed}
        relaxed = {r["pid"] for r in sub if r["budget_relaxed"] == "True"}
        block = {
            "practical_protected": pa.practical_reading(
                ff=ff, cc=arms["free_pp"], fc=arms["readout_pp"], **kw
            ),
            "practical_unprotected": pa.practical_reading(
                ff=ff, cc=arms["free_base"], fc=arms["readout_base"], **kw
            ),
            "shrinkage": pa.shrinkage_ratio(res_pp, res_b, **kw),
            "shrinkage_without_budget_relaxed": pa.shrinkage_ratio(
                {k: v for k, v in res_pp.items() if k[0] not in relaxed},
                {k: v for k, v in res_b.items() if k[0] not in relaxed},
                **kw,
            ),
        }
        stored = d["per_press"][pp]["primary"]
        block["reproduces_s1"] = {
            "residual_pp": [stored["residual_pp"]["mean"], equal_problem_mean(res_pp)],
            "residual_base": [
                stored["residual_base"]["mean"],
                equal_problem_mean(res_b),
            ],
        }
        block["reproduces_s1"]["match"] = all(
            abs(x - y) < 1e-12 for x, y in block["reproduces_s1"].values()
        )
        if not block["reproduces_s1"]["match"]:
            out["status"] = INVALID
        out["per_press"][pp] = block
    return out, out["status"]


# ------------------------------------------------------------------ matched
MATCHED_PAIRS = (("random", "h2o"), ("random", "streamingllm"), ("random", "rkv"))


MATCHED_GRID_RUNS = {
    "random": "grid-diss256-tok",
    "streamingllm": "grid-diss256-tok",
    "h2o": "grid-diss256-h2o",
    "rkv": "grid-diss256-rkv",
}
MATCHED_CAPTURE_RUN = "capture-b256"
MATCHED_CODE = (
    *ANALYSIS_CODE,
    "scripts/report_r17.py",
    "scripts/report_capture.py",
    "src/kvcache/read_checks.py",
)


def matched_dependencies(runs: Path) -> dict[str, Any]:
    import report_r17

    return dependency_fingerprint(
        {
            "capture_rows": (runs, tree_files(runs / MATCHED_CAPTURE_RUN, rs.BENCH)),
            "free_run_rows": (
                runs,
                [
                    f
                    for run in sorted(set(MATCHED_GRID_RUNS.values()))
                    for f in tree_files(runs / run, rs.BENCH)
                ],
            ),
            "specs": (
                PROJECT_ROOT,
                rs._project_files(
                    [report_r17.CAPTURE_CONFIGS[MATCHED_CAPTURE_RUN], *rs.FROZEN_SPECS]
                ),
            ),
            "analysis_code": (PROJECT_ROOT, rs._project_files(MATCHED_CODE)),
        }
    )


def matched(runs: Path, a: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """WP3: paired accuracy and lost-mass contrasts on the 25 capture problems.

    Accuracy is the budget-256 free run; the metric is the capture row's
    ``M_mean`` (lost attention mass, teacher-forced on the full-cache trace),
    with ``Delta_mean`` as a secondary shape metric. Both come from ONE joint
    problem draw per metric.

    Validation (review of fbf869f, R1): the capture rows pass the R.17
    boundary; the free-run rows are validated against the SAME frozen design
    before any dictionary is built (the grids are an allowed superset of the
    25 problems; relevant duplicates are INVALID, non-boolean ``correct`` is
    INVALID, a missing condition is PARTIAL). A capture row whose metric is
    absent is a declared exclusion and is listed; a missing accuracy row never
    shrinks the cohort silently. Anything short of FINAL withholds every label.
    """
    import report_r17

    cap = report_r17._validated_capture(
        runs, argparse.Namespace(capture_run=MATCHED_CAPTURE_RUN)
    )
    _design, expected = design_keys(
        report_r17.CAPTURE_CONFIGS[MATCHED_CAPTURE_RUN], root=PROJECT_ROOT
    )
    policies = sorted({p for pair in MATCHED_PAIRS for p in pair})
    acc_pop: dict[str, Any] = {}
    acc: dict[str, dict[tuple, float]] = {}
    for press in policies:
        exp = {k for k in expected if k[0] == press}
        pids = {k[2] for k in exp}
        rows = [
            r
            for r in rs._rows(runs / MATCHED_GRID_RUNS[press], rs.BENCH)
            if r.get("press") == press
            and str(r.get("budget")) == "256"
            and r.get("pid") in pids
        ]
        acc_pop[press] = validate_rows(
            rows,
            key=rs._grid_key,
            expected=exp,
            required=("pid", "press", "budget", "decoding.seed"),
            bool_fields=("correct",),
        )
        acc[press] = {rs._grid_key(r)[2:]: float(r["correct"]) for r in rows}
    statuses = [cap["status"]] + [
        FINAL if b["status"] == "COMPLETE" else b["status"] for b in acc_pop.values()
    ]
    status = (
        INVALID if INVALID in statuses else (PARTIAL if PARTIAL in statuses else FINAL)
    )
    out: dict[str, Any] = {
        "read": "matched-key policy contrasts: free-run accuracy and lost mass",
        "registration": "NEW ANALYSIS 2026-10-05 (execution plan WP3); "
        "descriptive, four policies cannot establish a per-trace predictor",
        "status": status,
        "dependencies": matched_dependencies(runs),
        "capture_validation": cap["validation"],
        "accuracy_population": acc_pop,
    }
    if status != FINAL:
        for blk in ("contrasts_M", "contrasts_Delta", "accuracy_by_policy"):
            out[blk] = f"WITHHELD: read is {status}"
        return out, status
    metric: dict[str, dict[str, dict[tuple, float]]] = {"M_mean": {}, "Delta_mean": {}}
    declared: list[list[Any]] = []
    for r in cap["rows"]:
        k = (str(r["pid"]), int(r["seed"]))
        for m in metric:
            if r.get(m) is None:
                declared.append([str(r["press"]), *k, m])
            else:
                metric[m].setdefault(str(r["press"]), {})[k] = float(r[m])
    design = sorted({k[2:] for k in expected})
    keys_sorted = [
        k
        for k in design
        if all(k in metric[m].get(p, {}) for m in metric for p in policies)
    ]
    accd = {p: {k: acc[p][k] for k in keys_sorted} for p in policies}
    kw = {"n_resamples": a.n_resamples, "seed": a.seed}
    out.update(
        {
            "n_conditions": len(keys_sorted),
            "n_problems": len({k[0] for k in keys_sorted}),
            "n_design_conditions": len(design),
            "declared_metric_exclusions": declared,
            "support": "the frozen capture design minus declared metric exclusions",
            "accuracy_by_policy": {
                p: equal_problem_bootstrap(accd[p], **kw) for p in policies
            },
            "contrasts_M": pa.policy_contrasts(
                accd,
                {p: {k: metric["M_mean"][p][k] for k in keys_sorted} for p in policies},
                MATCHED_PAIRS,
                **kw,
            ),
            "contrasts_Delta": pa.policy_contrasts(
                accd,
                {
                    p: {k: metric["Delta_mean"][p][k] for k in keys_sorted}
                    for p in policies
                },
                MATCHED_PAIRS,
                **kw,
            ),
        }
    )
    tsv = [
        [
            p,
            k[0],
            k[1],
            int(accd[p][k]),
            metric["M_mean"][p][k],
            metric["Delta_mean"][p][k],
        ]
        for p in policies
        for k in keys_sorted
    ]
    out.update(
        _write_tsv(
            "matched_b256",
            ["press", "pid", "seed", "free_correct", "M_mean", "Delta_mean"],
            tsv,
        )
    )
    return out, FINAL


READS = {"reverse": reverse, "decision": decision, "matched": matched}


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=None)
    ap.add_argument("--budget", type=int, choices=sorted(rs.CELLS), default=256)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("read", choices=sorted(READS))
    a = ap.parse_args()
    OUT = a.out_dir
    if a.dry_run:
        print(json.dumps({"command": "report_protocol_audit", "read": a.read}))
        return 0
    if a.read in ("reverse", "matched") and a.runs_root is None:
        ap.error(f"{a.read} needs --runs-root")
    name = f"{a.read}_b{a.budget}"
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

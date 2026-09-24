#!/usr/bin/env python3
"""Meeting #3 figures drawn ONLY from committed reader outputs (R.14).

Each function reads one reader JSON and draws nothing unless that read is
FINAL; a PARTIAL or INVALID read gets no figure, so a figure can never present
an incomplete grid as a result. Every mark carries its equal-problem 95% CI.

    python scripts/make_meeting3_figures.py \
        --capture results/diag/meeting3/capture_b256.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"
POLICY_LABEL = {
    "random": "Random",
    "streamingllm": "StreamingLLM",
    "h2o": "H2O-style",
    "rkv": "R-KV",
}
COLOR = {
    "random": "#0072B2",
    "streamingllm": "#D55E00",
    "h2o": "#CC79A7",
    "rkv": "#56B4E9",
}


def _load(path: Path) -> dict[str, Any] | None:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d if d.get("status") == "FINAL" else None


def _save(fig: Any, stem: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _bar(ax: Any, stats: dict[str, Any], title: str, ylabel: str) -> None:
    xs, names = [], []
    for i, p in enumerate(POLICY_LABEL):
        st = stats.get(p)
        if not st or st.get("mean") is None:
            continue
        lo, hi = st["ci95"]
        ax.errorbar(
            i,
            st["mean"],
            yerr=[[st["mean"] - lo], [hi - st["mean"]]],
            fmt="o",
            color=COLOR[p],
            capsize=3,
            ms=6,
        )
        ax.annotate(
            f"{st['n_problems']} pids",
            (i, st["mean"]),
            (6, 0),
            textcoords="offset points",
            fontsize=6.5,
        )
        xs.append(i)
        names.append(POLICY_LABEL[p])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", color="#dddddd", lw=0.6)


def fig_capture(path: Path, out_dir: Path = FIG_DIR) -> list[Path]:
    d = _load(path)
    if d is None:
        return []
    desc = d["descriptive"]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2))
    _bar(axes[0], desc["collateral"], "(a) GQA collateral (pooled)", "fraction")
    q3 = d["q3_registered_contrast"]
    axes[0].text(
        0.02,
        0.97,
        f"R-KV - H2O = {q3['mean']:+.3f} [{q3['ci95'][0]:+.3f}, "
        f"{q3['ci95'][1]:+.3f}]\n{q3['verdict']}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=6.5,
    )
    _bar(axes[1], desc["M"], "(b) M: full mass outside S", "mass")
    _bar(axes[2], desc["Delta"], "(c) Delta: shape TV on S", "TV")
    _bar(axes[3], desc["omega"], "(d) omega = |d_perp| / |d_par|", "omega")
    q6 = d["q6_registered_contrast"]
    axes[3].text(
        0.02,
        0.97,
        f"Random - SLM = {q6['mean']:+.3f} [{q6['ci95'][0]:+.3f}, "
        f"{q6['ci95'][1]:+.3f}]\n{q6['verdict']} (exploratory)",
        transform=axes[3].transAxes,
        va="top",
        fontsize=6.5,
    )
    fig.suptitle(f"Meeting #3 Tasks 2/3, budget {d['budget']}", fontsize=10)
    fig.tight_layout()
    return _save(fig, f"meeting3_capture_b{d['budget']}", out_dir)


def fig_dacc(path: Path, out_dir: Path = FIG_DIR) -> list[Path]:
    d = _load(path)
    if d is None:
        return []
    cells = [c for c in d["cells"].values() if c["status"] == "READ"]
    budgets = sorted({c["budget"] for c in cells})
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.2))
    axes = axes if len(budgets) > 1 else [axes]
    for ax, b in zip(axes, budgets, strict=True):
        for p in POLICY_LABEL:
            cs = sorted(
                (c for c in cells if c["budget"] == b and c["press"] == p),
                key=lambda c: c["onset"],
            )
            if not cs:
                continue
            xs = [c["onset"] for c in cs]
            ys = [c["damage_fc_minus_comp"]["mean"] for c in cs]
            err = [
                [
                    y - c["damage_fc_minus_comp"]["ci95"][0]
                    for y, c in zip(ys, cs, strict=True)
                ],
                [
                    c["damage_fc_minus_comp"]["ci95"][1] - y
                    for y, c in zip(ys, cs, strict=True)
                ],
            ]
            ax.errorbar(
                xs,
                ys,
                yerr=err,
                fmt="o-",
                color=COLOR[p],
                capsize=3,
                label=POLICY_LABEL[p],
            )
        ax.axhline(0, color="#555555", lw=0.6)
        ax.set_title(
            f"B = {b}" + ("  (exposure non-identifying, R.8.1)" if b == 2048 else ""),
            fontsize=9,
        )
        ax.set_xlabel("compression_start_step t0", fontsize=8)
        ax.set_ylabel("damage Acc_FC - Acc_comp", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save(fig, "meeting3_dacc", out_dir)


def fig_exact_lag(path: Path, out_dir: Path = FIG_DIR) -> list[Path]:
    d = _load(path)
    if d is None:
        return []
    pols = d["policies"]
    fig, axes = plt.subplots(1, len(pols), figsize=(4.4 * len(pols), 3.2))
    axes = axes if len(pols) > 1 else [axes]
    style = {"matched": "o-", "random": "s--", "relaxed": "^:"}
    for ax, (p, v) in zip(axes, sorted(pols.items()), strict=True):
        for arm, fmt in style.items():
            pts = [
                (int(lag), r["arms"][arm])
                for lag, r in v["per_lag"].items()
                if r.get("arms", {}).get(arm)
            ]
            if not pts:
                continue
            pts.sort()
            xs = [x for x, _ in pts]
            ys = [a["effect_vs_no_intervention"] for _, a in pts]
            err = [
                [y - a["effect_ci95"][0] for y, (_, a) in zip(ys, pts, strict=True)],
                [a["effect_ci95"][1] - y for y, (_, a) in zip(ys, pts, strict=True)],
            ]
            ax.errorbar(xs, ys, yerr=err, fmt=fmt, capsize=3, label=arm)
        ax.set_xscale("log", base=2)
        ax.axhline(0, color="#555555", lw=0.6)
        ax.set_title(
            f"{POLICY_LABEL.get(p, p)}: {v['sweep'].get('verdict')}", fontsize=9
        )
        ax.set_xlabel("lag (decode steps)", fontsize=8)
        ax.set_ylabel("none - arm, mean TV over 200 steps", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save(fig, "meeting3_exact_lag", out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--dacc", type=Path)
    ap.add_argument("--exact-lag", type=Path)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    a = ap.parse_args()
    made = []
    if a.capture:
        made += fig_capture(a.capture, a.out_dir)
    if a.dacc:
        made += fig_dacc(a.dacc, a.out_dir)
    if a.exact_lag:
        made += fig_exact_lag(a.exact_lag, a.out_dir)
    for p in made:
        print(f"wrote {p}")
    if not made:
        print("no FINAL read given: no figure drawn")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""R.17.1.4 Problem 1 figures, drawn only from committed reader artifacts.

(a) damage against budget, both policies, paired CIs, the realized ratio on a
    second axis, and AUC-TV against budget beside it (CIs where the per-trace
    profiles exist);
(b) two heatmaps of MEASURED cells only -- TV over (realized ratio, onset) and
    damage over (budget, onset) -- with unmeasured cells grey and never
    interpolated;
(c) the collapse plot: onset-0 window TV against realized ratio for every
    budget, matched-ratio pairs joined, once R.17.1.1 has been read.

No smoothing and no 3-D surface anywhere. Realized ratio is budget / (prompt +
position) or budget / (mean prompt + mean generated), never budget / decode step.

    python scripts/make_r17_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
M3 = PROJECT_ROOT / "results" / "diag" / "meeting3"
OUT = M3 / "figures"
LABEL = {"random": "Random", "streamingllm": "StreamingLLM"}
COLOR = {"random": "#0072B2", "streamingllm": "#D55E00"}
BUD_COLOR = {256: "#332288", 512: "#117733", 1024: "#DDAA33", 2048: "#CC6677"}
GREY = "#d9d9d9"


def _load(name: str) -> dict[str, Any] | None:
    p = M3 / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _save(fig: Any, stem: str, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def fig_supervisor(out: Path) -> list[Path]:
    curve = _load("t1_budget_curve.json")
    auc = _load("r17/auc_tv.json")
    if curve is None:
        return []
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    budgets = sorted({c["budget"] for c in curve["cells"]})
    ratios: dict[int, list[float]] = {b: [] for b in budgets}
    for p, dx in (("random", -0.03), ("streamingllm", 0.03)):
        cs = sorted(
            (c for c in curve["cells"] if c["press"] == p), key=lambda c: c["budget"]
        )
        xs = [c["budget"] * (2**dx) for c in cs]
        tc = [c["paired"]["total_change"] for c in cs]
        ys = [-t["mean"] for t in tc]
        err = [
            [y - (-t["ci95"][1]) for y, t in zip(ys, tc, strict=True)],
            [(-t["ci95"][0]) - y for y, t in zip(ys, tc, strict=True)],
        ]
        ax.errorbar(
            xs, ys, yerr=err, fmt="o-", color=COLOR[p], capsize=3, label=LABEL[p]
        )
        for c in cs:
            ratios[c["budget"]].append(100 * c["realized_ratio"])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.set_xlabel("cache budget (entries)", fontsize=8)
    ax.set_ylabel("damage: Acc_FC - Acc_comp", fontsize=8)
    ax.axhline(0, color="#555555", lw=0.6)
    top = ax.secondary_xaxis("top")
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top.set_xticks(
        budgets,
        [f"{min(ratios[b]):.1f}-{max(ratios[b]):.1f}%" for b in budgets],
        fontsize=7,
    )
    top.set_xlabel("realized ratio (Random-StreamingLLM)", fontsize=7)
    ax.set_title("(a) damage vs budget (100 problems x 4 seeds, paired CI)", fontsize=8)
    # AUC-TV beside it
    have = {int(k.split("-")[1]) for k in (auc or {}).get("cells", {})}
    # never join points across an unmeasured budget: that would draw an
    # interpolation the data do not contain
    contiguous = all(b in have for b in budgets)
    for p, dx in (("random", -0.03), ("streamingllm", 0.03)):
        pts = []
        for b in budgets:
            c = (auc or {}).get("cells", {}).get(f"{p}-{b}")
            if c and c.get("mean") is not None:
                pts.append((b * (2**dx), c["mean"], c["ci95"]))
        if pts:
            bx.errorbar(
                [x for x, _, _ in pts],
                [y for _, y, _ in pts],
                yerr=[[y - ci[0] for _, y, ci in pts], [ci[1] - y for _, y, ci in pts]],
                fmt="o-" if contiguous else "o",
                color=COLOR[p],
                capsize=3,
                label=LABEL[p],
            )
    measured = sorted({int(k.split("-")[1]) for k in (auc or {}).get("cells", {})})
    missing = [b for b in budgets if b not in measured]
    bx.set_xscale("log", base=2)
    bx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    bx.set_xticks(budgets, [str(b) for b in budgets])
    bx.set_xlabel("cache budget (entries)", fontsize=8)
    bx.set_ylabel("AUC-TV (per-trace mean TV)", fontsize=8)
    bx.set_title(
        "(b) divergence vs budget (25 problems x 4 seeds, clustered CI)"
        + (f"\nnot yet measured: {missing}" if missing else ""),
        fontsize=8,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    return _save(fig, "r17_supervisor_damage_and_tv", out)


def _heat(
    ax: Any,
    grid: list[list[float | None]],
    rows: list[str],
    cols: list[str],
    title: str,
    fmt: str,
    cmap: str,
) -> None:
    import numpy as np

    arr = np.array([[np.nan if v is None else v for v in r] for r in grid], float)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(GREY)
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cm, aspect="auto")
    for i, r in enumerate(grid):
        for j, v in enumerate(r):
            if v is None:
                color = "#555555"
            else:
                rgb = cm(im.norm(v))[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                color = "black" if lum > 0.5 else "white"
            ax.text(
                j,
                i,
                "not\nmeasured" if v is None else format(v, fmt),
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=6, rotation=40, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)


def fig_heatmaps(out: Path) -> list[Path]:
    grid = _load("q2_onset_grid.json")
    dacc = _load("dacc.json")
    if grid is None or dacc is None:
        return []
    fp = grid["full_panel"]
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 3.4), gridspec_kw={"width_ratios": [2.2, 1, 1]}
    )
    # (1) TV over (realized ratio, onset): columns = every (budget, position)
    # cell ordered by r, including the intermediate budgets never measured
    # at onset 512/2048 (grey), rows = onset
    cells = []
    for b in (256, 512, 1024, 2048):
        for t in (2304, 3328, 4352):
            key = f"b{b}@{t}"
            r = fp[key]["realized_ratio"] if key in fp else None
            cells.append((b, t, key, r))
    cells.sort(key=lambda c: (c[3] if c[3] is not None else c[0] / (115 + c[1])))
    rows = ["t0 = 0", "t0 = 512", "t0 = 2048"]
    tv = [
        [
            (fp[k]["arms"][o]["window_tv"]["mean"] if k in fp else None)
            for (_b, _t, k, _r) in cells
        ]
        for o in ("0", "512", "2048")
    ]
    cols = [
        (
            f"B{b}@{t}\nr={r:.3f}"
            if r is not None
            else f"B{b}@{t}\n(r approx {b / (115 + t):.2f})"
        )
        for (b, t, _k, r) in cells
    ]
    _heat(
        axes[0],
        tv,
        rows,
        cols,
        "(a) window-mean TV over (realized ratio, onset), measured cells only",
        ".3f",
        "viridis",
    )
    # (2,3) damage over (budget, onset), per policy
    for ax, p in zip(axes[1:], ("random", "streamingllm"), strict=True):
        g = [
            [
                (
                    dacc["cells"][f"{p}-{b}-t{o}"]["damage_fc_minus_comp"]["mean"]
                    if f"{p}-{b}-t{o}" in dacc["cells"]
                    else None
                )
                for b in (256, 512, 1024, 2048)
            ]
            for o in (0, 512, 2048)
        ]
        _heat(
            ax,
            g,
            rows,
            ["256", "512", "1024", "2048"],
            f"(b) damage over (budget, onset): {LABEL[p]}",
            "+.2f",
            "magma",
        )
    fig.tight_layout()
    return _save(fig, "r17_heatmaps_measured_cells", out)


def fig_collapse(out: Path) -> list[Path]:
    col = _load("r17/collapse.json")
    if not col or col.get("status") != "READ":
        return []
    pols = col["per_policy"]
    fig, axes = plt.subplots(1, len(pols), figsize=(5.2 * len(pols), 3.6))
    axes = axes if len(pols) > 1 else [axes]
    for ax, (p, v) in zip(axes, sorted(pols.items()), strict=True):
        for b in (256, 512, 1024, 2048):
            pts = sorted(
                (v["cell_r"][f"B{b}@{t}"], v["cell_means"][f"B{b}@{t}"])
                for t in (2304, 3328, 4352)
            )
            ax.plot(
                [x for x, _ in pts],
                [y for _, y in pts],
                "o-",
                color=BUD_COLOR[b],
                label=f"B = {b}",
            )
        for pr in v["pairs"]:
            small, large = pr["pair"].split(" vs ")
            xs = [v["cell_r"][small], v["cell_r"][large]]
            ys = [v["cell_means"][small], v["cell_means"][large]]
            ax.plot(xs, ys, "-", color="black", lw=2.2, alpha=0.35)
        ax.set_xscale("log")
        ax.set_xlabel("realized ratio r = B / (prompt + position)", fontsize=8)
        ax.set_ylabel("onset-0 window-mean TV", fontsize=8)
        ax.set_title(f"{LABEL.get(p, p)}: {v['label']}", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    return _save(fig, "r17_collapse", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    a = ap.parse_args()
    made = fig_supervisor(a.out_dir) + fig_heatmaps(a.out_dir) + fig_collapse(a.out_dir)
    for p in made:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

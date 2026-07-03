#!/usr/bin/env python3
"""Generate the five paper figures.

Two rendering paths per figure:
  * SUMMARY (always available): drawn from the verified headline constants below, so
    Figures 1 (bars), 2, 3 and a windowed Figure 4 are publication-ready anywhere.
  * DATA (when the run artifacts are reachable, i.e. on the remote): adds the
    per-problem paired-diff inset (Fig 1), the full onset-aligned per-step TV curve
    (Fig 4), and the per-problem ADD histogram (Fig 5).

Each figure prints which mode it used. Data paths are optional; missing/unparseable
data degrades gracefully to the SUMMARY rendering rather than crashing.

Numbers are the corrected, final values (residual-bracket fix, PREREGISTRATION
Amendment H). ADD's within-answered TV is a *renormalized channel diagnostic*, not an
additive part of ADD, so Figure 3 does NOT stack it onto ADD.

Usage:
  scripts/make_paper_figures.py --out-dir figures \
      [--runs-root <runs>] [--m4-summary <summary.json>] \
      [--profiles-dir <replay-diss256/profiles>] [--twofc-report <report.json>]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- verified headline constants (final; corrected brackets) ----
FC = 0.882
C = {
    "Random": dict(
        free=0.422, readout=0.765, readout_ci=(0.686, 0.817), reverse=0.636,
        residual=-0.343, bracket=(-0.343, -0.246),
        add=0.559, within=0.504, term=0.224, flip=-0.461, auc_tv=0.086, trunc=0.233,
    ),
    "StreamingLLM": dict(
        free=0.237, readout=0.584, readout_ci=(0.495, 0.647), reverse=0.417,
        residual=-0.347, bracket=(-0.466, -0.347),
        add=0.751, within=0.749, term=0.081, flip=-0.646, auc_tv=0.083, trunc=0.070,
    ),
}
PHEN = dict(gap=0.185, gap_ci=(0.130, 0.242), n_win=94, n_tot=100)
TVWIN = dict(before=0.0000, first50=0.0018, later=0.0884, onset=65)
TWOFC = dict(mean=0.068, median=0.000, p=0.976, frac_sig=0.0, n=250)

BLUE, ORANGE, GREY = "#3366cc", "#dd8452", "#8a8a8a"
MC = {"Random": BLUE, "StreamingLLM": ORANGE}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, out_dir: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------- Figure 1
def fig1_phenomenon(out_dir: Path, runs_root: Path | None):
    per_problem = _load_paired_diffs(runs_root) if runs_root else None
    mode = "DATA (with per-problem inset)" if per_problem else "SUMMARY"
    print(f"Figure 1 [{mode}]")
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    labels = ["Full cache", "Random\n(256)", "StreamingLLM\n(256)"]
    vals = [FC, C["Random"]["free"], C["StreamingLLM"]["free"]]
    ax.bar(labels, vals, color=[GREY, BLUE, ORANGE], width=0.62)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_ylabel("Free-running accuracy (MATH-500)")
    ax.set_ylim(0, 1.0)
    ax.annotate(
        f"Random $-$ SLM: $+{PHEN['gap']:.3f}$\n"
        f"95% CI [{PHEN['gap_ci'][0]:.3f}, {PHEN['gap_ci'][1]:.3f}], "
        f"{PHEN['n_win']}/{PHEN['n_tot']} problems",
        xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9),
    )
    if per_problem:
        ins = ax.inset_axes([0.60, 0.30, 0.36, 0.40])
        ins.hist(per_problem, bins=20, color=BLUE, alpha=0.8)
        ins.axvline(0, color="k", lw=0.8)
        ins.set_title("per-problem\nRandom$-$SLM", fontsize=6)
        ins.tick_params(labelsize=5)
    _save(fig, out_dir, "fig1_phenomenon")


# ---------------------------------------------------------------- Figure 2
def fig2_bracket(out_dir: Path, m4_summary: dict | None):
    cells = _cells_from_summary(m4_summary) if m4_summary else C
    print(f"Figure 2 [{'DATA' if m4_summary else 'SUMMARY'}]")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    rung_labels = ["Full cache", "Forward\nreadout", "Reverse\nreadout", "Free-run"]
    for ax, m in zip(axes, ["Random", "StreamingLLM"], strict=False):
        d = cells[m]
        vals = [FC, d["readout"], d["reverse"], d["free"]]
        ys = [3, 2, 1, 0]
        for y, v in zip(ys, vals, strict=True):
            ax.plot([0, v], [y, y], color=MC[m], lw=2)
            ax.plot(v, y, "o", color=MC[m], ms=6)
            ax.text(v + 0.02, y, f"{v:.3f}", va="center", fontsize=8)
        ax.set_yticks(ys)
        ax.set_yticklabels(rung_labels, fontsize=7.5)
        lo, hi = d["bracket"]
        ax.set_title(m, fontsize=9.5)
        ax.annotate(f"residual bracket\n[{lo:.3f}, {hi:.3f}]", xy=(0.97, 0.52),
                    xycoords="axes fraction", ha="right", va="center", fontsize=7.5,
                    bbox=dict(boxstyle="round", fc="white", ec=MC[m]))
        ax.set_xlim(0, 1.15)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlabel("Accuracy")
    fig.tight_layout()
    _save(fig, out_dir, "fig2_bracket")


# ---------------------------------------------------------------- Figure 3
def fig3_add(out_dir: Path):
    print("Figure 3 [SUMMARY]")
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    methods = ["StreamingLLM", "Random"]
    xs = [0, 1]
    w = 0.34
    within = [C[m]["within"] for m in methods]
    term = [C[m]["term"] for m in methods]
    ax.bar([x - w / 2 for x in xs], within, w, color="#4c78a8",
           label="within-answered TV (renorm. diagnostic)")
    ax.bar([x + w / 2 for x in xs], term, w, color=GREY,
           label="termination-mass shift")
    for i, m in enumerate(methods):
        ax.text(i - w / 2, within[i] + 0.015, f"{within[i]:.3f}", ha="center",
                fontsize=7)
        ax.text(i + w / 2, term[i] + 0.015, f"+{term[i]:.3f}", ha="center", fontsize=7)
        ax.text(i, 0.90, f"ADD {C[m]['add']:.3f}\nflip {C[m]['flip']:.3f}", ha="center",
                va="top", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="white", ec=MC[m], alpha=0.9))
    ax.axhline(TWOFC["mean"], color="green", ls="--", lw=1,
               label=f"two-FC noise floor {TWOFC['mean']:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Total variation")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=6.5,
              frameon=False)
    _save(fig, out_dir, "fig3_add")


# ---------------------------------------------------------------- Figure 4
def fig4_tvprofile(out_dir: Path, profiles_dir: Path | None):
    curve = _onset_aligned_curve(profiles_dir) if profiles_dir else None
    print(f"Figure 4 [{'DATA (full curve)' if curve else 'SUMMARY (windowed)'}]")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    if curve:
        xs, ys = curve
        ax.plot(xs, ys, color=BLUE, lw=1.6)
        ax.axvline(0, color="k", ls=":", lw=0.9)
        ax.set_xlabel("Decoding step relative to first eviction")
    else:
        labels = ["before\neviction", "first 50\nafter", "later\n(>50 after)"]
        vals = [TVWIN["before"], TVWIN["first50"], TVWIN["later"]]
        ax.bar(labels, vals, color=[GREY, "#9ecae1", BLUE], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)
        ax.set_xlabel(f"window rel. to first eviction (onset step {TVWIN['onset']})")
    ax.set_ylabel("Mean per-step total variation")
    ax.set_title("Divergence is ~0 at eviction and compounds over the trajectory",
                 fontsize=8)
    _save(fig, out_dir, "fig4_tvprofile")


# ---------------------------------------------------------------- Figure 5
def fig5_twofc(out_dir: Path, twofc_report: dict | None):
    vals = _twofc_values(twofc_report) if twofc_report else None
    print(f"Figure 5 [{'DATA (histogram)' if vals else 'SUMMARY'}]")
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    if vals:
        ax.hist(vals, bins=30, color=GREY, alpha=0.85)
        ax.axvline(TWOFC["mean"], color="green", ls="--", lw=1,
                   label=f"two-FC mean {TWOFC['mean']:.3f}")
        for m in ["Random", "StreamingLLM"]:
            ax.axvline(C[m]["add"], color=MC[m], lw=1.4,
                       label=f"{m} ADD {C[m]['add']:.2f}")
        ax.set_xlabel("Per-problem ADD")
        ax.set_ylabel("count")
    else:
        bars = ["two-FC\nfloor", "Random\nADD", "SLM\nADD"]
        bvals = [TWOFC["mean"], C["Random"]["add"], C["StreamingLLM"]["add"]]
        ax.bar(bars, bvals, color=[GREY, BLUE, ORANGE], width=0.5)
        for i, v in enumerate(bvals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylabel("ADD")
    ax.set_title(f"Negative control: ADD floor {TWOFC['mean']:.3f} "
                 f"({int(TWOFC['frac_sig']*100)}% sig., n={TWOFC['n']})", fontsize=8)
    ax.legend(fontsize=6.5)
    _save(fig, out_dir, "fig5_twofc")


# ---------------------------------------------------------------- data loaders
def _load_paired_diffs(runs_root: Path):
    try:
        acc = {"random": {}, "streamingllm": {}}
        for f in glob.glob(str(runs_root / "**/*.jsonl"), recursive=True):
            if "token_ids" in f:
                continue
            for ln in open(f, errors="ignore"):
                if not ln.strip():
                    continue
                d = json.loads(ln)
                p = d.get("press")
                if p in acc and d.get("budget") == 256:
                    acc[p].setdefault(d["pid"], []).append(bool(d["correct"]))
        pids = set(acc["random"]) & set(acc["streamingllm"])
        fm = statistics.fmean
        diffs = [fm(acc["random"][p]) - fm(acc["streamingllm"][p]) for p in pids]
        return diffs or None
    except Exception as e:  # noqa: BLE001
        print(f"  (paired-diff load failed: {e}; using SUMMARY)")
        return None


def _cells_from_summary(summ: dict):
    try:
        out = {}
        for c in summ["cells"]:
            name = {"random": "Random", "streamingllm": "StreamingLLM"}.get(c["press"])
            if name:
                out[name] = dict(
                    free=c["compressed_accuracy"], readout=c["readout_accuracy"],
                    reverse=c["reverse_readout_accuracy"],
                    residual=c["unexplained_residual"],
                    bracket=tuple(c["residual_bracket"]),
                )
        return out or C
    except Exception as e:  # noqa: BLE001
        print(f"  (summary parse failed: {e}; using constants)")
        return C


def _onset_aligned_curve(profiles_dir: Path, half=400):
    try:
        acc = {}
        for f in glob.glob(str(profiles_dir / "**/*.json"), recursive=True):
            d = json.load(open(f))
            est, onset = d.get("estimate"), d.get("onset_step")
            if not est or onset is None:
                continue
            for i, v in enumerate(est):
                off = i - onset
                if -half <= off <= half:
                    acc.setdefault(off, []).append(v)
        if not acc:
            return None
        xs = sorted(acc)
        return xs, [statistics.fmean(acc[x]) for x in xs]
    except Exception as e:  # noqa: BLE001
        print(f"  (TV curve load failed: {e}; using windowed)")
        return None


def _twofc_values(report: dict):
    try:
        for key in ("per_problem", "problems", "add_values"):
            if key in report:
                vals = [p["add"] if isinstance(p, dict) else p for p in report[key]]
                if vals:
                    return vals
        return None
    except Exception as e:  # noqa: BLE001
        print(f"  (two-FC values load failed: {e}; using SUMMARY)")
        return None


def _load_json(path: str | None):
    if not path:
        return None
    try:
        return json.load(open(path))
    except Exception as e:  # noqa: BLE001
        print(f"  (could not load {path}: {e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--runs-root", default=None, help="grid run dir (Fig 1 inset)")
    ap.add_argument("--m4-summary", default=None, help="report_m4 summary.json (Fig 2)")
    ap.add_argument("--profiles-dir", default=None, help="replay profiles (Fig 4)")
    ap.add_argument("--twofc-report", default=None, help="two-FC report json (Fig 5)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig1_phenomenon(out, Path(args.runs_root) if args.runs_root else None)
    fig2_bracket(out, _load_json(args.m4_summary))
    fig3_add(out)
    fig4_tvprofile(out, Path(args.profiles_dir) if args.profiles_dir else None)
    fig5_twofc(out, _load_json(args.twofc_report))
    print(f"\nAll figures written to {out}/")


if __name__ == "__main__":
    main()

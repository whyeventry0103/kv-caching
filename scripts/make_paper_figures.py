#!/usr/bin/env python3
"""Generate the five paper figures (publication styling, colorblind-safe palette).

Two rendering paths per figure:
  * SUMMARY (always available): drawn from the verified headline constants below, so
    every figure is publication-ready anywhere.
  * DATA (when the run artifacts are reachable, i.e. on the remote): adds the
    per-problem paired-diff histogram (Fig 1), report-exact bracket + CIs (Fig 2),
    the full onset-aligned per-step TV curve (Fig 4), and the per-problem ADD
    histogram (Fig 5). Missing/unparseable data degrades gracefully to SUMMARY.

Style: Okabe-Ito colorblind-safe hues (validated: worst adjacent-pair CVD dE 91.9),
Random=blue / StreamingLLM=vermillion, fixed assignment, never cycled; grey is the
neutral full-cache reference; green is reserved for the negative-control line.
Light tints of each method hue encode the 2048 budget (dose), always direct-labeled.

Numbers are the corrected, final values (residual-bracket fix + endpoint CIs,
PREREGISTRATION Amendments F/H). ADD's within-answered TV is a *renormalized channel
diagnostic*, not an additive part of ADD, so Figure 3 does NOT stack it onto ADD.

Usage:
  scripts/make_paper_figures.py --out-dir figures \
      [--runs-root <runs>] [--m4-summary <summary.json>] \
      [--profiles-dir <replay-diss256/profiles>] [--twofc-report <report.json>] \
      [--print-windows]
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

# ---- verified headline constants (final; corrected brackets + endpoint CIs) ----
FC = 0.882
C = {
    "Random": dict(
        free=0.422,
        readout=0.765,
        readout_ci=(0.686, 0.817),
        reverse=0.636,
        residual=-0.343,
        bracket=(-0.343, -0.246),
        fwd_ci=(-0.402, -0.258),
        rev_ci=(-0.383, -0.213),
        add=0.559,
        within=0.504,
        term=0.224,
        flip=-0.461,
        auc_tv=0.086,
        trunc=0.233,
    ),
    "StreamingLLM": dict(
        free=0.237,
        readout=0.584,
        readout_ci=(0.495, 0.647),
        reverse=0.417,
        residual=-0.347,
        bracket=(-0.466, -0.347),
        fwd_ci=(-0.403, -0.268),
        rev_ci=(-0.553, -0.410),
        add=0.751,
        within=0.749,
        term=0.081,
        flip=-0.646,
        auc_tv=0.083,
        trunc=0.070,
    ),
}
# Budget-2048 free-run (grid-diss2048-tok, 800/800, read 2026-07-04).
C2048 = {"Random": 0.850, "StreamingLLM": 0.808}
PHEN = dict(gap=0.185, gap_ci=(0.130, 0.242), win=44, tie=50, loss=6, n_tot=100)
TVWIN = dict(before=0.0000, first50=0.0018, later=0.0884, onset=65)
TWOFC = dict(mean=0.068, median=0.000, p=0.976, frac_sig=0.0, n=250)

# Palette (validated, see module docstring). GREEN reserved for the control line.
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#8a8a8a"
BLUE_LT, ORANGE_LT = "#9CC7E4", "#F2B27F"  # 2048-budget tints (dose)
GREEN = "#009E73"
INK, INK2, GRIDC, SPINE = "#1a1a1a", "#555555", "#dddddd", "#9a9a9a"
MC = {"Random": BLUE, "StreamingLLM": ORANGE}

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _ygrid(ax):
    ax.grid(axis="y", color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)


def _save(fig, out_dir: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------- Figure 1
def fig1_phenomenon(out_dir: Path, runs_root: Path | None):
    per_problem = _load_paired_diffs(runs_root) if runs_root else None
    print(f"Figure 1 [{'DATA (paired-diff panel)' if per_problem else 'SUMMARY'}]")
    fig, (axb, axh) = plt.subplots(
        1, 2, figsize=(6.9, 2.9), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    labels = ["Full\ncache", "Random\n256", "SLM\n256", "Random\n2048", "SLM\n2048"]
    vals = [
        FC,
        C["Random"]["free"],
        C["StreamingLLM"]["free"],
        C2048["Random"],
        C2048["StreamingLLM"],
    ]
    colors = [GREY, BLUE, ORANGE, BLUE_LT, ORANGE_LT]
    axb.bar(labels, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    for x, v in enumerate(vals):
        axb.text(x, v + 0.018, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    axb.axhline(FC, color=GREY, lw=0.8, ls=(0, (2, 3)), zorder=0)
    axb.set_ylabel("Free-running accuracy (MATH-500)")
    axb.set_ylim(0, 1.0)
    _ygrid(axb)
    gaptxt = (
        f"Random $-$ SLM @256: $+{PHEN['gap']:.3f}$\n"
        f"95% CI [{PHEN['gap_ci'][0]:.3f}, {PHEN['gap_ci'][1]:.3f}]\n"
        f"per problem: {PHEN['win']} win / {PHEN['tie']} tie / {PHEN['loss']} loss"
    )
    if per_problem:
        axh.hist(per_problem, bins=20, color=BLUE, edgecolor="white", linewidth=0.5)
        axh.axvline(0, color=INK, lw=0.9)
        axh.set_xlabel(r"Per-problem $\Delta$ (Random $-$ SLM) @256")
        axh.set_ylabel("Problems")
        _ygrid(axh)
        axh.annotate(
            gaptxt,
            xy=(0.03, 0.96),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=6.6,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=SPINE, lw=0.7),
        )
    else:
        axh.axis("off")
        axh.text(
            0.5,
            0.5,
            gaptxt,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=SPINE, lw=0.7),
        )
    fig.tight_layout(w_pad=2.0)
    _save(fig, out_dir, "fig1_phenomenon")


# ---------------------------------------------------------------- Figure 2
def fig2_bracket(out_dir: Path, m4_summary: dict | None):
    cells = _cells_from_summary(m4_summary) if m4_summary else C
    print(f"Figure 2 [{'DATA' if m4_summary else 'SUMMARY'}]")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), sharey=True)
    rung_labels = ["Full cache", "Forward\nreadout", "Reverse\nreadout", "Free-run"]
    for ax, m in zip(axes, ["Random", "StreamingLLM"], strict=False):
        d = cells[m]
        vals = [FC, d["readout"], d["reverse"], d["free"]]
        ys = [3, 2, 1, 0]
        ax.grid(axis="x", color=GRIDC, lw=0.6)
        ax.set_axisbelow(True)
        for y, v in zip(ys, vals, strict=True):
            color = GREY if y == 3 else MC[m]
            ax.plot([0, v], [y, y], color=color, lw=2.0, solid_capstyle="round")
            ax.plot(v, y, "o", color=color, ms=6.5, mec="white", mew=0.8)
            ax.text(v + 0.025, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
        ci = d.get("readout_ci")
        if ci:
            ax.plot([ci[0], ci[1]], [2, 2], color=MC[m], lw=0.9, alpha=0.55)
            for x in ci:
                ax.plot([x, x], [1.88, 2.12], color=MC[m], lw=0.9, alpha=0.55)
        # The key quantity as geometry: the forward residual is the horizontal gap
        # between free-run and forward readout.
        fwd_ci = d.get("fwd_ci")
        ax.annotate(
            "",
            xy=(d["free"], -0.55),
            xytext=(d["readout"], -0.55),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.0, shrinkA=0, shrinkB=0),
            annotation_clip=False,
        )
        lo, hi = d["bracket"]
        rev_ci = d.get("rev_ci")
        residual_txt = f"trajectory residual ${d['residual']:.3f}$"
        if fwd_ci:
            residual_txt += f"  (95% CI [{fwd_ci[0]:.3f}, {fwd_ci[1]:.3f}])"
        residual_txt += f"\nbracket [{lo:.3f}, {hi:.3f}]"
        if rev_ci:
            residual_txt += f" $\\cdot$ reverse CI [{rev_ci[0]:.3f}, {rev_ci[1]:.3f}]"
        ax.text(
            0.5,
            -0.98,
            residual_txt,
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.0,
            color=INK2,
            clip_on=False,
        )
        ax.set_yticks(ys)
        ax.set_yticklabels(rung_labels, fontsize=8)
        ax.set_title(m, fontsize=9.5, color=MC[m])
        ax.set_xlim(0, 1.12)
        ax.set_ylim(-1.9, 3.5)
        ax.set_xlabel("Accuracy")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=2.0)
    _save(fig, out_dir, "fig2_bracket")


# ---------------------------------------------------------------- Figure 3
def fig3_add(out_dir: Path):
    print("Figure 3 [SUMMARY]")
    fig, ax = plt.subplots(figsize=(4.9, 3.2))
    methods = ["StreamingLLM", "Random"]
    xs = [0, 1]
    w = 0.33
    within = [C[m]["within"] for m in methods]
    term = [C[m]["term"] for m in methods]
    ax.bar(
        [x - w / 2 - 0.01 for x in xs],
        within,
        w,
        color=[MC[m] for m in methods],
        edgecolor="white",
        linewidth=0.8,
        label="within-answered TV (renormalized)",
    )
    ax.bar(
        [x + w / 2 + 0.01 for x in xs],
        term,
        w,
        color=GREY,
        edgecolor="white",
        linewidth=0.8,
        label="termination-mass shift",
    )
    for i, m in enumerate(methods):
        ax.text(
            i - w / 2 - 0.01,
            within[i] + 0.015,
            f"{within[i]:.3f}",
            ha="center",
            fontsize=7.5,
            color=INK,
        )
        ax.text(
            i + w / 2 + 0.01,
            term[i] + 0.015,
            f"+{term[i]:.3f}",
            ha="center",
            fontsize=7.5,
            color=INK,
        )
        ax.text(
            i,
            0.945,
            f"ADD {C[m]['add']:.3f}   flip ${C[m]['flip']:.3f}$",
            ha="center",
            va="top",
            fontsize=7.5,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MC[m], lw=0.9),
        )
    ax.axhline(TWOFC["mean"], color=GREEN, ls=(0, (4, 3)), lw=1.1)
    ax.text(
        1.44,
        TWOFC["mean"] + 0.012,
        f"two-FC noise floor {TWOFC['mean']:.3f}",
        ha="right",
        fontsize=7,
        color=GREEN,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Total variation")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.55, 1.55)
    _ygrid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=2, frameon=False)
    _save(fig, out_dir, "fig3_add")


# ---------------------------------------------------------------- Figure 4
def fig4_tvprofile(out_dir: Path, profiles_dir: Path | None):
    curve = _onset_aligned_curve(profiles_dir) if profiles_dir else None
    print(f"Figure 4 [{'DATA (full curve)' if curve else 'SUMMARY (windowed)'}]")
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    if curve:
        xs, ys = curve
        ax.plot(xs, ys, color=BLUE, lw=1.3)
        ax.axvline(0, color=INK2, ls=(0, (3, 3)), lw=0.9)
        top = max(ys)
        ax.text(
            -8,
            top * 0.98,
            "first eviction",
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color=INK2,
        )
        step_x = 192
        step_y = ys[xs.index(step_x)] if step_x in xs else top * 0.35
        ax.annotate(
            "eviction events\nevery 64 steps",
            xy=(step_x, step_y),
            xytext=(60, top * 0.72),
            fontsize=7,
            color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8),
        )
        ax.set_xlabel("Decoding step relative to first eviction")
    else:
        labels = ["before\neviction", "first 50\nafter", "later\n(>50 after)"]
        vals = [TVWIN["before"], TVWIN["first50"], TVWIN["later"]]
        ax.bar(
            labels,
            vals,
            color=["#c9c9c9", "#7fb3d8", BLUE],
            width=0.6,
            edgecolor="white",
            linewidth=0.8,
        )
        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8, color=INK)
        ax.set_xlabel(f"Window rel. to first eviction (onset step {TVWIN['onset']})")
    ax.set_ylabel("Mean per-step total variation")
    _ygrid(ax)
    _save(fig, out_dir, "fig4_tvprofile")


# ---------------------------------------------------------------- Figure 5
def fig5_twofc(out_dir: Path, twofc_report: dict | None):
    vals = _twofc_values(twofc_report) if twofc_report else None
    print(f"Figure 5 [{'DATA (histogram)' if vals else 'SUMMARY'}]")
    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    if vals:
        ax.hist(vals, bins=30, color=GREY, edgecolor="white", linewidth=0.5)
        ax.axvline(
            TWOFC["mean"],
            color=GREEN,
            ls=(0, (4, 3)),
            lw=1.1,
            label=f"two-FC mean {TWOFC['mean']:.3f}",
        )
        for m in ["Random", "StreamingLLM"]:
            ax.axvline(
                C[m]["add"], color=MC[m], lw=1.3, label=f"{m} ADD {C[m]['add']:.2f}"
            )
        ax.set_xlabel("Per-problem ADD")
        ax.set_ylabel("Problems")
        ax.legend(frameon=False)
        _ygrid(ax)
    else:
        bars = ["two-FC\nnoise floor", "Random\nADD @256", "SLM\nADD @256"]
        bvals = [TWOFC["mean"], C["Random"]["add"], C["StreamingLLM"]["add"]]
        ax.bar(
            bars,
            bvals,
            color=[GREY, BLUE, ORANGE],
            width=0.55,
            edgecolor="white",
            linewidth=0.8,
        )
        for i, v in enumerate(bvals):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.set_ylabel("ADD (total variation)")
        ax.text(
            0,
            TWOFC["mean"] + 0.06,
            f"mean p = {TWOFC['p']:.3f}\n{int(TWOFC['frac_sig'] * 100)}% significant\n"
            f"n = {TWOFC['n']}",
            ha="center",
            fontsize=6.8,
            color=INK2,
        )
        _ygrid(ax)
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
                    free=c["compressed_accuracy"],
                    readout=c["readout_accuracy"],
                    readout_ci=tuple(c.get("readout_accuracy_ci95") or ())
                    or C[name]["readout_ci"],
                    reverse=c["reverse_readout_accuracy"],
                    residual=c["unexplained_residual"],
                    bracket=tuple(c["residual_bracket"]),
                    fwd_ci=tuple(c.get("unexplained_residual_ci95") or ())
                    or C[name]["fwd_ci"],
                    rev_ci=tuple(c.get("reverse_residual_ci95") or ())
                    or C[name]["rev_ci"],
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


def print_windowed_tv(profiles_dir: Path):
    """Committed provenance for the paper's windowed-TV table (Table: tab:tv).

    Full-horizon (NO +/-400 cap, unlike the figure curve) mean per-step TV in the
    three onset-aligned windows: before eviction (off < 0), first 50 steps after
    (0 <= off < 50), later (off >= 50). Two estimators are printed:
    per-profile = mean of per-profile window means (each trace counts once; the
    trace is the sampling unit, matching the problem-clustered convention);
    pooled = mean over all steps (long traces weigh more). The paper's table
    reports the per-profile estimator.
    """
    pooled = {"before": [], "first50": [], "later": []}
    per_profile = {"before": [], "first50": [], "later": []}
    n_profiles = 0
    for f in glob.glob(str(profiles_dir / "**/*.json"), recursive=True):
        d = json.load(open(f))
        est, onset = d.get("estimate"), d.get("onset_step")
        if not est or onset is None:
            continue
        n_profiles += 1
        local = {"before": [], "first50": [], "later": []}
        for i, v in enumerate(est):
            off = i - onset
            name = "before" if off < 0 else ("first50" if off < 50 else "later")
            pooled[name].append(v)
            local[name].append(v)
        for name, vals in local.items():
            if vals:
                per_profile[name].append(statistics.fmean(vals))
    if not n_profiles:
        print("windowed TV: no profiles found")
        return
    print(f"windowed TV over {n_profiles} profiles (full horizon):")
    for name in ("before", "first50", "later"):
        pooled_mean = statistics.fmean(pooled[name]) if pooled[name] else float("nan")
        prof_mean = (
            statistics.fmean(per_profile[name]) if per_profile[name] else float("nan")
        )
        print(
            f"  {name}: per-profile {prof_mean:.4f} over {len(per_profile[name])} "
            f"profiles; pooled {pooled_mean:.4f} over {len(pooled[name])} steps"
        )


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
    ap.add_argument(
        "--print-windows",
        action="store_true",
        help="print full-horizon windowed TV means from --profiles-dir "
        "(provenance for the paper's windowed-TV table)",
    )
    args = ap.parse_args()

    if args.print_windows:
        if not args.profiles_dir:
            ap.error("--print-windows requires --profiles-dir")
        print_windowed_tv(Path(args.profiles_dir))

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

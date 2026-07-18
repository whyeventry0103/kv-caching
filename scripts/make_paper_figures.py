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
# Display convention (pinned 2026-07-07): every 3-decimal label is f"{v:.3f}" of the
# report-artifact value — the same string numpy/R produce when recomputing from raw
# rows. Do NOT hand-round exact ties (99/400=0.2475, 323/400=0.8075) up to 0.248 /
# 0.808: that made Fig 1 (constants) disagree with Fig 2 (DATA mode) for the same
# quantity. Tie-valued constants below are stored at artifact precision so .3f
# renders them identically in both modes.
FC = 0.8825
C = {
    "Random": dict(
        free=0.4225,
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
        free=0.2375,
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
C["H2O-style"] = dict(
    add=0.715,
    within=0.713,
    term=0.101,
    flip=-0.636,
    free=0.2475,
    readout=0.597,
    readout_ci=(0.503, 0.668),
    reverse=0.419,
    residual=-0.349,
    bracket=(-0.463, -0.349),
    fwd_ci=(-0.403, -0.270),
    rev_ci=(-0.566, -0.420),
    trunc=0.105,
)
C["R-KV"] = dict(
    # Certified port (parity Jaccard 1.0 + generation smoke; Amendment I). Branch C
    # read 2026-07-12: best free-runner, same readout-invisible residual structure.
    free=0.52,
    readout=0.798,
    readout_ci=(0.710, 0.850),
    reverse=0.662,
    residual=-0.278,
    bracket=(-0.278, -0.221),
    fwd_ci=(-0.335, -0.195),
    rev_ci=(-0.355, -0.203),
    trunc=0.135,
    # ADD instrument values (report_add over grid-diss256-rkv, read 2026-07-12;
    # results/diag/add-256-rkv-summary.json). Lowest ADD of the four policies,
    # coherent with its highest free-run accuracy.
    add=0.459,
    within=0.450,
    term=0.149,
    flip=-0.364,
)
# Budget-2048 free-run (grid-diss2048-tok 800/800 read 2026-07-04; h2o 400/400
# read 2026-07-10; 0.8075/0.8225 stored at artifact precision -> .3f displays).
C2048 = {"Random": 0.850, "StreamingLLM": 0.8075, "H2O-style": 0.8225}
PHEN = dict(gap=0.185, gap_ci=(0.130, 0.242), win=44, tie=50, loss=6, n_tot=100)
TVWIN = dict(before=0.0000, first50=0.0018, later=0.0884, onset=65)
TWOFC = dict(mean=0.068, median=0.000, p=0.976, frac_sig=0.0, n=250)

# Palette (validated, see module docstring). GREEN reserved for the control line.
BLUE, ORANGE, PURPLE, GREY = "#0072B2", "#D55E00", "#CC79A7", "#8a8a8a"
SKY = "#56B4E9"  # R-KV (4th hue; validated 2026-07-12: worst adjacent-pair CVD
# dE 20.7 vs PURPLE, 15.7 vs BLUE_LT -- both above the 12 floor; contrast WARN
# relieved by direct value labels on every mark, per the dataviz palette rules)
BLUE_LT, ORANGE_LT, PURPLE_LT = "#9CC7E4", "#F2B27F", "#E5BCD3"  # 2048 dose tints
GREEN = "#009E73"
INK, INK2, GRIDC, SPINE = "#1a1a1a", "#555555", "#dddddd", "#9a9a9a"
MC = {"Random": BLUE, "StreamingLLM": ORANGE, "H2O-style": PURPLE, "R-KV": SKY}

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


# ---------------------------------------------------------------- Figure 0
def fig0_protocol(out_dir: Path):
    """Protocol schematic for the four estimands (SUMMARY-only; no data needed).

    Fill encodes the CACHE condition while a span is processed (grey = full cache,
    hatched = compression active); the in-span text says where the tokens come from
    (self-generated vs teacher-forced replay of stored token IDs). Texture, not a
    method hue, marks "compressed" so the schematic stays method-generic and does
    not collide with the fixed per-method palette."""
    print("Figure 0 [SUMMARY (protocol schematic)]")
    fig, ax = plt.subplots(figsize=(6.9, 2.7))
    trace_x, trace_w, ans_x, ans_w, bar_h = 0.0, 6.0, 6.15, 1.85, 0.52
    full = dict(facecolor="#c9c9c9", edgecolor=SPINE, linewidth=0.8)
    comp = dict(facecolor="white", edgecolor=INK, linewidth=0.8, hatch="////")
    rows = [
        # (label, trace style, trace text, answer style, right annotation)
        (
            "Full-cache\nreference",
            full,
            "self-generated trace",
            full,
            r"$\mathrm{Acc}_{\mathrm{FC}}$",
        ),
        (
            "Compressed\nfree-run",
            comp,
            "self-generated trace",
            comp,
            r"$\mathrm{Acc}_{\mathrm{free}}\;\rightarrow\;\Delta_{\mathrm{tot}}$",
        ),
        (
            "Forward\nreadout",
            comp,
            "reference token IDs, teacher-forced",
            comp,
            r"$\mathrm{Acc}_{\mathrm{ro}}\;\rightarrow\;\Delta_{\mathrm{ro}}$",
        ),
        (
            "Reverse\nreadout",
            full,
            "compressed free-run token IDs, teacher-forced",
            full,
            r"$\mathrm{Acc}_{\mathrm{rev}}$",
        ),
    ]
    for i, (label, tsty, ttxt, asty, right) in enumerate(rows):
        y = 3 - i
        ax.barh(y, trace_w, left=trace_x, height=bar_h, **tsty)
        ax.barh(y, ans_w, left=ans_x, height=bar_h, **asty)
        pad = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none")
        ax.text(
            trace_x + trace_w / 2,
            y,
            ttxt,
            ha="center",
            va="center",
            fontsize=7.2,
            color=INK,
            bbox=pad,
        )
        ax.text(
            ans_x + ans_w / 2,
            y,
            "answer",
            ha="center",
            va="center",
            fontsize=7.2,
            color=INK,
            bbox=pad,
        )
        ax.text(
            ans_x + ans_w + 0.18,
            y,
            right,
            ha="left",
            va="center",
            fontsize=8.2,
            color=INK,
        )
        ax.text(
            trace_x - 0.18, y, label, ha="right", va="center", fontsize=7.8, color=INK
        )
    ax.plot(
        [ans_x - 0.075] * 2, [-0.35, 3.42], color=INK2, lw=0.9, ls=(0, (2, 2)), zorder=0
    )
    ax.text(
        ans_x - 0.075,
        3.55,
        "reasoning/answer boundary\n(last </think>)",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=INK2,
    )
    ax.text(
        (trace_x + ans_x + ans_w) / 2,
        -0.92,
        r"$\rho(m) = \Delta_{\mathrm{tot}} - \Delta_{\mathrm{ro}}$ (residual);"
        " forward and reverse readout bracket the on-policy residual",
        ha="center",
        va="center",
        fontsize=7.8,
        color=INK,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, **full),
        plt.Rectangle((0, 0), 1, 1, **comp),
    ]
    ax.legend(
        handles,
        ["full cache", "compression active"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.24),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handlelength=1.4,
    )
    ax.set_xlim(-1.35, 10.4)
    ax.set_ylim(-1.35, 4.15)
    ax.axis("off")
    _save(fig, out_dir, "fig0_protocol")


# ---------------------------------------------------------------- Figure 1
def fig1_phenomenon(out_dir: Path, runs_root: Path | None):
    per_problem = _load_paired_diffs(runs_root) if runs_root else None
    print(f"Figure 1 [{'DATA (paired-diff panel)' if per_problem else 'SUMMARY'}]")
    fig, (axb, axh) = plt.subplots(
        1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.65, 1]}
    )
    labels = [
        "Full\ncache",
        "Random\n256",
        "SLM\n256",
        "H2O$^{\\dagger}$\n256",
        "R-KV$^{\\ddagger}$\n256",
        "Random\n2048",
        "SLM\n2048",
        "H2O$^{\\dagger}$\n2048",
    ]
    vals = [
        FC,
        C["Random"]["free"],
        C["StreamingLLM"]["free"],
        C["H2O-style"]["free"],
        C["R-KV"]["free"],
        C2048["Random"],
        C2048["StreamingLLM"],
        C2048["H2O-style"],
    ]
    colors = [GREY, BLUE, ORANGE, PURPLE, SKY, BLUE_LT, ORANGE_LT, PURPLE_LT]
    # Gap between the budget groups: secondary encoding on top of the tint scheme
    # (and it visually separates SKY from BLUE_LT, the closest validated pair).
    xs = [0, 1, 2, 3, 4, 5.55, 6.55, 7.55]
    axb.bar(xs, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    for x, v in zip(xs, vals, strict=True):
        axb.text(x, v + 0.018, f"{v:.3f}", ha="center", fontsize=7.6, color=INK)
    axb.axhline(FC, color=GREY, lw=0.8, ls=(0, (2, 3)), zorder=0)
    # Budget-group headers carry the realized compression ratios (pre-registered
    # Setup numbers), so the dose structure is legible from the figure alone.
    axb.text(
        2.5,
        1.045,
        "budget 256  (3.8% of mean trace)",
        ha="center",
        fontsize=7.2,
        color=INK2,
        clip_on=False,
    )
    axb.text(
        6.55,
        1.045,
        "budget 2048  (56.6%)",
        ha="center",
        fontsize=7.2,
        color=INK2,
        clip_on=False,
    )
    axb.set_xticks(xs)
    axb.set_xticklabels(labels, fontsize=7.6)
    axb.set_ylabel("Free-running accuracy (MATH-500)")
    axb.set_ylim(0, 1.0)
    axb.set_xlim(-0.6, 8.15)
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
    fig, axes = plt.subplots(2, 2, figsize=(9.9, 6.4), sharey=True)
    axes = axes.ravel()
    rung_labels = ["Full cache", "Forward\nreadout", "Reverse\nreadout", "Free-run"]
    methods = ["Random", "R-KV", "StreamingLLM", "H2O-style"]
    for ax, m in zip(axes, methods, strict=False):
        d = cells[m]
        vals = [FC, d["readout"], d["reverse"], d["free"]]
        ys = [3, 2, 1, 0]
        ax.grid(axis="x", color=GRIDC, lw=0.6)
        ax.set_axisbelow(True)
        # Faint reference vertical at Acc_FC so every rung reads against it.
        ax.axvline(FC, color=GREY, lw=0.7, ls=(0, (1, 2)), alpha=0.75, zorder=0)
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
    fig.tight_layout(w_pad=2.0, h_pad=2.4)
    _save(fig, out_dir, "fig2_bracket")


# ---------------------------------------------------------------- Figure 3
def fig3_add(out_dir: Path):
    print("Figure 3 [SUMMARY]")
    fig, ax = plt.subplots(figsize=(6.1, 3.2))
    methods = ["StreamingLLM", "H2O-style", "Random", "R-KV"]
    xs = [0, 1, 2, 3]
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
            f"ADD {C[m]['add']:.3f}\nflip ${C[m]['flip']:.3f}$",
            ha="center",
            va="top",
            fontsize=7.5,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MC[m], lw=0.9),
        )
    ax.axhline(TWOFC["mean"], color=GREEN, ls=(0, (4, 3)), lw=1.1)
    ax.text(
        3.44,
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
    ax.set_xlim(-0.55, 3.55)
    _ygrid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=2, frameon=False)
    _save(fig, out_dir, "fig3_add")


# ---------------------------------------------------------------- Figure 6
CAIME = dict(
    # m4-aime-2048 (read 2026-07-18; results/diag/m4-aime-2048-summary.json).
    # 30 problems x 4 seeds per cell vs the pinned AIME FC bank (0.3708).
    fc=0.3708,
    free={"Random": 0.2833, "R-KV": 0.2167, "StreamingLLM": 0.1417},
    readout=0.4624,  # IDENTICAL across all three presses (verified: same
    # 120-key correctness vectors; 85/93 defined rows converge to the same
    # extracted answer through different generations).
    readout_ci=(0.3274, 0.6607),
)


def fig6_aime(out_dir: Path):
    """AIME-24 @2048: free-running separates the presses; readout cannot.

    The identical readout bars ARE the message, so the readout bar is drawn in
    each method's hue with a shared texture (hatch = the readout protocol,
    consistent with fig0's compression-agnostic texture role), and free-run is
    the solid hue. Grey dashline = the AIME FC reference (0.371)."""
    print("Figure 6 [SUMMARY (AIME press-independence)]")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    methods = ["Random", "R-KV", "StreamingLLM"]
    xs = [0, 1, 2]
    w = 0.34
    for i, m in enumerate(methods):
        ax.bar(
            i - w / 2 - 0.01,
            CAIME["free"][m],
            w,
            color=MC[m],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar(
            i + w / 2 + 0.01,
            CAIME["readout"],
            w,
            facecolor="white",
            edgecolor=MC[m],
            linewidth=1.1,
            hatch="///",
        )
        ax.text(
            i - w / 2 - 0.01,
            CAIME["free"][m] + 0.012,
            f"{CAIME['free'][m]:.3f}",
            ha="center",
            fontsize=7.6,
            color=INK,
        )
        ax.text(
            i + w / 2 + 0.01,
            CAIME["readout"] + 0.012,
            f"{CAIME['readout']:.3f}",
            ha="center",
            fontsize=7.6,
            color=INK,
        )
    ax.axhline(CAIME["fc"], color=GREY, lw=0.9, ls=(0, (2, 3)), zorder=0)
    ax.text(
        2.52,
        CAIME["fc"] + 0.024,
        f"AIME FC reference {CAIME['fc']:.3f}",
        ha="right",
        fontsize=7,
        color=INK2,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#9a9a9a", edgecolor="white"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor=INK2, hatch="///", linewidth=1.0
        ),
    ]
    ax.legend(
        handles,
        ["free-run", "readout (identical across presses)"],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.14),
        ncol=2,
        frameon=False,
        fontsize=7.2,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Accuracy (AIME-24, budget 2048)")
    ax.set_ylim(0, 0.56)
    ax.set_xlim(-0.6, 2.6)
    _ygrid(ax)
    _save(fig, out_dir, "fig6_aime")


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
        # The instrument-validity reading, in the figure itself: the pre-onset
        # segment is EXACTLY zero (the replay harness adds no divergence).
        pre_x = min(xs)
        ax.annotate(
            "TV $= 0.0000$ before\nfirst eviction (self-control)",
            xy=(pre_x * 0.5, 0.0),
            xytext=(pre_x * 0.96, top * 0.30),
            fontsize=7,
            color=INK2,
            ha="left",
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
        cells = (
            summ["cells"]
            if isinstance(summ, dict)
            else [c for one in summ for c in one["cells"]]
        )
        for c in cells:
            name = {
                "random": "Random",
                "streamingllm": "StreamingLLM",
                "h2o": "H2O-style",
                "rkv": "R-KV",
            }.get(c["press"])
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
        return {**C, **out} if out else C
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
    ap.add_argument(
        "--m4-summary",
        default=None,
        nargs="*",
        help="one or more report_m4 summary.json files (Fig 2; cells are merged)",
    )
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
    fig0_protocol(out)
    fig6_aime(out)
    fig1_phenomenon(out, Path(args.runs_root) if args.runs_root else None)
    summaries = [x for x in (_load_json(p) for p in (args.m4_summary or [])) if x]
    fig2_bracket(out, summaries or None)
    fig3_add(out)
    fig4_tvprofile(out, Path(args.profiles_dir) if args.profiles_dir else None)
    fig5_twofc(out, _load_json(args.twofc_report))
    print(f"\nAll figures written to {out}/")


if __name__ == "__main__":
    main()

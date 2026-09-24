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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Every profile reader in the repo shares one predicate so that a new sidecar
# cannot reintroduce the double-count/KeyError class of bug. The four report
# scripts adopted it when it was written; these two figure readers were missed.
# They survive today only because .restoration.json / .ratio_trace.json happen
# to carry no "estimate" key, so the `continue` guard below skips them -- safe
# by accident, not by design. That accident expires the moment fig4 is pointed
# at replay-ratio-matched-m3-v2 for the ratio-matched overlay, which is a run
# that DOES write sidecars beside every profile.
from kvcache.replay import is_profile_file  # noqa: E402

# ---- headline constants, loaded from the number catalogue -------------------
# These were hardcoded until 2026-07-27, and that was the last surface in the
# repo where a paper number lived as a literal. It is exactly the surface that
# Amendment K's estimand correction could not reach: the summaries were
# regenerated, the .tex was regenerated, and the figures went on drawing the
# old values with no gate able to notice. So the figures now read the same
# catalogue the number audit gates the .tex against -- one source, two
# consumers, and a regenerated artifact moves both or neither.
#
# There is deliberately NO literal fallback. If the catalogue is missing or
# lacks a key, this raises: a figure drawn from stale constants is worse than
# no figure, because it looks finished.
#
# Display convention (pinned 2026-07-07): every 3-decimal label is f"{v:.3f}" of
# the artifact value. Loading the floats directly is what makes that automatic --
# hand-rounding exact ties (99/400 = 0.2475) once made Fig 1 disagree with Fig 2
# for the same quantity.
CATALOGUE = PROJECT_ROOT / "paper_numbers.json"


def _catalogue() -> dict[str, float]:
    if not CATALOGUE.exists():
        raise SystemExit(
            f"missing {CATALOGUE}; run scripts/build_paper_numbers.py first. "
            "The figures have no literal fallback by design."
        )
    payload = json.loads(CATALOGUE.read_text(encoding="utf-8"))
    return {e["key"]: float(e["value"]) for e in payload["entries"]}


K = _catalogue()


def _k(key: str) -> float:
    if key not in K:
        raise SystemExit(f"paper_numbers.json has no entry {key!r}")
    return K[key]


# Unit prefixes: the decomposition cell each press is reported from, and the ADD
# artifact its distribution numbers come from.
_CELL = {
    "Random": "diss256.random",
    "StreamingLLM": "diss256.streamingllm",
    "H2O-style": "h2o256.h2o",
    "R-KV": "rkv256.rkv",
}
_ADD = {
    "Random": "add256.random",
    "StreamingLLM": "add256.streamingllm",
    "H2O-style": "add256.h2o",
    "R-KV": "add256.rkv",
}
_B2048 = {
    "Random": "b2048.random",
    "StreamingLLM": "b2048.streamingllm",
    "H2O-style": "b2048h2o.h2o",
}
_AIME_PRESS = {"Random": "random", "R-KV": "rkv", "StreamingLLM": "streamingllm"}

# The bank accuracy (2000 rows) is the headline reference quoted in section 3.1;
# the cell reference (400 matched conditions) is the comparator every change is
# measured against. Amendment K's whole point is that these are different
# numbers with different jobs, so both are named rather than one standing in.
FC_BANK = _k("diss256.fc_bank")
FC = _k("diss256.random.fc_cell")


def _cell(name: str) -> dict:
    s, a = _CELL[name], _ADD[name]
    return dict(
        free=_k(f"{s}.free"),
        readout=_k(f"{s}.readout"),
        readout_ci=(_k(f"{s}.readout_ci0"), _k(f"{s}.readout_ci1")),
        reverse=_k(f"{s}.reverse"),
        residual=_k(f"{s}.residual"),
        bracket=(_k(f"{s}.bracket0"), _k(f"{s}.bracket1")),
        fwd_ci=(_k(f"{s}.residual_ci0"), _k(f"{s}.residual_ci1")),
        rev_ci=(_k(f"{s}.reverse_ci0"), _k(f"{s}.reverse_ci1")),
        trunc=_k(f"{s}.trunc"),
        add=_k(f"{a}.add"),
        within=_k(f"{a}.within"),
        term=_k(f"{a}.term"),
        flip=_k(f"{a}.flip_net"),
        **({"auc_tv": _k(f"{s}.auc_tv")} if f"{s}.auc_tv" in K else {}),
    )


C = {name: _cell(name) for name in _CELL}
C2048 = {name: _k(f"{s}.free") for name, s in _B2048.items()}
# Readout at 2048, paired with C2048's free-run values. Fig 1 shows both
# protocols side by side; free-running accuracy alone is what prior work already
# reported, so the paired series is what makes Fig 1 this paper's figure.
C2048_READOUT = {name: _k(f"{s}.readout") for name, s in _B2048.items()}
PHEN = dict(
    gap=_k("contrast256.mean"),
    gap_ci=(_k("contrast256.ci0"), _k("contrast256.ci1")),
    win=int(_k("contrast256.wins")),
    tie=int(_k("contrast256.ties")),
    loss=int(_k("contrast256.losses")),
    n_tot=int(_k("contrast256.n_problems")),
)
TVWIN = dict(
    before=_k("tv_diss256.before"),
    first50=_k("tv_diss256.first50"),
    later=_k("tv_diss256.later"),
    onset=_k("tv_diss256.median_onset"),
)
TWOFC = dict(
    mean=_k("twofc.mean_add"),
    median=_k("twofc.median_add"),
    p=_k("twofc.mean_perm_p"),
    frac_sig=_k("twofc.frac_sig"),
    n=250,
)

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
    # In SUMMARY mode the second panel used to be axis("off") plus a text box,
    # spending ~38% of the figure on three lines of text. Only build it when
    # there is a histogram to put in it.
    if per_problem:
        fig, (axb, axh) = plt.subplots(
            1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.65, 1]}
        )
    else:
        fig, axb = plt.subplots(1, 1, figsize=(7.6, 3.1))
        axh = None
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
    # Readout for the same cells, in the same order; None for full cache, which
    # IS the reference and has no readout condition to pair against.
    readouts = [
        None,
        C["Random"]["readout"],
        C["StreamingLLM"]["readout"],
        C["H2O-style"]["readout"],
        C["R-KV"]["readout"],
        C2048_READOUT["Random"],
        C2048_READOUT["StreamingLLM"],
        C2048_READOUT["H2O-style"],
    ]
    colors = [GREY, BLUE, ORANGE, PURPLE, SKY, BLUE_LT, ORANGE_LT, PURPLE_LT]
    # Gap between the budget groups: secondary encoding on top of the tint scheme
    # (and it visually separates SKY from BLUE_LT, the closest validated pair).
    xs = [0, 1, 2, 3, 4, 5.55, 6.55, 7.55]
    bw = 0.34
    # Headroom band, top to bottom: group headers, then the legend and the
    # contrast box side by side, then the tallest bar (0.903). Viewing the first
    # render is what showed both boxes striking the headers.
    HDR_Y = 1.045 if per_problem else 1.27
    # Value labels get a white backing box: the Acc_FC dashline runs at 0.885 and
    # struck through Random@2048's "0.850". Same fix as the protocol schematic's
    # in-bar labels -- back the text, don't move the line.
    LBL = dict(bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for x, v, r, c in zip(xs, vals, readouts, colors, strict=True):
        if r is None:
            axb.bar([x], [v], color=c, width=bw * 1.5, edgecolor="white", linewidth=0.8)
            axb.text(
                x, v + 0.018, f"{v:.3f}", ha="center", fontsize=6.8, color=INK, **LBL
            )
            continue
        # Free-run solid, readout hatched -- hatch means "the readout protocol"
        # everywhere in this paper (fig0's texture-not-hue role, and fig6).
        axb.bar([x - bw / 2], [v], color=c, width=bw, edgecolor="white", linewidth=0.8)
        axb.bar(
            [x + bw / 2],
            [r],
            color=c,
            width=bw,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.45,
            hatch="///",
        )
        axb.text(
            x - bw / 2,
            v + 0.018,
            f"{v:.3f}",
            ha="center",
            fontsize=6.8,
            color=INK,
            **LBL,
        )
        axb.text(
            x + bw / 2,
            r + 0.018,
            f"{r:.3f}",
            ha="center",
            fontsize=6.8,
            color=INK,
            **LBL,
        )
    axb.axhline(FC, color=GREY, lw=0.8, ls=(0, (2, 3)), zorder=0)
    # Budget-group headers carry the realized compression ratios (pre-registered
    # Setup numbers), so the dose structure is legible from the figure alone.
    axb.text(
        2.5,
        HDR_Y,
        "budget 256  (3.8% of mean trace)",
        ha="center",
        fontsize=7.2,
        color=INK2,
        clip_on=False,
    )
    axb.text(
        6.55,
        HDR_Y,
        "budget 2048  (56.6%)",
        ha="center",
        fontsize=7.2,
        color=INK2,
        clip_on=False,
    )
    axb.set_xticks(xs)
    axb.set_xticklabels(labels, fontsize=7.6)
    axb.set_ylabel("Accuracy (MATH-500)")
    axb.set_ylim(0, 1.0 if per_problem else 1.32)
    axb.set_xlim(-0.6, 8.15)
    _ygrid(axb)
    # Identity is never colour-alone: the two protocols are told apart by fill
    # texture, and the legend names them.
    from matplotlib.patches import Patch

    axb.legend(
        handles=[
            Patch(facecolor=GREY, edgecolor="white", label="Free-running"),
            Patch(
                facecolor=GREY,
                edgecolor="white",
                alpha=0.45,
                hatch="///",
                label="Held-trace readout",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0 if per_problem else 0.925),
        fontsize=6.8,
        frameon=True,
        framealpha=1.0,
        edgecolor=SPINE,
        ncol=2 if per_problem else 1,
        borderpad=0.4,
    )
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
        # No histogram to draw, so the contrast rides in the bar panel's own
        # headroom instead of on a dedicated blank axis. White fill so it reads
        # over the Acc_FC dashline (the established fix for that collision).
        axb.text(
            0.015,
            0.90,
            gaptxt,
            transform=axb.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=SPINE, lw=0.7),
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
        # SUBMISSION_STRATEGY Phase 4, applied at the venue conversion: draw the
        # residual bracket as a shaded band so it is visible GEOMETRY rather than
        # an annotation the reader has to assemble from two numbers. The bracket
        # is [min, max] of the forward and reverse residual estimates, and its
        # excluding zero is the paper's central robustness claim -- a claim that
        # should be seeable before it is read.
        # The band spans the REVERSE and FORWARD readout rungs, which is where
        # the bracket actually lives on this axis: the two estimates are biased
        # in opposite directions, so the truth is bounded between them. An
        # earlier version shaded FC + bracket, which plots a correct pair of
        # numbers at a position that corresponds to nothing on the axis.
        lo, hi = sorted((float(d["reverse"]), float(d["readout"])))
        ax.axvspan(lo, hi, color=MC[m], alpha=0.10, lw=0, zorder=0)
        ci = d.get("readout_ci")
        for y, v in zip(ys, vals, strict=True):
            color = GREY if y == 3 else MC[m]
            ax.plot([0, v], [y, y], color=color, lw=2.0, solid_capstyle="round")
            ax.plot(v, y, "o", color=color, ms=6.5, mec="white", mew=0.8)
            # Clear the label of the interval geometry. The forward-readout rung
            # (y=2) carries a CI whose upper whisker runs PAST the marker, so a
            # label pinned to the marker gets struck through by its own interval
            # -- visible on all four panels before this fix. Push the label past
            # whichever is further right, the mark or the whisker.
            label_x = max(v, ci[1]) if (ci and y == 2) else v
            ax.text(label_x + 0.025, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
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
    # 30 problems x 4 seeds per cell vs the pinned AIME FC bank. Readout is
    # IDENTICAL across all three presses -- same correctness on all 93 defined
    # rows, and a readout change of exactly 0.000 against the matched reference.
    fc=_k("aime.fc_bank"),
    free={m: _k(f"aime.{p}.free") for m, p in _AIME_PRESS.items()},
    readout=_k("aime.random.readout"),
    readout_ci=(_k("aime.random.readout_ci0"), _k("aime.random.readout_ci1")),
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


# ---------------------------------------------------------------- Figure 7
def fig7_ratiomatched(out_dir: Path, curves: dict | None):
    """The ratio-matched control, on axes that cannot assert the horizon reading.

    fig4 plots TV against "decoding step relative to first eviction". At a fixed
    budget the realized compression ratio is a deterministic function of that
    axis, so a single rising curve there asserts a horizon reading visually
    whatever the caption says -- the figure plots the confound as its x-axis.
    This figure is the control, on the paired above-floor window M.3 actually
    used, and it is DRIFT-relative on purpose: each trace is divided by its own
    first-quarter mean before averaging, so 1.0 means "no change across the
    window" and the plotted quantity is the one the verdict reports.

    Levels are deliberately NOT comparable between the two arms -- over this
    window the ratio-matched press has been compressing far harder than the
    fixed-budget press -- and normalising is what keeps a reader from making
    that comparison by eye.
    """
    if not curves:
        print("Figure 7 [SKIPPED: no curve artifact]")
        return
    print("Figure 7 [DATA]")
    presses = [p for p in ("random", "streamingllm") if p in curves["per_press"]]
    label = {"random": "Random", "streamingllm": "StreamingLLM"}
    fig, axes = plt.subplots(1, len(presses), figsize=(6.6, 2.7), sharey=True)
    if len(presses) == 1:
        axes = [axes]
    for ax, press in zip(axes, presses, strict=True):
        entry = curves["per_press"][press]
        hue = MC[label[press]]
        n = curves["n_bins"]
        xs = [(b + 0.5) / n for b in range(n)]
        ax.axhline(1.0, color=INK2, ls=(0, (3, 3)), lw=0.9)
        ax.plot(
            xs,
            entry["fixed"]["mean_curve"],
            color=hue,
            lw=1.6,
            label="fixed budget 256",
        )
        ax.plot(
            xs,
            entry["matched"]["mean_curve"],
            color=hue,
            lw=1.6,
            ls=(0, (5, 2)),
            alpha=0.75,
            label="ratio-matched",
        )
        ax.set_title(
            f"{label[press]}  (n={entry['fixed']['n_traces']} paired traces)",
            fontsize=8.5,
        )
        ax.set_xlabel("position in retained above-floor window")
        _ygrid(ax)
    axes[0].set_ylabel("per-step TV\n(relative to own first quarter)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir, "fig7_ratiomatched")


# ---------------------------------------------------------------- Figure 8
T1_BUDGET_CURVE = "results/diag/meeting3/t1_budget_curve.json"


def _err(mean: float, ci: list[float], sign: float = 1.0) -> list[list[float]]:
    """Asymmetric error bar for a (possibly sign-flipped) mean and its CI."""
    lo, hi = sorted(sign * x for x in ci)
    m = sign * mean
    return [[m - lo], [hi - m]]


def fig8_budget_curve(out_dir: Path):
    """Meeting #3 Task 1: damage against cache size, out to the full cache.

    Three panels, because "divergence/damage" is two different quantities and
    the project's standing rule is that they are never merged:

    (a) accuracy LOST against the matched full-cache reference, at four
        budgets, with the Amendment R.1 paired problem-clustered 95% CI on the
        total change. Plotted as a positive loss so the expected negative
        correlation with cache size reads directly; each point carries its
        realized compression ratio. The full cache is drawn as a hollow anchor
        at zero: that point is zero BY CONSTRUCTION (full cache against
        itself), not a measurement, and is marked as such.
    (b) the same loss split into its readout part and its trajectory residual
        (residual with its problem-clustered CI).
    (c) per-step divergence (AUC-TV). Only budgets 256 and 2048 have replay
        profiles -- the 512 and 1024 cells were generation-only -- so this
        panel has two points and draws no curve between them.

    Every value is read from the committed T1 artifact
    (scripts/report_budget_curve.py); nothing is typed in.
    """
    t1 = json.loads((PROJECT_ROOT / T1_BUDGET_CURVE).read_text(encoding="utf-8"))
    cells: dict[str, dict[int, dict]] = {"random": {}, "streamingllm": {}}
    for c in t1["cells"]:
        cells[c["press"]][c["budget"]] = c
    print("Figure 8 [DATA]")
    budgets = sorted(cells["random"])
    full_x = 8192  # visual anchor for the unbounded full cache, labelled as such
    label = {"random": "Random", "streamingllm": "StreamingLLM"}
    offset = {"random": 0.94, "streamingllm": 1.06}  # dodge overlapping CI bars
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    ax = axes[0]
    for press, by_b in cells.items():
        hue = MC[label[press]]
        xs = [b * offset[press] for b in budgets]
        tot = [by_b[b]["paired"]["total_change"] for b in budgets]
        loss = [-t["mean"] for t in tot]
        yerr = [[], []]
        for t in tot:
            e = _err(t["mean"], t["ci95"], sign=-1.0)
            yerr[0] += e[0]
            yerr[1] += e[1]
        ax.errorbar(
            xs,
            loss,
            yerr=yerr,
            fmt="o-",
            color=hue,
            lw=1.6,
            ms=5,
            capsize=2.5,
            elinewidth=0.9,
            label=label[press],
        )
        ax.plot([xs[-1], full_x], [loss[-1], 0.0], ls=(0, (2, 2)), color=hue, lw=1)
        for x, y, b in zip(xs, loss, budgets, strict=True):
            ax.annotate(
                f"{by_b[b]['realized_ratio_pct']:.0f}%",
                (x, y),
                (6 if press == "streamingllm" else -6, 4),
                textcoords="offset points",
                fontsize=6,
                color=hue,
                ha="left" if press == "streamingllm" else "right",
            )
    ax.plot(full_x, 0.0, "o", mfc="white", mec=INK2, ms=6)
    ax.annotate(
        "0 by\nconstruction",
        (full_x, 0.0),
        (full_x * 0.62, 0.14),
        fontsize=7,
        color=INK2,
        ha="center",
    )
    ax.set_ylabel("accuracy lost vs. full cache")
    ax.set_title("(a) total damage (paired 95% CI; % = realized ratio)", fontsize=8)
    ax.set_ylim(-0.02, 0.8)
    # The Random 256 ratio label sits left of its point; without this padding
    # it was clipped against the y axis (seen in the first render).
    ax.set_xlim(150, full_x * 1.3)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    for press, by_b in cells.items():
        hue = MC[label[press]]
        xs = [b * offset[press] for b in budgets]
        resid = [-by_b[b]["residual"] for b in budgets]
        yerr = [[], []]
        for b in budgets:
            e = _err(by_b[b]["residual"], by_b[b]["residual_ci95"], sign=-1.0)
            yerr[0] += e[0]
            yerr[1] += e[1]
        ro = [-by_b[b]["readout_change"] for b in budgets]
        ax.errorbar(
            xs,
            resid,
            yerr=yerr,
            fmt="o-",
            color=hue,
            lw=1.6,
            ms=5,
            capsize=2.5,
            elinewidth=0.9,
            label=f"{label[press]}: trajectory residual",
        )
        ax.plot(
            xs,
            ro,
            "s--",
            color=hue,
            lw=1.1,
            ms=4,
            alpha=0.7,
            label=f"{label[press]}: readout loss",
        )
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title("(b) readout closes, trajectory persists", fontsize=9)
    # Every series is high at small budgets and the labels are long, so any
    # in-data legend position overlapped the StreamingLLM residual at 256/512
    # (seen in two renders). Headroom above the data max gives the legend a
    # region no series enters. A two-column legend was tried and spilled
    # outside the axes over the y tick labels, so it stays one column.
    ax.set_ylim(-0.01, 0.66)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=1)

    ax = axes[2]
    for press in ("random", "streamingllm"):
        hue = MC[label[press]]
        pts = [(b, cells[press][b]["auc_tv"]) for b in (256, 2048)]
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "o",
            color=hue,
            ms=6,
            label=label[press],
        )
    ax.set_title("(c) divergence (2 budgets measured)", fontsize=9)
    ax.set_ylabel("mean per-step TV (AUC)")
    ax.legend(frameon=False, fontsize=7.5)

    for i, ax in enumerate(axes):
        ax.set_xscale("log", base=2)
        ticks = budgets + ([full_x] if i == 0 else [])
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in budgets] + (["full"] if i == 0 else []))
        ax.minorticks_off()
        ax.set_xlabel("cache budget (entries)")
        _ygrid(ax)
    fig.tight_layout()
    _save(fig, out_dir, "fig8_budget_curve")


# ---------------------------------------------------------------- Figure 9
Q2_SURFACE = "results/diag/amendment-q/onset_surface_q2.json"
Q2_GRID = "results/diag/meeting3/q2_onset_grid.json"


def fig9_onset_grid(out_dir: Path):
    """Amendment Q.2: the onset-staggered exposure grid, with its audit.

    Rows are budgets (256, 2048); every panel holds budget AND read position
    fixed, so the realized ratio is matched within a cell.

    (a) window-mean per-step TV for each onset arm at each read position, with
        problem-clustered 95% CIs and the problem/trace counts. Between-problem
        spread dwarfs the arm differences, which is why the read is a paired
        within-trace slope and not a comparison of these means.
    (b) the pre-registered quantity: paired slope of TV on NOMINAL exposure
        (full panel = the verdict; balanced fixed-population panel =
        diagnostic), plus the slope on ACTUAL exposure (diagnostic), per 1000
        steps with 95% CIs.
    (c) occupancy audit: nominal exposure t - t0 against actual exposure
        t - (first step with TV > 0). Enabling the press is not eviction: at
        B = 2048 the cache is under budget until decode step ~1976, so the
        t0 = 0 and t0 = 512 arms have the SAME actual exposure.

    Every value is read from committed artifacts; nothing is typed in.
    """
    surf = json.loads((PROJECT_ROOT / Q2_SURFACE).read_text(encoding="utf-8"))
    grid = json.loads((PROJECT_ROOT / Q2_GRID).read_text(encoding="utf-8"))
    print("Figure 9 [DATA]")
    arm_style = {
        "0": (INK, "o", "t0 = 0"),
        "512": (INK2, "s", "t0 = 512"),
        "2048": (GREY, "^", "t0 = 2048"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2))
    for row, budget in enumerate((256, 2048)):
        cells = {k: c for k, c in grid["full_panel"].items() if c["budget"] == budget}
        names = sorted(cells, key=lambda k: cells[k]["position"])
        positions = [cells[k]["position"] for k in names]
        idx = range(len(names))

        ax = axes[row, 0]
        for j, (t0, (hue, mk, lab)) in enumerate(arm_style.items()):
            xs = [i + (j - 1) * 0.18 for i in idx]
            ms = [cells[k]["arms"][t0]["window_tv"] for k in names]
            yerr = [
                [m["mean"] - m["ci95"][0] for m in ms],
                [m["ci95"][1] - m["mean"] for m in ms],
            ]
            ax.errorbar(
                xs,
                [m["mean"] for m in ms],
                yerr=yerr,
                fmt=mk,
                color=hue,
                ms=5,
                capsize=2.5,
                elinewidth=0.9,
                label=lab,
            )
        ax.set_xticks(list(idx))
        ax.set_xticklabels(
            [
                f"t={cells[k]['position']}\nr={cells[k]['realized_ratio']:.2f}\n"
                f"{cells[k]['n_problems']} pids / {cells[k]['n_traces']} tr"
                for k in names
            ],
            fontsize=7,
        )
        ax.set_ylabel(f"B = {budget}\nwindow-mean per-step TV")
        if row == 0:
            ax.set_title("(a) TV by onset arm (95% CI)", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=3)
        top = max(
            cells[k]["arms"][t0]["window_tv"]["ci95"][1]
            for k in names
            for t0 in arm_style
        )
        ax.set_ylim(0, top * 1.12)

        ax = axes[row, 1]
        series = [
            ("nominal, full panel (verdict)", INK, "o", "white", None),
            ("nominal, balanced (diag.)", INK2, "o", "none", "balanced"),
            ("actual exposure (diag.)", GREY, "D", None, "actual"),
        ]
        for j, (lab, hue, mk, mfc, src) in enumerate(series):
            xs, ys, lo, hi = [], [], [], []
            for i, k in enumerate(names):
                if src is None:
                    st = surf["per_cell"][k]["slope"]
                elif src == "balanced":
                    st = surf["balanced_panel"]["per_cell"].get(k, {}).get("slope")
                else:
                    st = cells[k]["actual_exposure_slope"]
                if st is None:
                    continue
                xs.append(i + (j - 1) * 0.2)
                ys.append(st["mean"] * 1e3)
                lo.append((st["mean"] - st["ci95"][0]) * 1e3)
                hi.append((st["ci95"][1] - st["mean"]) * 1e3)
            ax.errorbar(
                xs,
                ys,
                yerr=[lo, hi],
                fmt=mk,
                color=hue,
                mfc=hue if mfc is None or mfc == "white" else mfc,
                ms=5,
                capsize=2.5,
                elinewidth=0.9,
                label=lab,
            )
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(list(idx))
        ax.set_xticklabels([f"t={p}" for p in positions], fontsize=7.5)
        ax.set_ylabel("dTV / d(exposure) per 1k steps")
        if row == 0:
            ax.set_title(
                f"(b) paired exposure slope; verdict: {surf['verdict'].split()[0]}",
                fontsize=9,
            )
            ax.legend(frameon=False, fontsize=6.5, loc="upper right")

        ax = axes[row, 2]
        for t0, (hue, mk, lab) in arm_style.items():
            nom = [cells[k]["arms"][t0]["nominal_exposure"] for k in names]
            act = [cells[k]["arms"][t0]["actual_exposure"]["mean"] for k in names]
            ax.plot(nom, act, mk, color=hue, ms=5, label=lab)
        lim = max(positions) * 1.05
        ax.plot([0, lim], [0, lim], ls=(0, (2, 2)), color=GREY, lw=0.8)
        ax.annotate(
            "actual = nominal",
            (lim * 0.62, lim * 0.62),
            (lim * 0.34, lim * 0.8),
            fontsize=6.5,
            color=INK2,
        )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("nominal exposure t - t0 (steps)")
        ax.set_ylabel("actual exposure (steps)")
        if row == 0:
            ax.set_title("(c) occupancy audit: nominal vs actual", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        for a_ in axes[row]:
            _ygrid(a_)
    fig.tight_layout()
    _save(fig, out_dir, "fig9_onset_grid")


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
            if not is_profile_file(f):
                continue  # sidecar (.restoration.json / .ratio_trace.json)
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
        if not is_profile_file(f):
            continue  # sidecar (.restoration.json / .ratio_trace.json)
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
    ap.add_argument(
        "--m3-curves",
        default="results/diag/amendment-p/m3_drift_curves.json",
        help="M.3 drift-normalized curve artifact (Fig 7)",
    )
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
    fig7_ratiomatched(out, _load_json(args.m3_curves))
    fig8_budget_curve(out)
    fig9_onset_grid(out)
    print(f"\nAll figures written to {out}/")


if __name__ == "__main__":
    main()

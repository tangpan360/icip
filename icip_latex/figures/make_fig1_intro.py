"""
Figure 1 (Introduction): compact, single-column, two-panel conceptual diagram.

Output:
  - fig1_intro.png

Design goals:
  - Single-column friendly (about 8.6cm width)
  - Two small panels side-by-side
  - English labels only
  - "Premium" pastel blocks + clean typography (Times-like)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "fig1_intro.png"


def _pick_times_like_font() -> str:
    # Try Times New Roman first; fall back to common Times clones.
    candidates = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Nimbus Roman No9 L",
        "TeX Gyre Termes",
        "DejaVu Serif",
    ]
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return "DejaVu Serif"


def _style():
    font = _pick_times_like_font()
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )


def _rounded_box(ax, xy, wh, text, fc, ec, fontsize=8.8, lw=1.0, alpha=1.0, pad=0.02):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size=0.04",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def _arrow(ax, p0, p1, color="#444444", lw=1.1, style="-|>", mutation_scale=10, alpha=1.0, ls="-"):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        alpha=alpha,
        linestyle=ls,
    )
    ax.add_patch(arr)
    return arr


def _noise_badge(ax, center, r=0.04, fc="#FDE8E8", ec="#D66A6A", text="!"):
    cx, cy = center
    c = Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(c)
    ax.text(cx, cy - 0.002, text, ha="center", va="center", fontsize=9, color="#8B2F2F")


def draw_panel_naive(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Palette (soft, conference-style)
    c_in = "#EEF2FF"      # lavender
    c_fuse = "#FFEFD5"    # warm sand
    c_pred = "#E8F7EE"    # mint
    edge = "#2F2F2F"
    muted = "#6B7280"

    ax.text(0.02, 0.95, "Naïve Fusion", ha="left", va="top", fontsize=10, weight="bold")
    ax.text(0.02, 0.89, "no reliability control", ha="left", va="top", fontsize=8.3, color=muted)

    # Inputs
    _rounded_box(ax, (0.06, 0.64), (0.26, 0.18), "Text\n(T)", fc=c_in, ec=edge, fontsize=9)
    _rounded_box(ax, (0.06, 0.40), (0.26, 0.18), "Audio\n(A)", fc=c_in, ec=edge, fontsize=9)
    _rounded_box(ax, (0.06, 0.16), (0.26, 0.18), "Vision\n(V)", fc=c_in, ec=edge, fontsize=9)

    # Fusion + prediction
    _rounded_box(ax, (0.42, 0.40), (0.22, 0.24), "Fusion", fc=c_fuse, ec=edge, fontsize=9)
    _rounded_box(ax, (0.72, 0.42), (0.22, 0.20), "Prediction", fc=c_pred, ec=edge, fontsize=9)

    # Arrows into fusion
    _arrow(ax, (0.32, 0.73), (0.42, 0.56), color="#374151")
    _arrow(ax, (0.32, 0.49), (0.42, 0.52), color="#374151")
    _arrow(ax, (0.32, 0.25), (0.42, 0.48), color="#374151")
    _arrow(ax, (0.64, 0.52), (0.72, 0.52), color="#374151")

    # Noise/Conflict hints on A/V
    _noise_badge(ax, (0.30, 0.56))
    _noise_badge(ax, (0.30, 0.32))
    ax.text(0.35, 0.28, "noise / conflict\nmay leak", ha="left", va="center", fontsize=8.0, color=muted)


def draw_panel_coregate(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    edge = "#2F2F2F"
    muted = "#6B7280"

    # Pastel blocks
    c_in = "#EEF2FF"
    c_head = "#EAF4FF"     # light blue
    c_decomp = "#F3F4F6"   # cool gray
    c_gate = "#FFF1F2"     # rose
    c_out = "#E8F7EE"

    ax.text(0.02, 0.95, "CoReGate", ha="left", va="top", fontsize=10, weight="bold")
    ax.text(0.02, 0.89, "counterfactual deltas + gating", ha="left", va="top", fontsize=8.3, color=muted)

    # Inputs
    _rounded_box(ax, (0.06, 0.64), (0.26, 0.18), "Text\n(T)", fc=c_in, ec=edge, fontsize=9)
    _rounded_box(ax, (0.06, 0.40), (0.26, 0.18), "Audio\n(A)", fc=c_in, ec=edge, fontsize=9)
    _rounded_box(ax, (0.06, 0.16), (0.26, 0.18), "Vision\n(V)", fc=c_in, ec=edge, fontsize=9)

    # Heads (compact)
    _rounded_box(ax, (0.40, 0.62), (0.22, 0.16), "Head T", fc=c_head, ec=edge, fontsize=9)
    _rounded_box(ax, (0.40, 0.42), (0.22, 0.16), "Head TA", fc=c_head, ec=edge, fontsize=9)
    _rounded_box(ax, (0.40, 0.22), (0.22, 0.16), "Head TV", fc=c_head, ec=edge, fontsize=9)

    # Contribution decomposition
    _rounded_box(ax, (0.66, 0.46), (0.28, 0.18), r"$\Delta_A,\ \Delta_V$", fc=c_decomp, ec=edge, fontsize=10)

    # Gates
    _rounded_box(ax, (0.66, 0.26), (0.28, 0.14), r"Gates $r_A,\ r_V$", fc=c_gate, ec=edge, fontsize=9.2)

    # Output
    _rounded_box(ax, (0.72, 0.70), (0.22, 0.16), r"$\hat{y}$", fc=c_out, ec=edge, fontsize=11)

    # Arrows: inputs to heads
    _arrow(ax, (0.32, 0.73), (0.40, 0.70), color="#374151")
    _arrow(ax, (0.32, 0.49), (0.40, 0.50), color="#374151")
    _arrow(ax, (0.32, 0.25), (0.40, 0.30), color="#374151")

    # Heads to decomposition
    _arrow(ax, (0.62, 0.70), (0.66, 0.55), color="#374151")
    _arrow(ax, (0.62, 0.50), (0.66, 0.55), color="#374151", alpha=0.9)
    _arrow(ax, (0.62, 0.30), (0.66, 0.55), color="#374151", alpha=0.9)

    # Decomp to gates and to output
    _arrow(ax, (0.80, 0.46), (0.80, 0.40), color="#374151")
    _arrow(ax, (0.80, 0.40), (0.80, 0.70), color="#374151")

    # Visual hint: "suppress unreliable"
    _noise_badge(ax, (0.30, 0.56))
    _noise_badge(ax, (0.30, 0.32))
    ax.text(0.66, 0.18, "suppress\nunreliable deltas", ha="left", va="center", fontsize=8.0, color=muted)


def main():
    _style()

    # Single-column friendly size.
    # ICIP column width ~ 3.39 in. Use a bit narrower and leave margins for LaTeX.
    width_in = 3.35
    height_in = 1.55

    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in))
    plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.05, wspace=0.08)

    draw_panel_naive(axes[0])
    draw_panel_coregate(axes[1])

    # Panel labels (a)(b) at bottom, compact.
    fig.text(0.26, 0.01, "(a)", ha="center", va="bottom", fontsize=10)
    fig.text(0.74, 0.01, "(b)", ha="center", va="bottom", fontsize=10)

    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()


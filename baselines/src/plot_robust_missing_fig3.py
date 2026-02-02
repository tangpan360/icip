"""
Plot Fig.3: Robustness under missing modalities (gating ON vs OFF).

Designed for a paper-quality figure:
- Colorblind-friendly palette
- Vector PDF + high-res PNG
- Two panels: MOSI / MOSEI

Expected inputs (already produced by paper_eval_cenet_ccf.py):
  ../outputs/paper_runs/cenet_ccf/{mosei,mosi}/abl_full_seed1111/results/robustness/
    *_{missing=a|v|av}_disable_gating={0,1}.json

Run (from baselines/MMSA/src):
  conda run -n tp-icip python plot_robust_missing_fig3.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_missing_results(robust_dir: Path) -> dict[tuple[str, int], dict[str, float]]:
    """
    Returns:
      {(missing, disable_gating): {"MAE":..., "Corr":..., "Acc7":...}}
    """
    out: dict[tuple[str, int], dict[str, float]] = {}
    for missing in ("a", "v", "av"):
        for dg in (0, 1):
            p = robust_dir / f"{robust_dir.parent.parent.name}_missing={missing}_disable_gating={dg}.json"
            # Fallback: glob if dataset prefix differs.
            if not p.is_file():
                cand = list(robust_dir.glob(f"*_missing={missing}_disable_gating={dg}.json"))
                if not cand:
                    raise FileNotFoundError(f"Missing robustness json for missing={missing}, disable_gating={dg} in {robust_dir}")
                p = cand[0]
            d = _read_json(p)
            r = d["results"]
            out[(missing, dg)] = {
                "MAE": float(r["MAE"]),
                "Corr": float(r["Corr"]),
                "Acc7": float(r["Mult_acc_7"]),
            }
    return out


def _style_matplotlib():
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            # Times-like typography (paper style). If Times New Roman is not installed,
            # Matplotlib will fall back to the next available serif font.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,  # editable text in illustrator
            "ps.fonttype": 42,
        }
    )


def _plot_panel(ax, data: dict[tuple[str, int], dict[str, float]]):
    # Order: A, V, A+V
    labels = ["A", "V", "A+V"]
    miss_keys = ["a", "v", "av"]

    mae_on = np.array([data[(m, 0)]["MAE"] for m in miss_keys])
    mae_off = np.array([data[(m, 1)]["MAE"] for m in miss_keys])
    delta = mae_off - mae_on

    x = np.arange(len(labels))
    width = 0.34

    # Colorblind-friendly (Okabe-Ito-ish)
    c_on = "#1f77b4"   # deep blue
    c_off = "#ff7f0e"  # orange

    b1 = ax.bar(x - width / 2, mae_on, width, label="Gating ON", color=c_on, edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + width / 2, mae_off, width, label="Gating OFF", color=c_off, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE (lower is better)")

    # Tight y-range for readability
    ymin = float(min(mae_on.min(), mae_off.min()))
    ymax = float(max(mae_on.max(), mae_off.max()))
    pad = max(0.01, 0.25 * (ymax - ymin))
    ax.set_ylim(ymin - 0.25 * pad, ymax + 1.25 * pad)

    # Annotate delta above the OFF bar
    for i, (rect_off, d) in enumerate(zip(b2, delta)):
        ax.text(
            rect_off.get_x() + rect_off.get_width() / 2,
            rect_off.get_height() + 0.02 * (ymax - ymin + 1e-6),
            f"+{d:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )

    # Lighten spines
    for s in ("left", "bottom"):
        ax.spines[s].set_alpha(0.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paper_runs_dir",
        type=str,
        default="../outputs/paper_runs/cenet_ccf",
        help="Path to paper_runs/cenet_ccf (relative to baselines/MMSA/src by default).",
    )
    ap.add_argument(
        "--mosei_run",
        type=str,
        default="mosei/abl_full_seed1111",
        help="Subdir under paper_runs_dir for MOSEI run.",
    )
    ap.add_argument(
        "--mosi_run",
        type=str,
        default="mosi/abl_full_seed1111",
        help="Subdir under paper_runs_dir for MOSI run.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="../../../icip_latex/figures",
        help="Output directory for figures (relative to baselines/MMSA/src).",
    )
    ap.add_argument("--basename", type=str, default="fig3_missing_robustness")
    args = ap.parse_args()

    _style_matplotlib()

    paper_runs = Path(args.paper_runs_dir).expanduser().resolve()
    mosei_rob = (paper_runs / args.mosei_run / "results" / "robustness").resolve()
    mosi_rob = (paper_runs / args.mosi_run / "results" / "robustness").resolve()

    mosei_data = _load_missing_results(mosei_rob)
    mosi_data = _load_missing_results(mosi_rob)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=False)
    _plot_panel(axes[0], mosi_data)
    _plot_panel(axes[1], mosei_data)

    # Subfigure labels below each panel
    axes[0].text(0.5, -0.22, "(a) CMU-MOSI", transform=axes[0].transAxes, ha="center", va="top")
    axes[1].text(0.5, -0.22, "(b) CMU-MOSEI", transform=axes[1].transAxes, ha="center", va="top")

    # Shared legend at top-center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.985), frameon=False)

    # Leave space for legend (top) + subfigure labels (bottom)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.28, wspace=0.22)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{args.basename}.pdf"
    png_path = out_dir / f"{args.basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


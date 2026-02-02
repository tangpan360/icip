"""
Paper-friendly runner for CENET_CCF.

Goal: make it easy to reproduce the main results + minimal ablations with a
single copy-paste command (no inline python heredocs).

It calls MMSA.run.MMSA_run() and writes outputs under:
  ../outputs/paper_runs/cenet_ccf/<dataset>/<variant>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_seeds(s: str) -> list[int]:
    seeds = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        seeds.append(int(x))
    if not seeds:
        raise ValueError("--seeds is empty")
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["mosi", "mosei", "sims"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="1111,1112,1113")
    ap.add_argument("--num_workers", type=int, default=8)

    # Variants used in the draft
    ap.add_argument(
        "--variant",
        type=str,
        default="default",
        choices=["default", "ui_on", "blw0", "custom"],
        help="default: tuned MOSEI best; ui_on: enable interaction; blw0: no branch supervision; custom: use explicit args below",
    )

    # Explicit overrides (used when variant=custom, or to further override)
    ap.add_argument("--av_hidden", type=int, default=-1)
    ap.add_argument("--head_dropout", type=float, default=-1.0)
    ap.add_argument("--branch_loss_weight", type=float, default=-1.0)
    ap.add_argument("--use_interaction", type=str, default="", help="true/false (only for variant=custom or override)")
    ap.add_argument("--use_gating", type=str, default="", help="true/false (ablation: disable reliability gating)")
    ap.add_argument(
        "--fusion_mode",
        type=str,
        default="",
        help="ablation: coregate (default) | t_only | a_only | v_only | tav_only",
    )

    ap.add_argument("--outdir", type=str, default="", help="Optional output directory (default under ../outputs/paper_runs/...)")
    args = ap.parse_args()

    dataset = args.dataset.lower()
    seeds = _parse_seeds(args.seeds)

    # Tuned default from stage2 (MOSEI, 3 seeds)
    cfg: dict[str, Any] = dict(
        av_hidden=128,
        head_dropout=0.2,
        branch_loss_weight=0.05,
        use_interaction=False,
    )

    if args.variant == "ui_on":
        cfg["use_interaction"] = True
    elif args.variant == "blw0":
        cfg["branch_loss_weight"] = 0.0
        cfg["use_interaction"] = False
    elif args.variant == "custom":
        # keep defaults but allow explicit override below
        pass

    # Apply explicit overrides if provided
    if args.av_hidden > 0:
        cfg["av_hidden"] = args.av_hidden
    if args.head_dropout >= 0:
        cfg["head_dropout"] = float(args.head_dropout)
    if args.branch_loss_weight >= 0:
        cfg["branch_loss_weight"] = float(args.branch_loss_weight)
    if args.use_interaction.strip():
        v = args.use_interaction.strip().lower()
        if v in ("1", "true", "yes", "y", "t"):
            cfg["use_interaction"] = True
        elif v in ("0", "false", "no", "n", "f"):
            cfg["use_interaction"] = False
        else:
            raise ValueError("--use_interaction must be true/false")

    if args.use_gating.strip():
        v = args.use_gating.strip().lower()
        if v in ("1", "true", "yes", "y", "t"):
            cfg["use_gating"] = True
        elif v in ("0", "false", "no", "n", "f"):
            cfg["use_gating"] = False
        else:
            raise ValueError("--use_gating must be true/false")

    if args.fusion_mode.strip():
        cfg["fusion_mode"] = args.fusion_mode.strip().lower()

    here = Path(__file__).resolve().parent
    default_out = (here.parent / "outputs" / "paper_runs" / "cenet_ccf" / dataset / args.variant).resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else default_out
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_override.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    from MMSA.run import MMSA_run

    MMSA_run(
        model_name="cenet_ccf",
        dataset_name=dataset,
        config=cfg,
        seeds=seeds,
        is_tune=False,
        gpu_ids=[args.gpu] if args.gpu >= 0 else [-1],
        num_workers=args.num_workers,
        verbose_level=1,
        model_save_dir=str(outdir / "models"),
        res_save_dir=str(outdir / "results"),
        log_dir=str(outdir / "logs"),
    )

    print(f"Done. Outputs at: {outdir}")
    print(f"Results CSV: {outdir / 'results' / 'normal' / f'{dataset}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


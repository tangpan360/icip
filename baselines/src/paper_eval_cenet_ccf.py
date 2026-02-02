"""
Paper-friendly evaluator for CENET_CCF robustness tests.

Use case:
- You already trained a run with `paper_run_cenet_ccf.py` and have weights at:
    <run_dir>/models/cenet_ccf-<dataset>.pth
- You want to evaluate the SAME weights under modality missing and/or gating-off,
  without re-training.

It writes a JSON result under:
  <run_dir>/results/robustness/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from easydict import EasyDict as edict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["mosi", "mosei", "sims"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to an existing paper_run directory (contains config_override.json + models/).",
    )
    ap.add_argument(
        "--eval_missing",
        type=str,
        default="none",
        choices=["none", "a", "v", "av", "audio", "vision"],
        help="Apply modality missing ONLY during TEST evaluation.",
    )
    ap.add_argument(
        "--eval_disable_gating",
        action="store_true",
        help="Disable reliability gating ONLY during TEST evaluation (uses the same weights).",
    )
    args = ap.parse_args()

    dataset = args.dataset.lower()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg_path = run_dir / "config_override.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config_override.json: {cfg_path}")

    cfg: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["eval_missing"] = args.eval_missing
    cfg["eval_disable_gating"] = bool(args.eval_disable_gating)

    # Build args from default regression config, then override with cfg.
    from MMSA.config import get_config_regression
    from MMSA.data_loader import MMDataLoader
    from MMSA.models import AMIO
    from MMSA.trains import ATIO
    from MMSA.utils import assign_gpu

    config_file = Path(__file__).resolve().parent / "MMSA" / "config" / "config_regression.json"
    base = get_config_regression("cenet_ccf", dataset, config_file)
    base.update(cfg)
    base["train_mode"] = "regression"
    # Keep consistent with MMSA.run.MMSA_run(): these keys are expected by dataloader.
    base["custom_feature"] = bool(base.get("custom_feature", False))
    base["feature_T"] = base.get("feature_T", "")
    base["feature_A"] = base.get("feature_A", "")
    base["feature_V"] = base.get("feature_V", "")
    base["device"] = assign_gpu([args.gpu]) if args.gpu >= 0 else torch.device("cpu")
    base["cur_seed"] = 1  # for logging format compatibility

    if args.gpu >= 0:
        torch.cuda.set_device(base["device"])

    weights_path = run_dir / "models" / f"cenet_ccf-{dataset}.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    base["model_save_path"] = weights_path
    ed = edict(base)

    dataloader = MMDataLoader(ed, args.num_workers)
    model = AMIO(ed).to(ed["device"])
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.to(ed["device"])

    trainer = ATIO().getTrain(ed)
    results = trainer.do_test(model, dataloader["test"], mode="TEST")

    out_dir = run_dir / "results" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"missing={args.eval_missing}_disable_gating={int(args.eval_disable_gating)}"
    out_path = out_dir / f"{dataset}_{tag}.json"
    out_path.write_text(json.dumps({"tag": tag, "results": results}, indent=2), encoding="utf-8")

    print(f"Done. Robustness results saved to: {out_path}")
    print(f"Key metrics: MAE={results.get('MAE')} Corr={results.get('Corr')} Acc7={results.get('Mult_acc_7')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


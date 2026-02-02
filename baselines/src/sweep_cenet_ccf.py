"""
Grid sweep runner for CENET_CCF (counterfactual contribution fusion).

Why this exists:
- MMSA provides `MMSA_run()` which can run multiple seeds and save results,
  but its "normal/*.csv" format stores (mean, std) tuples as strings.
- For paper-ready tables, we often want a single aggregated CSV with:
    hyperparams + metric_mean + metric_std

This script runs a small, targeted grid over a few knobs that are directly tied
to the CENET_CCF mechanism:
  - branch_loss_weight
  - use_interaction
  - av_hidden
  - head_dropout

Example:
  python sweep_cenet_ccf.py --dataset mosei --gpu 0 --seeds 1111,1112,1113
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


def _parse_tuple_cell(x: str) -> tuple[float, float] | None:
    """
    Parse pandas-saved tuple cell like "(52.31, 1.02)".
    Returns (mean, std) in the *same scale* as the csv stores (usually *100).
    """
    x = (x or "").strip()
    if not x:
        return None
    if not (x.startswith("(") and x.endswith(")")):
        return None
    try:
        v = ast.literal_eval(x)
        if isinstance(v, tuple) and len(v) == 2:
            return float(v[0]), float(v[1])
    except Exception:
        return None
    return None


def _read_last_result_row(csv_path: Path) -> dict[str, Any]:
    """
    Reads the last row from MMSA "normal/{dataset}.csv" and parses (mean,std) cells.
    Returns a dict: {metric: (mean,std)} with metric names as keys.
    """
    import pandas as pd  # local import: keep base deps minimal

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise RuntimeError(f"Empty results csv: {csv_path}")
    row = df.iloc[-1].to_dict()
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k == "Model":
            out[k] = v
            continue
        tup = _parse_tuple_cell(str(v))
        out[k] = tup if tup is not None else v
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mosei", choices=["mosi", "mosei", "sims"])
    parser.add_argument("--gpu", type=int, default=0, help="Single GPU id (used if --gpus not provided).")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids. If set, shard_id maps to one GPU.")
    parser.add_argument("--seeds", type=str, default="1111,1112,1113")
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--scale_back_to_0_1", action="store_true", help="Convert mean/std from *100 to 0~1 where appropriate.")
    parser.add_argument("--max_runs", type=int, default=0, help="If >0, only run the first N grid points.")
    parser.add_argument("--dry_run", action="store_true", help="Only generate run folders/grid files; do not train.")
    parser.add_argument("--num_shards", type=int, default=1, help="Split grid into N disjoint shards (for multi-GPU parallel runs).")
    parser.add_argument("--shard_id", type=int, default=0, help="Which shard to run: 0..num_shards-1.")
    parser.add_argument("--skip_done", action="store_true", help="Skip runs that already have a non-empty normal/{dataset}.csv.")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    model = "cenet_ccf"
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    here = Path(__file__).resolve().parent
    default_outdir = here.parent / "outputs" / "sweeps" / model / dataset
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else default_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must be in [0, num_shards)")

    gpus: list[int] = []
    if args.gpus.strip():
        gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if len(gpus) < args.num_shards:
            raise ValueError(f"--gpus must provide >= num_shards ids (got {len(gpus)} vs {args.num_shards})")
    assigned_gpu = gpus[args.shard_id] if gpus else args.gpu

    # Targeted small grid (paper-friendly and mechanism-aligned).
    grid = {
        "branch_loss_weight": [0.0, 0.05, 0.1, 0.2],
        "use_interaction": [True, False],
        "av_hidden": [64, 128],
        "head_dropout": [0.1, 0.2],
    }

    # Persist the exact sweep plan for reproducibility.
    (outdir / "sweep_grid.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")

    # Import MMSA lazily so running `--help` stays fast.
    MMSA_run = None if args.dry_run else __import__("MMSA.run", fromlist=["MMSA_run"]).MMSA_run

    # Each shard writes its own summary file to avoid race conditions.
    summary_csv = outdir / f"sweep_summary_shard{args.shard_id}_of{args.num_shards}.csv"
    rows: list[dict[str, Any]] = []

    def _iter_grid():
        for blw in grid["branch_loss_weight"]:
            for ui in grid["use_interaction"]:
                for ah in grid["av_hidden"]:
                    for hd in grid["head_dropout"]:
                        yield {
                            "branch_loss_weight": blw,
                            "use_interaction": ui,
                            "av_hidden": ah,
                            "head_dropout": hd,
                        }

    for i, hp in enumerate(_iter_grid(), start=1):
        # Sharding: assign grid point i to shard by round-robin.
        if (i - 1) % args.num_shards != args.shard_id:
            continue
        if args.max_runs and i > args.max_runs:
            break
        tag = f"blw={hp['branch_loss_weight']}_ui={int(hp['use_interaction'])}_ah={hp['av_hidden']}_hd={hp['head_dropout']}"
        run_dir = outdir / tag
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "results").mkdir(parents=True, exist_ok=True)
        (run_dir / "models").mkdir(parents=True, exist_ok=True)

        # Save the actual overrides applied.
        (run_dir / "override.json").write_text(json.dumps(hp, indent=2), encoding="utf-8")

        normal_csv = run_dir / "results" / "normal" / f"{dataset}.csv"

        if args.dry_run:
            print(f"[{i}] shard {args.shard_id}/{args.num_shards} dry-run prepared: {tag}")
            continue

        if args.skip_done and normal_csv.is_file():
            try:
                # quick non-empty check
                if normal_csv.stat().st_size > 0:
                    print(f"[{i}] skip_done: {tag}")
                    continue
            except OSError:
                pass

        # Run training/testing (multi-seed). Results are written by MMSA.
        MMSA_run(
            model_name=model,
            dataset_name=dataset,
            config_file=None,  # use default regression config
            config=hp,  # override only the knobs we care about
            seeds=seeds,
            is_tune=False,
            gpu_ids=[assigned_gpu] if assigned_gpu >= 0 else [-1],
            num_workers=4,
            verbose_level=1,
            model_save_dir=str(run_dir / "models"),
            res_save_dir=str(run_dir / "results"),
            log_dir=str(run_dir / "logs"),
        )

        # Parse the MMSA-written csv row.
        parsed = _read_last_result_row(normal_csv)

        flat: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "run_tag": tag,
            **hp,
            "seeds": ",".join(map(str, seeds)),
            "results_csv": str(normal_csv),
        }
        for k, v in parsed.items():
            if k == "Model":
                continue
            if isinstance(v, tuple) and len(v) == 2:
                mean, std = v
                if args.scale_back_to_0_1:
                    # MMSA stores most metrics *100 in CSV; bring them back.
                    mean, std = mean / 100.0, std / 100.0
                flat[f"{k}_mean"] = mean
                flat[f"{k}_std"] = std
            else:
                flat[k] = v

        rows.append(flat)

        # Incremental write (safe for long sweeps).
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"[{i}] shard {args.shard_id}/{args.num_shards} done on gpu {assigned_gpu}: {tag} -> {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


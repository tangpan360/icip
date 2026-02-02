"""
Stage-2 rerun helper for CENET_CCF sweeps.

Workflow (recommended):
1) Stage-1 coarse sweep (1 seed) across full grid using sweep_cenet_ccf.py with sharding.
2) Run this script to:
   - merge shard summaries
   - select Top-K configs by a target metric (default: MAE_mean, lower is better)
   - rerun only those configs with multiple seeds (e.g., 3 seeds) for paper-ready mean±std

This script supports the same sharding pattern to utilize multiple GPUs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _read_csv_rows(p: Path) -> list[dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    # Handle MMSA tuple strings like:
    #   "(np.float64(52.68), np.float64(0.0))"
    #   "(52.68, 0.0)"
    # We want the first element (mean).
    if "np.float64(" in s:
        nums = re.findall(r"np\.float64\(([-0-9.]+)\)", s)
        if nums:
            try:
                return float(nums[0])
            except Exception:
                return None
    if s.startswith("(") and "," in s and s.endswith(")"):
        # try to extract first numeric token inside parentheses
        m = re.search(r"\(\s*([-0-9.]+)\s*,", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    try:
        return float(s)
    except Exception:
        return None


def _merge_shards(sweep_dir: Path) -> list[dict[str, Any]]:
    shard_files = sorted(sweep_dir.glob("sweep_summary_shard*_of*.csv"))
    if not shard_files:
        raise FileNotFoundError(f"No shard summary csv found under: {sweep_dir}")
    rows: list[dict[str, Any]] = []
    for f in shard_files:
        rows.extend(_read_csv_rows(f))
    # Dedup by run_tag (keep last)
    dedup: dict[str, dict[str, Any]] = {}
    for r in rows:
        tag = r.get("run_tag", "")
        if tag:
            dedup[tag] = r
    return list(dedup.values())


def _select_topk(
    rows: list[dict[str, Any]],
    metric: str,
    topk: int,
    minimize: bool = True,
    tie_breaker: str | None = "Corr_mean",
) -> list[dict[str, Any]]:
    # Backward compatibility:
    # Stage-1 summaries may have columns like "MAE"/"Corr" (tuple strings),
    # not "MAE_mean"/"Corr_mean". If user passes "*_mean", fall back.
    metric_used = metric
    tie_used = tie_breaker
    if metric_used.endswith("_mean"):
        base = metric_used[: -len("_mean")]
        if any(base in r for r in rows) and not any(metric_used in r for r in rows):
            metric_used = base
    if tie_used and tie_used.endswith("_mean"):
        base = tie_used[: -len("_mean")]
        if any(base in r for r in rows) and not any(tie_used in r for r in rows):
            tie_used = base

    scored: list[tuple[float, float, dict[str, Any]]] = []
    for r in rows:
        m = _to_float(r.get(metric_used))
        if m is None:
            continue
        tb = _to_float(r.get(tie_used)) if tie_used else None
        tbv = tb if tb is not None else 0.0
        scored.append((m, tbv, r))
    if not scored:
        raise RuntimeError(f"No rows contain metric '{metric}' (or fallback '{metric_used}')")

    # sort: metric asc/desc, tie-breaker desc (Corr higher better)
    if minimize:
        scored.sort(key=lambda t: (t[0], -t[1]))
    else:
        scored.sort(key=lambda t: (-t[0], -t[1]))
    return [t[2] for t in scored[:topk]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", type=str, default="", help="Directory containing stage-1 shard summaries and run folders.")
    ap.add_argument("--dataset", type=str, default="mosei", choices=["mosi", "mosei", "sims"])
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--metric", type=str, default="MAE_mean")
    ap.add_argument("--maximize", action="store_true", help="If set, higher metric is better (default is minimize).")
    ap.add_argument("--seeds", type=str, default="1111,1112,1113")
    ap.add_argument("--gpu", type=int, default=0, help="Single GPU id (used if --gpus not provided).")
    ap.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids. If set, shard_id maps to one GPU.")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--skip_done", action="store_true", help="Skip configs already rerun (summary exists).")
    args = ap.parse_args()

    dataset = args.dataset.lower()
    model = "cenet_ccf"
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    here = Path(__file__).resolve().parent
    default_sweep_dir = here.parent / "outputs" / "sweeps" / model / dataset
    sweep_dir = Path(args.sweep_dir).expanduser().resolve() if args.sweep_dir else default_sweep_dir
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"sweep_dir not found: {sweep_dir}")

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

    # Merge and select top-k based on stage-1 summaries.
    all_rows = _merge_shards(sweep_dir)
    selected = _select_topk(
        all_rows,
        metric=args.metric,
        topk=args.topk,
        minimize=not args.maximize,
        tie_breaker="Corr_mean",
    )

    # Stage-2 output
    stage2_dir = sweep_dir / f"stage2_topk{args.topk}_{args.metric}_{'max' if args.maximize else 'min'}"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    (stage2_dir / "selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    # Import MMSA lazily
    from MMSA.run import MMSA_run

    # Run only the selected list, but shard the *selected index* across GPUs.
    summary_csv = stage2_dir / f"stage2_summary_shard{args.shard_id}_of{args.num_shards}.csv"
    rows_out: list[dict[str, Any]] = []

    for j, r in enumerate(selected, start=1):
        if (j - 1) % args.num_shards != args.shard_id:
            continue
        tag = r.get("run_tag")
        if not tag:
            continue

        src_run_dir = sweep_dir / tag
        override_path = src_run_dir / "override.json"
        if not override_path.is_file():
            raise FileNotFoundError(f"Missing override.json for {tag}: {override_path}")
        hp = json.loads(override_path.read_text(encoding="utf-8"))

        run_dir = stage2_dir / tag
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "results").mkdir(parents=True, exist_ok=True)
        (run_dir / "models").mkdir(parents=True, exist_ok=True)
        (run_dir / "override.json").write_text(json.dumps(hp, indent=2), encoding="utf-8")

        normal_csv = run_dir / "results" / "normal" / f"{dataset}.csv"
        if args.skip_done and normal_csv.is_file() and normal_csv.stat().st_size > 0:
            print(f"[top{j}] skip_done: {tag}")
            continue

        MMSA_run(
            model_name=model,
            dataset_name=dataset,
            config_file=None,
            config=hp,
            seeds=seeds,
            is_tune=False,
            gpu_ids=[assigned_gpu] if assigned_gpu >= 0 else [-1],
            num_workers=4,
            verbose_level=1,
            model_save_dir=str(run_dir / "models"),
            res_save_dir=str(run_dir / "results"),
            log_dir=str(run_dir / "logs"),
        )

        # Parse the last row in normal csv (same parsing logic as sweep script)
        import pandas as pd

        df = pd.read_csv(normal_csv)
        last = df.iloc[-1].to_dict()
        flat: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "run_tag": tag,
            "rank_in_topk": j,
            **hp,
            "seeds": ",".join(map(str, seeds)),
            "results_csv": str(normal_csv),
        }
        # Keep the raw tuple-like strings; paper users can parse, but we also extract mean/std when possible.
        for k, v in last.items():
            if k == "Model":
                continue
            flat[k] = v

        rows_out.append(flat)

        # Incremental write
        fieldnames = sorted({k for rr in rows_out for k in rr.keys()})
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rr in rows_out:
                w.writerow(rr)

        print(f"[top{j}] shard {args.shard_id}/{args.num_shards} done on gpu {assigned_gpu}: {tag} -> {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


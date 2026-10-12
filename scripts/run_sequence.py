#!/usr/bin/env python
"""Phase 7: the sequence-model run on the widened, well-level split.

Two things happen here that the earlier phases did not do.

1. The split is *wide*: every real well takes part, and the honest number comes from
   leave-wells-out cross validation rather than one lucky fold (``wide_split.py``).
2. The model is a *sequence* model: it reads consecutive timesteps, not a bag of window
   statistics (``sequence.py`` + ``sequence_model.py``).

The arrays are built once and reused for every fold. That is safe because the only
per-well transform is an unsupervised median/IQR normalisation fitted on each instance's
own series, which involves no labels and no cross-well statistics.

Usage::

    python scripts/run_sequence.py --data ../amd-act3/data/3W/dataset \
        --downsample 60 --steps 48 --stride 30 --horizon 21600 --folds 5 \
        --max-sim 600 --out reports/sequence_wide.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from wellsentinel.data import discover  # noqa: E402
from wellsentinel.sequence import build_sequences, channel_names  # noqa: E402
from wellsentinel.sequence_model import (  # noqa: E402
    TrainConfig,
    lead_time_seconds,
    predict,
    score,
    train_sequence_model,
)
from wellsentinel.wide_split import (  # noqa: E402
    assert_disjoint,
    class_well_coverage,
    grouped_folds,
    split_by_well,
    split_summary,
)


def cache_path(cache_dir: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"seq_{digest}.npz"


def load_or_build(refs, channels, args, cache_dir: Path):
    """Build the sequence arrays once, or reuse the cached copy for the same config."""
    key = (
        f"v1|ds={args.downsample}|steps={args.steps}|stride={args.stride}"
        f"|hor={args.horizon}|sim={args.max_sim}|real={args.max_real}|n={len(refs)}"
    )
    path = cache_path(cache_dir, key)
    if path.exists():
        print(f"Reusing cached arrays {path.name}")
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print(f"Building sequences for {len(refs)} instances (this reads Parquet)...")
    seq = build_sequences(
        refs,
        channels=channels,
        downsample_seconds=args.downsample,
        steps=args.steps,
        stride=args.stride,
        horizon_seconds=args.horizon,
    )
    payload = {
        "x": seq.x,
        "mask": seq.mask.astype("uint8"),
        "target": seq.target.astype("int8"),
        "seconds_to_onset": seq.seconds_to_onset,
        "end_time": seq.end_time,
        "well_id": seq.well_id.astype(object),
        "instance": seq.instance.astype(object),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"Cached arrays to {path.name}  x={seq.x.shape}")
    return payload


def fold_membership(refs, k: int) -> dict[str, int]:
    """well id -> fold index, from the same rarest-first deal used by the split."""
    membership: dict[str, int] = {}
    for fold, (_, test) in enumerate(grouped_folds(refs, k=k)):
        for ref in test:
            membership[ref.well_id] = fold
    return membership


def real_well_set(refs) -> set[str]:
    return {r.well_id for r in refs if r.is_real}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequence model on the wide split.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--downsample", type=int, default=60, help="Seconds per timestep.")
    parser.add_argument("--steps", type=int, default=48, help="Timesteps per sequence.")
    parser.add_argument("--stride", type=int, default=30, help="Stride in timesteps.")
    parser.add_argument("--horizon", type=int, default=21600, help="Horizon seconds (6 h).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-sim", type=int, default=600, help="Cap simulated instances.")
    parser.add_argument("--max-real", type=int, default=0, help="Cap real instances (0 = all).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--cache-dir", default=str(Path(__file__).resolve().parents[1] / "cache"))
    parser.add_argument("--out", default="reports/sequence_wide.json")
    args = parser.parse_args()

    all_refs = discover(args.data)
    real = [r for r in all_refs if r.is_real]
    if args.max_real:
        real = real[: args.max_real]
    simulated = [r for r in all_refs if r.source == "simulated"][: args.max_sim]
    drawn = [r for r in all_refs if r.source == "drawn"]
    refs = real + simulated + drawn
    print(f"Real {len(real)}  simulated {len(simulated)}  drawn {len(drawn)}  total {len(refs)}")

    channels = channel_names()
    data = load_or_build(refs, channels, args, Path(args.cache_dir))

    x, mask = data["x"], data["mask"].astype("float32")
    target = data["target"].astype("int64")
    sto = data["seconds_to_onset"]
    well_id = data["well_id"]
    instance = data["instance"]
    print(f"Sequences {len(target)}  positives {int(target.sum())}  channels {len(channels)}")
    if len(target) == 0:
        print("No sequences built; check the data path and window sizes.")
        return 1

    # Reference view of the *single* wide split, so the report shows the split shape.
    wide = split_by_well(refs)
    assert_disjoint(wide)
    print("\nSingle wide split (60/20/20 by well):")
    print(split_summary(wide).to_string(index=False))
    coverage = class_well_coverage(wide, refs)
    print("\nClass -> wells per split:")
    print(coverage.to_string(index=False))

    membership = fold_membership(refs, args.folds)
    is_real = np.array([w in real_well_set(refs) for w in well_id])
    folds = np.array([membership.get(w, -1) for w in well_id])

    pooled_prob = np.full(len(target), np.nan)
    per_fold: list[dict[str, object]] = []
    rng = np.random.default_rng(42)

    for fold in range(args.folds):
        test_idx = np.where(is_real & (folds == fold))[0]
        train_pool = np.where((~is_real) | (folds != fold))[0]
        if len(test_idx) == 0 or len(train_pool) == 0:
            print(f"fold {fold}: empty, skipped")
            continue

        # Deterministic validation slice out of the training wells, for early stopping only.
        train_wells = np.unique(well_id[train_pool])
        val_wells = set(rng.choice(train_wells, size=max(1, len(train_wells) // 8), replace=False).tolist())
        val_idx = np.array([i for i in train_pool if well_id[i] in val_wells])
        fit_idx = np.array([i for i in train_pool if well_id[i] not in val_wells])

        config = TrainConfig(epochs=args.epochs, hidden=args.hidden, seed=42, device="cpu")
        result = train_sequence_model(
            x[fit_idx], mask[fit_idx], target[fit_idx].astype("float32"),
            x[val_idx], mask[val_idx], target[val_idx].astype("float32"),
            config,
        )
        if not result.ok:
            print(f"fold {fold}: training produced no usable model")
            continue

        probability = predict(result.model, x[test_idx], mask[test_idx])
        pooled_prob[test_idx] = probability
        fold_metrics = score(target[test_idx], probability)
        fold_metrics["fold"] = fold
        fold_metrics["test_wells"] = sorted(set(well_id[test_idx].tolist()))
        fold_metrics["mean_lead_seconds"] = round(
            lead_time_seconds(probability, sto[test_idx], instance[test_idx]), 1
        )
        per_fold.append(fold_metrics)
        print(
            f"fold {fold}: wells={len(fold_metrics['test_wells'])} n={len(test_idx)} "
            f"auc={fold_metrics.get('roc_auc')} ap={fold_metrics.get('average_precision')} "
            f"f1={fold_metrics.get('f1')} lead={fold_metrics['mean_lead_seconds']}s"
        )

    scored = ~np.isnan(pooled_prob)
    pooled = score(target[scored], pooled_prob[scored])
    pooled["mean_lead_seconds"] = round(lead_time_seconds(pooled_prob[scored], sto[scored], instance[scored]), 1)
    pooled["real_sequences_scored"] = int(scored.sum())
    pooled["real_wells_scored"] = len(set(well_id[scored].tolist()))

    print("\nPooled out-of-fold metrics (honest, every real sequence scored once):")
    for key, value in pooled.items():
        print(f"  {key}: {value}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "downsample_seconds": args.downsample,
                    "steps": args.steps,
                    "stride": args.stride,
                    "horizon_seconds": args.horizon,
                    "folds": args.folds,
                    "hidden": args.hidden,
                    "epochs": args.epochs,
                    "channels": channels,
                },
                "wide_split": split_summary(wide).to_dict(orient="records"),
                "class_well_coverage": coverage.to_dict(orient="records"),
                "per_fold": per_fold,
                "pooled": pooled,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Ablation: does reading the sequence help over summarising the same window?

The sequence path exists because a tree sees one window as a bag of statistics and never
sees *how* the window got there. This script tests that claim honestly.

It reuses the exact sequence arrays from ``run_sequence.py`` (same instances, same windows,
same split, same labels) and builds a bag-of-statistics table from each window - per channel
mean, standard deviation, min, max and last value - then scores it with LightGBM over the
same leave-wells-out folds. Same windows, same folds, one difference: the model sees
statistics instead of the ordered steps.

If the GRU does not beat this, the trajectory is not adding anything over a summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402

from wellsentinel.data import discover  # noqa: E402
from wellsentinel.sequence import channel_names  # noqa: E402
from wellsentinel.sequence_model import lead_time_seconds, score  # noqa: E402
from wellsentinel.wide_split import grouped_folds  # noqa: E402

CACHE = Path("/tmp/ws-seq/cache")
DATA = "/home/gmk/.openclaw/workspace/projects/amd-act3/data/3W/dataset"


def bag_features(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per channel: mean, std, min, max, last. Missing steps are ignored via the mask."""
    n, steps, channels = x.shape
    present = (mask < 0.5).astype("float64")  # 1 where measured
    count = present.sum(axis=1)
    safe = np.where(count == 0, 1.0, count)
    mean = (x * present).sum(axis=1) / safe
    centred = (x - mean[:, None, :]) ** 2
    std = np.sqrt((centred * present).sum(axis=1) / safe)
    filled = np.where(mask > 0.5, np.nan, x)
    with np.errstate(invalid="ignore"):
        low = np.nanmin(filled, axis=1)
        high = np.nanmax(filled, axis=1)
    last = x[:, -1, :]
    low = np.nan_to_num(low, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    return np.concatenate([mean, std, low, high, last], axis=1)


def main() -> int:
    npz = sorted(CACHE.glob("seq_*.npz"), key=lambda p: p.stat().st_mtime)[-1]
    print(f"Using cache {npz.name}")
    data = np.load(npz, allow_pickle=True)
    x, mask = data["x"], data["mask"].astype("float32")
    target = data["target"].astype("int64")
    well_id = data["well_id"]
    instance = data["instance"]
    sto = data["seconds_to_onset"]

    channels = channel_names()
    features = bag_features(x, mask)
    names = [
        f"{c}__{stat}" for stat in ("mean", "std", "min", "max", "last") for c in channels
    ]
    print(f"Bag features {features.shape} from windows {x.shape}")

    refs = discover(DATA)
    real = {r.well_id for r in refs if r.is_real}
    is_real = np.array([w in real for w in well_id])
    folds = grouped_folds(refs, k=5)

    pooled_prob = np.full(len(target), np.nan)
    per_fold = []
    for fold, (_, test_refs) in enumerate(folds):
        test_wells = {r.well_id for r in test_refs}
        test_idx = np.where(np.isin(well_id, list(test_wells)) & is_real)[0]
        train_idx = np.where(~(np.isin(well_id, list(test_wells)) & is_real))[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        y_train = target[train_idx]
        y_test = target[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        model = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(features[train_idx], y_train)
        prob = model.predict_proba(features[test_idx])[:, 1]
        pooled_prob[test_idx] = prob
        metrics = score(y_test, prob)
        metrics["fold"] = fold
        metrics["mean_lead_seconds"] = round(
            lead_time_seconds(prob, sto[test_idx], instance[test_idx]), 1
        )
        per_fold.append(metrics)
        print(f"fold {fold}: auc={metrics.get('roc_auc')} ap={metrics.get('average_precision')} "
              f"f1={metrics.get('f1')} base={metrics.get('base_rate')}")

    scored = ~np.isnan(pooled_prob)
    pooled = score(target[scored], pooled_prob[scored])
    pooled["mean_lead_seconds"] = round(
        lead_time_seconds(pooled_prob[scored], sto[scored], instance[scored]), 1
    )
    print("\nPooled bag-of-statistics metrics (same windows and folds as the GRU):")
    for key, value in pooled.items():
        print(f"  {key}: {value}")

    Path("/tmp/ws-seq/reports/ablation_bag.json").write_text(
        json.dumps({"per_fold": per_fold, "pooled": pooled}, indent=2, default=str)
    )
    print("wrote reports/ablation_bag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Phase 3+6: the prediction baseline. "Will an event start within H hours?"

This is the honest test of whether the data can predict the future, as opposed to
Phase 6's detection baseline which only asked "is an event happening now".

For each instance we window the signal, keep only windows whose end is still normal,
and label each one by whether a real onset lands inside the next ``--horizon`` seconds.
Train on real + simulated wells, score on real-only wells never seen in training.
Windows whose end is already inside an event are dropped, so the model is never asked
to predict the present.

Usage::

    python scripts/run_horizon.py --data ../amd-act3/data/3W/dataset \
        --per-class 20 --horizon 21600 --window 300 --stride 300 \
        --out reports/horizon_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from wellsentinel.cleansing import clean_instance  # noqa: E402
from wellsentinel.data import InstanceRef, discover, load_instance  # noqa: E402
from wellsentinel.features import add_derived_columns, window_features  # noqa: E402
from wellsentinel.horizon import horizon_targets, onset_times  # noqa: E402
from wellsentinel.split import assert_disjoint, split_by_well, split_summary  # noqa: E402

DEFAULT_CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
META_COLS = {"label", "well_id", "source", "instance", "n_rows"}


def build_horizon_frame(
    refs: list[InstanceRef],
    window_seconds: int,
    stride_seconds: int,
    horizon_seconds: int,
) -> pd.DataFrame:
    """Clean, derive, window, then label each still-normal window with its future."""
    frames: list[pd.DataFrame] = []
    for ref in refs:
        raw = load_instance(ref)
        cleaned, _ = clean_instance(raw)
        with_derived = add_derived_columns(cleaned)
        windowed = window_features(with_derived, window_seconds, stride_seconds)
        if windowed.empty:
            continue
        onsets = onset_times(cleaned["class"]) if "class" in cleaned.columns else np.array([])
        labelled = horizon_targets(windowed, onsets, window_seconds, horizon_seconds)
        if labelled.empty:
            continue
        labelled["well_id"] = ref.well_id
        labelled["source"] = ref.source
        labelled["instance"] = ref.name
        frames.append(labelled)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def take_subset(refs, per_class: int, classes=DEFAULT_CLASSES):
    chosen = []
    for label in classes:
        pool = [r for r in refs if r.label == label]
        reals = [r for r in pool if r.is_real]
        sims = [r for r in pool if not r.is_real]
        picked = (
            reals[:per_class]
            if len(reals) >= per_class
            else reals + sims[: per_class - len(reals)]
        )
        chosen.extend(picked[:per_class])
    return chosen


def train_and_score(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> dict:
    feature_names = [
        c
        for c in train.columns
        if c not in META_COLS | {"target", "seconds_to_onset"}
        and pd.api.types.is_numeric_dtype(train[c])
    ]
    x_train, y_train = train[feature_names], train["target"].astype(int)
    x_test, y_test = test[feature_names], test["target"].astype(int)

    metrics: dict[str, float] = {
        "rows_train": len(x_train),
        "rows_test": len(x_test),
        "positives_train": int(y_train.sum()),
        "positives_test": int(y_test.sum()),
        "base_rate_test": round(float(y_test.mean()), 4),
    }
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        metrics["note"] = "one class only; no score"
        return metrics

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)

    metrics.update(
        {
            "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
            "average_precision": round(float(average_precision_score(y_test, probability)), 4),
            "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Prediction (horizon) baseline.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=21600, help="Horizon seconds (6 h).")
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--stride", type=int, default=300)
    parser.add_argument("--out", default="reports/horizon_v1.json")
    args = parser.parse_args()

    refs = discover(args.data)
    subset = take_subset(refs, args.per_class)
    split = split_by_well(subset)
    assert_disjoint(split)
    print(f"Subset {len(subset)} instances; horizon {args.horizon/3600:.1f} h")
    print(split_summary(split).to_string(index=False))

    train = build_horizon_frame(split.train, args.window, args.stride, args.horizon)
    test = build_horizon_frame(split.test, args.window, args.stride, args.horizon)
    print(f"Train windows {len(train)}  Test windows {len(test)}")

    metrics = train_and_score(train, test)
    print("\nPrediction metrics (real-only test):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "per_class": args.per_class,
                    "horizon_seconds": args.horizon,
                    "window_seconds": args.window,
                    "stride_seconds": args.stride,
                },
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Export a deployable bundle for the demo UI.

Trains one model on real + simulated wells, scores it on real-only held-out wells,
and writes everything the Streamlit app needs to replay an unseen instance.

Bundle layout::

    bundle/model.joblib               trained model, feature names, metadata
    bundle/metrics.json               the honest score for this exact split
    bundle/importance.csv             top features by gain
    bundle/demo/<name>.parquet        raw held-out instance (sensor traces)
    bundle/demo/<name>.pred.parquet   precomputed per-window risk + labels
    bundle/manifest.json              the demo instances, with their event metadata
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/ws-p6/src")
sys.path.insert(0, "/tmp/ws-p6/scripts")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from wellsentinel.cleansing import clean_instance  # noqa: E402
from wellsentinel.constants import CLASS_NAMES, ANALOG_COLS  # noqa: E402
from wellsentinel.data import discover, load_instance  # noqa: E402
from wellsentinel.features import add_derived_columns, window_features  # noqa: E402
from wellsentinel.horizon import horizon_targets, onset_times  # noqa: E402
from wellsentinel.labels import fold_transient  # noqa: E402
from run_horizon import build_horizon_frame, take_subset  # noqa: E402
from wellsentinel.split import split_by_well  # noqa: E402

DATA = "/home/gmk/.openclaw/workspace/projects/amd-act3/data/3W/dataset"
HORIZON = 21600  # 6 h
OUT = Path("/tmp/ws-p6/bundle")
META = {"label", "well_id", "source", "instance", "n_rows", "target", "seconds_to_onset"}
MAX_DEMO = 6


def instance_summary(ref, frame: pd.DataFrame) -> dict:
    """Event metadata for one held-out instance, for the UI picker."""
    folded = frame["class"].map(
        lambda v: np.nan if pd.isna(v) else float(fold_transient(int(v)))
    )
    onsets = onset_times(frame["class"])
    span_h = (frame.index[-1] - frame.index[0]).total_seconds() / 3600
    lead_h = (onsets[0] - frame.index[0]).total_seconds() / 3600 if len(onsets) else None
    return {
        "name": ref.name,
        "well_id": ref.well_id,
        "label": ref.label,
        "event": CLASS_NAMES.get(ref.label, str(ref.label)),
        "rows": int(len(frame)),
        "span_hours": round(span_h, 2),
        "normal_lead_hours": round(lead_h, 2) if lead_h is not None else None,
        "onsets": [str(o) for o in onsets[:5]],
        "sensors_present": [c for c in ANALOG_COLS if c in frame.columns and frame[c].notna().any()],
    }


def main() -> int:
    refs = discover(DATA)
    subset = take_subset(refs, 20)
    split = split_by_well(subset)

    train = build_horizon_frame(split.train, 300, 300, HORIZON)
    test = build_horizon_frame(split.test, 300, 300, HORIZON)
    feature_names = [
        c
        for c in train.columns
        if c not in META and pd.api.types.is_numeric_dtype(train[c])
    ]
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train[feature_names], train["target"].astype(int))

    probability = model.predict_proba(test[feature_names])[:, 1]
    metrics = {
        "horizon_seconds": HORIZON,
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "positives_test": int(test["target"].sum()),
        "base_rate_test": round(float(test["target"].mean()), 4),
        "roc_auc": round(float(roc_auc_score(test["target"], probability)), 4),
        "average_precision": round(float(average_precision_score(test["target"], probability)), 4),
        "test_wells": sorted({r.well_id for r in split.test}),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "demo").mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "horizon_seconds": HORIZON,
            "window_seconds": 300,
            "stride_seconds": 300,
            "metrics": metrics,
        },
        OUT / "model.joblib",
    )
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    importance = pd.DataFrame(
        {"feature": feature_names, "gain": model.booster_.feature_importance("gain")}
    ).sort_values("gain", ascending=False)
    importance.head(60).to_csv(OUT / "importance.csv", index=False)

    # Precompute per-window risk for each held-out instance so the UI needs no pipeline.
    manifest = []
    for ref in split.test[:MAX_DEMO]:
        raw = load_instance(ref)
        cleaned, _ = clean_instance(raw)
        derived = add_derived_columns(cleaned)
        windowed = window_features(derived, 300, 300)
        if windowed.empty:
            continue
        onsets = onset_times(cleaned["class"])
        labelled = horizon_targets(windowed, onsets, 300, HORIZON)
        if labelled.empty:
            continue
        risk = model.predict_proba(labelled[feature_names])[:, 1]
        pred = pd.DataFrame(
            {
                "window_start": labelled.index,
                "risk": risk,
                "label": labelled["label"].to_numpy(),
                "seconds_to_onset": labelled["seconds_to_onset"].to_numpy(),
                "target": labelled["target"].to_numpy(),
            }
        )
        raw.reset_index().to_parquet(OUT / "demo" / f"{ref.name}.parquet", index=False)
        pred.to_parquet(OUT / "demo" / f"{ref.name}.pred.parquet", index=False)
        summary = instance_summary(ref, raw)
        summary["alert_hours_before"] = (
            round(float(pred.loc[pred["risk"] >= 0.5, "seconds_to_onset"].max()) / 3600, 2)
            if (pred["risk"] >= 0.5).any() and pred.loc[pred["risk"] >= 0.5, "seconds_to_onset"].notna().any()
            else None
        )
        manifest.append(summary)
        print(f"  exported {ref.name}  windows={len(pred)}  max_risk={risk.max():.2f}")

    (OUT / "manifest.json").write_text(json.dumps({"metrics": metrics, "instances": manifest}, indent=2))
    print(json.dumps(metrics, indent=2))
    print("bundle written to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

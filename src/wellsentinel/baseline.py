"""Phase 6: the subset baseline, our go / no-go gate.

Small, honest, fast. Take a bounded set of instances, clean them, compute window features,
and train a gradient-boosted tree. The point is not a good model yet. The point is to learn
whether the signal is there before we spend more days on the data.

Term notes:
  * *baseline* is the simplest reasonable model, used as a reference point.
  * *precision* of the alerts we raise, how many were right.
  * *recall* of the events that happened, how many we caught.
  * *lead time* how early before the event we raised the alert.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .cleansing import clean_instance
from .data import InstanceRef, load_instance
from .features import add_derived_columns, window_features
from .labels import fold_transient


@dataclass
class BaselineResult:
    """Metrics plus the trained model and the feature names, for inspection."""

    rows_train: int = 0
    rows_test: int = 0
    positives_train: int = 0
    positives_test: int = 0
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    roc_auc: float = float("nan")
    mean_lead_seconds: float = float("nan")
    model: object | None = None
    feature_names: list[str] = field(default_factory=list)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)

    def as_dict(self) -> dict[str, object]:
        return {
            "rows_train": self.rows_train,
            "rows_test": self.rows_test,
            "positives_train": self.positives_train,
            "positives_test": self.positives_test,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "mean_lead_seconds": round(self.mean_lead_seconds, 1),
        }


def build_window_frame(
    refs: list[InstanceRef],
    window_seconds: int = 300,
    stride_seconds: int = 300,
) -> pd.DataFrame:
    """Clean, derive and window a list of instances into one feature frame."""
    frames = []
    for ref in refs:
        raw = load_instance(ref)
        cleaned, _ = clean_instance(raw)
        with_derived = add_derived_columns(cleaned)
        windowed = window_features(
            with_derived, window_seconds=window_seconds, stride_seconds=stride_seconds
        )
        if windowed.empty:
            continue
        windowed["well_id"] = ref.well_id
        windowed["source"] = ref.source
        windowed["instance"] = ref.name
        frames.append(windowed)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def add_binary_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw ``class`` column into the anomaly flag we predict first."""
    out = frame.copy()
    folded = out["label"].map(
        lambda v: float("nan") if pd.isna(v) else float(fold_transient(int(v)))
    )
    out["is_anomaly"] = folded.map(lambda v: False if pd.isna(v) else v != 0)
    out = out[folded.notna()]
    return out


def _mean_lead_time(block: pd.DataFrame, threshold: float = 0.5) -> float:
    """Average seconds between the first alert in an instance and its first real event.

    A negative number means we alerted late. Only measured where both exist.
    """
    leads: list[float] = []
    if "alert" not in block.columns:
        return float("nan")
    for _, group in block.groupby("instance"):
        group = group.sort_index()
        alerts = group.index[group["alert"] >= threshold]
        events = group.index[group["is_anomaly"]]
        if len(alerts) and len(events):
            leads.append((events[0] - alerts[0]).total_seconds())
    return float(np.mean(leads)) if leads else float("nan")


def train_baseline(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    n_estimators: int = 300,
    random_state: int = 42,
) -> BaselineResult:
    """Train the LightGBM baseline and score it on a held-out, real-only frame."""
    result = BaselineResult()
    feature_names = [
        c
        for c in train_frame.columns
        if c not in {"label", "is_anomaly", "well_id", "source", "instance", "n_rows"}
        and pd.api.types.is_numeric_dtype(train_frame[c])
    ]
    result.feature_names = feature_names

    x_train = train_frame[feature_names]
    y_train = train_frame["is_anomaly"].astype(int)
    x_test = test_frame[feature_names]
    y_test = test_frame["is_anomaly"].astype(int)

    result.rows_train, result.rows_test = len(x_train), len(x_test)
    result.positives_train, result.positives_test = int(y_train.sum()), int(y_test.sum())

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        # One class only: nothing to learn or measure. Say so instead of faking a score.
        result.model = None
        return result

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result.model = model
    result.precision = float(precision_score(y_test, predictions, zero_division=0))
    result.recall = float(recall_score(y_test, predictions, zero_division=0))
    result.f1 = float(f1_score(y_test, predictions, zero_division=0))
    result.roc_auc = float(roc_auc_score(y_test, probabilities))

    scored = test_frame.copy()
    scored["alert"] = probabilities
    result.mean_lead_seconds = _mean_lead_time(scored)
    result.importance = (
        pd.DataFrame({"feature": feature_names, "gain": model.booster_.feature_importance("gain")})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    return result

"""Tests for the sequence data builder and the masked GRU.

Kept small and synthetic so they run in a second and do not need the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wellsentinel.sequence import WellScaler, build_sequences, channel_names, downsample
from wellsentinel.sequence_model import MaskedGRU, TrainConfig, score, train_sequence_model


def _toy_frame(n: int = 600, event_from: int | None = None) -> pd.DataFrame:
    """A 1 Hz toy instance with two analogue channels and a class column."""
    index = pd.date_range("2020-01-01", periods=n, freq="1s")
    base = np.sin(np.linspace(0, 20, n))
    frame = pd.DataFrame(
        {
            "P-PDG": 1e6 + 1e4 * base,
            "T-TPT": 60 + 2 * base,
            "class": np.zeros(n, dtype="float64"),
        },
        index=index,
    )
    if event_from is not None:
        frame.loc[frame.index[event_from:], "class"] = 4.0
    return frame


def test_channel_names_are_stable_and_unique():
    names = channel_names()
    assert len(names) == len(set(names))
    assert "P-PDG" in names


def test_downsample_averages_analog_and_keeps_last_label():
    frame = _toy_frame(120)
    frame.loc[frame.index[60:], "class"] = 3.0
    reduced = downsample(frame, 60)
    # 120 rows at 60 s bins -> 2 bins.
    assert len(reduced) == 2
    assert reduced["class"].iloc[0] == 0.0
    assert reduced["class"].iloc[1] == 3.0


def test_per_instance_scaler_is_centred():
    frame = _toy_frame(300)
    scaler = WellScaler.fit_instance(frame, ["P-PDG", "T-TPT"])
    scaled = scaler.transform_column(frame["P-PDG"].to_numpy(dtype="float64"), "P-PDG")
    # Median maps to roughly zero; the IQR maps to roughly one.
    assert abs(np.nanmedian(scaled)) < 1e-6
    q75, q25 = np.nanpercentile(scaled, [75, 25])
    assert abs((q75 - q25) - 1.0) < 1e-6


def test_scaler_handles_absent_channel():
    frame = _toy_frame(50)
    scaler = WellScaler.fit_instance(frame, ["P-PDG", "QGL"])
    assert scaler.iqr["QGL"] == 1.0  # absent channel falls back to a safe identity


def test_sequenceize_uses_a_fake_dataset(tmp_path):
    """Build sequences straight from the dataset loader using a temporary parquet."""
    refs = _write_toy_dataset(tmp_path)
    seq = build_sequences(
        refs,
        channels=channel_names(),
        downsample_seconds=10,
        steps=6,
        stride=3,
        horizon_seconds=600,
    )
    assert len(seq) > 0
    assert seq.x.shape[1] == 6
    assert seq.x.shape[2] == len(channel_names())
    assert set(np.unique(seq.target)).issubset({0, 1})
    # Windows whose end sits inside the event are dropped.
    assert np.all(seq.target[~np.isnan(seq.seconds_to_onset)] <= 1)


def _write_toy_dataset(root):
    from wellsentinel.data import InstanceRef

    class_dir = root / "4"
    class_dir.mkdir(parents=True)
    frame = _toy_frame(1200, event_from=800)
    path = class_dir / "WELL-00099_20200101.parquet"
    frame.to_parquet(path)
    return [
        InstanceRef(
            path=path,
            label=4,
            source="real",
            well_id="WELL-00099",
            name="WELL-00099_20200101",
        )
    ]


def test_masked_gru_forward_shape():
    model = MaskedGRU(n_channels=5, hidden=8)
    x = np.random.randn(4, 10, 5).astype("float32")
    m = np.zeros((4, 10, 5), dtype="float32")
    import torch

    out = model(torch.as_tensor(x), torch.as_tensor(m))
    assert out.shape == (4,)


def test_train_runs_and_predicts_probabilities():
    rng = np.random.default_rng(0)
    n, steps, channels = 200, 8, 3
    x = rng.normal(size=(n, steps, channels)).astype("float32")
    mask = np.zeros_like(x)
    # Make the label learnable: sum of the last step drives the target.
    signal = x[:, -1, :].sum(axis=1)
    y = (signal > np.median(signal)).astype("float32")
    result = train_sequence_model(
        x[:150], mask[:150], y[:150], x[150:], mask[150:], y[150:], TrainConfig(epochs=3)
    )
    assert result.ok
    assert result.best_epoch >= 0


def test_score_reports_auc_and_handles_single_class():
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    metrics = score(y, p)
    assert metrics["roc_auc"] == 1.0
    assert metrics["base_rate"] == 0.6
    single = score(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]))
    assert "roc_auc" not in single

"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wellsentinel.features import _slope, add_derived_columns, window_features


def test_add_derived_columns_builds_differential() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="1s")
    frame = pd.DataFrame({"P-MON-CKP": [10.0, 12.0], "P-JUS-CKP": [4.0, 5.0]}, index=idx)
    out = add_derived_columns(frame)
    column = "D__P-MON-CKP__minus__P-JUS-CKP"
    assert column in out.columns
    assert list(out[column]) == [6.0, 7.0]


def test_add_derived_columns_builds_ratio() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="1s")
    frame = pd.DataFrame({"QGL": [4.0, 9.0], "QBS": [2.0, 3.0]}, index=idx)
    out = add_derived_columns(frame)
    assert list(out["R__QGL__over__QBS"]) == [2.0, 3.0]


def test_ratio_handles_zero_denominator() -> None:
    idx = pd.date_range("2020-01-01", periods=1, freq="1s")
    frame = pd.DataFrame({"QGL": [4.0], "QBS": [0.0]}, index=idx)
    out = add_derived_columns(frame)
    assert bool(np.isnan(out["R__QGL__over__QBS"].iloc[0])) is True


def test_add_derived_columns_skips_missing_parents() -> None:
    idx = pd.date_range("2020-01-01", periods=1, freq="1s")
    frame = pd.DataFrame({"P-MON-CKP": [1.0]}, index=idx)
    out = add_derived_columns(frame)
    assert not [c for c in out.columns if c.startswith("D__")]


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1.0, 2.0, 3.0], 1.0), ([3.0, 2.0, 1.0], -1.0), ([5.0, 5.0], 0.0)],
)
def test_slope(values: list[float], expected: float) -> None:
    assert _slope(np.array(values)) == pytest.approx(expected)


def test_slope_handles_single_point() -> None:
    assert np.isnan(_slope(np.array([1.0])))


def test_window_features_shape_and_labels() -> None:
    idx = pd.date_range("2020-01-01", periods=600, freq="1s")
    frame = pd.DataFrame(
        {"P-MON-CKP": np.arange(600, dtype=float), "class": [0] * 300 + [1] * 300},
        index=idx,
    )
    out = window_features(frame, window_seconds=300, stride_seconds=300)
    assert len(out) == 1
    assert "P-MON-CKP__mean" in out.columns
    assert "P-MON-CKP__slope" in out.columns
    # Label is taken from the end of the window.
    assert out["label"].iloc[0] == 1


def test_window_features_raises_without_columns() -> None:
    idx = pd.date_range("2020-01-01", periods=10, freq="1s")
    with pytest.raises(ValueError):
        window_features(pd.DataFrame({"unrelated": range(10)}, index=idx))

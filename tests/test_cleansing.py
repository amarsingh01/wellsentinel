"""Tests for cleansing behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wellsentinel.cleansing import (
    add_missing_mask,
    clean_instance,
    clip_outliers,
    fill_short_gaps,
    find_frozen,
    standardize,
)


def _frame(values, freq="1s", class_values=None):
    idx = pd.date_range("2020-01-01", periods=len(values), freq=freq)
    data = {"P-MON-CKP": values}
    if class_values is not None:
        data["class"] = class_values
    return pd.DataFrame(data, index=idx)


def test_standardize_sorts_and_drops_duplicates() -> None:
    idx = pd.to_datetime(["2020-01-01T00:00:02", "2020-01-01T00:00:01", "2020-01-01T00:00:01"])
    frame = pd.DataFrame({"P-MON-CKP": [3.0, 2.0, 9.0]}, index=idx)
    out, dropped = standardize(frame)
    assert dropped == 1
    assert list(out["P-MON-CKP"]) == [2.0, 3.0]
    assert out.index.is_monotonic_increasing


def test_fill_short_gaps_interpolates_internal_gap() -> None:
    frame = _frame([1.0, np.nan, 3.0])
    out, filled, _clipped, left = fill_short_gaps(frame, limit_seconds=300)
    assert filled == 1
    assert left == 0
    assert out["P-MON-CKP"].iloc[1] == 2.0


def test_fill_short_gaps_leaves_short_limit_alone() -> None:
    frame = _frame([1.0, np.nan, 3.0, 4.0], freq="10s")
    # Limit of 5 s cannot bridge a 10 s sampling gap.
    out, _filled, _clipped, left = fill_short_gaps(frame, limit_seconds=5)
    assert left >= 1


def test_missing_mask_flags_absent_values() -> None:
    frame = _frame([1.0, np.nan, 3.0])
    masked = add_missing_mask(frame, columns=["P-MON-CKP"])
    assert "P-MON-CKP__missing" in masked.columns
    assert bool(masked["P-MON-CKP__missing"].iloc[1]) is True


def test_clip_outliers_counts_changes() -> None:
    frame = _frame([1.0, 500.0, 3.0])
    out, clipped = clip_outliers(frame, bounds={"P-MON-CKP": (0.0, 10.0)})
    assert clipped == 1
    assert out["P-MON-CKP"].max() == 10.0


def test_clip_outliers_without_bounds_is_noop() -> None:
    frame = _frame([1.0, 500.0])
    out, clipped = clip_outliers(frame, bounds=None)
    assert clipped == 0
    assert out["P-MON-CKP"].max() == 500.0


def test_find_frozen_detects_constant_column() -> None:
    frame = _frame([2.0, 2.0, 2.0])
    assert find_frozen(frame, columns=["P-MON-CKP"]) == ["P-MON-CKP"]


def test_clean_instance_reports_rows_and_never_edits_labels() -> None:
    frame = _frame([1.0, np.nan, 3.0], class_values=[0, np.nan, 1])
    out, report = clean_instance(frame)
    assert report.rows_in == 3
    assert report.rows_out == 3
    assert report.values_interpolated == 1
    # Label column untouched: the blank stays blank.
    assert bool(pd.isna(out["class"].iloc[1])) is True

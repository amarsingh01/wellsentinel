"""Tests for label handling."""

from __future__ import annotations

import pandas as pd
import pytest

from wellsentinel.labels import (
    fold_transient,
    is_anomaly,
    label_mask,
    to_binary,
    to_multiclass,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, 0), (1, 1), (9, 9), (101, 1), (107, 7), (109, 9)],
)
def test_fold_transient(raw: int, expected: int) -> None:
    assert fold_transient(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, False), (4, True), (109, True), (float("nan"), False)],
)
def test_is_anomaly(raw: float, expected: bool) -> None:
    assert is_anomaly(raw) is expected


def test_to_binary_drops_unlabelled_rows() -> None:
    series = pd.Series([0, 1, 101, None, 4])
    result = to_binary(series)
    assert list(result) == [False, True, True, False, True]


def test_to_multiclass_folds_transients() -> None:
    series = pd.Series([0, 1, 101, 109, None])
    result = to_multiclass(series)
    assert list(result.dropna()) == [0, 1, 1, 9]
    assert result.isna().sum() == 1


def test_label_mask_marks_blanks() -> None:
    series = pd.Series([0, None, 3])
    assert list(label_mask(series)) == [True, False, True]

"""Label handling for the 3W dataset.

3W ships two label columns:

``class``
    0 for normal operation, 1-9 for undesirable event types, and 101-109 for the
    *transient* onset of the event with the matching 1-9 id. Some trailing rows are
    blank, those observations carry no label.

``state``
    Well operational status. Context, not a target.

Nothing here touches the sensor values, so it is cheap to unit test.
"""

from __future__ import annotations

import pandas as pd

from .constants import NORMAL_CLASS, TRANSIENT_OFFSET


def fold_transient(class_id: int) -> int:
    """Fold a transient label back onto its parent event id.

    101 -> 1, 109 -> 9. Normal and plain event ids pass through unchanged.
    """
    if class_id >= TRANSIENT_OFFSET:
        return class_id - TRANSIENT_OFFSET
    return class_id


def is_anomaly(class_id: int) -> bool:
    """True when the label denotes an undesirable event (transient counts as one)."""
    if pd.isna(class_id):
        return False
    return fold_transient(int(class_id)) != NORMAL_CLASS


def to_binary(series: pd.Series) -> pd.Series:
    """Map a ``class`` series to a boolean anomaly flag, dropping unlabelled rows."""
    folded = series.map(_fold_or_nan)
    return folded.map(lambda c: False if pd.isna(c) else c != NORMAL_CLASS)


def to_multiclass(series: pd.Series) -> pd.Series:
    """Map a ``class`` series to folded event ids as nullable integers."""
    return series.map(_fold_or_nan).astype("Int64")


def _fold_or_nan(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(fold_transient(int(value)))


def label_mask(series: pd.Series) -> pd.Series:
    """True where the row carries a label; False for the unlabelled tail."""
    return series.notna()

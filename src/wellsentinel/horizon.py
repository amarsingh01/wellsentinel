"""Phase 3 (horizon labels): turn detection into prediction.

The subset baseline answered "am I inside an event right now". That is detection.
The question is "will an event start soon". That is prediction. The only difference
is the label.

For each window whose *end* time is still normal, the horizon label is 1 when a real
event onset lands inside ``(end, end + horizon]``, else 0. Windows already inside an
event are dropped: predicting the present is not predicting the future.

Leakage guard: the label only looks forward from the window end, and onset times come
from the label column (``class``), never from the sensor features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import fold_transient


def _folded(class_series: pd.Series) -> pd.Series:
    """Fold transient ids (101-109) onto their parents, keeping NaN as NaN."""
    return class_series.map(
        lambda value: np.nan if pd.isna(value) else float(fold_transient(int(value)))
    )


def onset_times(class_series: pd.Series) -> np.ndarray:
    """Timestamps where an event begins: a transition from normal/blank to non-zero."""
    folded = _folded(class_series)
    event = (folded.fillna(0.0).to_numpy() != 0.0)
    index = class_series.index.to_numpy()
    onsets: list[object] = []
    previous = False
    for timestamp, current in zip(index, event):
        if current and not previous:
            onsets.append(timestamp)
        previous = bool(current)
    return np.array(onsets, dtype="datetime64[ns]")


def horizon_targets(
    window_frame: pd.DataFrame,
    onsets: np.ndarray,
    window_seconds: int,
    horizon_seconds: int,
) -> pd.DataFrame:
    """Label each *normal* window by whether an onset is within ``horizon_seconds``.

    Adds:
      * ``target`` - 1 when an onset is inside the horizon, else 0;
      * ``seconds_to_onset`` - seconds until the next onset (NaN when none follows).
    Windows whose end sits inside an event are removed.
    """
    out = window_frame.copy()
    ends = out.index + pd.Timedelta(seconds=window_seconds)
    current = _folded(out["label"])
    keep = (current.fillna(-1.0) == 0.0).to_numpy()
    out = out[keep]
    ends = ends[keep]

    out["seconds_to_onset"] = np.nan
    out["target"] = 0

    if len(onsets) == 0 or len(out) == 0:
        return out

    end_ns = ends.to_numpy().astype("datetime64[ns]")
    positions = np.searchsorted(onsets, end_ns, side="right")
    valid = positions < len(onsets)

    gaps = np.full(len(out), np.nan)
    gaps[valid] = (onsets[positions[valid]] - end_ns[valid]) / np.timedelta64(1, "s")
    out["seconds_to_onset"] = gaps
    out["target"] = ((gaps > 0) & (gaps <= horizon_seconds)).astype(int)
    return out

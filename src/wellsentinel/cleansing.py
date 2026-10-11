"""Phase 2: cleansing.

The 3W authors deliberately leave frozen variables, missing variables and outliers
untreated, because handling them is part of the task. So this module is explicit about
what it does and what it refuses to do:

* we never invent a label;
* we only interpolate short gaps, and we count every value we fill;
* long gaps stay as gaps and get a mask column, so the model can see "no data here";
* outliers are clipped to physical bounds, and counted, never silently deleted.

Term note: *interpolation* means estimating a missing value from its neighbours in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import ANALOG_COLS, DEFAULT_MAX_GAP_SECONDS


@dataclass
class CleanReport:
    """What cleansing actually did to one instance. Counted, not guessed."""

    rows_in: int = 0
    rows_out: int = 0
    duplicate_timestamps_dropped: int = 0
    values_interpolated: int = 0
    values_clipped: int = 0
    long_gaps_flagged: int = 0
    frozen_columns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        data = self.__dict__.copy()
        data["frozen_columns"] = len(self.frozen_columns)
        return data


def standardize(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sort by time, drop duplicate timestamps, coerce numbers."""
    out = frame.copy()
    out = out[~out.index.duplicated(keep="first")]
    dropped = len(frame) - len(out)
    out = out.sort_index()
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out, dropped


def add_missing_mask(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Add a boolean mask column per analogue column: True where the value is absent."""
    out = frame.copy()
    for column in columns or list(ANALOG_COLS):
        if column in out.columns:
            out[f"{column}__missing"] = out[column].isna()
    return out


def estimate_cadence_seconds(frame: pd.DataFrame) -> float:
    """Median spacing between observations, in seconds.

    The dataset is nominally 1 Hz but the real instances wobble, so we measure rather
    than assume.
    """
    if len(frame.index) < 2:
        return float("nan")
    deltas = frame.index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return float("nan")
    return float(deltas.median())


def fill_short_gaps(
    frame: pd.DataFrame,
    limit_seconds: int = DEFAULT_MAX_GAP_SECONDS,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, int, int, int]:
    """Interpolate short gaps. Returns (frame, filled, clipped, missing_left).

    Interpolation is time aware, so an irregular cadence does not distort the result.
    ``limit_seconds`` is a real time budget, not a sample count: we convert it to a
    maximum number of consecutive missing samples using the measured cadence, so a
    60 s limit means 60 s regardless of how the rows are spaced.

    Gaps longer than the limit are left alone and only reported.
    """
    out = frame.copy()
    cols = [c for c in (columns or ANALOG_COLS) if c in out.columns]
    if not cols:
        return out, 0, 0, 0

    before = int(out[cols].isna().to_numpy().sum())

    cadence = estimate_cadence_seconds(frame)
    max_samples = 0 if cadence != cadence else int(limit_seconds / cadence)  # NaN-safe
    if max_samples >= 1:
        out[cols] = out[cols].interpolate(
            method="time", limit=max_samples, limit_area="inside"
        )

    after = int(out[cols].isna().to_numpy().sum())
    filled = before - after
    return out, filled, 0, after


def clip_outliers(
    frame: pd.DataFrame,
    bounds: dict[str, tuple[float, float]] | None = None,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Clip values to physical bounds. Counts how many changed.

    Physical bounds are supplied by the caller (from the paper or the historian tags).
    Without bounds this is a no-op, which is the honest default.
    """
    if not bounds:
        return frame, 0
    out = frame.copy()
    clipped = 0
    for column in columns or list(bounds):
        if column not in out.columns:
            continue
        low, high = bounds[column]
        original = out[column].to_numpy(dtype=float, copy=True)
        out[column] = out[column].clip(lower=low, upper=high)
        clipped += int(np.nansum(out[column].to_numpy(dtype=float) != original))
    return out, clipped


def find_frozen(frame: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    """Columns that never change over the instance. Context, not dynamics."""
    return [
        c
        for c in (columns or ANALOG_COLS)
        if c in frame.columns and frame[c].nunique(dropna=True) <= 1
    ]


def clean_instance(
    frame: pd.DataFrame,
    max_gap_seconds: int = DEFAULT_MAX_GAP_SECONDS,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, CleanReport]:
    """Run the whole cleansing pass on one instance."""
    report = CleanReport(rows_in=len(frame))
    out, duplicates = standardize(frame)

    out, filled, clipped, gaps_left = fill_short_gaps(out, limit_seconds=max_gap_seconds)
    out, clipped2 = clip_outliers(out, bounds=bounds)
    out = add_missing_mask(out)
    frozen = find_frozen(out)

    report.rows_out = len(out)
    report.duplicate_timestamps_dropped = duplicates
    report.values_interpolated = filled
    report.values_clipped = clipped + clipped2
    report.long_gaps_flagged = gaps_left
    report.frozen_columns = frozen
    return out, report

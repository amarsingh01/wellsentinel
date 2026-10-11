"""Phase 5: feature engineering.

Two layers:

1. *Derived columns* we compute once per instance. These are the domain features, the
   differences and ratios that carry the physics. A fault usually shows up as a change
   in the gap between two sensors, not in either sensor alone.
2. *Window features* computed over a time window, so a 1 minute event and a 1 hour event
   both leave a signature.

Term note: a *window* is a fixed slice of time, for example 5 minutes. We slide it along
the series and summarise each slice with a few numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ANALOG_COLS

#: Sensor pairs whose difference is physically meaningful (upstream vs downstream).
DIFFERENTIAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("P-MON-CKP", "P-JUS-CKP"),   # across the production choke
    ("P-MON-CKGL", "P-JUS-CKGL"),  # across the gas lift choke
    ("P-TPT", "P-PDG"),           # surface vs downhole
    ("T-MON-CKP", "T-JUS-CKP"),   # across the production choke (temperature)
    ("P-ANULAR", "P-PDG"),        # annulus vs downhole
    ("P-MON-CKP", "P-MON-SDV-P"),  # around the production shutdown valve
)

#: Ratios. Scale free, so they survive the big range differences between wells.
RATIO_PAIRS: tuple[tuple[str, str], ...] = (
    ("QGL", "QBS"),        # gas lift vs service pump
    ("P-PDG", "P-TPT"),    # downhole vs surface pressure
)


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add differential and ratio columns, only when both parents are present."""
    out = frame.copy()
    for left, right in DIFFERENTIAL_PAIRS:
        if left in out.columns and right in out.columns:
            out[f"D__{left}__minus__{right}"] = out[left] - out[right]
    for left, right in RATIO_PAIRS:
        if left in out.columns and right in out.columns:
            denominator = out[right].replace(0.0, np.nan)
            out[f"R__{left}__over__{right}"] = out[left] / denominator
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every numeric column we are willing to summarise in a window."""
    candidates = list(ANALOG_COLS) + [c for c in frame.columns if c.startswith(("D__", "R__"))]
    return [c for c in candidates if c in frame.columns]


def _as_float_array(series: pd.Series) -> np.ndarray:
    """Convert a (possibly nullable) column to a plain float64 array, NA becomes NaN."""
    return series.astype("float64").to_numpy()


def _slope(values: np.ndarray) -> float:
    """Least squares slope against a simple 0..n index. Trend direction and size."""
    n = values.size
    finite = values[~np.isnan(values)]
    if n < 2 or finite.size < 2:
        return float("nan")
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = finite.mean()
    denominator = ((x - x_mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    # Pairwise on the finite points only, so gaps do not pull the trend.
    mask = ~np.isnan(values)
    return float(((x[mask] - x_mean) * (values[mask] - y_mean)).sum() / denominator)


def _last_label(window: pd.DataFrame) -> float:
    """Label of the window, taken from its final row. NaN when the tail is unlabelled."""
    if "class" not in window.columns:
        return float("nan")
    value = window["class"].iloc[-1]
    if pd.isna(value):
        return float("nan")
    return float(value)


def _safe_stat(values: np.ndarray, reducer) -> float:
    """Run a nan-aware reducer, returning NaN for an all-missing window."""
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(reducer(values))


def window_features(
    frame: pd.DataFrame,
    window_seconds: int = 300,
    stride_seconds: int = 300,
) -> pd.DataFrame:
    """Summarise ``frame`` into fixed windows: mean, std, min, max and slope per column.

    The returned frame is indexed by window start time and carries a ``label`` column
    taken from the *end* of the window, plus ``n_rows`` so short windows are visible.
    """
    cols = feature_columns(frame)
    if not cols:
        raise ValueError("no feature columns available")

    index = frame.index
    start = index[0]
    end = index[-1]
    rows: list[dict[str, float]] = []
    starts: list[pd.Timestamp] = []

    step = pd.Timedelta(seconds=stride_seconds)
    width = pd.Timedelta(seconds=window_seconds)

    window_start = start
    while window_start + width <= end:
        window = frame.loc[window_start : window_start + width]
        if len(window) >= 2:
            record: dict[str, float] = {"n_rows": len(window)}
            for column in cols:
                values = _as_float_array(window[column])
                record[f"{column}__mean"] = _safe_stat(values, np.nanmean)
                record[f"{column}__std"] = _safe_stat(values, np.nanstd)
                record[f"{column}__min"] = _safe_stat(values, np.nanmin)
                record[f"{column}__max"] = _safe_stat(values, np.nanmax)
                record[f"{column}__slope"] = _slope(values)
            record["label"] = _last_label(window)
            rows.append(record)
            starts.append(window_start)
        window_start = window_start + step

    result = pd.DataFrame(rows, index=pd.Index(starts, name="window_start"))
    return result

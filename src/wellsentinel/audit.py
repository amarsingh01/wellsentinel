"""Phase 1: audit the raw dataset.

For every instance we record what is actually there, so later decisions are based on
measurements rather than assumptions:

* how many rows, over what time span, at what cadence;
* which of the 27 columns exist at all;
* which exist but never change (frozen);
* how much of each column is missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import ANALOG_COLS
from .data import InstanceRef, load_instance


def audit_instance(ref: InstanceRef) -> dict[str, object]:
    """Summarise one instance. Returns a flat row suitable for a CSV report."""
    frame = load_instance(ref)
    rows = len(frame)

    present = [c for c in ANALOG_COLS if c in frame.columns]
    missing_fraction = {
        c: float(frame[c].isna().mean()) if c in present else 1.0 for c in ANALOG_COLS
    }
    frozen = [c for c in present if frame[c].nunique(dropna=True) <= 1]

    span_seconds = float("nan")
    if rows > 1:
        span_seconds = (frame.index[-1] - frame.index[0]).total_seconds()

    labelled = frame["class"].notna().sum() if "class" in frame.columns else 0

    return {
        "name": ref.name,
        "label": ref.label,
        "source": ref.source,
        "well_id": ref.well_id,
        "rows": rows,
        "span_seconds": span_seconds,
        "cadence_seconds": (span_seconds / (rows - 1)) if rows > 1 else float("nan"),
        "columns_present": len(present),
        "columns_missing_entirely": len(ANALOG_COLS) - len(present),
        "frozen_columns": len(frozen),
        "mean_missing_fraction": (
            sum(missing_fraction.values()) / len(ANALOG_COLS) if ANALOG_COLS else float("nan")
        ),
        "labelled_rows": int(labelled),
        "unlabelled_rows": int(rows - labelled),
    }


def audit(refs: list[InstanceRef], limit: int | None = None) -> pd.DataFrame:
    """Audit many instances. ``limit`` caps how many are read, for a quick pass."""
    selected = refs[:limit] if limit else refs
    return pd.DataFrame(audit_instance(ref) for ref in selected)


def write_report(frame: pd.DataFrame, out_path: str | Path) -> Path:
    """Persist an audit report as CSV, creating parent directories as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return out_path

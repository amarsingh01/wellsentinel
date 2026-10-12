"""Phase 7: sequence data for a recurrent model.

Phase 6 summarised each 5 minute window into a fixed vector of statistics and handed that
to a tree. A tree sees one window at a time and never sees *how* the window got there.
This module builds the other kind of input: a short stack of consecutive timesteps, so a
recurrent model can read the trajectory itself - the pressure sliding, then turning.

Three ideas carry over unchanged from the earlier phases, and one is new.

* Clean the same way (:mod:`wellsentinel.cleansing`), no shortcuts for the model.
* Keep the derived differentials and ratios (:mod:`wellsentinel.features`); they are the
  scale-relative signatures a fault leaves behind.
* Never look past the window end when labelling (:mod:`wellsentinel.horizon`).
* New: normalise per well. Wells run at wildly different absolute levels, so a raw pressure
  means nothing across wells. We divide out each well's own median and spread. That is the
  "baseline deviation" idea from ``docs/FEATURE_IDEAS.md``, made concrete.

Term note: a *timestep* is one sample after downsampling; a *sequence* is a fixed number of
consecutive timesteps fed to the model at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cleansing import clean_instance
from .constants import ANALOG_COLS
from .data import InstanceRef, load_instance
from .features import add_derived_columns
from .horizon import onset_times

#: All channels a sequence may carry, in a fixed order so the model input is stable.
DERIVED_DIFFERENTIAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("P-MON-CKP", "P-JUS-CKP"),
    ("P-MON-CKGL", "P-JUS-CKGL"),
    ("P-TPT", "P-PDG"),
    ("T-MON-CKP", "T-JUS-CKP"),
    ("P-ANULAR", "P-PDG"),
    ("P-MON-CKP", "P-MON-SDV-P"),
)
DERIVED_RATIO_PAIRS: tuple[tuple[str, str], ...] = (
    ("QGL", "QBS"),
    ("P-PDG", "P-TPT"),
)


def channel_names() -> list[str]:
    """The value channels every instance is projected onto, present or not.

    Missing channels are filled and flagged rather than dropped, so instances from
    different wells line up column for column.
    """
    names = list(ANALOG_COLS)
    names += [f"D__{a}__minus__{b}" for a, b in DERIVED_DIFFERENTIAL_PAIRS]
    names += [f"R__{a}__over__{b}" for a, b in DERIVED_RATIO_PAIRS]
    return names


@dataclass
class WellScaler:
    """Robust per-instance normaliser: subtract the median, divide by the interquartile range.

    Median and IQR are used instead of mean and standard deviation because the sensor
    series carry spikes and long frozen stretches, and we do not want either to set the
    scale. Crucially this is fitted on each well's *own* series, with no labels and no
    other well involved. Wells differ by orders of magnitude in absolute level, so a global
    scaler is meaningless here; a per-well scaler is the standard fix and it leaks nothing
    from the test fold, which is why the arrays can be built once and reused across folds.
    """

    median: dict[str, float] = field(default_factory=dict)
    iqr: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit_instance(cls, frame: pd.DataFrame, channels: list[str]) -> "WellScaler":
        """Fit on one instance's own (already cleaned, already derived) values."""
        scaler = cls()
        for channel in channels:
            if channel in frame.columns:
                column = frame[channel].to_numpy(dtype="float64")
                normal = column[~np.isnan(column)]
            else:
                normal = np.empty(0)
            if not normal.size:
                scaler.median[channel] = 0.0
                scaler.iqr[channel] = 1.0
                continue
            median = float(np.median(normal))
            q75, q25 = np.percentile(normal, [75, 25])
            spread = float(q75 - q25)
            scaler.median[channel] = median
            scaler.iqr[channel] = spread if spread > 0 else 1.0
        return scaler

    def transform_column(self, values: np.ndarray, channel: str) -> np.ndarray:
        median = self.median.get(channel, 0.0)
        spread = self.iqr.get(channel, 1.0) or 1.0
        return (values - median) / spread


def downsample(frame: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Collapse a 1 Hz instance to one row per ``seconds``.

    Sensor columns take the mean of their bin; the label takes the last non-blank value in
    the bin, because the label describes the state the bin ended in. Bins with no readings
    are left out entirely.
    """
    if seconds <= 1:
        return frame
    rule = f"{int(seconds)}s"

    analog_present = [c for c in frame.columns if c in set(channel_names())]
    out = frame[analog_present].resample(rule).mean()
    if "class" in frame.columns:
        out["class"] = frame["class"].resample(rule).last()
    if "state" in frame.columns:
        out["state"] = frame["state"].resample(rule).last()
    return out


@dataclass
class SequenceSet:
    """Model-ready arrays plus the metadata needed to score and explain them."""

    x: np.ndarray                       # (n_sequences, steps, channels)
    mask: np.ndarray                    # (n_sequences, steps, channels) 1 where input was absent
    target: np.ndarray                  # (n_sequences,) 0/1
    seconds_to_onset: np.ndarray        # (n_sequences,) NaN when no onset follows
    end_time: np.ndarray                # (n_sequences,) datetime64 of each sequence end
    well_id: np.ndarray                 # (n_sequences,) object
    instance: np.ndarray                # (n_sequences,) object
    feature_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.x.shape[0])


def sequenceize_instance(
    frame: pd.DataFrame,
    scaler: WellScaler,
    channels: list[str],
    steps: int,
    stride: int,
    horizon_seconds: int,
) -> dict[str, np.ndarray] | None:
    """Turn one cleaned instance into sequences, labelled from the window end forward.

    ``scaler`` must already be fitted for this instance. Returns ``None`` when the instance
    is too short for a single sequence.
    """
    if frame.empty or len(frame) < steps:
        return None

    values = np.full((len(frame), len(channels)), np.nan, dtype="float64")
    masks = np.zeros((len(frame), len(channels)), dtype="float32")
    for j, channel in enumerate(channels):
        if channel in frame.columns:
            raw = frame[channel].to_numpy(dtype="float64")
            masks[:, j] = np.isnan(raw).astype("float32")
            values[:, j] = scaler.transform_column(raw, channel)
        else:
            masks[:, j] = 1.0

    # A NaN that survives scaling cannot be fed to the model; zero it and flag it.
    absent = np.isnan(values)
    values[absent] = 0.0
    masks[absent] = 1.0

    onsets = onset_times(frame["class"]) if "class" in frame.columns else np.array([])
    labels = frame["class"].to_numpy(dtype="float64") if "class" in frame.columns else None

    index = frame.index.to_numpy()
    starts = range(0, len(frame) - steps + 1, stride)
    x_rows, m_rows, ends, targets, gaps = [], [], [], [], []
    for start in starts:
        stop = start + steps
        end_time = index[stop - 1]
        # Drop any window whose end already sits inside an event: predicting the present
        # is not predicting the future, and it would leak the answer.
        end_label = labels[stop - 1] if labels is not None else np.nan
        if not np.isnan(end_label) and int(end_label) != 0:
            continue
        if np.isnan(end_label):
            continue
        x_rows.append(values[start:stop])
        m_rows.append(masks[start:stop])
        ends.append(end_time)
        if len(onsets):
            gap = np.nan
            position = int(np.searchsorted(onsets, end_time, side="right"))
            if position < len(onsets):
                gap = (onsets[position] - end_time) / np.timedelta64(1, "s")
            gaps.append(gap)
            targets.append(1 if (gap == gap and 0 < gap <= horizon_seconds) else 0)
        else:
            gaps.append(np.nan)
            targets.append(0)

    if not x_rows:
        return None
    return {
        "x": np.stack(x_rows).astype("float32"),
        "mask": np.stack(m_rows).astype("float32"),
        "target": np.asarray(targets, dtype="float32"),
        "seconds_to_onset": np.asarray(gaps, dtype="float64"),
        "end_time": np.asarray(ends, dtype="datetime64[ns]"),
    }


def build_sequences(
    refs: list[InstanceRef],
    channels: list[str],
    downsample_seconds: int,
    steps: int,
    stride: int,
    horizon_seconds: int,
    max_instances: int | None = None,
) -> SequenceSet:
    """Clean, downsample and sequence a list of instances into one set of arrays.

    The normaliser is fitted per instance here, unsupervised, so the result is independent
    of which fold the instance lands in and can be built once.
    """
    chunks: list[dict[str, np.ndarray]] = []
    wells: list[str] = []
    instances: list[str] = []
    used = refs[:max_instances] if max_instances else refs
    for ref in used:
        raw = load_instance(ref)
        cleaned, _ = clean_instance(raw)
        derived = add_derived_columns(cleaned)
        reduced = downsample(derived, downsample_seconds)
        scaler = WellScaler.fit_instance(reduced, channels)
        chunk = sequenceize_instance(reduced, scaler, channels, steps, stride, horizon_seconds)
        if chunk is None:
            continue
        chunks.append(chunk)
        wells.extend([ref.well_id] * len(chunk["target"]))
        instances.extend([ref.name] * len(chunk["target"]))

    if not chunks:
        empty = np.empty((0, steps, len(channels)), dtype="float32")
        return SequenceSet(
            x=empty,
            mask=np.empty_like(empty),
            target=np.empty(0, dtype="float32"),
            seconds_to_onset=np.empty(0, dtype="float64"),
            end_time=np.empty(0, dtype="datetime64[ns]"),
            well_id=np.empty(0, dtype=object),
            instance=np.empty(0, dtype=object),
            feature_names=channels,
        )

    return SequenceSet(
        x=np.concatenate([c["x"] for c in chunks]),
        mask=np.concatenate([c["mask"] for c in chunks]),
        target=np.concatenate([c["target"] for c in chunks]),
        seconds_to_onset=np.concatenate([c["seconds_to_onset"] for c in chunks]),
        end_time=np.concatenate([c["end_time"] for c in chunks]),
        well_id=np.asarray(wells, dtype=object),
        instance=np.asarray(instances, dtype=object),
        feature_names=channels,
    )

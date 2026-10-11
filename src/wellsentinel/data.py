"""Reading the 3W dataset.

The dataset ships as one Parquet file per instance::

    <root>/<class>/<name>.parquet

The file name tells you the source of the instance (real, simulated, hand-drawn)
and, for real instances, the well id. Nothing here loads the whole dataset into
memory: 2,228 instances and ~1.9 GB of Parquet is too much for a laptop.

Term note: an *instance* is one recorded time window from one well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import SOURCE_DRAWN, SOURCE_REAL, SOURCE_SIMULATED

#: WELL-00001_20170201010207.parquet  ->  well id "WELL-00001"
_WELL_RE = re.compile(r"^(WELL-\d+)_")


@dataclass(frozen=True)
class InstanceRef:
    """A pointer to one instance on disk, with cheap metadata read from its name."""

    path: Path
    label: int
    source: str
    well_id: str
    name: str

    @property
    def is_real(self) -> bool:
        return self.source == SOURCE_REAL


def infer_source(name: str) -> str:
    """Classify a file name by where its data came from."""
    if name.startswith("WELL-"):
        return SOURCE_REAL
    if name.startswith("SIMULATED_"):
        return SOURCE_SIMULATED
    if name.startswith("DRAWN_"):
        return SOURCE_DRAWN
    return "unknown"


def infer_well_id(name: str) -> str:
    """Recover the well id when the file name carries one, else fall back to the name."""
    match = _WELL_RE.match(name)
    if match:
        return match.group(1)
    # Simulated and hand-drawn instances are not tied to a real well.
    return name


def discover(root: str | Path) -> list[InstanceRef]:
    """List every instance under ``root`` with its label and source."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"3W dataset root not found: {root}")

    refs: list[InstanceRef] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        label = int(class_dir.name)
        for parquet in sorted(class_dir.glob("*.parquet")):
            refs.append(
                InstanceRef(
                    path=parquet,
                    label=label,
                    source=infer_source(parquet.name),
                    well_id=infer_well_id(parquet.name),
                    name=parquet.stem,
                )
            )
    return refs


def load_instance(ref: InstanceRef) -> pd.DataFrame:
    """Load one instance, labelling columns as integers where they should be."""
    frame = pd.read_parquet(ref.path)
    frame.index.name = "timestamp"
    for column in ("class", "state"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame

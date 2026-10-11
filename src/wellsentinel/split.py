"""Phase 4: train / validation / test splits.

The one rule that matters: split by well, not by row. Two windows from the same well are
near duplicates, so if the same well appears in train and test, the score is a lie.

Second rule: simulated instances never reach validation or test. They are useful for
bulk training, but they are not real wells, so grading on them grades us on the simulator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from .data import InstanceRef


@dataclass
class Split:
    """Three disjoint slices of instances, plus a per-well view."""

    train: list[InstanceRef] = field(default_factory=list)
    validation: list[InstanceRef] = field(default_factory=list)
    test: list[InstanceRef] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[InstanceRef]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def wells(self) -> dict[str, set[str]]:
        return {
            name: {ref.well_id for ref in refs} for name, refs in self.as_dict().items()
        }


def _stable_unit_interval(key: str) -> float:
    """Hash a well id into [0, 1). Deterministic across machines and runs.

    Term note: hashing turns a string into a number. Same string, same number, always,
    so the split is reproducible without shipping an index file.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_by_well(
    refs: list[InstanceRef],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: str = "wellsentinel-v1",
) -> Split:
    """Assign instances to train/validation/test by hashing their well id.

    Real wells are split. Simulated and hand-drawn instances all go to train, by design.
    """
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must leave room for test")

    split = Split()
    seen: dict[str, str] = {}

    for ref in refs:
        if not ref.is_real:
            split.train.append(ref)
            continue

        bucket_key = f"{seed}:{ref.well_id}"
        position = _stable_unit_interval(bucket_key)
        if position < train_fraction:
            where = "train"
        elif position < train_fraction + validation_fraction:
            where = "validation"
        else:
            where = "test"
        seen[ref.well_id] = where
        getattr(split, where).append(ref)

    return split


def assert_disjoint(split: Split) -> None:
    """Raise if any well leaks across splits. Cheap, and worth calling every run."""
    wells = split.wells()
    names = list(wells)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = wells[left] & wells[right]
            if overlap:
                raise AssertionError(f"well leak between {left} and {right}: {sorted(overlap)}")


def split_summary(split: Split) -> pd.DataFrame:
    """A small table for the report: instances, wells and class spread per split."""
    rows = []
    for name, refs in split.as_dict().items():
        labels = pd.Series([ref.label for ref in refs])
        rows.append(
            {
                "split": name,
                "instances": len(refs),
                "wells": len({ref.well_id for ref in refs}),
                "real": sum(ref.is_real for ref in refs),
                "anomalous": int((labels != 0).sum()),
                "normal": int((labels == 0).sum()),
            }
        )
    return pd.DataFrame(rows)

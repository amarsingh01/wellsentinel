"""Wide, well-level splits for the 3W dataset.

Why this module exists. The first split we shipped (``split.py``) walked a per-class
subset of 160 instances and hashed each well id into train / validation / test. It obeyed
the one rule that matters - no well spans two splits - but it had two failures we only saw
after scoring:

* the test set collapsed to 4 wells and every instance in it was anomalous, so the model
  was never scored on a normal well and precision could not be read at all;
* the hash scattered wells at random, so a rare class could miss a split entirely.

The fix has two parts.

1. *Widen.* Use every real well, not a per-class slice. There are 40 real wells and 1,119
   real instances; the per-class subset used 160 instances and touched a handful of wells.
2. *Balance.* Order wells from rarest to most common, then deal them into the splits in a
   fixed, evenly spread cycle. Each split then gets rare classes, common classes and, most
   importantly, normal wells.

Rare classes are thin at the *well* level - classes 1, 2, 5, 6 and 8 have only 2-3 real
wells each - so a single held-out fold is still a small-sample estimate. That is why the
honest number for this project comes from leave-wells-out cross validation
(:func:`grouped_folds`), not from one lucky fold.

Term note: a *fold* is one slice used as the test set while the rest trains the model.
Cross validation repeats that with every fold taking a turn as test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .data import InstanceRef

#: WELL-00007 -> 7, so wells can be ordered deterministically by number.
_WELL_NUMBER_RE = re.compile(r"WELL-(\d+)")


@dataclass
class WellInfo:
    """What one real well looks like: the classes it carries and how big it is."""

    well_id: str
    classes: set[int] = field(default_factory=set)
    n_real: int = 0
    n_normal: int = 0
    class_instances: dict[int, int] = field(default_factory=dict)

    @property
    def has_normal(self) -> bool:
        return 0 in self.classes


@dataclass
class WideSplit:
    """A train / validation / test split plus the per-well detail behind it."""

    train: list[InstanceRef] = field(default_factory=list)
    validation: list[InstanceRef] = field(default_factory=list)
    test: list[InstanceRef] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[InstanceRef]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def wells(self) -> dict[str, set[str]]:
        return {name: {r.well_id for r in refs} for name, refs in self.as_dict().items()}


def well_number(well_id: str) -> int:
    """Numeric part of a well id, or a large number when it is not a real well."""
    match = _WELL_NUMBER_RE.search(well_id)
    return int(match.group(1)) if match else 10**9


def well_table(refs: list[InstanceRef]) -> dict[str, WellInfo]:
    """Group real instances by well; simulated and hand-drawn are excluded by design."""
    table: dict[str, WellInfo] = {}
    for ref in refs:
        if ref.is_real:
            info = table.setdefault(ref.well_id, WellInfo(ref.well_id))
            label = int(ref.label)
            info.classes.add(label)
            info.n_real += 1
            info.class_instances[label] = info.class_instances.get(label, 0) + 1
            if label == 0:
                info.n_normal += 1
    return table


def _class_well_counts(table: dict[str, WellInfo]) -> dict[int, int]:
    """For each class, how many distinct wells carry it. Drives the rarity ordering."""
    counts: dict[int, int] = {}
    for info in table.values():
        for label in info.classes:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _strata(table: dict[str, WellInfo]) -> list[list[str]]:
    """Group wells by the rarity of their rarest class, rarest stratum first.

    Used only to order wells for the allocator; the allocator itself does the balancing.
    """
    counts = _class_well_counts(table)
    grouped: dict[int, list[str]] = {}
    for well_id, info in table.items():
        rarest = min((counts[label] for label in info.classes), default=10**9)
        grouped.setdefault(rarest, []).append(well_id)
    ordered: list[list[str]] = []
    for rarest in sorted(grouped):
        ordered.append(sorted(grouped[rarest], key=well_number))
    return ordered


def _rarity_rank(table: dict[str, WellInfo]) -> dict[str, tuple[int, int, int]]:
    """Sort key per well: rarest class first, then smallest, then well number.

    Deterministic on purpose. Two machines running the same code must produce the same
    split without shipping an index file.
    """
    counts = _class_well_counts(table)
    ranks: dict[str, tuple[int, int, int]] = {}
    for well_id, info in table.items():
        rarest = min((counts[label] for label in info.classes), default=10**9)
        ranks[well_id] = (rarest, -info.n_real, well_number(well_id))
    return ranks


def _greedy_allocate(
    well_ids: list[str],
    n_buckets: int,
    weights: list[float],
    table: dict[str, WellInfo],
) -> dict[str, int]:
    """Assign wells to buckets, each to whichever bucket is currently most deficient.

    Five deficit terms, all normalised to fractions so they are comparable:

    * *class coverage* (weight 2) - how far the bucket is below its share of the wells
      carrying each of the well's classes, with rare classes counting for more via
      ``1 / n_wells_with_class``.
    * *normal coverage* (weight 1) - how far the bucket is below its share of normal
      instances. This is what stops the test bucket ending up with no normal data.
    * *normal-well coverage* (weight 1) - the same idea at well level, so a bucket does not
      rely on one well for all its normal data.
    * *well-count balance* (weight 1) - how far the bucket is below its share of wells, so
      no bucket ends up with one giant well while others hold ten.
    * *instance balance* (weight 0.5) - a tie-breaker on total instances, so buckets are
      comparable in size without a single huge well monopolising one.

    The well-count term leads on purpose: balancing the number of wells is what keeps each
    leave-wells-out fold a real test set rather than one colossal well.

    A single held-out split cannot satisfy every class: classes 1, 3 and 6 have only 2-3
    wells dataset-wide, so there may be no well to place. That is a property of 3W, not a
    bug, and the report states it plainly instead of hiding it.
    """
    class_wells = _class_well_counts(table)
    total_instances = sum(info.n_real for info in table.values()) or 1
    total_normal = sum(info.n_normal for info in table.values()) or 1
    total_normal_wells = sum(1 for info in table.values() if info.has_normal) or 1
    total_wells = len(table) or 1
    weight_sum = sum(weights) or 1.0
    fractions = [w / weight_sum for w in weights]

    have_instances = [0] * n_buckets
    have_normal = [0] * n_buckets
    have_normal_wells = [0] * n_buckets
    have_wells = [0] * n_buckets
    have_class_wells: list[dict[int, int]] = [{} for _ in range(n_buckets)]

    assignment: dict[str, int] = {}
    for well_id in well_ids:
        info = table[well_id]
        scores = []
        for b in range(n_buckets):
            class_def = 0.0
            for label in info.classes:
                n_wells = class_wells.get(label, 0)
                if n_wells >= n_buckets:
                    target = fractions[b] * n_wells
                    class_def += (target - have_class_wells[b].get(label, 0)) / n_wells
            normal_def = (fractions[b] * total_normal - have_normal[b]) / total_normal
            normal_well_def = 0.0
            if info.has_normal:
                normal_well_def = (
                    fractions[b] * total_normal_wells - have_normal_wells[b]
                ) / total_normal_wells
            well_def = (fractions[b] * total_wells - have_wells[b]) / total_wells
            size_def = (fractions[b] * total_instances - have_instances[b]) / total_instances
            scores.append(
                2.0 * class_def + normal_def + normal_well_def + well_def + 0.5 * size_def
            )
        best = max(range(n_buckets), key=lambda b: (scores[b], -have_wells[b], -b))
        assignment[well_id] = best
        have_instances[best] += info.n_real
        have_normal[best] += info.n_normal
        have_wells[best] += 1
        if info.has_normal:
            have_normal_wells[best] += 1
        for label in info.classes:
            have_class_wells[best][label] = have_class_wells[best].get(label, 0) + 1
    return assignment


def _ordered_wells(table: dict[str, WellInfo]) -> list[str]:
    """Wells ordered rarest-class-first, then by normal size, then by well number.

    Deterministic. Rare wells are placed while every bucket is still starved of them, and
    the large normal-bearing wells are placed early so their normal data is spread rather
    than dumped in whichever bucket happens to be last.
    """
    counts = _class_well_counts(table)

    def key(well_id: str):
        info = table[well_id]
        rarest = min((counts[label] for label in info.classes), default=10**9)
        return (rarest, -info.n_normal, -info.n_real, well_number(well_id))

    return sorted(table, key=key)


def _bucket_names(train_fraction: float, validation_fraction: float, test_fraction: float):
    weights = _fraction_weights(train_fraction, validation_fraction, test_fraction)
    return [weights["train"], weights["validation"], weights["test"]], ["train", "validation", "test"]


def split_by_well(
    refs: list[InstanceRef],
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> WideSplit:
    """Split real wells across train / validation / test, simulated always into train.

    A greedy allocator sends each well to the split that needs it most - for class
    coverage, for normal data and for size - so rare classes and normal wells reach every
    split wherever the dataset allows it. Simulated and hand-drawn instances are not tied
    to a real well and never belong in validation or test.
    """
    weights, names = _bucket_names(train_fraction, validation_fraction, test_fraction)
    table = well_table(refs)
    buckets = _greedy_allocate(_ordered_wells(table), len(names), weights, table)
    assignment = {well_id: names[b] for well_id, b in buckets.items()}

    split = WideSplit()
    for ref in refs:
        if not ref.is_real:
            split.train.append(ref)
            continue
        getattr(split, assignment[ref.well_id]).append(ref)
    return split


def grouped_folds(
    refs: list[InstanceRef], k: int = 5
) -> list[tuple[list[InstanceRef], list[InstanceRef]]]:
    """Leave-wells-out folds: ``k`` rounds, each yielding (train_refs, test_refs).

    Wells are dealt round-robin over the same rarest-first order used by
    :func:`split_by_well`, so every fold carries rare classes and normal wells. Real wells
    only; simulated and hand-drawn instances stay in every training set.
    """
    if k < 2:
        raise ValueError("need at least 2 folds")

    table = well_table(refs)
    ordered = _ordered_wells(table)
    # Reuse the same allocator with k equal-weight buckets: this is what makes the folds
    # balanced in class coverage and normal data, not just in well count.
    buckets = _greedy_allocate(ordered, k, [1.0] * k, table)
    fold_of = {well_id: b for well_id, b in buckets.items()}

    other = [r for r in refs if not r.is_real]
    folds: list[tuple[list[InstanceRef], list[InstanceRef]]] = []
    for fold in range(k):
        test = [r for r in refs if r.is_real and fold_of[r.well_id] == fold]
        train = [r for r in refs if r.is_real and fold_of[r.well_id] != fold] + other
        folds.append((train, test))
    return folds


def assert_disjoint(split: WideSplit) -> None:
    """Raise if any well leaks across splits. Cheap, and worth calling every run."""
    wells = split.wells()
    names = list(wells)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = wells[left] & wells[right]
            if overlap:
                raise AssertionError(
                    f"well leak between {left} and {right}: {sorted(overlap)}"
                )


def _fraction_weights(
    train_fraction: float, validation_fraction: float, test_fraction: float
) -> dict[str, int]:
    """Turn fractions into integer weights used by the greedy allocator."""
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-6:
        raise ValueError("train + validation + test fractions must sum to 1")
    scale = 20
    return {
        "train": max(1, round(train_fraction * scale)),
        "validation": max(1, round(validation_fraction * scale)),
        "test": max(1, round(test_fraction * scale)),
    }


def split_summary(split: WideSplit) -> pd.DataFrame:
    """Per split: instances, real wells, real count, normals, anomalies and class count.

    Wells are counted among real instances only; simulated and hand-drawn files each carry
    a synthetic id and would otherwise inflate the count.
    """
    rows = []
    for name, refs in split.as_dict().items():
        labels = pd.Series([ref.label for ref in refs])
        real_refs = [ref for ref in refs if ref.is_real]
        rows.append(
            {
                "split": name,
                "instances": len(refs),
                "real_instances": len(real_refs),
                "real_wells": len({ref.well_id for ref in real_refs}),
                "real": int(sum(ref.is_real for ref in refs)),
                "normal_instances": int((labels == 0).sum()),
                "anomalous_instances": int((labels != 0).sum()),
                "classes": len(set(labels.tolist())),
            }
        )
    return pd.DataFrame(rows)


def class_well_coverage(split: WideSplit, refs: list[InstanceRef]) -> pd.DataFrame:
    """Which real wells of each class landed in which split. The 'did we widen it' check."""
    table = well_table(refs)
    wells_by_class: dict[int, set[str]] = {}
    for info in table.values():
        for label in info.classes:
            wells_by_class.setdefault(label, set()).add(info.well_id)

    wells = split.wells()
    rows = []
    for label in sorted(wells_by_class):
        total = wells_by_class[label]
        row = {"class": label, "real_wells": len(total)}
        for name in ("train", "validation", "test"):
            row[f"{name}_wells"] = len(total & wells[name])
        rows.append(row)
    return pd.DataFrame(rows)

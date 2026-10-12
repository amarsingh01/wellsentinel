"""Tests for the wide well-level splits."""

from __future__ import annotations

from pathlib import Path

import pytest

from wellsentinel.data import InstanceRef
from wellsentinel.wide_split import (
    assert_disjoint,
    class_well_coverage,
    grouped_folds,
    split_by_well,
    split_summary,
    well_table,
)


def _ref(well: str, label: int, idx: int = 0, source: str = "real") -> InstanceRef:
    name = f"{well}_{idx:05d}" if source == "real" else f"SIMULATED_{idx:05d}"
    return InstanceRef(
        path=Path(f"/nonexistent/{name}.parquet"),
        label=label,
        source=source,
        well_id=well if source == "real" else name,
        name=name,
    )


def _toy_refs() -> list[InstanceRef]:
    """Six wells: three normal, one rare class, one common class, one mixed."""
    refs: list[InstanceRef] = []
    for i in range(3):
        refs.append(_ref(f"WELL-0000{i + 1}", 0, i))
    refs.append(_ref("WELL-00007", 1, 0))  # rare class 1
    refs.append(_ref("WELL-00008", 4, 0))  # common class 4
    refs.append(_ref("WELL-00009", 0, 0))  # mixed well
    refs.append(_ref("WELL-00009", 1, 1))
    refs.append(_ref("SIMULATED_1", 5, 0, source="simulated"))
    return refs


def test_no_well_spans_splits():
    split = split_by_well(_toy_refs())
    assert_disjoint(split)


def test_simulated_never_leaves_train():
    split = split_by_well(_toy_refs())
    assert all(r.is_real for r in split.validation)
    assert all(r.is_real for r in split.test)
    assert any(not r.is_real for r in split.train)


def test_every_real_instance_is_used_exactly_once():
    refs = _toy_refs()
    split = split_by_well(refs)
    seen: list[str] = []
    for name in ("train", "validation", "test"):
        seen.extend(r.name for r in split.as_dict()[name] if r.is_real)
    assert sorted(seen) == sorted(r.name for r in refs if r.is_real)


def test_rare_wells_reach_more_than_one_split():
    """With the old hash split a rare class could miss a split entirely."""
    refs = [_ref(f"WELL-{i + 1:05d}", 1, 0) for i in range(6)]
    refs.append(_ref("WELL-00099", 0, 0))
    split = split_by_well(refs)
    wells = split.wells()
    rare = {f"WELL-{i + 1:05d}" for i in range(6)}
    where = [name for name in ("train", "validation", "test") if wells[name] & rare]
    assert len(where) >= 2


def test_deterministic():
    refs = _toy_refs()
    a, b = split_by_well(refs), split_by_well(refs)
    assert sorted(r.name for r in a.test) == sorted(r.name for r in b.test)


def test_grouped_folds_cover_every_well_once():
    refs = _toy_refs()
    folds = grouped_folds(refs, k=3)
    assert len(folds) == 3
    per_fold: list[set[str]] = []
    for _, test in folds:
        assert all(r.is_real for r in test)
        per_fold.append({r.well_id for r in test})
    real_wells = {r.well_id for r in refs if r.is_real}
    # Every real well is tested exactly once, and no well is tested in two folds.
    assert set().union(*per_fold) == real_wells
    assert sum(len(fold) for fold in per_fold) == len(real_wells)


def test_grouped_folds_keep_simulated_in_train():
    refs = _toy_refs()
    for train, _ in grouped_folds(refs, k=3):
        assert any(not r.is_real for r in train)


def test_grouped_folds_reject_single_fold():
    with pytest.raises(ValueError):
        grouped_folds(_toy_refs(), k=1)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        split_by_well(_toy_refs(), 0.5, 0.2, 0.2)


def test_summary_and_coverage_shapes():
    refs = _toy_refs()
    split = split_by_well(refs)
    summary = split_summary(split)
    assert set(summary["split"]) == {"train", "validation", "test"}
    coverage = class_well_coverage(split, refs)
    assert set(coverage.columns) >= {"class", "real_wells", "test_wells"}
    assert len(well_table(refs)) == 6

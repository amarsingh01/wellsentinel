"""Tests for dataset discovery and loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wellsentinel.constants import SOURCE_DRAWN, SOURCE_REAL, SOURCE_SIMULATED
from wellsentinel.data import infer_source, infer_well_id, discover, load_instance


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("WELL-00001_20170201010207", SOURCE_REAL),
        ("SIMULATED_00001", SOURCE_SIMULATED),
        ("DRAWN_00001", SOURCE_DRAWN),
        ("something_else", "unknown"),
    ],
)
def test_infer_source(name: str, expected: str) -> None:
    assert infer_source(name) == expected


def test_infer_well_id_from_real_name() -> None:
    assert infer_well_id("WELL-00042_20170201010207") == "WELL-00042"


def test_infer_well_id_falls_back_to_name() -> None:
    assert infer_well_id("SIMULATED_00001") == "SIMULATED_00001"


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """A miniature dataset laid out exactly like 3W."""
    (tmp_path / "0").mkdir()
    (tmp_path / "1").mkdir()
    idx = pd.date_range("2020-01-01", periods=5, freq="1s")
    normal = pd.DataFrame({"P-MON-CKP": [1.0, 1.0, 1.0, 1.0, 1.0], "class": 0}, index=idx)
    fault = pd.DataFrame({"P-MON-CKP": [2.0, 2.0, 2.0, 2.0, 2.0], "class": 101}, index=idx)
    normal.to_parquet(tmp_path / "0" / "WELL-00001_20200101000000.parquet")
    fault.to_parquet(tmp_path / "1" / "SIMULATED_00001.parquet")
    return tmp_path


def test_discover_finds_labelled_instances(tiny_dataset: Path) -> None:
    refs = discover(tiny_dataset)
    assert len(refs) == 2
    by_name = {ref.name: ref for ref in refs}
    real = by_name["WELL-00001_20200101000000"]
    assert (real.label, real.source, real.is_real) == (0, SOURCE_REAL, True)
    sim = by_name["SIMULATED_00001"]
    assert (sim.label, sim.source, sim.is_real) == (1, SOURCE_SIMULATED, False)


def test_discover_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "nope")


def test_load_instance_labels_timestamps(tiny_dataset: Path) -> None:
    ref = next(r for r in discover(tiny_dataset) if r.name.startswith("WELL-"))
    frame = load_instance(ref)
    assert frame.index.name == "timestamp"
    assert len(frame) == 5

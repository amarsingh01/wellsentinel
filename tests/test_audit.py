"""Tests for the raw-data audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wellsentinel.audit import audit, audit_instance, write_report
from wellsentinel.data import InstanceRef, discover


@pytest.fixture()
def one_instance(tmp_path: Path) -> InstanceRef:
    idx = pd.date_range("2020-01-01", periods=4, freq="1s")
    frame = pd.DataFrame(
        {
            "P-MON-CKP": [1.0, 2.0, 3.0, 4.0],   # varies
            "P-JUS-CKP": [1.0, 1.0, 1.0, 1.0],   # frozen
            "T-PDG": [None, None, None, None],   # entirely missing
            "class": [0, 0, None, 1],
        },
        index=idx,
    )
    path = tmp_path / "WELL-00009_20200101000000.parquet"
    frame.to_parquet(path)
    return InstanceRef(
        path=path, label=0, source="real", well_id="WELL-00009", name=path.stem
    )


def test_audit_counts_rows_and_span(one_instance: InstanceRef) -> None:
    row = audit_instance(one_instance)
    assert row["rows"] == 4
    assert row["span_seconds"] == 3.0
    assert row["cadence_seconds"] == pytest.approx(1.0)


def test_audit_detects_frozen_and_missing(one_instance: InstanceRef) -> None:
    row = audit_instance(one_instance)
    # P-JUS-CKP is constant (frozen); T-PDG exists as a column but is all-NaN.
    # 16 analogue columns total, 3 of them are present in this file.
    assert row["frozen_columns"] >= 1
    assert row["columns_missing_entirely"] == 13


def test_audit_counts_unlabelled_tail(one_instance: InstanceRef) -> None:
    row = audit_instance(one_instance)
    assert row["labelled_rows"] == 3
    assert row["unlabelled_rows"] == 1


def test_write_report_creates_parents(tmp_path: Path, one_instance: InstanceRef) -> None:
    frame = pd.DataFrame([audit_instance(one_instance)])
    out = write_report(frame, tmp_path / "nested" / "audit.csv")
    assert out.exists()
    assert len(pd.read_csv(out)) == 1

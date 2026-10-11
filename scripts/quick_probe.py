#!/usr/bin/env python
"""Quick probe: audit a sample of the 3W dataset and print what we actually have.

This is the "are we wasting our time" pass. It reads a small number of instances,
reports per-instance reality, and writes ``reports/data_audit.csv``.

Usage::

    python scripts/quick_probe.py --data ../amd-act3/data/3W/dataset --limit 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from wellsentinel.audit import audit, write_report  # noqa: E402
from wellsentinel.data import discover  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a sample of the 3W dataset.")
    parser.add_argument("--data", required=True, help="Path to the 3W dataset directory.")
    parser.add_argument("--limit", type=int, default=40, help="How many instances to read.")
    parser.add_argument(
        "--out",
        default="reports/data_audit.csv",
        help="Where to write the CSV audit report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    refs = discover(args.data)
    print(f"Discovered {len(refs)} instances under {args.data}")

    by_source: dict[str, int] = {}
    for ref in refs:
        by_source[ref.source] = by_source.get(ref.source, 0) + 1
    print(f"By source: {by_source}")

    # One instance per class, so the sample is representative rather than the first N files.
    seen: set[int] = set()
    sample = []
    for ref in refs:
        if ref.label not in seen:
            seen.add(ref.label)
            sample.append(ref)
    remaining = [r for r in refs if r not in sample][: max(0, args.limit - len(sample))]
    sample.extend(remaining)
    print(f"Sampling {len(sample)} instances across {len(seen)} classes")

    report = audit(sample)
    out = write_report(report, args.out)

    printable = [
        "name",
        "label",
        "source",
        "rows",
        "span_seconds",
        "cadence_seconds",
        "columns_present",
        "frozen_columns",
        "unlabelled_rows",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(report[printable].to_string(index=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

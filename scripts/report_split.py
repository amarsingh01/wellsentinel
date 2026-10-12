#!/usr/bin/env python
"""Report the widened well-level split, and compare it with the old hash split.

This is the "is the split trustworthy now?" check. It prints, for the full dataset:

* how many wells and instances land in train / validation / test;
* whether every split has normal wells (the failure that broke the first baseline);
* which real wells of each class reach the test set;
* the same shape for the leave-wells-out folds used by the sequence model.

Usage::

    python scripts/report_split.py --data ../amd-act3/data/3W/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wellsentinel.data import discover  # noqa: E402
from wellsentinel.split import split_by_well as old_split  # noqa: E402
from wellsentinel.wide_split import (  # noqa: E402
    assert_disjoint,
    class_well_coverage,
    grouped_folds,
    split_by_well,
    split_summary,
    well_table,
)


def old_summary(refs) -> list[dict[str, object]]:
    """Reproduce the old per-class-160 hash split shape for comparison."""
    classes = (0, 1, 3, 4, 5, 6, 8, 9)
    subset = []
    for label in classes:
        pool = [r for r in refs if r.label == label]
        reals = [r for r in pool if r.is_real]
        sims = [r for r in pool if not r.is_real]
        picked = reals[:20] if len(reals) >= 20 else reals + sims[: 20 - len(reals)]
        subset.extend(picked[:20])
    split = old_split(subset)
    rows = []
    for name, refs_ in split.as_dict().items():
        labels = [r.label for r in refs_]
        rows.append(
            {
                "split": name,
                "instances": len(refs_),
                "wells": len({r.well_id for r in refs_}),
                "normal_instances": sum(1 for x in labels if x == 0),
                "anomalous_instances": sum(1 for x in labels if x != 0),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Report the wide split.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default="reports/split_report.json")
    args = parser.parse_args()

    refs = discover(args.data)
    table = well_table(refs)
    real = [r for r in refs if r.is_real]
    print(f"Instances {len(refs)}  real {len(real)}  real wells {len(table)}")

    wide = split_by_well(refs)
    assert_disjoint(wide)
    print("\nWide split (60/20/20 by well, rarest-first deal):")
    print(split_summary(wide).to_string(index=False))

    print("\nClass -> real wells reaching each split:")
    coverage = class_well_coverage(wide, refs)
    print(coverage.to_string(index=False))

    old = old_summary(refs)
    print("\nOld per-class-160 hash split, for comparison:")
    for row in old:
        print(f"  {row}")

    folds = []
    print(f"\nLeave-wells-out folds (k={args.folds}):")
    for i, (train, test) in enumerate(grouped_folds(refs, k=args.folds)):
        test_labels = [r.label for r in test]
        row = {
            "fold": i,
            "test_instances": len(test),
            "test_wells": len({r.well_id for r in test}),
            "test_normal_instances": sum(1 for x in test_labels if x == 0),
            "test_anomalous_instances": sum(1 for x in test_labels if x != 0),
            "train_instances": len(train),
        }
        folds.append(row)
        print(
            f"  fold {i}: test {row['test_instances']} inst / {row['test_wells']} wells "
            f"(normal {row['test_normal_instances']}, anomaly {row['test_anomalous_instances']}) "
            f"train {row['train_instances']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "wide_split": split_summary(wide).to_dict(orient="records"),
                "class_well_coverage": coverage.to_dict(orient="records"),
                "old_split": old,
                "folds": folds,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

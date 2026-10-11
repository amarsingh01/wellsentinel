#!/usr/bin/env python
"""Phase 6: the subset baseline. Fast, honest, and the go / no-go gate.

Take a bounded subset of 3W, clean it, window it, train a LightGBM baseline on real +
simulated instances, and score it on real-only wells that the model never saw.

Read the printed numbers as "is there signal here", not as a final result.

Usage::

    python scripts/run_baseline.py --data ../amd-act3/data/3W/dataset \
        --per-class 6 --window 300 --stride 300 --out reports/baseline_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from wellsentinel.baseline import (  # noqa: E402
    add_binary_target,
    build_window_frame,
    train_baseline,
)
from wellsentinel.data import discover  # noqa: E402
from wellsentinel.split import assert_disjoint, split_by_well, split_summary  # noqa: E402

#: A handful of classes we can read comfortably in a subset pass.
DEFAULT_CLASSES = (0, 1, 3, 4, 5, 6, 8, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 6 subset baseline.")
    parser.add_argument("--data", required=True, help="Path to the 3W dataset directory.")
    parser.add_argument("--per-class", type=int, default=6, help="Instances per class.")
    parser.add_argument("--window", type=int, default=300, help="Window seconds.")
    parser.add_argument("--stride", type=int, default=300, help="Stride seconds.")
    parser.add_argument("--out", default="reports/baseline_v1.json", help="JSON output path.")
    return parser.parse_args()


def take_subset(refs, per_class: int, classes=DEFAULT_CLASSES):
    """Take up to ``per_class`` instances per class, mixing real first."""
    chosen = []
    for label in classes:
        pool = [r for r in refs if r.label == label]
        reals = [r for r in pool if r.is_real]
        sims = [r for r in pool if not r.is_real]
        picked = (reals[:per_class] if len(reals) >= per_class else reals + sims[: per_class - len(reals)])
        chosen.extend(picked[:per_class])
    return chosen


def main() -> int:
    args = parse_args()
    refs = discover(args.data)
    subset = take_subset(refs, args.per_class)
    print(f"Subset: {len(subset)} instances from {args.data}")

    split = split_by_well(subset)
    assert_disjoint(split)
    print(split_summary(split).to_string(index=False))

    print("\nBuilding window features (this reads Parquet, give it a moment)...")
    train_frame = add_binary_target(build_window_frame(split.train, args.window, args.stride))
    test_frame = add_binary_target(build_window_frame(split.test, args.window, args.stride))
    print(f"Train windows: {len(train_frame)}  Test windows: {len(test_frame)}")

    result = train_baseline(train_frame, test_frame)
    metrics = result.as_dict()
    print("\nBaseline metrics (real-only test):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    if not result.importance.empty:
        print("\nTop 15 features by gain:")
        print(result.importance.head(15).to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "per_class": args.per_class,
            "window_seconds": args.window,
            "stride_seconds": args.stride,
        },
        "metrics": metrics,
        "top_features": result.importance.head(25).to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

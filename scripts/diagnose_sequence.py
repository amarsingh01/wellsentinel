#!/usr/bin/env python
"""Diagnostic: is the sequence model under-trained or genuinely signal-free?

Trains on the largest fold's training wells and reports average precision on the *training*
subset and on the held-out test wells. If train AP is high and test AP is at chance, the
model memorises wells and does not generalise. If both are at chance, there is no usable
precursor signal at this horizon with these features.

Reuses the cached arrays from ``run_sequence.py`` so it does not re-read Parquet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from wellsentinel.data import discover  # noqa: E402
from wellsentinel.sequence_model import (  # noqa: E402
    TrainConfig,
    _average_precision,
    predict,
    train_sequence_model,
)
from wellsentinel.wide_split import grouped_folds  # noqa: E402

CACHE = Path("/tmp/ws-seq/cache")
DATA = "/home/gmk/.openclaw/workspace/projects/amd-act3/data/3W/dataset"


def main() -> int:
    npz = sorted(CACHE.glob("seq_*.npz"), key=lambda p: p.stat().st_mtime)[-1]
    print(f"Using cache {npz.name}")
    data = np.load(npz, allow_pickle=True)
    x, mask = data["x"], data["mask"].astype("float32")
    target = data["target"].astype("int64")
    well_id = data["well_id"]

    refs = discover(DATA)
    real = {r.well_id for r in refs if r.is_real}
    is_real = np.array([w in real for w in well_id])
    folds = grouped_folds(refs, k=5)

    # Pick the fold with the most training wells carrying both classes.
    report = {}
    for fold, (train_refs, test_refs) in enumerate(folds):
        test_wells = {r.well_id for r in test_refs}
        train_wells = {r.well_id for r in train_refs if r.is_real}
        test_idx = np.where(np.isin(well_id, list(test_wells)) & is_real)[0]
        train_pool = np.where(np.isin(well_id, list(train_wells)) & is_real)[0]
        if len(test_idx) == 0 or len(train_pool) == 0:
            continue

        # Hold out a slice of training *wells* for early stopping.
        rng = np.random.default_rng(42)
        tw = sorted(train_wells)
        val_wells = set(rng.choice(tw, size=max(1, len(tw) // 8), replace=False).tolist())
        val_idx = np.array([i for i in train_pool if well_id[i] in val_wells])
        fit_idx = np.array([i for i in train_pool if well_id[i] not in val_wells])

        result = train_sequence_model(
            x[fit_idx], mask[fit_idx], target[fit_idx].astype("float32"),
            x[val_idx], mask[val_idx], target[val_idx].astype("float32"),
            TrainConfig(epochs=15, hidden=64, seed=42),
        )
        if not result.ok:
            report[f"fold{fold}"] = {"note": "no model"}
            continue
        fit_prob = predict(result.model, x[fit_idx], mask[fit_idx])
        test_prob = predict(result.model, x[test_idx], mask[test_idx])
        report[f"fold{fold}"] = {
            "train_wells": len(train_wells),
            "test_wells": len(test_wells),
            "fit_n": int(len(fit_idx)),
            "test_n": int(len(test_idx)),
            "train_pos_rate": round(float(target[fit_idx].mean()), 4),
            "test_pos_rate": round(float(target[test_idx].mean()), 4),
            "train_ap": round(_average_precision(target[fit_idx], fit_prob), 4),
            "test_ap": round(_average_precision(target[test_idx], test_prob), 4),
            "best_epoch": result.best_epoch,
            "history_tail": result.history[-2:],
        }
        r = report[f"fold{fold}"]
        print(f"fold {fold}: trainAP {r['train_ap']} (pos {r['train_pos_rate']}) "
              f"testAP {r['test_ap']} (pos {r['test_pos_rate']}) best_ep {r['best_epoch']}")

    Path("/tmp/ws-seq/reports/sequence_diagnostic.json").write_text(json.dumps(report, indent=2))
    print("wrote reports/sequence_diagnostic.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

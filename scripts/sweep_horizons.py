#!/usr/bin/env python
"""Sweep the prediction horizon: 1 h, 6 h, 24 h. Same data, same split, only the
label changes. Answers "how far ahead can we actually see".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/ws-p6/src")
sys.path.insert(0, "/tmp/ws-p6/scripts")

from run_horizon import build_horizon_frame, take_subset, train_and_score  # noqa: E402
from wellsentinel.data import discover  # noqa: E402
from wellsentinel.split import split_by_well  # noqa: E402

DATA = "/home/gmk/.openclaw/workspace/projects/amd-act3/data/3W/dataset"
out = {}
refs = discover(DATA)
subset = take_subset(refs, 20)
split = split_by_well(subset)
for horizon in (3600, 21600, 86400):
    train = build_horizon_frame(split.train, 300, 300, horizon)
    test = build_horizon_frame(split.test, 300, 300, horizon)
    m = train_and_score(train, test)
    out[f"{horizon}s"] = m
    print(f"horizon={horizon}s -> {m}", flush=True)
Path("/tmp/ws-p6/horizon_sweep.json").write_text(json.dumps(out, indent=2))
print("DONE")

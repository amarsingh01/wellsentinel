# WellSentinel

Early warning for undesirable events in oil wells. Ops copilot for flow assurance.

Built for the **AMD Developer Hackathon ACT III**, Track 1 (Intelligent Industry).

- **ML core** (this repo): data, cleansing, features, model training.
- **UI**: https://github.com/amarsingh01/wellsentinel-ui

## What it does

Reads live well sensor data (pressures, temperatures, flows, valve states), detects a
developing undesirable event, and gives the operator a warning *with lead time* plus a
plain explanation and a recommended action. Human review gate before any action.

## Data

3W Dataset 2.0.0 (Petrobras). 2,228 instances, 27 variables, 1 Hz.
Real (1,119), OLGA-simulated (1,089), hand-drawn (20). CC BY 4.0.

Raw data is **not** in this repo. Fetch it separately (see `docs/DATA_AND_SPLIT.md`).

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
pytest -q
python scripts/quick_probe.py --data ../amd-act3/data/3W/dataset --limit 40
```

## Layout

```
src/wellsentinel/   importable package (data, labels, features, split, train)
scripts/            runnable entry points
docs/               phase plan, data plan, feature ideas
tests/              unit + integration tests
```

## Status

Phase 0-1 in progress. See `docs/PHASES.md`.

# Phase plan

Rule for every phase: one deliverable, one check. No phase is done until the check passes.

## Phase 0 - Scaffold (today)

Do:
- Create both repos, public.
- Python 3.12 env, pinned dependencies.
- Package skeleton, tests, CI-ready layout.

Check: `pytest -q` green. Repos exist and are pushed.

## Phase 1 - Raw data audit (today)

Do:
- Load a sample of instances from all 10 classes.
- Record per instance: row count, time span, which of the 27 columns are present,
  which are frozen, missing fraction per column, real vs simulated vs drawn.
- Write the audit to `reports/data_audit.csv`.

Check: the audit file exists and the numbers match what we measured earlier
(2,228 files, 1,119 real / 1,089 simulated / 20 drawn).

## Phase 2 - Cleansing

Do:
- Standardise column names and types (labels as integers).
- Handle the label tail: drop or mask the trailing rows where class is blank.
- Fill short gaps: interpolate gaps up to N seconds (default 60).
- Flag, do not fake, long gaps: add a missing mask column.
- Frozen columns: keep as state, mark with a frozen flag, exclude from dynamics.
- Outliers: clip to physical bounds, keep a count. Do not silently delete rows.

Check: cleansed output is reproducible from raw with one command, and row counts
are accounted for (input rows = output rows + dropped label tail).

## Phase 3 - Label engineering

Do:
- Binary label: normal vs any undesirable event.
- Multi-class label: 10 classes, transient labels 101-109 folded into their parent.
- Horizon labels (later): for each window, is an event starting within 1 / 2 / 6 / 12 / 24 h.
- Never label a window using data from after the event starts. That is leakage.

Check: label distribution printed; a human can trace any single label back to raw rows.

## Phase 4 - Splits

Do:
- Split by well id, never by row. Rows from one well must not span train and test.
- Real-only holdout set for the honest number.
- Simulated instances in training only.
- Stratify so rare classes appear in test.

Check: no well id appears in two splits. Rare-class counts printed per split.

## Phase 5 - Feature engineering v1

Do:
- Window features (see `FEATURE_IDEAS.md`), v1 = the obvious, high-value ones only.
- Domain differentials and ratios first, then rolling stats.
- Every feature block is ablated. If it does not help, it does not ship.

Check: feature matrix builds for all splits with no NaN explosion, and a feature
importance plot exists.

## Phase 6 - Subset baseline (the go / no-go gate)

Do:
- Take a small subset: a few wells, binary label first.
- Train a gradient-boosted tree baseline (LightGBM).
- Report precision, recall, F1, and a lead-time number.

Check: beats a random / majority baseline clearly. If it does not, we stop and rethink
before spending more time on the data. This is the waste-of-time test you asked for.

## Phase 7 - Model improvements

Do:
- Multi-class, then horizon heads.
- Add feature blocks that earned their place in Phase 5.
- Tune, calibrate, and measure false alarms per day (the business number).

Check: false alarms per day and lead time stated, on the real-only holdout.

## Phase 8 - AMD

Do:
- Move training to AMD Developer Cloud (ROCm + PyTorch).
- Serve the model and the explainer on the AMD GPU.
- Show live GPU usage in the demo.

Check: a prediction round-trip goes through the AMD endpoint, visible in the UI.

## Phase 9 - UI

Do:
- Streamlit app: live sensor view, risk gauge, lead time, explanation, action, review gate.
- Streams a held-out well so the demo is honest, not memorised.

Check: live URL works, calls the AMD endpoint, and a reviewer can use it unaided.

## Phase 10 - Submission

Do: repo public, README that runs, cover image 16:9, demo MP4, pitch PDF, live URL.

Check: every hard gate in `PLAN.md` ticked off one by one.

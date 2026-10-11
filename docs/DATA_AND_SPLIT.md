# Data and split plan

## Source

3W Dataset 2.0.0 (Petrobras), CC BY 4.0. Live in a separate folder, never in this repo.
Default local path used by the scripts: `../amd-act3/data/3W/dataset`.

Layout: `dataset/<class>/<name>.parquet`, one file per instance.

- `WELL-*` real, 1,119 instances
- `SIMULATED_*` OLGA simulator, 1,089
- `DRAWN_*` hand-drawn by experts, 20

Labels live in the `class` column: 0 normal, 1-9 event types, 101-109 transient onset.
A separate `state` column holds well operational status.

## Columns (27 + 2 labels)

10 pressures, 4 temperatures, 2 flows, 11 valve/choke states, plus class and state.

## Known realities (measured, not assumed)

- A given real well exposes only about 6 to 9 of the 16 analogue sensors. Which set differs per well.
- Absolute ranges vary by orders of magnitude across wells. Pressure in Pa, millions.
- Valves are often constant for a whole instance (frozen). Use as context, not dynamics.
- Missing, frozen and outlier values are left untreated on purpose. Handling them is the task.
- Class imbalance is severe. Classes 0 and 4 dominate. Classes 1, 2, 7, 8 are thin.

## Cleansing rules

1. Rename and cast. Labels become integers. Timestamps stay as the index.
2. Label tail. Some instances end with rows where class is blank. Mask or drop them, and
   report the count. Never guess a label.
3. Short gaps. Interpolate gaps up to a threshold (default 60 s). Record how many values
   were filled per instance.
4. Long gaps. Do not fill. Add a missing mask column so the model can see "no data here".
5. Frozen columns. Keep the value, add a frozen flag, exclude from dynamic features.
6. Outliers. Clip to physical bounds and count them. Do not silently delete rows.

## Backfill policy

Backfill means filling gaps in the time series. We allow it only short range.

- Time based interpolation for gaps up to 60 s. This is safe, the physics moves slowly.
- Forward fill for valve and choke states. They hold their value until changed.
- Never backfill across a gap longer than 60 s, and never across a label boundary.
- Everything filled is counted, so the report can say how much of the data is synthetic fill.

## Split rules

- Split by well id, never by row. Rows from one well must not appear in two splits.
- Train: real plus simulated. Test: real only, so the number is honest.
- Keep a small real validation fold for tuning.
- Stratify so rare classes reach the test set.
- Simulated instances must never appear in test, or we are grading ourselves on fake data.

Suggested starting point: 70 / 15 / 15 by well, real only in val and test.

## Test data

A held-out set of real wells, never seen in training, used for every reported number.
The UI demo streams from this same held-out set so the demo is honest.

## Leakage checklist

- No well id across splits.
- No window labelled using data after the event starts.
- No feature computed with global statistics that saw the test set.
- Normalisation statistics fit on train only.

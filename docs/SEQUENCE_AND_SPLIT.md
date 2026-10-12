# Phase 7 - Sequence model and the widened split

Two things happen in this phase, in this order.

1. **Widen the split.** The Phase 6 subset scored on 4 wells and every one of them was
   anomalous, so precision could not be read and the whole number was unreliable. We
   rebuild the split to use every real well and to put normal wells in every split.
2. **Sequence model.** Replace the bag-of-window-statistics input with a short stack of
   consecutive timesteps, and read it with a masked GRU.

## 1. The split

`wide_split.py`. Rules, in order of priority:

- Split by **well**, never by row. A well's windows are near duplicates; if a well spans
  two splits the score is a memory test.
- Use **every** real well. The Phase 6 subset walked `per_class=20` over 8 classes and
  reached only a handful of wells.
- Put **normal wells in every split**. This is the failure that broke the first baseline.
- Real and simulated may train together. **Simulated and hand-drawn never enter validation
  or test.** They are not real wells.

Wells are placed by a greedy allocator: each well goes to the split that is currently most
deficient, weighted toward (a) covering the well's classes, counting rare classes for more,
(b) covering normal instances and normal-bearing wells, (c) balancing the number of wells,
and (d) balancing size. Wells are considered rarest-class-first, so scarce-class wells are
placed while every split is still starved of them.

There are only 40 real wells and the sizes are wildly uneven; `WELL-00002` alone holds 326
of the 1,119 real instances. The allocator cannot make a perfectly balanced three-way cut
out of that, but it does put normals in every split:

| split | real instances | real wells | normal | anomalous | classes |
|-------|---------------:|-----------:|-------:|----------:|--------:|
| train | 857 | 20 | 454 | 1512 | 10 |
| validation | 143 | 9 | 81 | 62 | 7 |
| test | 119 | 11 | 59 | 60 | 6 |

Because one held-out fold is still a small sample of 40 wells, the number we trust is not
that fold but **leave-wells-out cross validation** (`grouped_folds`, k=5). Every real
instance is scored exactly once, by a model that never saw its well. The folds are built by
the same allocator with equal weights, so each fold has normal data.

### The honest limit

Classes 1, 3 and 6 have only 2-3 real wells in the whole dataset, and class 1 has zero in
the test split. On a 40-well dataset that is arithmetic, not a modelling choice: a class
carried by three wells can sit in at most three splits, and it will often sit in fewer once
the other classes are balanced too. Every score for those classes is a small-sample guess
and is reported with that caveat rather than smoothed over.

## 2. The sequence model

`sequence.py` builds the input:

- Clean exactly as Phase 2 does. No shortcuts for the model.
- Keep the Phase 5 differentials and ratios. A fault shows up in the *gap* between two
  sensors, not in either one alone.
- **Per-instance robust normalisation**: subtract the instance's median, divide by its IQR.
  Wells run at orders-of-magnitude different absolute levels, so a raw pressure means
  nothing across wells. Fitting this on each well's own series uses no labels and no other
  well, so the arrays can be built once and reused for every fold.
- Downsample to one row per minute by default, label taken from the last row of each bin.

`sequence_model.py` reads it:

- A **masked GRU**. The mask (which channels were actually measured) is concatenated to the
  input, so the cell can tell "value is zero" from "value is absent". Real wells are missing
  about 9-11 of 16 analogue sensors, and which ones differs per well.
- Positive-class weighting in the loss, because events are the minority.
- Early stopping on validation average precision. The test fold is never touched during
  training.

`scripts/run_sequence.py` ties it together and prints both per-fold and pooled out-of-fold
metrics (`roc_auc`, `average_precision`, `precision`, `recall`, `f1`, and a lead-time
number), plus the split shape, so the run reports its own trustworthiness.

## What to read in the output

- `reports/split_report.json` - the split shape and the leave-wells-out folds.
- `reports/sequence_wide.json` - per-fold and pooled sequence-model metrics.

# Phase 7 findings - sequence model on the widened split

Date: 2026-10-13. Branch: `feat/wide-split-sequence`.

## What we did, in order

1. Widened the split so the numbers are trustworthy.
2. Built the sequence-model path (masked GRU over consecutive timesteps).
3. Scored it honestly (leave-wells-out over all 40 real wells).
4. Ablated it against a bag-of-statistics tree on the exact same windows.

## Result: the sequence model does not find onset precursors

Predicting "an undesirable event starts within 6 h", pooled out-of-fold over all 40 real
wells (7,710 real sequences, 687 positive, base rate 0.089):

| model | ROC AUC | average precision | base rate |
|-------|--------:|------------------:|----------:|
| masked GRU (sequence) | 0.412 | 0.072 | 0.089 |
| LightGBM on window statistics (same windows) | 0.436 | 0.081 | 0.089 |

Both sit at chance. A model with no skill would score ROC AUC 0.5 and average precision
equal to the base rate; the GRU is *below* chance on AUC, and neither beats the base rate on
average precision. The trajectory is not adding anything a summary does not already carry.

Per fold the AUC ranges from 0.34 to 0.61 and average precision from 0.01 to 0.57. The
spread tracks the base rate (wells differ a lot in how much event time they contain), not
transferable skill.

## Why: the model memorises wells instead of learning precursors

Trained on the largest fold's wells, the GRU reaches training average precision of about
0.31 and test average precision of 0.01 to 0.22 on wells it never saw. High on training,
near chance off it. With only 40 real wells, and a handful of them holding most of the
instances, a sequence model has plenty of ways to key on a well's signature and almost no
wells to prove it generalises.

## What this means

The earlier Phase 6 numbers (ROC AUC 0.62 at 24 h, "mean lead 12,000 s") were not a real
signal. They came from a test set of 3-4 wells where every instance was anomalous, and from
a base rate so high that "an event is coming" was usually the correct guess. The wide split
removes that illusion, and what is left is a negative result. Better to know now.

## Where signal might still be

- A few folds do reach average precision above their base rate (fold 3: 0.40 to 0.57 vs
  0.28 base). Some wells' onsets may be predictable; the question is whether a model can
  learn the pattern from 40 wells, and so far it cannot.
- Rare classes (1, 3, 6) have 2-3 real wells each. No split can give them an honest test.

## What we would try next

1. Scale the wells, not just the instances. A few large wells drive every score; a
   well-balanced or well-capped design would tell us whether the pattern is in the wells
   that currently contribute most.
2. Predict a different target: time-to-event as a regression, or event type, rather than a
   fixed-horizon binary.
3. Add the physics features the earlier phases queued (baseline deviation, spectral energy)
   and re-run the same ablation, keeping the wide split as the referee.
4. Treat the leave-wells-out number as the only number from here on.

# Feature ideas

Goal: squeeze insight out of the raw columns. Start with the obvious, high value ones,
prove each block helps, drop the rest.

Term note. A **feature** is an input number we compute from the raw data and feed to the model.
Raw columns are features too. A **differential** is simply one column minus another.

## Block 1 - Obvious, do first

1. Pressure differences across each choke and valve.
   Example: upstream minus downstream pressure at the production choke (P-MON-CKP minus P-JUS-CKP).
   A growing difference means the choke is restricting flow. This is the single most
   informative idea, because faults show up as pressure drops, not absolute pressure.
2. Pressure ratios, same pairs. Ratio is scale free, so it survives well to well range drift.
3. Temperature differences, same idea. Downstream minus upstream at the production choke.
4. Flow ratio. Gas lift flow divided by service pump flow, and gas lift flow divided by choke opening.
5. Downhole versus surface gap. P-PDG versus P-TPT tells you about the column of fluid in between.

## Block 2 - Rolling window statistics

For each analogue column, over windows of 1, 5, 15 and 60 minutes:
mean, standard deviation, min, max, range, slope (trend), and start to end change.

Multi scale matters because we predict over 1, 2, 6, 12 and 24 hour horizons.
A 1 minute window catches a fast event, a 60 minute window catches a slow one.

## Block 3 - Dynamics

First difference (how fast a value changes), lagged values, and exponential moving average.
Change rate catches spikes that a plain average hides.

## Block 4 - Spectral, for oscillatory faults

Severe slugging and flow instability show up as repeating waves. Use an FFT or wavelet
energy to describe the rhythm. This block earns its place only if the oscillatory classes
are in scope.

## Block 5 - Baseline deviation

For each well, compute a normal baseline, then measure how far the current window sits
from it. This turns an absolute value into a "compared to its own normal" number, which
is the right frame because wells differ so much.

## Block 6 - Data quality as features

Missing mask, frozen flag, gap length. Let the model know when it is guessing.

## Block 7 - Valve and choke state encodings

One hot the valve states, and treat choke openings as numbers. Context, not dynamics.

## Rules

- Every block is ablated. Add, measure, keep only what helps.
- Fit any scaler or baseline on train only.
- Report the top features each time we retrain. If a block never appears, kill it.

# Flow-cytometry data (not distributed)

The flow-cytometry application is the one part of this repository that cannot be
reproduced from what is committed here.

The flow-cytometry data analyzed in the real-data application were collected
through an EQAPOL collaboration with the National Institutes of Health
(HHSN272201700061C), as described in [Staats et al.
(2014)](https://www.sciencedirect.com/science/article/abs/pii/S0022175914001860).
The data are not included in the repository because they are not publicly
available for redistribution, so this directory is empty.

Everything else — all six simulation figures — reproduces from committed data
with no external inputs.

This file documents what the published analysis did, and so the same code can be pointed at other data.

## What the analysis read

Normalised marker measurements for 10,000 cells across 14 markers, each scaled
to [0, 1] — the estimator partitions the unit square.

The published figure uses two of those markers, `CD45RO` and `CD27`, which are
`fit_flow.py`'s defaults: run with no arguments, it reproduces the published fit.
Markers are named by column, so another pair is selected by name with
`--markers` (for example `--markers CD3 CD8` for the CD3 and CD8 markers); any
two of the fourteen can be fitted.


## How the figure was produced

`src/data_analysis/fit_flow.py` runs the whole fit -- load the data, select a marker
pair, fit both OPT estimators and the Beta-tree histogram -- and caches the
result; `src/data_analysis/plot_flow.py` then draws the figure from that cache in
seconds. Fitting also needs R with the BetaTree package:

```r
devtools::install_github("zq00/BetaTree")
```

## Settings

The values behind the published figure, and `fit_flow.py`'s defaults:

| Setting | Value |
|---|---|
| `n_mc_samples` | 10000 Monte Carlo draws |
| `n_states` | 2 |
| `RHO` | `[[0.5, 0.5], [0, 1]]`, i.e. prior stopping probability 0.5 |
| `lambda_vals` | `[0.5, 0.5]` -- prior on which dimension a node splits |
| `alpha0` | `exp(20)` -- concentration in the stopping state |
| grid | 400 x 400 on `[0, 1]^2` |
| base seed | 23 |
| depths | 8 and 10 |


## The published figure

`results/figures/2D_CD45RO_CD27_betatree.jpg` is committed, so the figure the
paper shows is present in the repository even though it cannot be rebuilt here.

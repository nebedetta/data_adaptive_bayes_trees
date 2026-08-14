# Pólya Trees with Data-adaptive Splitting

Code and data accompanying the paper **"A partial likelihood approach to
tree-based density modeling and its applications in Bayesian inference"**.

The repository contains the code and processed results needed to reproduce the
simulation figures reported in the paper. All simulation figures can be
regenerated from the committed files in **under one minute on a standard
laptop**, without access to a computing cluster or external data downloads.

The flow-cytometry application is the only exception: the underlying data cannot
be redistributed or shared on request. The corresponding figure is therefore
committed to the repository, and the code that produced it is provided. See
[Flow-cytometry application](#flow-cytometry-application).

## Quick start

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then regenerate all reproducible figures:

```bash
./make_all_figures.sh
```

The figures are written to `results/figures/`.

## Repository structure

| Path                 | Contents                                                                                                                    |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `results/`           | Processed results shipped with the repository, including aggregated simulation metrics and the figures generated from them. |
| `data/`              | Data required to generate the figures. See [Data](#data).                                                                   |
| `src/core/`          | Implementations of the 1D, 2D, and *n*D Optional Pólya Trees, together with the KDE and DPM baselines and common utilities. |
| `src/make_figures/`  | Scripts for generating the simulation figures.                                                                              |
| `src/data_analysis/` | Code for the flow-cytometry application.                                                                                    |
| `src/`               | Simulation and data-processing entry points, including `aggregate_tables.py`.                                               |
| `scripts/`           | SLURM scripts used to run the simulation experiments.                                                                       |
| `distributions/`     | Simulation scenarios and sample/density generators.                                                                         |

The 1D, 2D, and 3D experiments use separate simulation scripts. These
implementations were developed incrementally as the dimensionality of the
experiments increased and are therefore maintained separately in the repository.

## Reproducing the figures

`./make_all_figures.sh` regenerates the six simulation figures in
`results/figures/`.

Individual figures can also be generated with:

```bash
python src/make_figures/make_1D_figures.py
python src/make_figures/make_2D_metric_figures.py
python src/make_figures/make_2D_posterior_figure.py
python src/make_figures/make_3D_figure.py
```

The scripts generate:

* `1D_L1_complete`
* `1D_L2_complete`
* `2D_L1`
* `2D_L2`
* `2D_itern15`
* `3D_L1_L2`

The flow-cytometry figure is handled separately as described below.

All figure-generation scripts resolve paths relative to the repository and can
therefore be run from any working directory.

## Flow-cytometry application

`2D_CD45RO_CD27_betatree.jpg` is the only figure in the paper that cannot be
regenerated from the committed data.

The flow-cytometry data analyzed in the real-data application were collected
through an EQAPOL collaboration with the National Institutes of Health
(HHSN272201700061C), as described in [Staats et al.
(2014)](https://www.sciencedirect.com/science/article/abs/pii/S0022175914001860).
The data are not included in the repository because they are not publicly
available for redistribution.

`data/flow/` contains a README documenting the data requirements and the
settings the published run used.

The code that produced it is provided in:

```text
src/data_analysis/fit_flow.py
src/data_analysis/plot_flow.py
```

`fit_flow.py` fits the estimators and caches the result; `plot_flow.py` draws the
figure from that cache. The analysis can therefore be read, and the same code
applied to other measurements: `--data-file` points at the data and `--markers`
selects the pair of columns to fit.

`make_all_figures.sh` skips this figure with an explanation when the data is
absent, so a clean checkout still completes. The committed figure is retained in
`results/figures/` so that a clean checkout still contains all figures reported
in the paper.

## Data

The repository contains approximately 15 MB of committed data.

| Directory                           | Contents                                                                                                                                                                                   |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `results/aggregated_metric_tables/` | Aggregated simulation metrics used to generate the metric figures. These contain the mean, standard deviation, and 2.5% and 97.5% quantiles of the log distances by depth and sample size. |
| `data/true_density/`                | The six 2D true densities used in the figures. They are stored at six significant figures.                                                                                                 |
| `data/2D/fits/`                     | Six per-iteration 2D simulation outputs required for `2D_itern15`, which displays posterior mean surfaces rather than aggregated metrics.                                                  |
| `data/flow/`                        | README documenting the unavailable flow-cytometry data.                                                                                                                                    |
| `distributions/grid_points_*`       | The evaluation grids the true densities are computed on: 100,000 points on [0, 1] in 1D, 500 x 500 in 2D and 50^3 in 3D, the latter two on [1e-5, 1-1e-5] to avoid the boundary singularities of the Beta-family densities. |

The seven 1D true densities are analytic and are evaluated directly from the
scenario specifications when the figures are generated rather than stored as
separate files.

The full per-iteration simulation outputs are not distributed because of their
size. A complete simulation run produces:

```text
<dim>/fits/<scenario>/       fitted trees
<dim>/metrics/<scenario>/    per-iteration distances
```

The fitted trees occupy approximately 4.7 GB and the per-iteration metric files
approximately 37 MB. They can be regenerated with the scripts in `scripts/`
(see [Full simulations](#3-full-simulations)), and are also available from the
authors upon request.

## Reproducing the full simulation results

The repository supports three levels of the computational workflow:

### 1. Figures from committed results

This is the default reproducibility workflow described above. It requires no
cluster and takes under one minute.

```text
committed results/data
        ↓
   figure scripts
        ↓
      figures
```

### 2. Aggregated metrics from per-iteration results

The aggregated tables shipped in `results/aggregated_metric_tables/` were
computed from the per-iteration metric files.

The aggregation procedure is provided in:

```text
src/aggregate_tables.py
```

Given the complete simulation output, the tables can be regenerated with:

```bash
python src/aggregate_tables.py --input-root /path/to/output
```

To check the resulting tables against the committed versions without overwriting
them:

```bash
python src/aggregate_tables.py --input-root /path/to/output --check
```

The `--check` option re-aggregates the supplied per-iteration results and
compares them with the committed tables. Timing columns are excluded from the
comparison because they record wall-clock time and are not used in the reported
results.

### 3. Full simulations

The full simulations require a computing cluster with SLURM. The array-job
scripts used to generate the reported results are provided in `scripts/`. Each
configuration was run as a 200-job SLURM array.

The sample-generation scripts are in `distributions/`. They take no arguments;
the scenarios to write are set at the top of each file. For example:

```bash
Rscript distributions/generating_1D_samples.R
```

The corresponding 2D and 3D generators are organized analogously.

The estimator simulations are run using the scripts in `scripts/1D/`,
`scripts/2D/`, and `scripts/3D/`.

A 1D simulation can be submitted with:

```bash
sbatch --export=ALL,DIST_NAME=1D_beta64 scripts/1D/1D_sim_speed.sh
```

A 2D simulation can be submitted with:

```bash
./scripts/2D/2D_run_sequentially.sh 2D_mix14
```

`2D_run_sequentially.sh` submits the three 2D sample sizes as a dependent chain
(500, then 5000, then 50000, each starting only if the previous finished) with
the arguments the three array scripts expect. It is run from a login node, since
it submits rather than computes.

A 3D simulation can be submitted with:

```bash
sbatch scripts/3D/3D_sim.sh 3D_mix1_input distributions/3D_mix1_input.py \
    output/3D/fits output/3D/metrics 8 17 1000 \
    distributions/true_density distributions/samples
```

The 1D benchmarks are four methods — KDE with Scott's rule, KDE with
cross-validated bandwidth, a Dirichlet process mixture, and DR-BART — run as
three arrays, one per method family, each taking a scenario and a sample size.
DR-BART is fitted through R: `src/sim_run_baselines_1D_drbart.py` shells out to
`src/run_drbart_1D.R` once per fit.

```bash
sbatch --export=ALL,DIST_NAME=1D_beta64,N=5000 scripts/1D/1D_baselines_kde_sim.sh
sbatch --export=ALL,DIST_NAME=1D_beta64,N=5000 scripts/1D/1D_baselines_dpm_sim.sh
sbatch --export=ALL,DIST_NAME=1D_beta64,N=5000 scripts/1D/1D_baselines_drbart_sim.sh
```

The baseline jobs have no trees to write, so their densities go to
`1D/fits_baselines/` and their distances to `1D/metrics_baselines/`.

The exact arguments and output locations are documented in the corresponding
scripts.

## Requirements

The main environment uses:

* Python 3.13.5
* R 4.3.2 for sample generation for selected scenarios and the Beta-tree baseline

Python dependencies are pinned in `requirements.txt`. A Conda environment is also
provided:

```bash
conda env create -f environment.yml
conda activate pt_sim
```

Only the following Python packages are required to regenerate the figures from
the committed results:

* numpy
* pandas
* matplotlib
* scipy

The additional dependencies listed in `requirements.txt` are required for
rerunning the simulations and/or the flow-cytometry analysis.

Two baselines come from external R packages, both needed only to rerun the
benchmarks. Neither is on CRAN, so there is no R lock file; the commits the
reported results were produced with are pinned here instead:

```r
# Beta-tree baseline (BetaTree 1.0.0)
devtools::install_github("zq00/BetaTree@9e4f31212732d21abe249e7a0138bd4a0eca71ff")

# DR-BART baseline (drbart 0.0.0.9000)
devtools::install_github("vittorioorlandi/drbart@66178ffdbf75c2a0a499cea917f9526073561f98")
```

Omitting the `@<commit>` suffix installs each package's current default
branch, which may differ from what was used here.

## Notes on the simulation outputs

The 1D experiments can be run two ways, and the aggregated tables come in two
matching layouts:

* `by_statistic/`: one file per statistic, with sample size included in the
  column names. Produced by `scripts/1D/1D_sim_speed.sh`, which fits and reduces
  in a single job.
* `by_sample_size/`: one file per scenario and sample size. Produced by the
  two-stage route — `scripts/1D/1D_sim.sh` fits the trees with
  `src/sim_run_1D.py`, then `scripts/1D/1D_sim_process.sh` reduces them with
  `src/sim_process_1D.py`, whose sample sizes and depths are set inside
  `sim_run_1D.py` rather than passed in.

The two are complementary: `by_statistic/` covers all seven scenarios, and
`by_sample_size/` supplies the 500/5000 runs for three of them.
`src/make_figures/make_1D_figures.py` reconciles the layouts when generating the
figures; the mapping is documented in `BY_SAMPLE_SIZE_RUNS`.

`KDE_scott` is fitted with the bandwidth computed in
`src/core/baseline_functions.py`, `n^{-1/(d+4)}` times the mean standard
deviation of the data. sklearn's own `bandwidth="scott"` string omits the
standard-deviation factor and oversmooths badly on unstandardized data.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).

`BetaTree` and `drbart` are external dependencies and are not distributed as part
of this repository; their respective licenses therefore apply to those
dependencies.

"""
sim_run_baselines_1D_drbart.py
--------------------------------
Runs DR-BART (Orlandi &
Murray, https://github.com/vittorioorlandi/drbart) on one (scenario,
sample size, iteration) combination, coerced into unconditional density
estimation by passing a constant covariate x (see src/run_drbart_1D.R).

Scenario, sample size, and iteration are independent CLI arguments --
each combination is run as its own invocation (one process per n, not a
loop over a comma-separated sample_sizes list), so scenarios and sample
sizes can be parallelized separately (e.g. one SLURM array task per n).
Mirrors sim_run_baselines_1D_kde.py / sim_run_baselines_1D_dpm.py's
structure/conventions exactly.

There's no Python binding for DR-BART and no in-process R bridge here
(rpy2 isn't installed), so this shells out to src/run_drbart_1D.R
via subprocess per fit.

Metrics are written as a long/tidy CSV -- one row per method ("DRBART"
here), one column group per sample size -- consistent with the KDE/DPM-only
files' format, so per-baseline metric CSVs stay column-compatible if ever
merged. Fitted density arrays are saved separately as compressed .npz
files, one per n.

Usage (cluster / command line):
    python sim_run_baselines_1D_drbart.py <iteration> <n>
        <dist_module_name> <dist_module_path>
        <metric_output_path> <sim_output_path>
        <density_folder> <samples_folder>
        [<drbart_nburn> <drbart_nsim> <drbart_variance> <drbart_max_n>]

    Optional arguments (defaults: 2000, 2000, "ux", 50000).
    drbart_max_n: max sample size for which DR-BART is run (set to 0 to skip).

Example (one scenario, one sample size, one iteration):
    python sim_run_baselines_1D_drbart.py 0 5000 \\
        1D_spikemix distributions/1D_spikemix_input.py \\
        output/baselines/metrics output/baselines/sim \\
        distributions/true_density distributions/samples

To run several sample sizes or several scenarios, invoke this script once
per (scenario, n) pair -- e.g. from a shell loop or SLURM array -- rather
than looping inside Python. run_one() / main() below can also be imported
and called directly in a notebook for interactive use across a list of n.

Output CSV (one file per iteration, shared across all n for that
scenario/iteration -- repeated invocations at different n append column
groups to the same file):
    <metric_output_path>/<prefix>/<prefix>_iter<iteration>_baseline_drbart_metric.csv

    One row (method="DRBART"); columns per sample size n:
        {n}_L1, {n}_L2, {n}_Linf, {n}_time
            (NaN if n > drbart_max_n, or if the R fit fails)

Output .npz files (one per n per iteration), compressed:
    <sim_output_path>/<prefix>/
        {prefix}_n{n}_iter{iteration}_DRBART.npz  -> density
            (only written if n <= drbart_max_n and the R fit succeeds)
"""

import sys
import os
import time
import shutil
import tempfile
import subprocess
import importlib.util
import numpy as np
import pandas as pd

from src.core import numerical_analysis_functions as na

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRBART_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_drbart_1D.R")

# subprocess.run() doesn't inherit an interactive shell's PATH (Jupyter/Anaconda
# kernels are commonly launched without it), so Rscript must be located explicitly
# rather than relying on the bare command name resolving via PATH.
RSCRIPT_BIN = shutil.which("Rscript") or "/usr/local/bin/Rscript"

METHODS = ["DRBART"]


# ── helpers ───────────────────────────────────────────────────────────────────

def import_distribution_module(dist_module_name, dist_module_path):
    spec = importlib.util.spec_from_file_location(dist_module_name, dist_module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_metrics(den, grid, true_den):
    """
    Return (L1, L2, Linf) using na.l1_distance/l2_distance/linfty_distance --
    the same distance functions sim_run_1D_speed.py uses for OPT, so
    baseline metrics are numerically comparable to OPT's.
    dx is the true grid spacing (grid[1] - grid[0]), matching
    na.distance_metric's own convention -- NOT 1/n_eval, which silently
    assumes a unit-length domain.
    """
    dx = grid[1] - grid[0]
    L1   = na.l1_distance(true_den, den, dx)
    L2   = na.l2_distance(true_den, den, dx)
    Linf = na.linfty_distance(true_den, den, dx)
    return L1, L2, Linf


def save_density(path, **arrays):
    """Save density arrays to a compressed .npz file, overwriting if it already exists."""
    np.savez_compressed(path, **arrays)


def load_or_init_metric_df(metric_filepath):
    if os.path.exists(metric_filepath):
        try:
            df = pd.read_csv(metric_filepath)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if "method" not in df.columns or len(df) == 0:
        df = pd.DataFrame({"method": METHODS})
    for method in METHODS:
        if method not in df["method"].values:
            df = pd.concat(
                [df, pd.DataFrame({"method": [method]})], ignore_index=True
            )
    return df


def method_row_idx(df, method):
    return df.index[df["method"] == method][0]


def run_drbart(samples, grid, nburn, nsim, nthin, variance, seed):
    """
    Fit DR-BART unconditionally by shelling out to src/run_drbart_1D.R.

    There's no Python binding for DR-BART and no in-process R bridge here
    (rpy2 isn't installed), so this writes the sample and grid to temp CSVs,
    calls Rscript run_drbart_1D.R \ <y_csv> <grid_csv> <out_csv> \
    <nburn> <nsim> <nthin> <seed>
    and reads the resulting density CSV back in.

    Returns
    -------
    density : (n_eval,) array, or None if the R fit failed
    elapsed : wall-clock time in seconds
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        y_path    = os.path.join(tmpdir, "y.csv")
        grid_path = os.path.join(tmpdir, "grid.csv")
        out_path  = os.path.join(tmpdir, "density.csv")
        pd.DataFrame({"y": samples.ravel()}).to_csv(y_path, index=False)
        pd.DataFrame({"y": grid}).to_csv(grid_path, index=False)

        result = subprocess.run(
            [
                RSCRIPT_BIN,
                DRBART_SCRIPT,
                y_path,
                grid_path,
                out_path,
                str(nburn),
                str(nsim),
                str(nthin),
                str(seed),
            ],
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        if result.returncode != 0 or not os.path.exists(out_path):
            print("  [DR-BART FAILED]")
            print(result.stderr[-2000:])
            return None, elapsed

        # Surface the R script's own sampling/predict phase timing (see
        # run_drbart_1D.R's cat() call) so it lands in the SLURM .out log --
        # otherwise subprocess.run(capture_output=True) discards stdout on success.
        if result.stdout.strip():
            print(f"  [R] {result.stdout.strip()}")

        density = pd.read_csv(out_path)["density"].to_numpy()
        return density, elapsed


# ── core single-(scenario, n, iteration) run ────────────────────────────────────

def run_one(iteration, n,
            dist_module_name, dist_module_path,
            metric_output_path, sim_output_path,
            density_folder, samples_folder,
            drbart_nburn=10000,
            drbart_nsim=1000,
            drbart_nthin=10,
            drbart_variance="ux",
            drbart_max_n=50000):
    """
    Run DR-BART for exactly one (scenario, sample size, iteration)
    combination, updating that scenario/iteration's metric CSV in place
    (adding/overwriting only the {n}_* columns) and writing that
    combination's .npz density file.

    This is the single source of truth for DR-BART's default
    nburn/nsim/nthin/variance/max_n -- main() and the CLI entry point below
    both pass these through rather than re-declaring their own defaults.

    Parameters
    ----------
    iteration            : int, index of the dataset (0-199)
    n                     : int, sample size, e.g. 500, 5000, or 50000
    dist_module_name      : str, e.g. '1D_beta64'
    dist_module_path      : str, path to the distribution .py file
    metric_output_path    : str, root folder for metric output CSVs
    sim_output_path       : str, root folder for density .npz files
    density_folder        : str, folder containing true density CSVs
    samples_folder        : str, root folder containing per-distribution sample CSVs
    drbart_nburn          : int, MCMC burn-in draws (default 10000)
    drbart_nsim           : int, retained posterior draws (default 1000)
    drbart_nthin          : int, thinning interval; total post-burn iterations
                             are drbart_nsim * drbart_nthin (default 10)
    drbart_variance       : str, drbart's variance mode -- "ux", "x", or
                             "const" (default "ux")
    drbart_max_n          : int, max n for which DR-BART is run (default 50000)
                            Set to 0 to skip DR-BART entirely.

    Returns
    -------
    prefix : str, the scenario prefix (for logging by callers that loop
             over scenarios themselves)
    """

    # ── load distribution module ──────────────────────────────────────────────
    dist_module = import_distribution_module(dist_module_name, dist_module_path)
    prefix = dist_module.prefix

    # ── load true density and grid (1D) ───────────────────────────────────────
    density_path = os.path.join(density_folder, f"{prefix}_true_density.csv")
    density_file = pd.read_csv(density_path)
    grid     = density_file["y"].to_numpy()
    true_den = density_file["true_density"].to_numpy()

    # ── output paths ──────────────────────────────────────────────────────────
    dir_metric_path = os.path.join(metric_output_path, prefix)
    os.makedirs(dir_metric_path, exist_ok=True)

    dir_sim_path = os.path.join(sim_output_path, prefix)
    os.makedirs(dir_sim_path, exist_ok=True)

    metric_filepath = os.path.join(
        dir_metric_path, f"{prefix}_iter{iteration}_baseline_drbart_metric.csv"
    )
    # Loaded (not overwritten) so repeated invocations at different n for the
    # same scenario/iteration accumulate column groups in one shared file.
    metric_df = load_or_init_metric_df(metric_filepath)

    start_time = time.time()

    sample_path = os.path.join(
        samples_folder, prefix, f"{prefix}_n{n}_iter{iteration}.csv"
    )
    samples = pd.read_csv(sample_path).to_numpy()   # (n, 1)

    npz_drbart = os.path.join(dir_sim_path,
        f"{prefix}_n{n}_iter{iteration}_DRBART.npz")

    row = method_row_idx(metric_df, "DRBART")

    # ── DR-BART ──────────────────────────────────────────────────────────────
    run_drbart_now = (drbart_max_n > 0) and (n <= drbart_max_n)
    if run_drbart_now:
        print(f"  [{prefix}, iter={iteration}, n={n}] Running DR-BART "
              f"(nburn={drbart_nburn}, nsim={drbart_nsim}, nthin={drbart_nthin}, "
              f"variance={drbart_variance}) ...", flush=True)
        drbart_den, drbart_time = run_drbart(
            samples, grid, drbart_nburn, drbart_nsim, drbart_nthin, drbart_variance, seed=iteration,
        )
        if drbart_den is None:
            drbart_L1 = drbart_L2 = drbart_Linf = np.nan
        else:
            drbart_L1, drbart_L2, drbart_Linf = compute_metrics(drbart_den, grid, true_den)
            save_density(npz_drbart, density=drbart_den)
            print(f"    DR-BART    "
                  f"L1={drbart_L1:.4f}  L2={drbart_L2:.4f}  "
                  f"Linf={drbart_Linf:.4f}  time={drbart_time:.2f}s", flush=True)
    else:
        drbart_L1 = drbart_L2 = drbart_Linf = np.nan
        drbart_time = np.nan
        print(f"  [{prefix}, iter={iteration}, n={n}] Skipping DR-BART "
              f"(n={n} > drbart_max_n={drbart_max_n})", flush=True)

    metric_df.at[row, f"{n}_L1"]   = drbart_L1
    metric_df.at[row, f"{n}_L2"]   = drbart_L2
    metric_df.at[row, f"{n}_Linf"] = drbart_Linf
    metric_df.at[row, f"{n}_time"] = drbart_time

    metric_df.to_csv(metric_filepath, index=False, header=True)

    elapsed = time.time() - start_time
    print(f"  [{prefix}, iter={iteration}, n={n}] done in {elapsed:.1f}s", flush=True)
    return prefix


# ── main (convenience wrapper for looping locally over several n) ──────────────

def main(iteration, sample_size_vec,
         dist_module_name, dist_module_path,
         metric_output_path, sim_output_path,
         density_folder, samples_folder,
         drbart_nburn=None,
         drbart_nsim=None,
         drbart_nthin=None,
         drbart_variance=None,
         drbart_max_n=None):
    """
    Convenience wrapper that calls run_one() once per n in sample_size_vec,
    for interactive/notebook use. On the cluster, prefer invoking this
    script once per (scenario, n) pair directly (see module docstring)
    so scenarios and sample sizes can be parallelized independently.

    drbart_nburn/nsim/nthin/variance/max_n default to None here and are
    omitted from the run_one() call when unset, so run_one()'s own defaults
    (the single source of truth) apply rather than being re-declared.
    """
    start_time = time.time()
    overrides = {
        "drbart_nburn": drbart_nburn,
        "drbart_nsim": drbart_nsim,
        "drbart_nthin": drbart_nthin,
        "drbart_variance": drbart_variance,
        "drbart_max_n": drbart_max_n,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    for n in sample_size_vec:
        run_one(
            iteration=iteration, n=n,
            dist_module_name=dist_module_name, dist_module_path=dist_module_path,
            metric_output_path=metric_output_path, sim_output_path=sim_output_path,
            density_folder=density_folder, samples_folder=samples_folder,
            **overrides,
        )
    end_time = time.time()
    print(f"Total execution time for iteration {iteration}, "
          f"n in {sample_size_vec}: {end_time - start_time:.1f}s", flush=True)


# ── cluster entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":

    REQUIRED_ARGS = 8

    if len(sys.argv) < REQUIRED_ARGS + 1:
        print(
            "Usage: python sim_run_baselines_1D_drbart.py "
            "<iteration> <n> "
            "<dist_module_name> <dist_module_path> "
            "<metric_output_path> <sim_output_path> "
            "<density_folder> <samples_folder> "
            "[<drbart_nburn> <drbart_nsim> <drbart_nthin> <drbart_variance> <drbart_max_n>]"
        )
        sys.exit(1)

    iteration          = int(sys.argv[1])
    n                  = int(sys.argv[2])
    dist_module_name   = sys.argv[3]
    dist_module_path   = sys.argv[4]
    metric_output_path = sys.argv[5]
    sim_output_path    = sys.argv[6]
    density_folder     = sys.argv[7]
    samples_folder     = sys.argv[8]

    # Only override run_one()'s own defaults (the single source of truth)
    # when a CLI arg is actually supplied.
    overrides = {}
    if len(sys.argv) > 9:  overrides["drbart_nburn"]    = int(sys.argv[9])
    if len(sys.argv) > 10: overrides["drbart_nsim"]     = int(sys.argv[10])
    if len(sys.argv) > 11: overrides["drbart_nthin"]    = int(sys.argv[11])
    if len(sys.argv) > 12: overrides["drbart_variance"] = sys.argv[12]
    if len(sys.argv) > 13: overrides["drbart_max_n"]    = int(sys.argv[13])

    run_one(
        iteration           = iteration,
        n                   = n,
        dist_module_name    = dist_module_name,
        dist_module_path    = dist_module_path,
        metric_output_path  = metric_output_path,
        sim_output_path     = sim_output_path,
        density_folder      = density_folder,
        samples_folder      = samples_folder,
        **overrides,
    )

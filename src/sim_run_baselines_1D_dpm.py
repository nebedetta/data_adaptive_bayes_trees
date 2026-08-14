"""
sim_run_baselines_1D_dpm.py
----------------------------
Runs the variational Bayes DPGMM (BayesianGaussianMixture) on one
(scenario, sample size, iteration) combination.

Scenario, sample size, and iteration are independent CLI arguments --
each combination is run as its own invocation (one process per n, not a
loop over a comma-separated sample_sizes list), so scenarios and sample
sizes can be parallelized separately (e.g. one SLURM array task per n).
Mirrors sim_run_baselines_1D_kde.py's structure/conventions exactly.

Metrics are written as a long/tidy CSV -- one row per method ("DPM" here),
one column group per sample size -- consistent with the KDE-only file's
format, so per-baseline metric CSVs stay column-compatible if ever merged.
Fitted density arrays and fit diagnostics are saved separately as
compressed .npz files, one per n.

Usage (cluster / command line):
    python sim_run_baselines_1D_dpm.py <iteration> <n>
        <dist_module_name> <dist_module_path>
        <metric_output_path> <sim_output_path>
        <density_folder> <samples_folder>
        [<dpm_max_components> <dpm_max_iter> <dpm_covariance>
         <dpm_alpha> <dpm_n_init> <dpm_init_params> <dpm_tol>]

    Optional arguments default to src.core.baseline_functions's
    DPM_DEFAULT_* constants (currently K=50, max_iter=5000, cov="full",
    alpha=1.0, n_init=1, init_params="kmeans", tol=1e-4).

Example (one scenario, one sample size, one iteration):
    python sim_run_baselines_1D_dpm.py 0 5000 \\
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
    <metric_output_path>/<prefix>/<prefix>_iter<iteration>_baseline_dpm_metric.csv

    One row (method="DPM"); columns per sample size n:
        {n}_L1, {n}_L2, {n}_Linf, {n}_time,
        {n}_active_k, {n}_converged, {n}_n_iter

Output .npz files (one per n per iteration), compressed:
    <sim_output_path>/<prefix>/
        {prefix}_n{n}_iter{iteration}_DPM.npz  -> density, active_k, converged, n_iter
"""

import sys
import os
import time
import importlib.util
import numpy as np
import pandas as pd

from src.core import numerical_analysis_functions as na
from src.core import baseline_functions as bf

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

METHODS = ["DPM"]


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


# ── core single-(scenario, n, iteration) run ────────────────────────────────────

def run_one(iteration, n,
            dist_module_name, dist_module_path,
            metric_output_path, sim_output_path,
            density_folder, samples_folder,
            dpm_max_components=bf.DPM_DEFAULT_K,
            dpm_max_iter=bf.DPM_DEFAULT_MAX_ITER,
            dpm_covariance=bf.DPM_DEFAULT_COV,
            dpm_alpha=bf.DPM_DEFAULT_ALPHA,
            dpm_n_init=bf.DPM_DEFAULT_N_INIT,
            dpm_init_params=bf.DPM_DEFAULT_INIT_PARAMS,
            dpm_tol=bf.DPM_DEFAULT_TOL):
    """
    Run DPM for exactly one (scenario, sample size, iteration) combination,
    updating that scenario/iteration's metric CSV in place (adding/
    overwriting only the {n}_* columns) and writing that combination's
    .npz density file.

    Parameters
    ----------
    iteration           : int, index of the dataset (0-199)
    n                    : int, sample size, e.g. 500, 5000, or 50000
    dist_module_name     : str, e.g. '1D_beta64'
    dist_module_path     : str, path to the distribution .py file
    metric_output_path   : str, root folder for metric output CSVs
    sim_output_path      : str, root folder for density .npz files
    density_folder       : str, folder containing true density CSVs
    samples_folder       : str, root folder containing per-distribution sample CSVs
    dpm_max_components   : int, truncation level K (default bf.DPM_DEFAULT_K)
    dpm_max_iter         : int, max EM iterations (default bf.DPM_DEFAULT_MAX_ITER)
    dpm_covariance       : str, covariance type (default bf.DPM_DEFAULT_COV)
    dpm_alpha            : float, weight_concentration_prior (default bf.DPM_DEFAULT_ALPHA)
    dpm_n_init           : int, number of initializations (default bf.DPM_DEFAULT_N_INIT)
    dpm_init_params      : str, init method (default bf.DPM_DEFAULT_INIT_PARAMS)
    dpm_tol              : float, ELBO convergence tolerance (default bf.DPM_DEFAULT_TOL)

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
    grid        = density_file["y"].to_numpy()
    true_den    = density_file["true_density"].to_numpy()
    grid_points = grid.reshape(-1, 1)

    # ── output paths ──────────────────────────────────────────────────────────
    dir_metric_path = os.path.join(metric_output_path, prefix)
    os.makedirs(dir_metric_path, exist_ok=True)

    dir_sim_path = os.path.join(sim_output_path, prefix)
    os.makedirs(dir_sim_path, exist_ok=True)

    metric_filepath = os.path.join(
        dir_metric_path, f"{prefix}_iter{iteration}_baseline_dpm_metric.csv"
    )
    # Loaded (not overwritten) so repeated invocations at different n for the
    # same scenario/iteration accumulate column groups in one shared file.
    metric_df = load_or_init_metric_df(metric_filepath)

    start_time = time.time()

    sample_path = os.path.join(
        samples_folder, prefix, f"{prefix}_n{n}_iter{iteration}.csv"
    )
    samples = pd.read_csv(sample_path).to_numpy()   # (n, 1)
    seed = 50000 * iteration + n

    npz_dpm = os.path.join(dir_sim_path,
        f"{prefix}_n{n}_iter{iteration}_DPM.npz")

    # ── DPM ──────────────────────────────────────────────────────────────────
    print(f"  [{prefix}, iter={iteration}, n={n}] Running DPM "
          f"(K={dpm_max_components}, max_iter={dpm_max_iter}) ...", flush=True)
    dpm_den, dpm_time, active_k, converged, n_iter_used = bf.run_dpm(
        samples, grid_points,
        max_components  = dpm_max_components,
        max_iter         = dpm_max_iter,
        covariance_type  = dpm_covariance,
        seed             = seed,
        tol              = dpm_tol,
        alpha            = dpm_alpha,
        n_init           = dpm_n_init,
        init_params      = dpm_init_params,
    )
    if not converged:
        print(f"    WARNING: DPM did not converge after {n_iter_used} iterations",
              flush=True)
    dpm_L1, dpm_L2, dpm_Linf = compute_metrics(dpm_den, grid, true_den)
    save_density(npz_dpm, density=dpm_den, active_k=np.int64(active_k),
                 converged=np.bool_(converged), n_iter=np.int64(n_iter_used))
    print(f"    DPM        active_K={active_k}  converged={converged}  "
          f"n_iter={n_iter_used}  "
          f"L1={dpm_L1:.4f}  L2={dpm_L2:.4f}  Linf={dpm_Linf:.4f}  "
          f"time={dpm_time:.2f}s", flush=True)

    row = method_row_idx(metric_df, "DPM")
    metric_df.at[row, f"{n}_L1"]        = dpm_L1
    metric_df.at[row, f"{n}_L2"]        = dpm_L2
    metric_df.at[row, f"{n}_Linf"]      = dpm_Linf
    metric_df.at[row, f"{n}_time"]      = dpm_time
    metric_df.at[row, f"{n}_active_k"]  = active_k
    metric_df.at[row, f"{n}_converged"] = int(converged)
    metric_df.at[row, f"{n}_n_iter"]    = n_iter_used

    metric_df.to_csv(metric_filepath, index=False, header=True)

    elapsed = time.time() - start_time
    print(f"  [{prefix}, iter={iteration}, n={n}] done in {elapsed:.1f}s", flush=True)
    return prefix


# ── main (convenience wrapper for looping locally over several n) ──────────────

def main(iteration, sample_size_vec,
         dist_module_name, dist_module_path,
         metric_output_path, sim_output_path,
         density_folder, samples_folder,
         dpm_max_components=bf.DPM_DEFAULT_K,
         dpm_max_iter=bf.DPM_DEFAULT_MAX_ITER,
         dpm_covariance=bf.DPM_DEFAULT_COV,
         dpm_alpha=bf.DPM_DEFAULT_ALPHA,
         dpm_n_init=bf.DPM_DEFAULT_N_INIT,
         dpm_init_params=bf.DPM_DEFAULT_INIT_PARAMS,
         dpm_tol=bf.DPM_DEFAULT_TOL):
    """
    Convenience wrapper that calls run_one() once per n in sample_size_vec,
    for interactive/notebook use. On the cluster, prefer invoking this
    script once per (scenario, n) pair directly (see module docstring)
    so scenarios and sample sizes can be parallelized independently.
    """
    start_time = time.time()
    for n in sample_size_vec:
        run_one(
            iteration=iteration, n=n,
            dist_module_name=dist_module_name, dist_module_path=dist_module_path,
            metric_output_path=metric_output_path, sim_output_path=sim_output_path,
            density_folder=density_folder, samples_folder=samples_folder,
            dpm_max_components=dpm_max_components,
            dpm_max_iter=dpm_max_iter,
            dpm_covariance=dpm_covariance,
            dpm_alpha=dpm_alpha,
            dpm_n_init=dpm_n_init,
            dpm_init_params=dpm_init_params,
            dpm_tol=dpm_tol,
        )
    end_time = time.time()
    print(f"Total execution time for iteration {iteration}, "
          f"n in {sample_size_vec}: {end_time - start_time:.1f}s", flush=True)


# ── cluster entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":

    REQUIRED_ARGS = 8

    if len(sys.argv) < REQUIRED_ARGS + 1:
        print(
            "Usage: python sim_run_baselines_1D_dpm.py "
            "<iteration> <n> "
            "<dist_module_name> <dist_module_path> "
            "<metric_output_path> <sim_output_path> "
            "<density_folder> <samples_folder> "
            "[<dpm_max_components> <dpm_max_iter> <dpm_covariance> "
            "<dpm_alpha> <dpm_n_init> <dpm_init_params> <dpm_tol>]"
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

    dpm_max_components = int(sys.argv[9])    if len(sys.argv) > 9  else bf.DPM_DEFAULT_K
    dpm_max_iter        = int(sys.argv[10])   if len(sys.argv) > 10 else bf.DPM_DEFAULT_MAX_ITER
    dpm_covariance      = sys.argv[11]        if len(sys.argv) > 11 else bf.DPM_DEFAULT_COV
    dpm_alpha           = float(sys.argv[12]) if len(sys.argv) > 12 else bf.DPM_DEFAULT_ALPHA
    dpm_n_init          = int(sys.argv[13])   if len(sys.argv) > 13 else bf.DPM_DEFAULT_N_INIT
    dpm_init_params     = sys.argv[14]        if len(sys.argv) > 14 else bf.DPM_DEFAULT_INIT_PARAMS
    dpm_tol             = float(sys.argv[15]) if len(sys.argv) > 15 else bf.DPM_DEFAULT_TOL

    run_one(
        iteration           = iteration,
        n                   = n,
        dist_module_name    = dist_module_name,
        dist_module_path    = dist_module_path,
        metric_output_path  = metric_output_path,
        sim_output_path     = sim_output_path,
        density_folder      = density_folder,
        samples_folder      = samples_folder,
        dpm_max_components  = dpm_max_components,
        dpm_max_iter        = dpm_max_iter,
        dpm_covariance      = dpm_covariance,
        dpm_alpha           = dpm_alpha,
        dpm_n_init          = dpm_n_init,
        dpm_init_params     = dpm_init_params,
        dpm_tol             = dpm_tol,
    )

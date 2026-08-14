"""
sim_run_baselines_1D_kde.py
----------------------------
Runs KDE (Scott's rule) and KDE (CV bandwidth) on one (scenario, sample
size, iteration) combination.

Scenario, sample size, and iteration are independent CLI arguments --
each combination is run as its own invocation (one process per n, not a
loop over a comma-separated sample_sizes list), so scenarios and sample
sizes can be parallelized separately (e.g. one SLURM array task per n).

Metrics are written as a long/tidy CSV -- one row per method, one column
group per sample size -- so that (a) other baselines (DPM, DR-BART) can be
appended as additional rows later without reshaping the file, and (b)
separate runs over different n for the same (scenario, iteration) merge
into the same file by adding column groups, not rows. Fitted density
arrays and other per-fit diagnostics are saved separately as compressed
.npz files, one per (method, n).

Usage (cluster / command line):
    python sim_run_baselines_1D_kde.py <iteration> <n>
        <dist_module_name> <dist_module_path>
        <metric_output_path> <sim_output_path>
        <density_folder> <samples_folder>
        [<kde_cv_folds> <kde_cv_n_bandwidths> <kde_cv_max_n>]

    Optional arguments (defaults: 5, 30, 50000).
    kde_cv_max_n:   max sample size for which KDE-CV is run (set to 0 to skip).

Example (one scenario, one sample size, one iteration):
    python sim_run_baselines_1D_kde.py 0 5000 \\
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
    <metric_output_path>/<prefix>/<prefix>_iter<iteration>_baseline_kde_metric.csv

    One row per method ("KDE_scott", "KDE_cv"); columns per sample size n:
        {n}_L1, {n}_L2, {n}_Linf, {n}_time, {n}_bw
        {n}_cv_score, {n}_bw_search_min, {n}_bw_search_max,
        {n}_bw_at_lower_edge, {n}_bw_at_upper_edge
            (KDE_cv only -- NaN on the KDE_scott row; also NaN if n > kde_cv_max_n)

Output .npz files (one per method per n per iteration), compressed:
    <sim_output_path>/<prefix>/
        {prefix}_n{n}_iter{iteration}_KDE_scott.npz  -> density, bw
        {prefix}_n{n}_iter{iteration}_KDE_cv.npz     -> density, bw, cv_score,
                                                          bw_search_min, bw_search_max,
                                                          bw_at_lower_edge, bw_at_upper_edge
            (only written if n <= kde_cv_max_n)
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

METHODS = ["KDE_scott", "KDE_cv"]


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
            kde_cv_folds=5,
            kde_cv_n_bandwidths=30,
            kde_cv_max_n=50000,
            kde_rtol=0.0):
    """
    Run KDE-Scott and KDE-CV for exactly one (scenario, sample size,
    iteration) combination, updating that scenario/iteration's metric CSV
    in place (adding/overwriting only the {n}_* columns) and writing that
    combination's .npz density files.

    Parameters
    ----------
    iteration            : int, index of the dataset (0-199)
    n                     : int, sample size, e.g. 500, 5000, or 50000
    dist_module_name     : str, e.g. '1D_beta64'
    dist_module_path     : str, path to the distribution .py file
    metric_output_path   : str, root folder for metric output CSVs
    sim_output_path      : str, root folder for density .npz files
    density_folder       : str, folder containing true density CSVs
    samples_folder       : str, root folder containing per-distribution sample CSVs
    kde_cv_folds         : int, number of CV folds for KDE-CV (default 5)
    kde_cv_n_bandwidths  : int, number of bandwidths to search (default 30)
    kde_cv_max_n         : int, max n for which KDE-CV is run (default 50000)
                           Set to 0 to skip KDE-CV entirely.
    kde_rtol             : float, relative tolerance for the Ball Tree
                           approximate density evaluation, passed to both
                           KDE-Scott and KDE-CV (default 0.0 -- exact).

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
        dir_metric_path, f"{prefix}_iter{iteration}_baseline_kde_metric.csv"
    )
    # Loaded (not overwritten) so repeated invocations at different n for the
    # same scenario/iteration accumulate column groups in one shared file.
    metric_df = load_or_init_metric_df(metric_filepath)

    start_time = time.time()

    sample_path = os.path.join(
        samples_folder, prefix, f"{prefix}_n{n}_iter{iteration}.csv"
    )
    samples = pd.read_csv(sample_path).to_numpy()   # (n, 1)

    npz_kde_scott = os.path.join(dir_sim_path,
        f"{prefix}_n{n}_iter{iteration}_KDE_scott.npz")
    npz_kde_cv    = os.path.join(dir_sim_path,
        f"{prefix}_n{n}_iter{iteration}_KDE_cv.npz")

    # ── KDE — Scott's rule ────────────────────────────────────────────────────
    print(f"  [{prefix}, iter={iteration}, n={n}] Running KDE (Scott) ...", flush=True)
    kde_scott_den, kde_scott_time, kde_scott_bw = bf.run_kde(samples, grid_points, rtol=kde_rtol)
    kde_scott_L1, kde_scott_L2, kde_scott_Linf = compute_metrics(
        kde_scott_den, grid, true_den
    )
    save_density(npz_kde_scott, density=kde_scott_den, bw=np.float64(kde_scott_bw))
    print(f"    KDE-Scott  bw={kde_scott_bw:.4f}  "
          f"L1={kde_scott_L1:.4f}  L2={kde_scott_L2:.4f}  "
          f"Linf={kde_scott_Linf:.4f}  time={kde_scott_time:.2f}s", flush=True)

    row = method_row_idx(metric_df, "KDE_scott")
    metric_df.at[row, f"{n}_L1"]   = kde_scott_L1
    metric_df.at[row, f"{n}_L2"]   = kde_scott_L2
    metric_df.at[row, f"{n}_Linf"] = kde_scott_Linf
    metric_df.at[row, f"{n}_time"] = kde_scott_time
    metric_df.at[row, f"{n}_bw"]   = kde_scott_bw

    # ── KDE — CV bandwidth ────────────────────────────────────────────────────
    run_cv = (kde_cv_max_n > 0) and (n <= kde_cv_max_n)
    row = method_row_idx(metric_df, "KDE_cv")
    if run_cv:
        cv_folds = min(kde_cv_folds, n)
        print(f"  [{prefix}, iter={iteration}, n={n}] Running KDE (CV, "
              f"folds={cv_folds}, n_bw={kde_cv_n_bandwidths}) ...", flush=True)
        kde_cv_den, kde_cv_time, kde_cv_bw, kde_cv_score, kde_cv_info = bf.run_kde_cv(
            samples, grid_points,
            cv=cv_folds,
            n_bandwidths=kde_cv_n_bandwidths,
            rtol=kde_rtol,
        )
        kde_cv_L1, kde_cv_L2, kde_cv_Linf = compute_metrics(
            kde_cv_den, grid, true_den
        )
        save_density(
            npz_kde_cv, density=kde_cv_den,
            bw=np.float64(kde_cv_bw), cv_score=np.float64(kde_cv_score),
            h_ref=np.float64(kde_cv_info["h_ref"]),
            bw_search_min=np.float64(kde_cv_info["bw_search_min"]),
            bw_search_max=np.float64(kde_cv_info["bw_search_max"]),
            bw_at_lower_edge=np.bool_(kde_cv_info["bw_at_lower_edge"]),
            bw_at_upper_edge=np.bool_(kde_cv_info["bw_at_upper_edge"]),
        )
        print(f"    KDE-CV     bw={kde_cv_bw:.4f}  cv_score={kde_cv_score:.4f}  "
              f"L1={kde_cv_L1:.4f}  L2={kde_cv_L2:.4f}  "
              f"Linf={kde_cv_Linf:.4f}  time={kde_cv_time:.2f}s  "
              f"at_lower_edge={kde_cv_info['bw_at_lower_edge']}  "
              f"at_upper_edge={kde_cv_info['bw_at_upper_edge']}", flush=True)
        if kde_cv_info["bw_at_lower_edge"] or kde_cv_info["bw_at_upper_edge"]:
            edge = "lower" if kde_cv_info["bw_at_lower_edge"] else "upper"
            print(f"    WARNING: KDE-CV selected bandwidth at {edge} search edge "
                  f"(bw={kde_cv_bw:.4g}, range=[{kde_cv_info['bw_search_min']:.4g}, "
                  f"{kde_cv_info['bw_search_max']:.4g}]) -- not a genuine optimum, "
                  f"widen bw_factor_range", flush=True)

        metric_df.at[row, f"{n}_L1"]                = kde_cv_L1
        metric_df.at[row, f"{n}_L2"]                = kde_cv_L2
        metric_df.at[row, f"{n}_Linf"]               = kde_cv_Linf
        metric_df.at[row, f"{n}_time"]               = kde_cv_time
        metric_df.at[row, f"{n}_bw"]                 = kde_cv_bw
        metric_df.at[row, f"{n}_cv_score"]           = kde_cv_score
        metric_df.at[row, f"{n}_h_ref"]              = kde_cv_info["h_ref"]
        metric_df.at[row, f"{n}_bw_search_min"]      = kde_cv_info["bw_search_min"]
        metric_df.at[row, f"{n}_bw_search_max"]      = kde_cv_info["bw_search_max"]
        metric_df.at[row, f"{n}_bw_at_lower_edge"]   = kde_cv_info["bw_at_lower_edge"]
        metric_df.at[row, f"{n}_bw_at_upper_edge"]   = kde_cv_info["bw_at_upper_edge"]
    else:
        print(f"  [{prefix}, iter={iteration}, n={n}] Skipping KDE-CV "
              f"(n={n} > kde_cv_max_n={kde_cv_max_n})", flush=True)
        for col in ("L1", "L2", "Linf", "time", "bw", "cv_score", "h_ref",
                    "bw_search_min", "bw_search_max",
                    "bw_at_lower_edge", "bw_at_upper_edge"):
            metric_df.at[row, f"{n}_{col}"] = np.nan

    metric_df.to_csv(metric_filepath, index=False, header=True)

    elapsed = time.time() - start_time
    print(f"  [{prefix}, iter={iteration}, n={n}] done in {elapsed:.1f}s", flush=True)
    return prefix


# ── main (convenience wrapper for looping locally over several n) ──────────────

def main(iteration, sample_size_vec,
         dist_module_name, dist_module_path,
         metric_output_path, sim_output_path,
         density_folder, samples_folder,
         kde_cv_folds=5,
         kde_cv_n_bandwidths=30,
         kde_cv_max_n=50000,
         kde_rtol=0.0):
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
            kde_cv_folds=kde_cv_folds,
            kde_cv_n_bandwidths=kde_cv_n_bandwidths,
            kde_cv_max_n=kde_cv_max_n,
            kde_rtol=kde_rtol,
        )
    end_time = time.time()
    print(f"Total execution time for iteration {iteration}, "
          f"n in {sample_size_vec}: {end_time - start_time:.1f}s", flush=True)


# ── cluster entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":

    REQUIRED_ARGS = 8

    if len(sys.argv) < REQUIRED_ARGS + 1:
        print(
            "Usage: python sim_run_baselines_1D_kde.py "
            "<iteration> <n> "
            "<dist_module_name> <dist_module_path> "
            "<metric_output_path> <sim_output_path> "
            "<density_folder> <samples_folder> "
            "[<kde_cv_folds> <kde_cv_n_bandwidths> <kde_cv_max_n> <kde_rtol>]"
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

    kde_cv_folds        = int(sys.argv[9])    if len(sys.argv) > 9  else 5
    kde_cv_n_bandwidths = int(sys.argv[10])   if len(sys.argv) > 10 else 30
    kde_cv_max_n        = int(sys.argv[11])   if len(sys.argv) > 11 else 50000
    kde_rtol            = float(sys.argv[12]) if len(sys.argv) > 12 else 0.0

    run_one(
        iteration           = iteration,
        n                   = n,
        dist_module_name    = dist_module_name,
        dist_module_path    = dist_module_path,
        metric_output_path  = metric_output_path,
        sim_output_path     = sim_output_path,
        density_folder      = density_folder,
        samples_folder      = samples_folder,
        kde_cv_folds        = kde_cv_folds,
        kde_cv_n_bandwidths = kde_cv_n_bandwidths,
        kde_rtol            = kde_rtol,
        kde_cv_max_n        = kde_cv_max_n,
    )

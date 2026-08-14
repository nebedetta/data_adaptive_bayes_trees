"""
sim_run_1D_speed.py
--------------------
1D pipeline with the same output structure as sim_run.py (the nD pipeline):
reads pre-generated samples from disk (rather than drawing fresh ones
in-process, as sim_run_1D.py does), saves a gzipped pickle per (n, depth),
and appends L1/L2/Linf/time metrics to one combined CSV per iteration.

Uses the same 1D-specific functions as sim_run_1D.py / sim_process_1D.py:
    - src.core.OPT_1D_partial_functions.BT_Optional_Stopping_fdepth  (partial/latent model)
    - src.core.OPT_1D_full_functions.partition_full / OPT_full_model  (full/Polya-tree model)
    - src.core.numerical_analysis_functions.distance_metric / l1_distance / l2_distance / linfty_distance

Note: sim_run_1D.py and sim_process_1D.py both import
`from src.core import OPT_1D_partial_funtions as latent` -- a typo (missing
"c") that does not match the actual module name on disk,
`OPT_1D_partial_functions.py`. That import is broken in both of those
scripts as they currently stand; this script uses the correctly-spelled
module name and does not touch src/core/.

Output structure matches sim_run.py exactly:
    - one gzipped pickle per (n, depth), containing samples + both models'
      step-function fits (heights/intervals) + fit times:
          <sim_output_path>/<prefix>/<prefix>_n<n>_depth<depth>_itern<iteration>.pkl.gz
    - one combined metric CSV per iteration, appended across n/depth,
      row index = depth - 1, columns {n}_L1P/L1F/L2P/L2F/LIP/LIF/TP/TF:
          <metric_output_path>/<prefix>/<prefix>_iter<iteration>_metric.csv

Usage:
    python -m src.sim_run_1D_speed <iteration> <sample_sizes> <depths> \\
        <dist_module_name> <dist_module_path> \\
        <sim_output_path> <metric_output_path> \\
        <density_folder> <samples_folder>

Example:
    python -m src.sim_run_1D_speed 0 "500,5000" "1,2,3,4,5,6" \\
        1D_spikemix_input distributions/1D_spikemix_input.py \\
        output/1D/fits output/1D/metrics \\
        distributions/true_density distributions/samples
"""

import gzip
import importlib.util
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from src.core import OPT_1D_partial_functions as latent
from src.core import OPT_1D_full_functions as polya
from src.core import numerical_analysis_functions as na

P0    = 0.5
A     = 1
B     = 1
ALPHA = 2
ALPHA0 = np.exp(20)


def import_distribution_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_partial_fdepth(X, depth):
    """Partial/latent model (BT_Optional_Stopping_fdepth), timed like generating_fdepth does."""
    start_time = time.time()
    bt = latent.BT_Optional_Stopping_fdepth(X, P0, ALPHA, A, B, ALPHA0, P0, depth)
    elapsed = time.time() - start_time
    return bt, elapsed


def fit_full_fdepth(X, depth):
    """Full/Polya-tree model (OPT_full_model), timed like generating_fdepth does."""
    start_time = time.time()
    pvec_size = polya.partition_full(depth)
    pt_s = polya.OPT_full_model(X, pvec_size, depth, ALPHA, A, B, ALPHA0, P0)
    elapsed = time.time() - start_time
    return pt_s, elapsed


def get_or_create_column(df, col_name, max_depth):
    if col_name in df.columns:
        col_values = df[col_name].tolist()
    else:
        col_values = []
    if len(col_values) < max_depth:
        col_values.extend([None] * (max_depth - len(col_values)))
    return col_values


def extend_dataframe(df, target_length):
    current_length = len(df)
    if current_length < target_length:
        additional_rows = pd.DataFrame(index=range(current_length, target_length))
        df = pd.concat([df, additional_rows], ignore_index=True)
    return df


def main(iteration, sample_size_vec, depths, dist_module_name, dist_module_path,
         sim_output_path, metric_output_path, density_folder, samples_folder):

    dist_module = import_distribution_module(dist_module_name, dist_module_path)
    prefix = dist_module.prefix

    # Load true density file -- 1D convention: columns are "y", "true_density".
    density_file = pd.read_csv(f"{density_folder}/{prefix}_true_density.csv")
    X0 = density_file["y"].to_numpy()
    y  = density_file["true_density"].to_numpy()

    dir_sim_path = os.path.join(sim_output_path, prefix)
    os.makedirs(dir_sim_path, exist_ok=True)

    dir_metric_path = os.path.join(metric_output_path, prefix)
    os.makedirs(dir_metric_path, exist_ok=True)

    metric_filepath = os.path.join(dir_metric_path, f"{prefix}_iter{iteration}_metric.csv")
    if os.path.exists(metric_filepath):
        try:
            metric_df = pd.read_csv(metric_filepath)
        except pd.errors.EmptyDataError:
            metric_df = pd.DataFrame()
    else:
        metric_df = pd.DataFrame()
        metric_df.to_csv(metric_filepath, index=False, header=True)

    metric_df = extend_dataframe(metric_df, np.max(depths))

    start_time = time.time()
    for n in sample_size_vec:

        # 1D sample CSVs have a "y" header.
        samples_df = pd.read_csv(
            f"{samples_folder}/{prefix}/{prefix}_n{n}_iter{iteration}.csv"
        )
        X = samples_df["y"].to_numpy()

        metric_suffixes = ['L1P', 'L1F', 'L2P', 'L2F', 'LIP', 'LIF', 'TP', 'TF']
        metric_vecs = {}
        for suffix in metric_suffixes:
            col = f'{n}_{suffix}'
            metric_vecs[suffix] = get_or_create_column(metric_df, col, np.max(depths))

        for depth in depths:

            sim_filepath = os.path.join(
                dir_sim_path, f'{prefix}_n{n}_depth{depth}_itern{iteration}.pkl.gz'
            )

            bt, time_partial = fit_partial_fdepth(X, depth)
            pt_s, time_full  = fit_full_fdepth(X, depth)

            data_dict = {
                "samples":      X,
                "bt":           bt,     # partial/latent model -- bt[5]=heights, bt[6]=intervals
                "pt_s":         pt_s,   # full/Polya-tree model -- pt_s[5]=heights, pt_s[6]=intervals
                "time_partial": time_partial,
                "time_full":    time_full,
            }

            with gzip.open(sim_filepath, 'wb') as file:
                pickle.dump(data_dict, file)

            L1_partial,     L1_full     = na.distance_metric(X0, bt[5], bt[6], pt_s[5], pt_s[6], y, na.l1_distance)
            L2_partial,     L2_full     = na.distance_metric(X0, bt[5], bt[6], pt_s[5], pt_s[6], y, na.l2_distance)
            Linfty_partial, Linfty_full = na.distance_metric(X0, bt[5], bt[6], pt_s[5], pt_s[6], y, na.linfty_distance)

            metric_vecs['L1P'][depth - 1] = L1_partial
            metric_vecs['L1F'][depth - 1] = L1_full
            metric_vecs['L2P'][depth - 1] = L2_partial
            metric_vecs['L2F'][depth - 1] = L2_full
            metric_vecs['LIP'][depth - 1] = Linfty_partial
            metric_vecs['LIF'][depth - 1] = Linfty_full
            metric_vecs['TP'][depth - 1]  = time_partial
            metric_vecs['TF'][depth - 1]  = time_full

        for suffix in metric_suffixes:
            metric_df[f'{n}_{suffix}'] = pd.Series(metric_vecs[suffix])

        metric_df.to_csv(metric_filepath, index=False, header=True)

    end_time = time.time()
    print(f"Total execution time for iteration {iteration}: {end_time - start_time:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) != 10:
        print("Usage: python -m src.sim_run_1D_speed <iteration> <sample_sizes> <depths> "
              "<dist_module_name> <dist_module_path> <sim_output_path> <metric_output_path> "
              "<density_folder> <samples_folder>")
        sys.exit(1)

    iteration           = int(sys.argv[1])
    sample_sizes        = sys.argv[2]
    depths_arg          = sys.argv[3]
    dist_module_name    = sys.argv[4]
    dist_module_path    = sys.argv[5]
    sim_output_path     = sys.argv[6]
    metric_output_path  = sys.argv[7]
    density_folder      = sys.argv[8]
    samples_folder      = sys.argv[9]

    sample_size_vec = list(map(int, sample_sizes.split(',')))
    depths_vec      = list(map(int, depths_arg.split(',')))

    main(iteration, sample_size_vec, depths_vec, dist_module_name, dist_module_path,
         sim_output_path, metric_output_path, density_folder, samples_folder)

import numpy as np
import pandas as pd
import time
import os
import sys
import gzip
import pickle
import importlib.util
from joblib import Parallel, delayed

from src import OPT_functions_speed_emptynode_fix as f


def import_distribution_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def recursion_partial_model(samples, d, p0, lambdas, alpha, alpha0, grids, base_seed, depth_parallelization, chunk_number):
    start_time = time.time()
    r = f.bottom_up_recursion_2(samples, d, p0, lambdas, alpha, alpha0, depth_parallelization, chunk_number)
    end_time = time.time()
    return r, end_time - start_time


def recursion_full_model(samples, d, p0, lambdas, alpha, alpha0, grids, base_seed):
    start_time = time.time()
    r = f.binary_bottom_up_recursion(samples, d, p0, lambdas, alpha, alpha0)
    end_time = time.time()
    return r, end_time - start_time


def sampling_partial(r, time_partition, samples, depth, p0, lambdas, alpha0, grids, base_seed, n_iter, njobs):
    start_time = time.time()
    part_vec   = r["part_vec"]
    u1         = r["set_size"]
    ratio1     = r["ratio1"]
    rho00      = r["rho00"]
    rho01      = r["rho01"]
    alpha_vec  = r["alpha_vec"]
    pm_mcmc = f.sampling_MCt(n_iter, grids, depth, part_vec, rho00, rho01,
                              lambdas, u1, ratio1, alpha_vec, alpha0, base_seed, njobs,
                              d=samples.shape[1],                        # fixed
                              interpolate_fn=f.interpolate_step_function) # fixed
    end_time = time.time()
    total_time = time_partition + (end_time - start_time)
    return pm_mcmc, part_vec, total_time


def sampling_full(r, time_partition, samples, depth, p0, lambdas, alpha0, grids, base_seed, n_iter, njobs):
    start_time = time.time()
    part_vec   = r["part_vec"]
    u1         = r["set_size"]
    ratio1     = r["ratio1"]
    rho00      = r["rho00"]
    rho01      = r["rho01"]
    alpha_vec  = r["alpha_vec"]
    pm_mcmc = f.sampling_MCt(n_iter, grids, depth, part_vec, rho00, rho01,
                              lambdas, u1, ratio1, alpha_vec, alpha0, base_seed, njobs,
                              d=samples.shape[1],                        # fixed
                              interpolate_fn=f.interpolate_step_function) # fixed
    end_time = time.time()
    total_time = time_partition + (end_time - start_time)
    return pm_mcmc, total_time


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


def main(iteration, sample_size_vec, depths, d, dist_module_name, dist_module_path,
         sim_output_path, metric_output_path, depth_parallelization, ncores, n_iter, 
         density_folder, samples_folder):

    dist_module = import_distribution_module(dist_module_name, dist_module_path)
    prefix = dist_module.prefix

    # Load true density file — columns are x0, x1, ..., x{d-1}, true_density
    density_file = pd.read_csv(
        f"{density_folder}/{prefix}_true_density.csv"
    ).to_numpy()   # shape (n_eval, d+1)

    # grids: list of d flat arrays, one per dimension
    grids = [density_file[:, i] for i in range(d)]   # each shape (n_eval,)
    true_den = density_file[:, d]                     # shape (n_eval,)

    p0 = 0.5
    lambdas = [1.0 / d] * d
    alpha  = 2
    alpha0 = np.exp(20)

    base_seed = 50000 * iteration

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

        samples_df = pd.read_csv(
            f"{samples_folder}/{prefix}/{prefix}_n{n}_iter{iteration}.csv"
        )
        samples = samples_df.to_numpy()

        metric_suffixes = ['L1P', 'L1F', 'L2P', 'L2F', 'LIP', 'LIF', 'TP', 'TF']
        metric_vecs = {}
        for suffix in metric_suffixes:
            col = f'{n}_{suffix}'
            metric_vecs[suffix] = get_or_create_column(metric_df, col, np.max(depths))

        for depth in depths:

            sim_filepath = os.path.join(
                dir_sim_path, f'{prefix}_n{n}_depth{depth}_itern{iteration}.pkl.gz'
            )

            chunk_number = ncores - 1

            recursion_partial, recursion_full_res = Parallel(n_jobs=2)(
                [
                    delayed(recursion_partial_model)(
                        samples, depth, p0, lambdas, alpha, alpha0, grids,
                        base_seed + depth * 1000, depth_parallelization, chunk_number
                    ),
                    delayed(recursion_full_model)(
                        samples, depth, p0, lambdas, alpha, alpha0, grids,
                        base_seed + depth * 1000
                    )
                ]
            )

            r_partial, time_partial = recursion_partial
            r_full,    time_full    = recursion_full_res

            njobs = int((ncores - 1) / 2)

            result_partial, result_full = Parallel(n_jobs=2)(
                [
                    delayed(sampling_partial)(
                        r_partial, time_partial, samples, depth, p0, lambdas, alpha0,
                        grids, base_seed + depth * 1000, n_iter, njobs
                    ),
                    delayed(sampling_full)(
                        r_full, time_full, samples, depth, p0, lambdas, alpha0,
                        grids, base_seed + depth * 1000, n_iter, njobs
                    )
                ]
            )

            pm_mcmc_partial, part_vec_partial, total_time_partial = result_partial
            pm_mcmc_full,    total_time_full                       = result_full

            data_dict = {
                "samples":          samples,
                "pm_mcmc_partial":  pm_mcmc_partial,
                "pm_mcmc_full":     pm_mcmc_full,
                "time_partial":     time_partial,
                "time_full":        time_full,
            }

            with gzip.open(sim_filepath, 'wb') as file:
                pickle.dump(data_dict, file)

            L1_partial,    L1_full    = f.distance_metric(grids, pm_mcmc_partial, pm_mcmc_full, true_den, f.l1_distance)
            L2_partial,    L2_full    = f.distance_metric(grids, pm_mcmc_partial, pm_mcmc_full, true_den, f.l2_distance)
            Linfty_partial, Linfty_full = f.distance_metric(grids, pm_mcmc_partial, pm_mcmc_full, true_den, f.linfty_distance)

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
    if len(sys.argv) != 14:
        print("Usage: python sim_run_nD.py <iteration> <sample_sizes> <depths> <d> "
              "<dist_module_name> <dist_module_path> <sim_output_path> <metric_output_path> "
              "<depth_parallelization> <ncores> <n_iter> <eval_grid_path>")
        sys.exit(1)

    iteration             = int(sys.argv[1])
    sample_sizes          = sys.argv[2]
    depths                = sys.argv[3]
    d                     = int(sys.argv[4])
    dist_module_name      = sys.argv[5]
    dist_module_path      = sys.argv[6]
    sim_output_path       = sys.argv[7]
    metric_output_path    = sys.argv[8]
    depth_parallelization = int(sys.argv[9])
    ncores                = int(sys.argv[10])
    n_iter                = int(sys.argv[11])
    density_folder        = sys.argv[12]
    samples_folder        = sys.argv[13]

    sample_size_vec = list(map(int, sample_sizes.split(',')))
    depths_vec      = list(map(int, depths.split(',')))

    main(iteration, sample_size_vec, depths_vec, d, dist_module_name, dist_module_path,
         sim_output_path, metric_output_path, depth_parallelization, ncores, n_iter,
         density_folder, samples_folder)
import numpy as np
import random
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
import time
import warnings
from bisect import bisect_left, bisect_right
from IPython.display import display
from scipy.special import beta as bf
from scipy.special import comb
from scipy.stats import beta
from matplotlib.patches import Rectangle
import os
import sys
import gzip
import pickle
import importlib.util
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed


from src.core import OPT_2D_functions  as f2D
from src.core import io_utils


def recursion_partial_model(samples, d, p0, lx, alpha0, x0, y0, base_seed, depth_parallelization, chunk_number):

    start_time = time.time()
    r = f2D.bottom_up_recursion_2(samples, d, p0, lx, alpha0, depth_parallelization, chunk_number)
    end_time = time.time()
    time_partition = end_time - start_time
    
    return r, time_partition
    

def recursion_full_model(samples, d, p0, lx, alpha0, x0, y0, base_seed):

    start_time = time.time()
    r = f2D.binary_bottom_up_recursion(samples, d, p0, lx, alpha0)
    end_time = time.time()
    time_partition = end_time - start_time
    
    return r, time_partition


def sampling_partial(r, time_partition, samples, d, p0, lx, alpha0, x0, y0, base_seed, n_iter, njobs):
    
    start_time = time.time()  
    part_vec = r["part_vec"]
    u1 = r["set_size"]
    ratio1 = r["ratio1"]
    rho0_0x = r["rho00_x"]
    rho0_1x = r["rho01_x"]
    rho0_0y = r["rho00_y"]
    rho0_1y = r["rho01_y"]
    alpha_vec = r["alpha_vec"]
    pm_mcmc = f2D.sampling_MCt(n_iter, x0, y0, d, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, base_seed, njobs)
    end_time = time.time()
    time_recursion = end_time - start_time
    total_time = time_partition + time_recursion
    
    return pm_mcmc, part_vec, total_time

def sampling_full(r, time_partition, samples, d, p0, lx, alpha0, x0, y0, base_seed, n_iter, njobs):
    
    start_time = time.time()  
    part_vec = r["part_vec"]
    u1 = r["set_size"]
    ratio1 = r["ratio1"]
    rho0_0x = r["rho00_x"]
    rho0_1x = r["rho01_x"]
    rho0_0y = r["rho00_y"]
    rho0_1y = r["rho01_y"]
    alpha_vec = r["alpha_vec"]
    pm_mcmc = f2D.sampling_MCt(n_iter, x0, y0, d, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, base_seed, njobs)
    end_time = time.time()
    time_recursion = end_time - start_time
    total_time = time_partition + time_recursion
    
    return pm_mcmc, total_time

def get_or_create_column(df, col_name, max_depth):
    # Check if column exists in the DataFrame
    if col_name in df.columns:
        # If the column exists, convert it to a list
        col_values = df[col_name].tolist()
    else:
        # If the column doesn't exist, initialize it as an empty list
        col_values = []
    
    # Extend the list to the length of max_depth if necessary
    if len(col_values) < max_depth:
        col_values.extend([None] * (max_depth - len(col_values)))
    
    return col_values

def extend_dataframe(df, target_length):
    """Extend the DataFrame to the target length by adding rows if necessary."""
    current_length = len(df)
    if current_length < target_length:
        # Create new rows with NaN values and append them to the DataFrame
        additional_rows = pd.DataFrame(index=range(current_length, target_length))
        df = pd.concat([df, additional_rows], ignore_index=True)
    return df



def main(iteration, sample_size_vec, depths, dist_module_name, dist_module_path, sim_output_path, metric_output_path, depth_parallelization, ncores, n_iter):
    
        #x0, y0 = np.meshgrid(np.linspace(0.00001, 0.99999, 500), np.linspace(0.00001, 0.99999, 500))
    #grid_points = np.vstack([x0.ravel(), y0.ravel()]).T
  
    dist_module = io_utils.import_distribution_module(dist_module_name, dist_module_path)
    #extract_samples = dist_module.extract_samples
    #compute_pdf = dist_module.compute_pdf
    prefix = dist_module.prefix

    x0_original, y0_original = np.meshgrid(np.linspace(0.00001, 0.99999, 500), np.linspace(0.00001, 0.99999, 500))
    f, g = x0_original.shape
    density_file = pd.read_csv(f"partial_likelihood_repo/distributions/true_density/{prefix}_true_density.csv")
    grid_points = density_file.to_numpy()
    x0 = grid_points[:,0].reshape(f, g)
    y0 = grid_points[:,1].reshape(f, g)
    true_den = grid_points[:,2].reshape(f, g)
    #true_den = compute_pdf(x0, y0)
    
    p0 = 0.5
    lx = 0.5
    alpha0 = np.exp(20)

    base_seed = 50000 * iteration
    
    dir_sim_path = os.path.join(sim_output_path, prefix)
    os.makedirs(dir_sim_path, exist_ok=True)

    dir_metric_path = os.path.join(metric_output_path, prefix)
    os.makedirs(dir_metric_path, exist_ok=True)

    # Creating a metric output csv for each iteration ID
    metric_filepath = os.path.join(dir_metric_path, f"{prefix}_iter{iteration}_metric.csv") 
    if os.path.exists(metric_filepath):
        try:
            metric_df = pd.read_csv(metric_filepath)
        except pd.errors.EmptyDataError:
            metric_df = pd.DataFrame()  # In case of an empty file
    else:
        metric_df = pd.DataFrame()
        metric_df.to_csv(metric_filepath, index=False, header=True)

    metric_df = extend_dataframe(metric_df, np.max(depths))

    
    start_time = time.time()
    for n in sample_size_vec:
       
        #samples = extract_samples(n, base_seed)
        samples_df = pd.read_csv(f"partial_likelihood_repo/distributions/samples/{prefix}/{prefix}_n{n}_iter{iteration}.csv")
        samples = samples_df.to_numpy()

        # Generating vectors to save the metrices. Retrieve them from the cvs file if it is available or just create empty ones. 
        # Extend them to accomodate for possible greater depths. 
        metric_vecs = {}
        
        metric_suffixes = ['L1P', 'L1F', 'L2P', 'L2F', 'LIP', 'LIF', 'TP', 'TF']
        for metric_suffix in metric_suffixes:
            column_name = f'{n}_{metric_suffix}'
            globals()[metric_suffix] = get_or_create_column(metric_df, column_name, np.max(depths))

        for d in depths: 
            
            sim_filepath = os.path.join(dir_sim_path, f'{prefix}_n{n}_depth{d}_itern{iteration}.pkl.gz')
            
#            if os.path.exists(sim_filepath):
#                print(f"File {sim_filepath} already exists in server. Skipping...")
#                continue

            chunk_number = ncores - 1
            
            recursion_partial, recursion_full = Parallel(n_jobs=2)(
                [
                    delayed(recursion_partial_model)(samples, d, p0, lx, alpha0, x0, y0, base_seed + d*1000, depth_parallelization, chunk_number),
                    delayed(recursion_full_model)(samples, d, p0, lx, alpha0, x0, y0, base_seed + d*1000)
                ]
            )
            

            r_partial, time_partial = recursion_partial 
            r_full, time_full = recursion_full 

            njobs = int((ncores - 1)/2)
            
            result_partial, result_full = Parallel(n_jobs=2)(
                [
                    delayed(sampling_partial)(r_partial, time_partial, samples, d, p0, lx, alpha0, x0, y0, base_seed + d*1000, n_iter, njobs),
                    delayed(sampling_full)(r_full, time_full, samples, d, p0, lx, alpha0, x0, y0, base_seed + d*1000, n_iter, njobs)
                ]
            )
            
            pm_mcmc_partial, part_vec_partial, total_time_partial = result_partial
            pm_mcmc_full, total_time_full = result_full
    
            data_dict = {
                "samples": samples,
               # "part_vec_partial": part_vec_partial,
               # "part_vec_full": part_vec_full,
                "pm_mcmc_partial": pm_mcmc_partial,
                "pm_mcmc_full": pm_mcmc_full,
                "time_partial": time_partial,
                "time_full": time_full
            }
                    
            with gzip.open(sim_filepath, 'wb') as file:
                pickle.dump(data_dict, file)


            L1_partial, L1_full = f2D.distance_metric(x0, y0, pm_mcmc_partial, pm_mcmc_full, true_den, f2D.l1_distance)
            L2_partial, L2_full = f2D.distance_metric(x0, y0, pm_mcmc_partial, pm_mcmc_full, true_den, f2D.l2_distance)
            Linfty_partial, Linfty_full = f2D.distance_metric(x0, y0, pm_mcmc_partial, pm_mcmc_full, true_den, f2D.linfty_distance)

            L1P[d-1] = L1_partial
            L1F[d-1] = L1_full
            L2P[d-1] = L2_partial
            L2F[d-1] = L2_full
            LIP[d-1] = Linfty_partial
            LIF[d-1] = Linfty_full
            TP[d-1] = time_partial
            TF[d-1] = time_full

        metric_columns = [(globals()[suffix], f'{n}_{suffix}') for suffix in metric_suffixes]
        for metric_data, metric_name in metric_columns:
            series = pd.Series(metric_data)
            metric_df[metric_name] = series
         
        metric_df.to_csv(metric_filepath, index=False, header=True)
        
        
    end_time = time.time()
    print(f"Total execution time for iteration {iteration}: {end_time - start_time} seconds")


if __name__ == "__main__":
    if len(sys.argv) != 11:
        print("Usage: python run_simulation.py <iteration> <distribution_module> <parameter_file>")
        sys.exit(1)

    
    # Read command line arguments
    iteration = int(sys.argv[1])
    sample_sizes = sys.argv[2]
    depths = sys.argv[3]
    
    dist_module_name = sys.argv[4]
    dist_module_path = sys.argv[5]
    sim_output_path = sys.argv[6]
    metric_output_path = sys.argv[7] 
    depth_parallelization = int(sys.argv[8])
    ncores = int(sys.argv[9])
    n_iter = int(sys.argv[10])

    # Convert some arguments to lists
    sample_size_vec = list(map(int, sample_sizes.split(',')))
    depths_vec = list(map(int, depths.split(',')))
    
    main(iteration, sample_size_vec, depths_vec, dist_module_name, dist_module_path, sim_output_path, metric_output_path, depth_parallelization, ncores, n_iter)



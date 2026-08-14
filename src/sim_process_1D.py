import numpy as np
import pandas as pd
import random
import scipy as sp
import scipy.stats as stats
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
import sys
import pickle
from tabulate import tabulate 
import time
import gzip
import importlib.util

from src.core import partition_1D_functions as basic
from src.core import OPT_1D_partial_functions as latent
from src.core import OPT_1D_full_functions as polya
from src.core import numerical_analysis_functions as na

def import_distribution_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def metric_computation(output_from_pickle, X0, y):
    
    # For OPT
    bt, pt_s, time_partial, time_full, X = output_from_pickle
    L1_partial, L1_full = na.distance_metric(X0, bt[5], bt[6], pt_s[5], pt_s[6], y, na.l1_distance)
    L2_partial, L2_full = na.distance_metric(X0, bt[5], bt[6], pt_s[5], pt_s[6], y, na.l2_distance)
    Linfty_partial, Linfty_full = na.distance_metric(X0, bt[5], bt[6], pt_s[5], pt_s[6], y, na.linfty_distance)
    

    return L1_partial, L1_full, L2_partial, L2_full, Linfty_partial, Linfty_full, time_partial, time_full#, X


def process_pickle_files(sample_size, dist_module_name, dist_module_path, pickle_path, output_path, title):
    # Ensure the output directory exists
    #if not os.path.exists(output_directory):
    #    os.makedirs(output_directory)
    
    # Retrieve distribution-specific functions and parameters
    dist_module =  import_distribution_module(dist_module_name, dist_module_path)
    extract_samples = dist_module.extract_samples
    compute_pdf = dist_module.compute_pdf
    prefix = dist_module.prefix
    
    X0 = np.linspace(0, 1, 100000)
    y = compute_pdf(X0)

    # Initialize sets to collect unique depths and iterations
    depths_set = set()
    iterations_set = set()
    
    pickle_directory = os.path.join(pickle_path, prefix)
    
    X = None
    # First pass to determine unique depths and iterations
    for filename in os.listdir(pickle_directory):
        if filename.startswith(prefix) and filename.endswith('.pkl.gz'):
            parts = filename.split('_')
            try:
                sample_size_in_file = int(parts[2].replace('n', ''))
                depth = int(parts[3].replace('depth', ''))
                itern = int(parts[4].replace('itern', '').replace('.pkl.gz', ''))
                
                if sample_size_in_file == sample_size:
                    depths_set.add(depth)
                    iterations_set.add(itern)
            except ValueError:
                # Skip files that don't match expected format
                continue
    
    # Sort the unique depths and iterations
    depths = sorted(depths_set)
    iterations = sorted(iterations_set)
    
    # Initialize a 3D NumPy array to store results: dimensions are [depths, iterations, metrics]
    num_metrics = 4*2
    num_depths = len(depths)
    num_iterations = len(iterations)
    results_array = np.full((num_depths, num_iterations, num_metrics), np.nan)  # Using NaN for missing values
    
    # Create a mapping from depth and iteration to indices in the array
    depth_index = {depth: idx for idx, depth in enumerate(depths)}
    iteration_index = {itern: idx for idx, itern in enumerate(iterations)}
    
    # Second pass to process files and fill the array
    for filename in os.listdir(pickle_directory):
        if filename.startswith(prefix) and filename.endswith('.pkl.gz'):
            parts = filename.split('_')
            try:
                sample_size_in_file = int(parts[2].replace('n', ''))
                depth = int(parts[3].replace('depth', ''))
                itern = int(parts[4].replace('itern', '').replace('.pkl.gz', ''))
                
                if sample_size_in_file == sample_size and depth in depths:
                    # Load the pickle file
                    filepath = os.path.join(pickle_directory, filename)
                    with gzip.open(filepath, 'rb') as file:
                        data = pickle.load(file)
                    
                    # Perform computations for all metrics
                    #L1_b, L1_p, time_b, time_p, X = metric_computation(data, X0, y)
                    L1_partial, L1_full, L2_partial, L2_full, Linfty_partial, Linfty_full, time_partial, time_full = metric_computation(data, X0, y)
                    
                    # Store the results in the array
                    if itern in iteration_index:
                        depth_idx = depth_index[depth]
                        itern_idx = iteration_index[itern]
                        results_array[depth_idx, itern_idx, :] = L1_partial, L1_full, L2_partial, L2_full, Linfty_partial, Linfty_full, time_partial, time_full
            except ValueError:
                # Skip files that don't match expected format
                continue
    
    # Generate the output file name
    output_directory = os.path.join(output_path, prefix)
    os.makedirs(output_directory, exist_ok=True)
    output_filename = f"{prefix}_n{sample_size}_{title}.pkl.gz"
    output_filepath = os.path.join(output_directory, output_filename)
    
    # Save the results array to a pickle file
    with gzip.open(output_filepath, 'wb') as file:
        #pickle.dump((results_array, X), file)
        pickle.dump((results_array), file)


if __name__ == "__main__":
    if len(sys.argv) != 7:  # Expects 6 arguments plus the script name
            print("Usage: python script.py sample_size dist_module_name dist_module_path pickle_path output_path title")
            sys.exit(1)

    # Read sample size from the command line argument
    sample_size = int(sys.argv[1])
    dist_module_name = sys.argv[2]
    dist_module_path = sys.argv[3]
    pickle_path = sys.argv[4]
    output_path = sys.argv[5]
    title = sys.argv[6]
    
    process_pickle_files(sample_size, dist_module_name, dist_module_path, pickle_path, output_path, title)

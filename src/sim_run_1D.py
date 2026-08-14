import numpy as np
import pandas as pd
import random
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
from scipy.special import beta as bfahah
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


def main(iteration, dist_module_name, dist_module_path, output_path):
    
    X0 = np.linspace(0, 1, 10000)
    p = 0.5
    a = 1 
    b = 1 
    alpha = 2 
    alpha0 = np.exp(20)
    p0 =  0.5
    start_time = time.time()
    a1 = 8
    b1 = 2
    
    # Retrieve distribution-specific functions and parameters
    dist_module =  import_distribution_module(dist_module_name, dist_module_path)
    extract_samples = dist_module.extract_samples
    compute_pdf = dist_module.compute_pdf
    prefix = dist_module.prefix
    
    
    random.seed(90 + iteration)
    np.random.seed(90 + iteration)
    y = compute_pdf(X0) if compute_pdf else None
    
    dir_path = os.path.join(output_path, f'{prefix}')
    os.makedirs(dir_path, exist_ok=True)
    
    #sample_size_vec = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    sample_size_vec = [500, 5000, 50000]
    for n in sample_size_vec:
        #md = basic.partition(n, 1, 0.5)[0]
    
        X = extract_samples(n)
        
#        if n == 100:
#            depths = [1, 2, 3, 4, 5, 6]
#        
#        elif n == 200:
#            depths = [1, 2, 3, 4, 5, 6, 7]
            
        if n == 500:
            depths = [1, 2, 3, 4, 5, 6, 7, 8]
        
#        elif n == 1000:
#            depths = [1, 2, 3, 4, 5, 6, 7, 9]
#            
#        elif n == 2000:
#            depths = [1, 2, 3, 4, 5, 6, 7, 9, 10]
            
        elif n == 5000:
            depths = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
            
#        elif n == 10000:
#            depths = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
#            
#        elif n == 20000:
#            depths = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
            
        elif n == 50000:
            depths = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
        
        for d in depths:   #range(1, md + 1):
            file_path = os.path.join(dir_path, f'{prefix}_n{n}_depth{d}_itern{iteration}.pkl.gz')
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Skipping...")
                continue
            
            bt, pt_s, time_b, time_p= na.generating_fdepth(X, X0, y, p, a, b, alpha, alpha0, p0, d, d)
        
            # Save the results using pickle
            with gzip.open(file_path, 'wb') as file:
                pickle.dump((bt, pt_s, time_b, time_p, X), file)
    
    end_time = time.time()
    print(f"Total execution time for iteration {iteration}: {end_time - start_time} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run_simulation.py <iteration> <distribution_module> <parameter_file>")
        sys.exit(1)
    
    # Read command line arguments
    iteration = int(sys.argv[1])
    dist_module_name = sys.argv[2]
    dist_module_path = sys.argv[3]
    output_path = sys.argv[4]
    
    # Call the main function with the three arguments
    main(iteration, dist_module_name, dist_module_path, output_path)

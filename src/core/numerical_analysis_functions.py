import importlib.util
import math
import timeit

import matplotlib.pyplot as plt
import numpy as np

from src.core import partition_1D_functions as basic
from src.core import OPT_1D_partial_functions as latent
from src.core import OPT_1D_full_functions as polya



def import_distribution_module(module_name, module_path):
    """Load a distribution spec as a module.

    `io_utils.import_distribution_module` is the one the pipeline calls: it
    resolves paths relative to the repository. This copy keeps the module
    self-contained.
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def generating_PT(X, p, a, b, depth):
    """Fit the fixed-depth Polya tree alone, without the optional-stopping model.

    Currently unused: the simulations call `generating_fdepth`, which fits both
    models and times them.
    """
    
    X = np.sort(X)
    n = len(X)
    #X_aug = list(X)
    #X_aug.insert(0, 0)
    #X_aug.append(1)

    start_time_partial = timeit.default_timer()
    partial_output = latent.PT_partial_model(X, depth, a, b, p)
    end_time_partial = timeit.default_timer()

    start_time_full = timeit.default_timer()
    full_output = polya.PT_full_model(X, depth, a, b)
    end_time_full = timeit.default_timer()
    
    time_partial = end_time_partial - start_time_partial
    time_full = end_time_full - start_time_full
    
    return partial_output, full_output, time_partial, time_full


def generating(X, X0, y, p, a, b, alpha, alpha0, p0, threshold):
    """Fit both models on a threshold-grown partition, timing each.

    The partial model picks the depth, which the full model is then matched to,
    so the two are compared on the same resolution.

    Returns (bt, pt_s, time_bt, time_pt): the two fits and their wall-clock
    times in seconds.
    """
    
    X = np.sort(X)
    n = len(X)
    X_aug = list(X)
    X_aug.insert(0, 0)
    X_aug.append(1)
    
    
    bt, time_bt = basic.measure_execution_time(latent.BT_Optional_Stopping, X, threshold, p, alpha, a, b, alpha0, p0)  
    depth = bt[7]
    
    start_time = timeit.default_timer()
    pvec_size = polya.partition(depth)
    #pvec_size, depth_size = polya.partition_size(X, n, threshold, 0)
    pt_s = polya.OPT(X, pvec_size, depth, alpha, a, b, alpha0, p0)
    end_time = timeit.default_timer()
    
    time_pt = end_time - start_time
    
    return bt, pt_s, time_bt, time_pt
    

# def l1_distance(vector1, vector2): # this formula only works in the [0, 1] interval, otherwise the integral has to be specified
#     if len(vector1) != len(vector2):
#         raise ValueError("Vectors must have the same length")

#     distance = 0
#     for x, y in zip(vector1, vector2):
#         distance += abs(x - y)

#     return distance/len(vector1)

def l1_distance(true_density, estimated_density, dx):
    """
    Compute the L1 distance between the true density and the estimated density.
    
    Parameters:
    - true_density: 1D array representing the true density values at sample points
    - estimated_density: 1D array representing the estimated density values at the same sample points
    - dx: The grid spacing in the x-dimension
    
    Returns:
    - L1 distance as a float
    """
    # Ensure true_density and estimated_density are numpy arrays
    true_density = np.array(true_density)
    estimated_density = np.array(estimated_density)
    
    # Compute the absolute difference between the true and estimated densities
    diff = np.abs(true_density - estimated_density)
    
    # Compute the L1 distance (numerical integration over the grid)
    l1_dist = np.sum(diff * dx) # NOTE THAT THIS FORMULA IS NOT GENERAL!! WE ARE ASSUMING AN EQUALLY SPACED GRID, THUS CAN STILL ACCEfull THIS FORMULA HERE. 
    
    return l1_dist

def l2_distance(true_density, estimated_density, dx):
    """
    Compute the L2 distance between the true density and the estimated density.
    
    Parameters:
    - true_density: 1D array representing the true density values at sample points
    - estimated_density: 1D array representing the estimated density values at the same sample points
    - dx: The grid spacing in the x-dimension
    
    Returns:
    - L2 distance as a float
    """
    # Ensure true_density and estimated_density are numpy arrays
    true_density = np.array(true_density)
    estimated_density = np.array(estimated_density)
    
    # Compute the absolute difference between the true and estimated densities
    square_diff = (true_density - estimated_density)**2
    
    # Compute the L1 distance (numerical integration over the grid)
    l2_dist = np.sum(square_diff * dx)
    
    return np.sqrt(l2_dist)

def linfty_distance(true_density, estimated_density, dx): # We keep dx as argument of the function (although it is not needed for the Linfty computation) becasue it is easier for the distance metric computation later. 
    """
    Compute the L infinity distance between the true density and the estimated density.
    
    Parameters:
    - true_density: 1D array representing the true density values at sample points
    - estimated_density: 1D array representing the estimated density values at the same sample points
    - dx: The grid spacing in the x-dimension
    - dy: The grid spacing in the y-dimension
    
    Returns:
    - Linfty distance as a float
    """
    # Ensure true_density and estimated_density are numpy arrays
    true_density = np.array(true_density)
    estimated_density = np.array(estimated_density)
    
    # Compute the absolute difference between the true and estimated densities
    abs_diff = np.abs(true_density - estimated_density)
    
    # Compute the L1 distance (numerical integration over the grid)
    linfty_dist = np.max(abs_diff)
    
    return linfty_dist


def interpolate_step_function(intervals, heights, grid):
    """Evaluate a step function on `grid`.

    `intervals` holds the len(heights)+1 breakpoints; each height fills its
    interval. Used to put both fits on the common grid the distances are
    computed over.
    """
    if len(intervals) != len(heights) + 1:
        raise ValueError("Number of intervals must be one more than the number of heights")
    num_points = len(grid)
    interpolated_heights = np.zeros(num_points)

    for i in range(len(heights)):
        start, end = intervals[i], intervals[i + 1]
        height = heights[i]
        
        # Interpolate heights linearly within the interval
        mask = (grid >= start) & (grid <= end)
        interpolated_heights[mask] = np.interp(grid[mask], [start, end], [height, height])

    return interpolated_heights


# For zooming maybe just cut the vectors on the interesting region
def distance_metric(grid, partial_prob, partial_int, full_prob, full_int ,true_den, distance_func):
    """Distance from the true density to each of the two fits.

    Interpolates both step functions onto `grid`, then applies `distance_func`
    -- one of l1_distance, l2_distance, linfty_distance -- to each. The grid
    must be equally spaced: the spacing is taken from its first two points.

    Returns (partial, full).
    """

    dx = grid[1] - grid[0] # EQUALLY SPACED GRIDPOINTS
    
    partial_prob = interpolate_step_function(partial_int, partial_prob, grid)
    full_prob = interpolate_step_function(full_int, full_prob, grid)
    
    dist_partial = distance_func(true_den, partial_prob, dx)
    dist_full = distance_func(true_den, full_prob, dx)
     
    return dist_partial, dist_full


def plotting(pt_prob, pt_int, bt_prob, bt_int, X0, X, y, n, alpha, threshold, depth, p, p0, L1_b, L1_p, time_b, time_p):
    """Plot both fitted densities against the truth. Currently unused."""
    
    fig,  ax = plt.subplots()
    line_OPT = ax.stairs(pt_prob, pt_int, color='blue');
    line_BT = ax.stairs(bt_prob, bt_int, color='red');
    line_true = ax.plot(np.sort(X0), y, alpha = 0.5, color='#1f1e1e');
    hist = ax.hist(X, alpha = 0.5, color='#A0A0A0',  bins = 70, density = True);
    
    #ax.fill_between(pt_int[1:len(pt_int)], lower_p, upper_p, step = 'pre', color = 'blue', alpha=0.2)
    #ax.fill_between(bt_int[1:len(bt_int)], lower_b, upper_b, step = 'pre', color = 'red', alpha=0.2)
    
    ax.set_title(f'Approximate Posterior Mean Distribution, n ={n}');
    #fig.suptitle(f'Approximate Posterior Mean Distribution, n ={n}')
    plt.figtext(0.1, -0.2, f'Prior Concentration = {alpha}, Set Size Threshold = {threshold}, \nMaximum Resolution = {depth}, \nOrder statistic = {p},  \nPrior Stopping Probability = {p0}, \nL1 Full = {L1_p}, L1 Partial = {L1_b}, \nTime Full = {time_p}, Time Partial = {time_b}');
    ax.set_ylabel('Probability');
    fig.legend([line_OPT, line_BT], ['Full', 'Partial'], loc = 'lower right');
    
    #ax2.stairs(pt_prob, pt_int, color='blue');
    #ax2.stairs(bt_prob, pt_int, color='red');
    #ax2.hist(X, alpha = 0.5, color='#A0A0A0',  bins = 70, density = True);
    
    return fig, ax, hist


def confidence_bands(nsamples, posterior_sampler, *args):
    """Pointwise credible bands from `nsamples` posterior draws.

    `posterior_sampler` is passed in -- `latent.posterior_sampler` or the full
    model's equivalent -- and called with `*args` once per draw. Currently
    unused: the shipped pipeline reports posterior means rather than bands.
    """
    
    all_samples_p = []
    for iter in range(1, nsamples + 1):
        sample = posterior_sampler(*args)
        all_samples_p.append(sample[0])
        
    lower = np.quantile(all_samples_p, 0.025, axis=0)
    upper = np.quantile(all_samples_p, 0.975, axis=0)
 
    return lower, upper



def plotting_bands(pt_prob, pt_int, lower_p, upper_p,  bt_prob, bt_int, lower_b, upper_b, X0, X, y, n, alpha, threshold, depth, p, p0):
    """Plot both fits with their credible bands. Currently unused."""

    fig, ax = plt.subplots()
    line_OPT = fig.stairs(pt_prob, pt_int, color='blue');
    line_BT = fig.stairs(bt_prob, bt_int, color='red');
    line_true = fig.plot(np.sort(X0), y, alpha = 0.5, color='#1f1e1e')
    #ax.hist(X, alpha = 0.5, color='#A0A0A0',  bins = 70, density = True);

    ax.fill_between(pt_int[1:len(pt_int)], lower_p, upper_p, step = 'pre', color = 'blue', alpha=0.2)
    ax.fill_between(bt_int[1:len(bt_int)], lower_b, upper_b, step = 'pre', color = 'red', alpha=0.2)

    ax.set_title(f'Approximate Posterior Mean Distribution in blue, n ={n}')
    plt.figtext(0.5, 0, f'Alpha = {alpha}, set size = {threshold}, Maximum Resolution = {depth}, \n Order statistic = {p},  Prob = {p0}')

    ax.set_ylabel('Probability')
    ax.legend([line_OPT, line_BT], ['Full', 'Partial'], loc = 'center left')
    
    return fig, ax

def repetitions(iter, X0, y, p, a, b, alpha, alpha0, p0, function, *args, **kwargs):
    """Repeat `function` over `iter` samples, collecting the L1 distances.

    The threshold-grown counterpart of `repetitions_fdepth`.
    """
    L1B = []
    L1P = []
    TB = []
    TP = []
    
    for i in range(0, iter):
        X = function(*args, **kwargs)
        n = np.size(X)
        threshold = np.ceil(n*0.02)
        
        bt, pt_s, time_b, time_p = generating(X, X0, y, p, a, b, alpha, alpha0, p0, threshold)
        
        L1_b, L1_p = distance_metric(X0, bt[5], bt[6], pt_s[5], pt_s[6], y, l1_distance)
        
        L1B.append(L1_b)
        L1P.append(L1_p)
        TB.append(time_b)
        TP.append(time_p)
    
    return np.mean(L1B), np.std(L1B), np.mean(L1P), np.std(L1P), np.mean(TB), np.mean(TP)



def generating_fdepth(X, p, a, b, alpha, alpha0, p0, depthbt, depthpt):
    """Fit both models at fixed depths, timing each.

    The entry point the 1D simulations call: `depthbt` sets the partial model's
    depth and `depthpt` the full model's, so the two can be swept independently.

    Returns (bt, pt_s, time_bt, time_pt): the two fits and their wall-clock
    times in seconds.
    """
    
    X = np.sort(X)
    n = len(X)
    X_aug = list(X)
    X_aug.insert(0, 0)
    X_aug.append(1)
    
    
    bt, time_bt = basic.measure_execution_time(latent.BT_Optional_Stopping_fdepth, X, p, alpha, a, b, alpha0, p0, depthbt)  
    
    start_time = timeit.default_timer()
    pvec_size = polya.partition_full(depthpt)
    pt_s = polya.OPT_full_model(X, pvec_size, depthpt, alpha, a, b, alpha0, p0)
    end_time = timeit.default_timer()
    
    time_pt = end_time - start_time
    
    return bt, pt_s, time_bt, time_pt


def repetitions_fdepth(iter, X0, y, p, a, b, alpha, alpha0, p0, depthbt, depthpt, function, *args, **kwargs):
    """Repeat `function` over `iter` samples at fixed depth.

    Currently unused: the simulations parallelise iterations across SLURM array
    tasks rather than looping here.
    """
    
    L1B = np.zeros((iter, min(depthbt, depthpt)))
    L1P = np.zeros((iter, min(depthbt, depthpt)))
    TB = np.zeros((iter, min(depthbt, depthpt)))
    TP = np.zeros((iter, min(depthbt, depthpt)))
    
    for i in range(0, iter):
        X = function(*args, **kwargs)
        n = np.size(X)
    
        for d in range(1, min(depthbt, depthpt) + 1):
            bt, pt_s, time_b, time_p = generating_fdepth(X, p, a, b, alpha, alpha0, p0, d, d)
            L1_b, L1_p = distance_metric(X0, bt[5], bt[6], pt_s[5], pt_s[6], y, l1_distance)
            
            L1B[i, d-1] = L1_b
            L1P[i, d-1] = L1_p
            TB[i, d-1] = time_b
            TP[i, d-1] = time_p
    
        
    return {"L1B_mean_by_depth" : np.mean(L1B, axis = 0), "L1B_sd_by_depth" : np.std(L1B, axis = 0, ddof = 1), \
            "L1B_l_by_depth" : np.percentile(L1B, 2.5, axis = 0), "L1B_u_by_depth" : np.percentile(L1B, 97.5, axis = 0), \
            "L1P_mean_by_depth" : np.mean(L1P, axis = 0), "L1P_sd_by_depth" : np.std(L1P, axis = 0, ddof = 1), \
            "L1P_l_by_depth" : np.percentile(L1P, 2.5, axis = 0), "L1P_u_by_depth" : np.percentile(L1P, 97.5, axis = 0), \
            "TB_mean_by_depth" : np.mean(TB, axis = 0), "TB_sd_by_depth" : np.std(TB, axis = 0, ddof = 1), \
            "TB_l_by_depth" : np.percentile(TB, 2.5, axis = 0), "TB_u_by_depth" : np.percentile(TB, 97.5, axis = 0), \
            "TP_mean_by_depth" : np.mean(TP, axis = 0), "TP_sd_by_depth" : np.std(TP, axis = 0, ddof = 1), \
            "TP_l_by_depth" : np.percentile(TP, 2.5, axis = 0), "TP_u_by_depth" : np.percentile(TP, 97.5, axis = 0)}

def plots_with_errorbar(x, yP, yP_lower_quantile, yP_upper_quantile, yF, yF_lower_quantile, yF_upper_quantile, shift, markersize, alpha, ylabel, model_name, sample_size, itern, title ="{model_name}, n = {sample_size}, iter = {itern}"):
    """Plot a metric against depth with quantile error bars. Currently unused."""

    x_shifted_l = [t - shift for t in x]
    x_shifted_u = [t + shift for t in x]
    
    fig,  ax = plt.subplots()

    yP_lower_errors = [y - lower for y, lower in zip(yP, yP_lower_quantile)]
    yP_upper_errors = [upper - y for y, upper in zip(yP, yP_upper_quantile)]
    
    
    ax.plot(x_shifted_l, yP, 'o', label='Partial', color='red', markersize=markersize)
    ax.errorbar(x_shifted_l, yP, yerr=[yP_lower_errors, yP_upper_errors], fmt='none', ecolor='red', alpha=alpha, capsize=3.5)

    yF_lower_errors = [y - lower for y, lower in zip(yF, yF_lower_quantile)]
    yF_upper_errors = [upper - y for y, upper in zip(yF, yF_upper_quantile)]

    ax.plot(x_shifted_u, yF, 's', label='Full', color='blue', markersize=markersize)
    ax.errorbar(x_shifted_u, yF, yerr=[yF_lower_errors, yF_upper_errors], fmt='none', ecolor='blue', alpha=alpha, capsize=3.5)
    
    # Add labels and a legend
    ax.set_xlabel('Depth')
    ax.set_ylabel(ylabel)
    formatted_title = title.format(model_name=model_name, sample_size=sample_size, itern = itern)
    ax.set_title(formatted_title)
    ax.legend()

    # Show the plot

    return fig, ax

         

def plots_with_se(x, yP, yPse, yF, yFse, shift, markersize, alpha, ylabel, model_name, sample_size, itern, title="{model_name}, n = {sample_size}, iter = {itern}"):
    """Plot a metric against depth with standard-error bars. Currently unused."""
   # Apply horizontal shift to avoid overlap
    x_shifted_l = [t - shift for t in x]
    x_shifted_u = [t + shift for t in x]
    
    fig, ax = plt.subplots()

    # Plot the first vector with error bars
    ax.plot(x_shifted_l, yP, 'o', label='Partial', color='red', markersize=markersize)
    ax.errorbar(x_shifted_l, yP, yerr=yPse, fmt='none', ecolor='red', alpha=alpha, capsize=3.5)

    # Plot the second vector with error bars
    ax.plot(x_shifted_u, yF, 's', label='Full', color='blue', markersize=markersize)
    ax.errorbar(x_shifted_u, yF, yerr=yFse, fmt='none', ecolor='blue', alpha=alpha, capsize=3.5)
    
    # Add labels and a legend
    ax.set_xlabel('Depth')
    ax.set_ylabel(ylabel)
    formatted_title = title.format(model_name=model_name, sample_size=sample_size, itern=itern)
    ax.set_title(formatted_title)
    ax.legend()

    return fig, ax

import numpy as np
import random
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from bisect import bisect_left, bisect_right
from IPython.display import display
from scipy.special import beta as bf
from scipy.special import comb
import timeit
from scipy.stats import beta
from joblib import Parallel, delayed
import gc

######################################################################################################
############################################# HELPERS ################################################


# def generate_modelscount(dim, max_res):
#     """
#     Generate counts (how many available models) and configurations (how many possible configurations conditionally on the previous recursion  level) for models based on the given dimensions and maximum resolution level.
    
#     Parameters:
#         dim (int): The dimension of the model.
#         max_res (int): The maximum resolution level.
    
#     Returns:
#         tuple: Three lists containing:
#             - nsets_per_reclevel (list): Number of sets per recursion level.
#             - conditional_config (list): Conditional configurations at each level.
#             - total_config (list): Total configurations at each level.
#     """
#     nsets_per_reclevel = [0] * (max_res + 1) 
#     conditional_config = [0] * (max_res + 1) 
#     total_config = [0] * (max_res + 1) 
    
#     for i in range(max_res + 1):
#         nsets_per_reclevel[i] = 2**i
#         conditional_config[i] = 2**nsets_per_reclevel[i-1]
#         total_config[i] = conditional_config[i]*nsets_per_reclevel[i-1]
        
#     return nsets_per_reclevel, conditional_config, total_config

def order_stat(n, p): 
    """
    Calculate the order statistic for a given position in a sorted list.
    
    Parameters:
        n (int): The total number of elements in the list.
        p (float): The desired percentile (a value between 0 and 1).
    
    Returns:
        int: The position in the list corresponding to the given percentile.
    """
    return int(np.ceil(n * p))

def sort_sample(sample, dimension): #ok
    """
    Sort the 2D sample points by the specified dimension.
    
    Parameters:
        sample (numpy.ndarray): The 2D sample points as an array of shape (n, 2).
        dimension (int): The dimension to sort by (0 for x, 1 for y).
    
    Returns:
        numpy.ndarray: The sorted sample.
    """
    return sample[np.argsort(sample[:, dimension])]


def find_bounds(sorted_data, lower_bound, upper_bound):
    """
    Find the indices of the lower and upper bounds in a sorted dataset (giving intervals of this type (a, b] ). 
    
    Parameters:
        sorted_data (list): A list of sorted data elements.
        lower_bound (any): The lower bound value to find the index for.
        upper_bound (any): The upper bound value to find the index for.
    
    Returns:
        tuple: A tuple containing:
            - lower_idx (int): The index of the first element greater than lower_bound.
            - upper_idx (int): The index of the first element not less than upper_bound.
    """

    lower_idx = bisect_right(sorted_data, lower_bound)
    upper_idx = bisect_left(sorted_data, upper_bound) 
    return lower_idx, upper_idx


def create_repeated_vector(max_value): 
    """
    Create a vector with repeated values where each value i is repeated 4^i times.
    
    Parameters:
        max_value (int): The maximum value to be included in the vector.
    
    Returns:
        numpy.ndarray: A vector with values from 0 to max_value, where each value i is repeated 4^i times.
    """
    vector = []
    for i in range(max_value + 1):
        repetitions = 4 ** i
        vector.extend([i] * repetitions)
    return np.array(vector)

def log_sum_exp_multiple(log_values):
    """
    Compute the log of the sum of exponentials of input elements.

    Parameters:
        log_values (list): List of log-values.

    Returns:
        float: The log of the sum of the exponentials of the input values.
    """
    if not log_values:
        return float('-inf')  # log(0) = -inf

    max_log_value = max(log_values)
    
    # Handle cases where max_log_value is infinity
    if math.isinf(max_log_value):
        if all(math.isinf(v) and v < 0 for v in log_values):
            return float('-inf')  # If all are -inf, return -inf
        if all(math.isinf(v) and v > 0 for v in log_values):
            return float('inf')  # If all are +inf, return +inf
        # If mix of +inf and other values, return +inf
        return float('inf')

    sum_exp = 0
    for log_val in log_values:
        if max_log_value - log_val < 50:
            sum_exp += math.exp(log_val - max_log_value)
        elif log_val == max_log_value:
            sum_exp += 1  # When values are equal, avoid computing exp(0)

    return max_log_value + math.log(sum_exp)


def exchange_adjacent_elements_with_offset(vector):
    """
    Exchange adjacent elements in pairs with an offset in a vector (i.e. starting from index 2).

    Parameters:
        vector (list): The input vector.

    Returns:
        list: The modified vector with adjacent elements swapped.
    """
    # Check if the vector has at least two elements
    if len(vector) % 2 == 0:
        raise ValueError("Vector length must be odd.")

    new_vec =  [0]*len(vector)
    new_vec[0] = vector[0]
    # Iterate over the vector starting from index 2 and swap adjacent elements in pairs
    for i in range(1, len(vector) + 1, 2):
        if i + 1 < len(vector):  # Check if there is a next element to swap with
            new_vec[i] = vector[i + 1]
            new_vec[i + 1] = vector[i]

    return new_vec


def round_element(element, decimals):
    """
    Recursively round elements of a list or a single numeric element to the specified number of decimal places.
    
    Parameters:
        element (list, int, float, or other): The element to be rounded. Can be a list of elements or a single numeric value.
        decimals (int): The number of decimal places to round to.
    
    Returns:
        list, int, float, or other: The rounded element. If the input is a list, each numeric element within the list is rounded.
    """
    if isinstance(element, list):
        return [round_element(sub_element, decimals) for sub_element in element]
    elif isinstance(element, (int, float)):
        return round(element, decimals)
    else:
        return element


def round_dataframe(df, decimals):
    """
    Round all numeric elements in a pandas DataFrame to the specified number of decimal places, via the custom rounding function above. 
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the elements to be rounded.
        decimals (int): The number of decimal places to round to.
    
    Returns:
        pandas.DataFrame: The DataFrame with all numeric elements rounded to the specified decimal places.
    """
    return df.apply(lambda col: col.apply(lambda x: round_element(x, decimals)))


def remove_points_from_sample(node_sorted_x, node_sorted_y, removed_points):
    """
    Remove specified points from sorted sample lists of x and y coordinates.
    
    Parameters:
        node_sorted_x (list or numpy.ndarray): The observations falling in the node bounds sorted by x coordinate. 
        node_sorted_y (list or numpy.ndarray):  The observations falling in the node bounds sorted by y coordinate. 
        removed_points (set or list): The points to be removed from the samples (not restricted to the node bounds).
    
    Returns:
        tuple: Two numpy arrays containing the filtered x and y coordinates with the specified points removed.
    """
    # Convert lists of lists to numpy arrays
    node_sorted_x = np.array(node_sorted_x)
    node_sorted_y = np.array(node_sorted_y)
    removed_points = np.array(list(removed_points))
    
    # # Create a mask for sorted_sample_x
    mask_x = ~np.isin(node_sorted_x, removed_points).all(axis=1)
    current_sample_x = node_sorted_x[mask_x]
    
    # # Create a mask for sorted_sample_y
    mask_y = ~np.isin(node_sorted_y, removed_points).all(axis=1)
    current_sample_y = node_sorted_y[mask_y]

    return current_sample_x, current_sample_y


######################################################################################################
########### PARTITION GENERATION REMOVING ONE POINT PER DIMENSION (ORDER STATISTIC) ##################

def order_stat_node(sorted_sample_x, sorted_sample_y, x_bounds, y_bounds, p = 0.5): 
    """
    Compute the order statistic within a specific bounding box for a given percentile in x and y sorted samples.
    
    Parameters:
        sorted_sample_x (numpy.ndarray): 2D array with sorted x and y coordinates, where each row is [x, y].
        sorted_sample_y (numpy.ndarray): 2D array with sorted y and x coordinates, where each row is [y, x].
        x_bounds (tuple): Tuple specifying the (min, max) bounds for x coordinates.
        y_bounds (tuple): Tuple specifying the (min, max) bounds for y coordinates.
        p (float): The percentile for which to find the order statistic (default is 0.5 for median).
    
    Returns:
        dict: A dictionary with keys:
            - "x": The x-coordinate of the order statistic.
            - "y": The y-coordinate of the order statistic.
            - "set_size_lb": Number of elements in the subset less than the order statistic.
            - "set_size_rt": Number of elements in the subset greater than the order statistic.
    """

  
    sorted_x_filtered_by_y = sorted_sample_x[
        (sorted_sample_x[:, 1] > y_bounds[0]) & (sorted_sample_x[:, 1] < y_bounds[1])]
    
    sorted_y_filtered_by_x = sorted_sample_y[
        (sorted_sample_y[:, 0] > x_bounds[0]) & (sorted_sample_y[:, 0] < x_bounds[1])]
    
    # Find bounds within the filtered data
    lower_idx_x, upper_idx_x = find_bounds(sorted_x_filtered_by_y[:, 0], x_bounds[0], x_bounds[1])
    lower_idx_y, upper_idx_y = find_bounds(sorted_y_filtered_by_x[:, 1], y_bounds[0], y_bounds[1])
    
    # Extract subsets for each dimension
    subset_sorted_x = sorted_x_filtered_by_y[lower_idx_x:upper_idx_x]
    subset_sorted_y = sorted_y_filtered_by_x[lower_idx_y:upper_idx_y]
    
    
    # Extract the values from the specified dimension
    values_x = subset_sorted_x[:, 0]
    values_y = subset_sorted_y[:, 1]

    set_size = len(values_x)
    os = order_stat(set_size, p)
    os_index = os - 1
    
    # Return the order statistic (1-based index, so order - 1 for 0-based index)
    
    return {"x": values_x[os_index], "y": values_y[os_index], "set_size_lb": os-1, "set_size_rt": set_size-os}


def node_children(node, sorted_sample_x, sorted_sample_y, p = 0.5):
    """
    Compute the children nodes and their sizes based on the order statistic for a given node in a spatial partition.
    
    Parameters:
        node (list): A 2D list defining the bounds of the current node in the format [[x_min, x_max], [y_min, y_max]].
        sorted_sample_x (numpy.ndarray): 2D array with sorted x and y coordinates, where each row is [x, y].
        sorted_sample_y (numpy.ndarray): 2D array with sorted y and x coordinates, where each row is [y, x].
        p (float): The percentile for which to find the order statistic (default is 0.5 for median).
    
    Returns:
        dict: A dictionary with the following keys:
            - "l_child": The bounds of the left child node.
            - "l_child_size": The size of the left child node.
            - "r_child": The bounds of the right child node.
            - "r_child_size": The size of the right child node.
            - "b_child": The bounds of the bottom child node.
            - "b_child_size": The size of the bottom child node.
            - "t_child": The bounds of the top child node.
            - "t_child_size": The size of the top child node.
    """
    
    t = order_stat_node(sorted_sample_x, sorted_sample_y, node[0], node[1], p)
    x_orderstat = t["x"]
    y_orderstat = t["y"]
    set_size_lb = t["set_size_lb"]
    set_size_rt = t["set_size_rt"]

    l_child = [[node[0][0], x_orderstat], node[1]]
    r_child = [[x_orderstat, node[0][1]], node[1]]
    b_child = [node[0], [node[1][0], y_orderstat]]
    t_child = [node[0], [y_orderstat, node[1][1]]]

    return {"l_child": l_child, "l_child_size": set_size_lb , \
            "r_child": r_child, "r_child_size": set_size_rt, \
            "b_child": b_child, "b_child_size": set_size_lb, \
            "t_child": t_child, "t_child_size": set_size_rt}  


def generate_node_list(sample, max_res): 
    """
    Generate a list of nodes representing a spatial partitioning of the sample.
    
    Parameters:
        sample (numpy.ndarray): The 2D sample points as an array of shape (n, 2).
        max_res (int): The maximum resolution level for partitioning.
    
    Returns:
        dict: A dictionary containing:
            - "part_vec": List of nodes representing the spatial partitions.
            - "set_size_vec": List of sizes of the sets associated with each node.
            - "ratio1": List of ratios related to the partitioning areas for the first dimension.
            - "ratio2": List of ratios related to the partitioning areas for the second dimension.
            - "parent_index_list": List of indices of parent nodes for each node.
            - "res_level": List of resolution levels for each node.
    """
    
#    node_res = create_repeated_vector(max_res)
#    node_list = [0]*len(node_res)
#    parent_index_list = [0]*len(node_res)
#    set_size_list = [0]*len(node_res)


    prev = [[[0, 1], [0, 1]]]
    set_size_list = [len(sample)]
    ratio1 = [np.log(1)]
    ratio2 = [np.log(1)]
    parent_index_list = [0]
    res_level = [0]

    sorted_sample_x = sort_sample(sample, 0)  # Sort by x-coordinate
    sorted_sample_y = sort_sample(sample, 1) 
    

    for m in range(0, max_res):
        l = 4**m
        new = []
        for idx, node in enumerate(prev[-l:]):
            
            node_index = len(prev) - l + idx
            
            sub = node_children(node, sorted_sample_x, sorted_sample_y, p = 0.5)

            left = sub["l_child"]
            right = sub["r_child"]
            bottom = sub["b_child"]
            top = sub["t_child"]

            node_area = np.log(node[0][1] - node[0][0])+ np.log(node[1][1] - node[1][0])
            l_area = np.log(left[0][1] - left[0][0])+ np.log(left[1][1] - left[1][0])
            r_area = np.log(right[0][1] - right[0][0])+ np.log(right[1][1] - right[1][0])
            b_area = np.log(bottom[0][1] - bottom[0][0])+np.log(bottom[1][1] - bottom[1][0])
            t_area = np.log(top[0][1] - top[0][0])+np.log(top[1][1] - top[1][0])
            
     
            # Updating the nodes vector
            new.append(left)
            new.append(right)
            new.append(bottom)
            new.append(top)

            # Updating the nodes prior beta parameters without alpha 
            ratio1.append(l_area - node_area)
            ratio2.append(r_area - node_area)
            
            ratio1.append(r_area - node_area)
            ratio2.append(l_area - node_area)
            
            ratio1.append(b_area - node_area)
            ratio2.append(t_area - node_area)
            
            ratio1.append(t_area - node_area)
            ratio2.append(b_area - node_area)
            
            # Updating the set_size vector
            set_size_list.append(sub["l_child_size"])
            set_size_list.append(sub["r_child_size"])
            set_size_list.append(sub["b_child_size"])
            set_size_list.append(sub["t_child_size"])

            # Update the parent index vector
            parent_index_list.extend([node_index]*4)
            res_level.extend([m+1]*4)
        
        prev.extend(new)
        l = len(new)


    return  {"part_vec": prev, "set_size_vec":set_size_list,\
             "ratio1": np.exp(ratio1), "ratio2": np.exp(ratio2), \
            "parent_index_list": parent_index_list, "res_level": res_level}


def rec_level(node_index): 
    """
    Determine the resolution level of a node based on its index in the 2D partitioning vector containing all possible sets.
    
    Parameters:
        node_index (int): The index of the node whose resolution level is to be determined.
    
    Returns:
        int: The resolution level of the node.
    """
 
    power = 0
    sum_power = 4**power
    while sum_power <= node_index:
        power += 1
        sum_power += 4**power 
    
    level = power
    
    return level

def children_index(node_index):
    """
    Compute the indices of the child nodes for a given node in the 2D partitioning vector containing all possible sets.

    Parameters:
        node_index (int): The index of the node whose children indices are to be determined.
    
    Returns:
        tuple: A tuple containing the indices of the left, right, bottom, and top child nodes.
    """
    level = rec_level(node_index)
    sum = 0
    for i in range(0, level+1) :
        sum += 4**i
    lchild_i = int(sum + 4 * (node_index - (sum - 4**level)))
    rchild_i = lchild_i + 1
    bchild_i = lchild_i + 2
    tchild_i = lchild_i + 3

    return lchild_i, rchild_i, bchild_i, tchild_i

######################################################################################################
###################################### UPDATED PARTITION GENERATION ##################################

def order_stat_node_2(subset_sorted_x, subset_sorted_y, p = 0.5):
    """
    Compute the order statistic within a specific bounding box for a given percentile in x and y sorted samples; 
    this function returns the order statistic points of both dimensions as points to be conditioned upon and adjustst the 
    set count to the case when we condition on two points per set. 
    
    Parameters:
        subset_sorted_x (numpy.ndarray): 
        subset_sorted_y (numpy.ndarray): 
        x_bounds (tuple): Tuple specifying the (min, max) bounds for x coordinates.
        y_bounds (tuple): Tuple specifying the (min, max) bounds for y coordinates.
        p (float): The percentile for which to find the order statistic (default is 0.5 for median).
    
    Returns:
        dict: A dictionary with keys:
            - "x": The x-coordinate of the order statistic.
            - "y": The y-coordinate of the order statistic.
            - "set_size_lb": Number of elements in the subset less than the order statistic.
            - "set_size_rt": Number of elements in the subset greater than the order statistic.
            - "new_removed_points": Order statistic points of each dimension to be conditioned upon.   
    """

     
    # Extract the values from the specified dimension
    values_x = subset_sorted_x[:, 0]
    values_y = subset_sorted_y[:, 1]

    set_size = len(values_x)
    os = order_stat(set_size, p)
    os_index = os - 1
    
    # Return the order statistic (1-based index, so order - 1 for 0-based index)

    #x_point = [t for t in subset_sorted_x if t[0] == values_x[os_index]][0]
    #y_point = [t for t in subset_sorted_y if t[1] == values_y[os_index]][0]

    # Changing approach to find order statistic points to a more efficient version: 
    x_point_idx = np.searchsorted(subset_sorted_x[:, 0], values_x[os_index])
    x_point = subset_sorted_x[x_point_idx]

    y_point_idx = np.searchsorted(subset_sorted_y[:, 1], values_y[os_index])
    y_point = subset_sorted_y[y_point_idx]

    if y_point[0] < x_point[0]:
        lsize = os-2
        rsize = set_size-os
    else:
        lsize = os-1
        rsize = set_size - os - 1

    if x_point[1] < y_point[1]:
        bsize = os-2
        tsize = set_size-os
    else:
        bsize = os-1
        tsize = set_size - os - 1

    
    return {"x": values_x[os_index], "y": values_y[os_index], "x_point": x_point, "y_point": y_point, "lsize": lsize, "rsize": rsize, "bsize": bsize, "tsize": tsize}


def node_children_2(node, sorted_sample_x, sorted_sample_y, removed_points, p=0.5):
    """
    Compute the children nodes and their sizes based on the order statistic for a given node in a spatial partition.
    This function computes node children of a node, excluding the splitting points for both dimensions of parent sets; moreover it returns
    the order statistic points of the node to be split. 
    
    Parameters:
        node (list): A 2D list defining the bounds of the current node in the format [[x_min, x_max], [y_min, y_max]].
        sorted_sample_x (numpy.ndarray): 2D array with sorted x and y coordinates, where each row is [x, y].
        sorted_sample_y (numpy.ndarray): 2D array with sorted y and x coordinates, where each row is [y, x].
        removed_points: removed points up to the node to be split.
        p (float): The percentile for which to find the order statistic (default is 0.5 for median).
    
    Returns:
        dict: A dictionary with the following keys:
            - "l_child": The bounds of the left child node.
            - "l_child_size": The size of the left child node.
            - "r_child": The bounds of the right child node.
            - "r_child_size": The size of the right child node.
            - "b_child": The bounds of the bottom child node.
            - "b_child_size": The size of the bottom child node.
            - "t_child": The bounds of the top child node.
            - "t_child_size": The size of the top child node.
            - "new_removed_points": The order statistics of the node to be split, with respect to each dimension. 
    """
    # Filter the sample by the node
    x_bounds = node[0]
    y_bounds = node[1]
    sorted_x_filtered_by_y = sorted_sample_x[(sorted_sample_x[:, 1] > y_bounds[0]) & (sorted_sample_x[:, 1] < y_bounds[1])]
    
    sorted_y_filtered_by_x = sorted_sample_y[(sorted_sample_y[:, 0] > x_bounds[0]) & (sorted_sample_y[:, 0] < x_bounds[1])]
    
    # Find bounds within the filtered data
    lower_idx_x, upper_idx_x = find_bounds(sorted_x_filtered_by_y[:, 0], x_bounds[0], x_bounds[1])
    lower_idx_y, upper_idx_y = find_bounds(sorted_y_filtered_by_x[:, 1], y_bounds[0], y_bounds[1])
    
    # Extract subsets for each dimension: these are the samples restricted to the node
    subset_sorted_x = sorted_x_filtered_by_y[lower_idx_x:upper_idx_x]
    subset_sorted_y = sorted_y_filtered_by_x[lower_idx_y:upper_idx_y]

    # Remove the points from the sample before processing
    current_sample_x, current_sample_y = remove_points_from_sample(subset_sorted_x, subset_sorted_y, removed_points)
    
    t = order_stat_node_2(current_sample_x, current_sample_y, p)
    
    x_orderstat = t["x"]
    y_orderstat = t["y"]
    x_point = t["x_point"]
    y_point = t["y_point"]
    lsize = t["lsize"]
    rsize = t["rsize"]
    bsize = t["bsize"]
    tsize = t["tsize"]
    
    l_child = [[node[0][0], x_orderstat], node[1]]
    r_child = [[x_orderstat, node[0][1]], node[1]]
    b_child = [node[0], [node[1][0], y_orderstat]]
    t_child = [node[0], [y_orderstat, node[1][1]]]

    # Update the removed points list
    new_removed_points = removed_points.copy()
    new_removed_points.add(tuple(x_point))
    new_removed_points.add(tuple(y_point))

    return {
        "l_child": l_child, "l_child_size": lsize, 
        "r_child": r_child, "r_child_size": rsize, 
        "b_child": b_child, "b_child_size": bsize, 
        "t_child": t_child, "t_child_size": tsize,
        "new_removed_points": new_removed_points
    }

def process_node(length_prev, node, idx, l, sorted_sample_x, sorted_sample_y, removed_points_per_node):
    """Count points and find bounds for one node. Worker for `process_nodes_chunk`."""
    node_index = length_prev - l + idx
    removed_points = removed_points_per_node[node_index]
    sub = node_children_2(node, sorted_sample_x, sorted_sample_y, removed_points)

    left = sub["l_child"]
    right = sub["r_child"]
    bottom = sub["b_child"]
    top = sub["t_child"]

    node_area = np.log(node[0][1] - node[0][0]) + np.log(node[1][1] - node[1][0])
    l_area = np.log(left[0][1] - left[0][0]) + np.log(left[1][1] - left[1][0])
    r_area = np.log(right[0][1] - right[0][0]) + np.log(right[1][1] - right[1][0])
    b_area = np.log(bottom[0][1] - bottom[0][0]) + np.log(bottom[1][1] - bottom[1][0])
    t_area = np.log(top[0][1] - top[0][0]) + np.log(top[1][1] - top[1][0])

    new_removed_points = sub["new_removed_points"]
    l_child_size = sub["l_child_size"]
    r_child_size = sub["r_child_size"]
    b_child_size = sub["b_child_size"]
    t_child_size = sub["t_child_size"]

    return (node_index, new_removed_points, sub, left, right, bottom, top, node_area, l_area, r_area, b_area, t_area, l_child_size, r_child_size, b_child_size, t_child_size)



def process_nodes_chunk(chunk, prev, start_idx, l, sorted_sample_x, sorted_sample_y, removed_points_per_node):
    """Process a chunk of nodes in one parallel task. Passed to joblib `delayed`."""
    results = []
    for idx, node in enumerate(chunk):
        result = process_node(len(prev), node, start_idx + idx, l, sorted_sample_x, sorted_sample_y, removed_points_per_node)
        results.append(result)
    
    return results
 

def generate_node_list_2(sample, max_res):
    """
    Generate a list of nodes representing a spatial partitioning of the sample. It recursively excludes two points per node (order statistics 
    of the node).
    
    Parameters:
        sample (numpy.ndarray): The 2D sample points as an array of shape (n, 2).
        max_res (int): The maximum resolution level for partitioning.
    
    Returns:
        dict: A dictionary containing:
            - "part_vec": List of nodes representing the spatial partitions.
            - "set_size_vec": List of sizes of the sets associated with each node.
            - "ratio1": List of ratios related to the partitioning areas for the first dimension.
            - "ratio2": List of ratios related to the partitioning areas for the second dimension.
            - "parent_index_list": List of indices of parent nodes for each node.
            - "res_level": List of resolution levels for each node.
    """
    prev = [[[0, 1], [0, 1]]]
    set_size_list = [len(sample)]
    ratio1 = [np.log(1)]
    ratio2 = [np.log(1)]
    parent_index_list = [0]
    res_level = [0]

    sorted_sample_x = np.array(sort_sample(sample, 0))  # Sort by x-coordinate
    sorted_sample_y = np.array(sort_sample(sample, 1))

    removed_points_per_node = [set()]

    for m in range(0, max_res):
        l = 4**m
        new = []
        new_removed_points_per_node = []
        current_nodes = prev[-l:]

        if m <= 7: 
            l = 4**m
            new = []
            new_removed_points_per_node = []
            for idx, node in enumerate(prev[-l:]):
                node_index = len(prev) - l + idx
                removed_points = removed_points_per_node[node_index]
                
                sub = node_children_2(node, sorted_sample_x, sorted_sample_y, removed_points, p=0.5)
    
                left = sub["l_child"]
                right = sub["r_child"]
                bottom = sub["b_child"]
                top = sub["t_child"]
    
                node_area = np.log(node[0][1] - node[0][0]) + np.log(node[1][1] - node[1][0])
                l_area = np.log(left[0][1] - left[0][0]) + np.log(left[1][1] - left[1][0])
                r_area = np.log(right[0][1] - right[0][0]) + np.log(right[1][1] - right[1][0])
                b_area = np.log(bottom[0][1] - bottom[0][0]) + np.log(bottom[1][1] - bottom[1][0])
                t_area = np.log(top[0][1] - top[0][0]) + np.log(top[1][1] - top[1][0])
    
                # Updating the nodes vector
                new.append(left)
                new.append(right)
                new.append(bottom)
                new.append(top)
    
                # Adding new removed points for each child node
                new_removed_points_per_node.append(sub["new_removed_points"])
                new_removed_points_per_node.append(sub["new_removed_points"])
                new_removed_points_per_node.append(sub["new_removed_points"])
                new_removed_points_per_node.append(sub["new_removed_points"])
    
                # Updating the nodes prior beta parameters without alpha 
                ratio1.append(l_area - node_area)
                ratio2.append(r_area - node_area)
                
                ratio1.append(r_area - node_area)
                ratio2.append(l_area - node_area)
                
                ratio1.append(b_area - node_area)
                ratio2.append(t_area - node_area)
                
                ratio1.append(t_area - node_area)
                ratio2.append(b_area - node_area)
                
                # Updating the set_size vector
                set_size_list.append(sub["l_child_size"])
                set_size_list.append(sub["r_child_size"])
                set_size_list.append(sub["b_child_size"])
                set_size_list.append(sub["t_child_size"])
    
                # Update the parent index vector
                parent_index_list.extend([node_index] * 4)
                res_level.extend([m + 1] * 4)
            
            prev.extend(new)
            removed_points_per_node.extend(new_removed_points_per_node)
            l = len(new)

        else:
            chunk_number = 4
            chunk_size = int(len(current_nodes)/chunk_number)
    
            chunks = [current_nodes[i:i + chunk_size] for i in range(0, len(current_nodes), chunk_size)]
            start_indices = [i * chunk_size for i in range(len(chunks))]
    
            all_results = Parallel(n_jobs=chunk_number)(delayed(process_nodes_chunk)(chunk, prev, start_idx, l, sorted_sample_x, sorted_sample_y, removed_points_per_node) for chunk, start_idx in zip(chunks, start_indices))
        
            all_results_flat = [result for chunk_results in all_results for result in chunk_results]
            sorted_results = sorted(all_results_flat, key=lambda x: x[0])
                    
            for result in sorted_results:
                (node_index, removed_points, sub, left, right, bottom, top, node_area, l_area, r_area, b_area, t_area, l_child_size, r_child_size, b_child_size, t_child_size) = result
    
                # Updating the nodes vector
                new.append(left)
                new.append(right)
                new.append(bottom)
                new.append(top)
        
                # Adding new removed points for each child node
                new_removed_points_per_node.append(removed_points)
                new_removed_points_per_node.append(removed_points)
                new_removed_points_per_node.append(removed_points)
                new_removed_points_per_node.append(removed_points)
        
                # Updating the nodes prior beta parameters without alpha 
                ratio1.append(l_area - node_area)
                ratio2.append(r_area - node_area)
            
                ratio1.append(r_area - node_area)
                ratio2.append(l_area - node_area)
            
                ratio1.append(b_area - node_area)
                ratio2.append(t_area - node_area)
            
                ratio1.append(t_area - node_area)
                ratio2.append(b_area - node_area)
            
                # Updating the set_size vector
                set_size_list.append(l_child_size)
                set_size_list.append(r_child_size)
                set_size_list.append(b_child_size)
                set_size_list.append(t_child_size)
        
                # Update the parent index vector
                parent_index_list.extend([node_index] * 4)
                res_level.extend([m + 1] * 4)
              
            prev.extend(new)
            removed_points_per_node.extend(new_removed_points_per_node)
            l = len(new)


    return {"part_vec": prev, "set_size_vec": set_size_list, "ratio1": np.exp(ratio1), "ratio2": np.exp(ratio2), "parent_index_list": parent_index_list, "res_level": res_level}



######################################################################################################
######################################### BOTTOM UP REC ##############################################

# Arguments not in log, output in log
def eta_f(base_al, base_ar, n_al, n_ar, c):
    """Log marginal likelihood of a node's split, per dimension.

    The 2D counterpart of the 1D `eta_f`, evaluated separately for an x-split
    and a y-split. Written via lgamma for numerical stability.
    """

    t = - n_al * np.log(base_al) - n_ar*np.log(base_ar) + \
    math.lgamma(c*base_al + n_al) + math.lgamma(c*base_ar + n_ar) - \
    math.lgamma(c + n_al + n_ar) - \
    math.lgamma(c*base_al) - math.lgamma(c*base_ar) + math.lgamma(c)

    return t

# Arguments in log, output in log. this function computes phi, the state is not specified, 
# it depends on the parameters added and particularly in the parameter rho

def phi_f(lambda_x, lambda_y, RHO, eta0_x, eta1_x, eta0_y, eta1_y, \
          PHI0_l, PHI0_r, PHI1_l, PHI1_r, \
          PHI0_b, PHI0_t, PHI1_b, PHI1_t):
    """Combine children's phi values into a node's, over both stopping states.

    Mixes the x-split, y-split and stop branches with their prior log-weights.
    Logs throughout.
    """

    rho00 = RHO[0]
    rho01 = RHO[1]

    t_x0 = lambda_x + rho00 + eta0_x + PHI0_l + PHI0_r
    t_x1 = lambda_x + rho01 + eta1_x + PHI1_l + PHI1_r
    t_y0 = lambda_y + rho00 + eta0_y + PHI0_b + PHI0_t
    t_y1 = lambda_y + rho01 + eta1_y + PHI1_b + PHI1_t

    phi0 = log_sum_exp_multiple([t_x0, t_x1, t_y0, t_y1])
    phi1 = log_sum_exp_multiple([lambda_x + eta1_x + PHI1_l + PHI1_r, lambda_y + eta1_y + PHI1_b + PHI1_t])
    
    return phi0, phi1


def posterior_prob(lambda_j1, rho_j_j1, \
                  eta_j1_s1, phi_s1_l, phi_s1_r, phi_s):
    """Posterior log-probability of each state at a node, normalised by phi0."""

    pp = lambda_j1 + rho_j_j1 + eta_j1_s1 + phi_s1_l + phi_s1_r - phi_s
    
    return pp


def beta_prior(beta1, beta2, c): 
    """Scale the base-measure ratios by the per-depth pseudocount."""
    beta1_c = [c * element for element in beta1]
    beta2_c = [c * element for element in beta2]
    
    return beta1_c, beta2_c

def non_stopping_c(res_level):
    """Pseudocount per node by depth: 2 for the root and its children, depth**2 below.

    The 2D partition has 2*d = 4 children, so the first 5 entries are the root
    plus its depth-1 children. `OPT_nD_functions.non_stopping_c` generalises
    this cutoff to 1 + 2*d.
    """
    nsc = res_level
    nsc[0:5] = [2]*5 
    
    for i in range(5, len(res_level)):
        nsc[i] = res_level[i]**2

    
    return nsc


def process_node_recursion(part_vec, k, ratio1, u1, alpha_vec, alpha0, lambda_x, lambda_y, RHO, phi0, phi1):
    """Bottom-up step for one node. Worker for `process_chunk_recursion`."""
    
    node      = part_vec[-k]
    lchild_i, rchild_i, bchild_i, tchild_i = children_index(len(part_vec) - k) 
    lchild = part_vec[lchild_i]
    rchild = part_vec[rchild_i]
    tchild = part_vec[tchild_i]
    bchild = part_vec[bchild_i]
    
    eta0_x_node = eta_f(ratio1[lchild_i], ratio1[rchild_i], u1[lchild_i], u1[rchild_i], alpha_vec[-k])
    eta0_y_node = eta_f(ratio1[bchild_i], ratio1[tchild_i], u1[bchild_i], u1[tchild_i], alpha_vec[-k])
    
    eta1_x_node = eta_f(ratio1[lchild_i], ratio1[rchild_i], u1[lchild_i], u1[rchild_i], alpha0)
    eta1_y_node = eta_f(ratio1[bchild_i], ratio1[tchild_i], u1[bchild_i], u1[tchild_i], alpha0)
    
    
    phi0_node, phi1_node = phi_f(lambda_x, lambda_y, RHO, eta0_x_node, eta1_x_node, eta0_y_node, eta1_y_node, \
                     phi0[lchild_i], phi0[rchild_i], phi1[lchild_i], phi1[rchild_i], \
                     phi0[bchild_i], phi0[tchild_i], phi1[bchild_i], phi1[tchild_i])
    

    rho00_x_node = posterior_prob(lambda_x, RHO[0], eta0_x_node, phi0[lchild_i], phi0[rchild_i], phi0_node)
    rho01_x_node= posterior_prob(lambda_x, RHO[1], eta1_x_node, phi1[lchild_i], phi1[rchild_i], phi0_node)
  #  rho11_x[-k] = posterior_prob(lambda_x, 0, eta1_x[-k], phi1[lchild_i], phi1[rchild_i], phi1[-k])

    rho00_y_node = posterior_prob(lambda_y, RHO[0], eta0_y_node, phi0[bchild_i], phi0[tchild_i], phi0_node)
    rho01_y_node = posterior_prob(lambda_y, RHO[1], eta1_y_node, phi1[bchild_i], phi1[tchild_i], phi0_node) 

    return (k, eta0_x_node, eta0_y_node, eta1_x_node, eta1_y_node, phi0_node, phi1_node, rho00_x_node, rho01_x_node, rho00_y_node, rho01_y_node)


def process_chunk_recursion(chunk, part_vec, ratio1, u1, alpha_vec, alpha0, lambda_x, lambda_y, RHO, phi0, phi1):
    """Run the bottom-up step over a chunk of nodes. Passed to joblib `delayed`."""

    results = []
    for k in chunk: 
        result = process_node_recursion(part_vec, k, ratio1, u1, alpha_vec, alpha0, lambda_x, lambda_y, RHO, phi0, phi1)
        results.append(result)
        
    return results

def bottom_up_recursion_2(X, max_res, p0, lx, alpha0, depth_parallelization, chunk_number): #ok
    """Run the optional-stopping recursion over the whole 2D partition.

    Sweeps depth by depth from the leaves up, computing each node's eta, phi
    and posterior state probabilities. Nodes at a depth are split into chunks
    and processed in parallel when `chunk_number` > 1.

    This is the main entry point the 2D simulations call.
    """
    
    n = len(X)

    si = generate_node_list_2(X, max_res)
    part_vec = si["part_vec"]
    u1 = si["set_size_vec"]
    u2 = exchange_adjacent_elements_with_offset(u1)
    ratio1 = si["ratio1"]
    ratio2 = si["ratio2"]
    res_level = si["res_level"]
    alpha_vec = non_stopping_c(res_level)

   # b1, b2 = basic.beta_prior(ratio1, ratio2, alpha)
   # b1_u = [x + y for x, y in zip(b1, u1)]
   # b2_u = [x + y for x, y in zip(b2, u2)]

    eta0_x = [0]*(len(part_vec))
    eta0_y = [0]*(len(part_vec))
    
    eta1_x = [0]*(len(part_vec))
    eta1_y = [0]*(len(part_vec))
    
    phi0 = [0]*(len(part_vec))
    phi1 = [0]*(len(part_vec))
    
    rho01_x  = [0]*(len(part_vec))
    rho00_x  = [0]*(len(part_vec))
    #rho11_x  = [0]*(len(part_vec))

    rho01_y  = [0]*(len(part_vec))
    rho00_y  = [0]*(len(part_vec))
    #rho11_y  = [0]*(len(part_vec))

    RHO = [np.log(1-p0), np.log(p0)]

    lambda_x = np.log(lx)
    lambda_y = np.log(1 - lx)

    #data = []

    start = 4**max_res +1 
    for j in reversed(range(0, max_res)):
        
        end = start + 4**j - 1

        if j > depth_parallelization:
        
            current_indices = list(range(start, end+1))
            chunk_size = int(len(current_indices)/chunk_number)
    
            chunks = [current_indices[i:i + chunk_size] for i in range(0, len(current_indices), chunk_size)]
            
            all_results = Parallel(n_jobs=chunk_number)(delayed(process_chunk_recursion)\
                        (chunk, part_vec, ratio1, u1, alpha_vec, alpha0, lambda_x, lambda_y, RHO, phi0, phi1) \
                                                        for chunk in chunks)
            all_results_flat = [result for chunk_results in all_results for result in chunk_results]
            sorted_results = sorted(all_results_flat, key=lambda x: x[0])
             
            for result in sorted_results:
    
                k = result[0]
                eta0_x[-k], eta0_y[-k], eta1_x[-k], eta1_y[-k], phi0[-k], phi1[-k], rho00_x[-k], \
                rho01_x[-k], rho00_y[-k], rho01_y[-k] = result[1:]
                
            start = start + 4**j


        else: 
            for k in range(start, end +1):
    
                node = part_vec[-k]
                lchild_i, rchild_i, bchild_i, tchild_i = children_index(len(part_vec) - k) 
                lchild = part_vec[lchild_i]
                rchild = part_vec[rchild_i]
                tchild = part_vec[tchild_i]
                bchild = part_vec[bchild_i]
                
                eta0_x[-k] = eta_f(ratio1[lchild_i], ratio1[rchild_i], u1[lchild_i], u1[rchild_i], alpha_vec[-k])
                eta0_y[-k] = eta_f(ratio1[bchild_i], ratio1[tchild_i], u1[bchild_i], u1[tchild_i], alpha_vec[-k])
                
                eta1_x[-k] = eta_f(ratio1[lchild_i], ratio1[rchild_i], u1[lchild_i], u1[rchild_i], alpha0)
                eta1_y[-k] = eta_f(ratio1[bchild_i], ratio1[tchild_i], u1[bchild_i], u1[tchild_i], alpha0)
                
                
                phi0[-k], phi1[-k] = phi_f(lambda_x, lambda_y, RHO, eta0_x[-k], eta1_x[-k], eta0_y[-k], eta1_y[-k], \
                                 phi0[lchild_i], phi0[rchild_i], phi1[lchild_i], phi1[rchild_i], \
                                 phi0[bchild_i], phi0[tchild_i], phi1[bchild_i], phi1[tchild_i])
                
    
                rho00_x[-k] = posterior_prob(lambda_x, RHO[0], eta0_x[-k], phi0[lchild_i], phi0[rchild_i], phi0[-k])
                rho01_x[-k] = posterior_prob(lambda_x, RHO[1], eta1_x[-k], phi1[lchild_i], phi1[rchild_i], phi0[-k])
    
                rho00_y[-k] = posterior_prob(lambda_y, RHO[0], eta0_y[-k], phi0[bchild_i], phi0[tchild_i], phi0[-k])
                rho01_y[-k] = posterior_prob(lambda_y, RHO[1], eta1_y[-k], phi1[bchild_i], phi1[tchild_i], phi0[-k]) 
    
            start = start + 4**j

    
    return {"part_vec": part_vec, "ratio1": ratio1, "ratio2": ratio2, "set_size": u1, \
            "eta0_x": np.longdouble(eta0_x), "eta0_y": np.longdouble(eta0_y), \
            "eta1_x": np.longdouble(eta1_x), "eta1_y": np.longdouble(eta1_y), \
            "phi0": np.longdouble(phi0), "phi1": np.longdouble(phi1), \
            "rho01_x": np.longdouble(rho01_x), "rho00_x": np.longdouble(rho00_x),\
            "rho01_y": np.longdouble(rho01_y), "rho00_y": np.longdouble(rho00_y), \
            "alpha_vec": alpha_vec}

def posterior_rho_s_s1(rho_s_s1_x, rho_s_s1_y):
    """Total log-probability of splitting, pooling the x and y branches."""
    return log_sum_exp_multiple([rho_s_s1_x, rho_s_s1_y])

def posterior_ss1_dim(rho_s_s1_dim, rho_s_s1):
    """Log-probability of splitting on one dimension, given that a split occurs."""
    return rho_s_s1_dim - rho_s_s1

def map_parent_index(i):
    """
    Given an index `i` in a binary tree represented as a vector,
    return the index of the parent node.

    Parameters:
        i (int): The index of the node whose parent is to be found.

    Returns:
        int: The index of the parent node, or None if `i` is the root (0).
    """
    if i == 0:
        return 0  # The root node has no parent
    return (i - 1) // 2

######################################################################################################
################################# OPT PARTITION ######################################################


def find_bounds_binary(sorted_data, lower_bound, upper_bound): #ok
    """Index range of `sorted_data` falling within [lower_bound, upper_bound]."""

    lower_idx = bisect_left(sorted_data, lower_bound)
    upper_idx = bisect_left(sorted_data, upper_bound) 
    return lower_idx, upper_idx

def binary_set_count(node, sorted_sample_x):
    """Number of sample points inside a node, from the sorted-index bounds."""
    
    lower_idx_x, upper_idx_x = find_bounds_binary(sorted_sample_x[:, 0], node[0][0], node[0][1])
    filtered_x = sorted_sample_x[lower_idx_x:upper_idx_x]

    node_count = ((filtered_x[:, 1] >= node[1][0]) & (filtered_x[:, 1] < node[1][1])).sum()

    return node_count


def binary_node_children(node, node_size, sorted_sample_x): 
    """The four children of a node under an x- then y-split."""
    
    x_midpoint = np.mean([node[0][0], node[0][1]])
    y_midpoint = np.mean([node[1][0], node[1][1]]) 

    l_child = [[node[0][0], x_midpoint], node[1]]
    r_child = [[x_midpoint, node[0][1]], node[1]]
    b_child = [node[0], [node[1][0], y_midpoint]]
    t_child = [node[0], [y_midpoint, node[1][1]]]

    l_child_size = binary_set_count(l_child, sorted_sample_x)
    r_child_size = node_size - l_child_size
    b_child_size = binary_set_count(b_child, sorted_sample_x)
    t_child_size = node_size - b_child_size
    

    return {"l_child": l_child, "l_child_size": l_child_size , \
            "r_child": r_child, "r_child_size": r_child_size, \
            "b_child": b_child, "b_child_size": b_child_size, \
            "t_child": t_child, "t_child_size": t_child_size} 


def binary_generate_node_list(sample, max_res): 
    """Enumerate the partition nodes to `max_res`, with their counts and bounds."""
    
    prev = [[[0, 1], [0, 1]]]
    set_size_list = [len(sample)]
    parent_index_list = [0]
    res_level = [0]

    sorted_sample_x = sort_sample(sample, 0)  # Sort by x-coordinate
    
    for m in range(0, max_res):
        l = 4**m
        new = []
        for idx, node in enumerate(prev[-l:]):
            
            node_index = len(prev) - l + idx
            
            sub = binary_node_children(node, set_size_list[node_index], sorted_sample_x)

            left = sub["l_child"]
            right = sub["r_child"]
            bottom = sub["b_child"]
            top = sub["t_child"]
             
            # Updating the nodes vector
            new.append(left)
            new.append(right)
            new.append(bottom)
            new.append(top)

            # Updating the nodes prior beta parameters without alpha 
            #ratio1.append(l_area - node_area)
            #ratio2.append(r_area - node_area)
            
            #ratio1.append(r_area - node_area)
            #ratio2.append(l_area - node_area)
            
            #ratio1.append(b_area - node_area)
            #ratio2.append(t_area - node_area)
            
            #ratio1.append(t_area - node_area)
            #ratio2.append(b_area - node_area)
            
            # Updating the set_size vector
            set_size_list.append(sub["l_child_size"])
            set_size_list.append(sub["r_child_size"])
            set_size_list.append(sub["b_child_size"])
            set_size_list.append(sub["t_child_size"])

            # Update the parent index vector
            parent_index_list.extend([node_index]*4)
            res_level.extend([m+1]*4)
        
        prev.extend(new)
        l = len(new)


    return  {"part_vec": prev, "set_size_vec":set_size_list,\
            # "ratio1": np.exp(ratio1), "ratio2": np.exp(ratio2), \
            "parent_index_list": parent_index_list, "res_level": res_level}



################################# OPT RECURSION ######################################################

def binary_bottom_up_recursion(X, max_res, p0, lx, alpha0): 
    """Bottom-up recursion on the binary-search partition.

    As `bottom_up_recursion_2`, but locating points by binary search on the
    sorted sample rather than carrying per-node subsets.
    """
    
    n = len(X)

    si = binary_generate_node_list(X, max_res)
    part_vec = si["part_vec"]
    u1 = si["set_size_vec"]
    u2 = exchange_adjacent_elements_with_offset(u1)
    res_level = si["res_level"]
    alpha_vec = non_stopping_c(res_level)

    #ratio1 = si["ratio1"]
    #ratio2 = si["ratio2"]

    eta0_x = [0]*(len(part_vec))
    eta0_y = [0]*(len(part_vec))
    
    eta1_x = [0]*(len(part_vec))
    eta1_y = [0]*(len(part_vec))
    
    phi0 = [0]*(len(part_vec))
    phi1 = [0]*(len(part_vec))
    
    rho01_x  = [0]*(len(part_vec))
    rho00_x  = [0]*(len(part_vec))
   
    rho01_y  = [0]*(len(part_vec))
    rho00_y  = [0]*(len(part_vec))
   
    RHO = [np.log(1-p0), np.log(p0)]

    lambda_x = np.log(lx)
    lambda_y = np.log(1 - lx)

    #data = []

    start = 4**max_res +1 
    for j in reversed(range(0, max_res)):

        end = start + 4**j - 1
        for k in range(start, end +1):

            node      = part_vec[-k]
            lchild_i, rchild_i, bchild_i, tchild_i = children_index(len(part_vec) - k) 
            lchild = part_vec[lchild_i]
            rchild = part_vec[rchild_i]
            tchild = part_vec[tchild_i]
            bchild = part_vec[bchild_i]
            
            eta0_x[-k] = eta_f(0.5, 0.5, u1[lchild_i], u1[rchild_i], alpha_vec[-k])
            eta0_y[-k] = eta_f(0.5, 0.5, u1[bchild_i], u1[tchild_i], alpha_vec[-k])
            
            eta1_x[-k] = eta_f(0.5, 0.5, u1[lchild_i], u1[rchild_i], alpha0)
            eta1_y[-k] = eta_f(0.5, 0.5, u1[bchild_i], u1[tchild_i], alpha0)
            
            
            phi0[-k], phi1[-k] = phi_f(lambda_x, lambda_y, RHO, eta0_x[-k], eta1_x[-k], eta0_y[-k], eta1_y[-k], \
                             phi0[lchild_i], phi0[rchild_i], phi1[lchild_i], phi1[rchild_i], \
                             phi0[bchild_i], phi0[tchild_i], phi1[bchild_i], phi1[tchild_i])
            

            rho00_x[-k] = posterior_prob(lambda_x, RHO[0], eta0_x[-k], phi0[lchild_i], phi0[rchild_i], phi0[-k])
            rho01_x[-k] = posterior_prob(lambda_x, RHO[1], eta1_x[-k], phi1[lchild_i], phi1[rchild_i], phi0[-k])

            rho00_y[-k] = posterior_prob(lambda_y, RHO[0], eta0_y[-k], phi0[bchild_i], phi0[tchild_i], phi0[-k])
            rho01_y[-k] = posterior_prob(lambda_y, RHO[1], eta1_y[-k], phi1[bchild_i], phi1[tchild_i], phi0[-k]) 


        start = start + 4**j

   
    
    return {"part_vec": part_vec, "ratio1": [0.5] * len(u1), "set_size": u1, \
            "eta0_x": np.longdouble(eta0_x), "eta0_y": np.longdouble(eta0_y), \
            "eta1_x": np.longdouble(eta1_x), "eta1_y": np.longdouble(eta1_y), \
            "phi0": np.longdouble(phi0), "phi1": np.longdouble(phi1), \
            "rho01_x": np.longdouble(rho01_x), "rho00_x": np.longdouble(rho00_x),\
            "rho01_y": np.longdouble(rho01_y), "rho00_y": np.longdouble(rho00_y), \
            "alpha_vec": alpha_vec}




def log_diff_exp(log_a, log_b):
    """
    Compute the log of the difference of exponentials of input elements.

    Parameters:
        log_a (float): The logarithm of the first number (should be greater than or equal to log_b).
        log_b (float): The logarithm of the second number.

    Returns:
        float: The log of the difference of the exponentials of the input values.
    """
    if log_a < log_b:
        raise ValueError("log_a must be greater than or equal to log_b to ensure the result is real and positive.")

    # If log_a and log_b are very close, use a numerically stable approach
    if log_a == log_b:
        return float('-inf')  # log(a - a) = log(0) = -inf

    # Compute the difference in a numerically stable way
    return log_a + math.log1p(-math.exp(log_b - log_a))



#################################################################################################
############################### MC APPROACH ###################################################

def sample_tree_and_probability(max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0):
    """Draw one tree from the posterior and return its log-probability.

    Walks down from the root, sampling each node's state (stop, split on x,
    split on y) from its posterior.
    """

    tree = []           # this vector collects all the nodes in the sampled tree
    tree.append(part_vec[0])
    original_index = [0]    # this vector collects the indices in the original partition vector for the nodes in the sampled tree
    parent_index = [0]  # this vector collects the index of the parent node for vectors in the sampled tree
    node_state = []     # this vector collects the state of a node in the map tree (0 if state 0, 1 if state 1)
    split_dim = []
    prob_assignment = [0]
    
    m = 0
    l = 1
    while m <= max_res-1:
        new = []
        
        for node in tree[-l:]:
            # define the node indices in the map tree vector and in the full model vector
            k = tree.index(node)
            node_pvec_idx = original_index[k]
            
            # define the node size as in the full model set size vector 
            node_size = u1[node_pvec_idx]
        
            # define the parent node in the map tree vector and in the original full model vector 
            parent_idx = parent_index[k]
            node_prob = prob_assignment[k]
            
            if k == 0:
                parent_state = 0
            else: 
                parent_state = node_state[parent_idx]
           
            lchild_i, rchild_i, bchild_i, tchild_i = children_index(node_pvec_idx)  #define the node children
    
            if parent_state == 0:
                
                rho0_0 = posterior_rho_s_s1(rho0_0x[node_pvec_idx], rho0_0y[node_pvec_idx])  
                rho0_1 = posterior_rho_s_s1(rho0_1x[node_pvec_idx], rho0_1y[node_pvec_idx])
                rho0_1_exp = np.clip(np.exp(rho0_1), 0, 1)
                s = np.random.binomial(1, rho0_1_exp)
                    
                if s == 0: # no stopping
                    node_state.append(0)
                    cpar = alpha_vec[node_pvec_idx]
                    
                    rho00_x = rho0_0x[node_pvec_idx] - rho0_0 # splitting dimension
    
                    rho00_x_exp = np.clip(np.exp(rho00_x), 0, 1)
                    dx = np.random.binomial(1, rho00_x_exp)
                    
    
                    if dx == 1:
                        split_dim.append("x")
                        child1_i = lchild_i
                        child2_i = rchild_i
                        prob1 = float(np.random.beta(cpar*ratio1[lchild_i] + u1[lchild_i],\
                                        cpar*ratio1[rchild_i] + u1[rchild_i], size=1))
                        prob2 = 1-prob1  # numerical issue? 
                        
                    
                    elif dx == 0:
                        split_dim.append("y")
                        child1_i = bchild_i
                        child2_i = tchild_i
                        prob1 = float(np.random.beta(cpar*ratio1[bchild_i] + u1[bchild_i],\
                                        cpar*ratio1[tchild_i] + u1[tchild_i], size=1))
                        prob2 = 1-prob1  # numerical issue? 
                  
        
                elif s == 1: # stopping
                    node_state.append(1)
                    cpar = alpha0
                    
                    rho01_x = rho0_1x[node_pvec_idx] - rho0_1
    
                    rho01_x_exp = np.clip(np.exp(rho01_x), 0, 1)
                    dx = np.random.binomial(1, rho01_x_exp)
                    
    
                    if dx == 1:
                        split_dim.append("x")
                        child1_i = lchild_i
                        child2_i = rchild_i
                        prob1 = float(np.random.beta(cpar*ratio1[lchild_i] + u1[lchild_i],\
                                        cpar*ratio1[rchild_i] + u1[rchild_i], size=1))
                        prob2 = 1-prob1  # numerical issue? 
                        
                    
                    elif dx == 0:
                        split_dim.append("y")
                        child1_i = bchild_i
                        child2_i = tchild_i
                        prob1 = float(np.random.beta(cpar*ratio1[bchild_i] + u1[bchild_i],\
                                        cpar*ratio1[tchild_i] + u1[tchild_i], size=1))
                        prob2 = 1-prob1  # numerical issue? 
             
    
            elif parent_state == 1:
                node_state.append(1)
                cpar = alpha0
                
                rho11_x = np.log(lx) 
                rho11_x_exp = np.clip(np.exp(rho11_x), 0, 1)
                dx = np.random.binomial(1, rho11_x_exp)
    
                if dx == 1:
                        split_dim.append("x")
                        child1_i = lchild_i
                        child2_i = rchild_i
                        prob1 = float(np.random.beta(cpar*ratio1[lchild_i] + u1[lchild_i],\
                                        cpar*ratio1[rchild_i] + u1[rchild_i], size=1))
                        prob2 = 1-prob1  # numerical issue? 
                        
                    
                elif dx == 0:
                        split_dim.append("y")
                        child1_i = bchild_i
                        child2_i = tchild_i
                        prob1 = float(np.random.beta(cpar*ratio1[bchild_i] + u1[bchild_i],\
                                        cpar*ratio1[tchild_i] + u1[tchild_i], size=1))
                        prob2 = 1-prob1
    
            child1 = part_vec[child1_i]
            child2 = part_vec[child2_i]
                
            new.append(child1)
            new.append(child2)
            original_index.append(child1_i)
            original_index.append(child2_i)
            parent_index.append(k)
            parent_index.append(k)
            prob_assignment.append(node_prob + np.log(prob1))
            prob_assignment.append(node_prob + np.log(prob2))
                
    
        tree.extend(new)
        l = len(new)
         
        m = m + 1
    
    split_dim.extend(["t"]*len(new))

    last_sets = [tree[i] for i in range(len(tree)) if split_dim[i] == "t"]
    sample_prob = [prob_assignment[i] for i in range(len(tree)) if split_dim[i] == "t"]
    
    for i in range(0, len(last_sets)):
        node = last_sets[i]
        node_area = np.log(node[0][1] - node[0][0])+ np.log(node[1][1] - node[1][0])
    
        sample_prob[i] = sample_prob[i] - node_area

    return {"sets": last_sets, "prob": np.exp(sample_prob), \
            "tree": tree, "node_states": node_state, "original_index": original_index,\
            "parent_index": parent_index, "split_dimension": split_dim}





# Function to get a posterior sample on a given tree

def posterior_prob_cond_tree(tree, original_index, parent_index, node_states, alpha_mat, ratio1, set_sizes, max_res):
    """
    Compute posterior probability conditional on tree by computing and tracking beta assignments
    
    Parameters:
    - tree: list of nodes
    - original_index: indices in partition vector
    - parent_index: parent indices in tree
    - node_states: states for each node
    - alpha_mat: array of alpha values for each state and node
    - ratio1: array of ratios
    - set_sizes: array of node sizes
    - max_res: maximum resolution level
    """
    n_leaves = 2**max_res
    prob_values = np.zeros(n_leaves)  # store probabilities for leaves
    
    # For each leaf pair
    for j in range(0, n_leaves, 2):
        # Start from leaf nodes
        index_leaf = len(tree) - n_leaves + j
        
        # Initialize probabilities for this path
        accumulated_log_prob_left = 0
        accumulated_log_prob_right = 0
        temp_index = index_leaf
        
        # Track back through the tree
        while temp_index > 0:  # stop at root
            parent_idx = parent_index[temp_index]
            parent_state = node_states[parent_idx]
            node_pvec_idx = original_index[parent_idx]
            
            # Get the two children indices
            left_idx = temp_index if temp_index % 2 == 1 else temp_index - 1
            right_idx = left_idx + 1
            
            # Compute beta probability
            left_pvec_idx = original_index[left_idx]
            right_pvec_idx = original_index[right_idx]
            alpha_val = alpha_mat[parent_state][node_pvec_idx]
            
            prob = float(np.random.beta(alpha_val*ratio1[left_pvec_idx] + set_sizes[left_pvec_idx],
                                      alpha_val*ratio1[right_pvec_idx] + set_sizes[right_pvec_idx], size=1))
            
            # Accumulate probabilities based on whether current node is left or right child
            if temp_index % 2 == 1:  # left child
                accumulated_log_prob_left += np.log(prob)
                accumulated_log_prob_right += np.log(prob)
            else:  # right child
                accumulated_log_prob_left += np.log(1 - prob)
                accumulated_log_prob_right += np.log(1 - prob)
            
            temp_index = parent_idx
        
        # Store the probabilities
        prob_values[j] = np.exp(accumulated_log_prob_left)
        prob_values[j + 1] = np.exp(accumulated_log_prob_right)
    
    last_sets = tree[-n_leaves:]

    for i in range(len(last_sets)):
        node = last_sets[i]
        # Compute log volume by summing over all dimensions
        node_volume = sum(np.log(node[dim][1] - node[dim][0]) for dim in range(len(node)))
        
        prob_values[i] = prob_values[i] - node_volume
    
    return {
        "sets": last_sets,
        "prob": prob_values
    }

def process_sample_MCp(max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid, seed):
    """One posterior draw of the density. Worker for `process_seed_chunk_MCp`."""
    
    random.seed(seed)
    np.random.seed(seed)
    
    sample = sample_tree_and_probability(max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0)
    return interpolate_step_function_2d(sample["sets"], sample["prob"], x_grid, y_grid)

# def sampling(iter, x_grid, y_grid, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, base_seed, n_jobs):

#     seeds = [base_seed + i for i in range(iter)]

#     sum_probs = np.zeros(x_grid.shape)
#     results = Parallel(n_jobs)(delayed(process_sample_MCp)(max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid, seeds[i]) for i in range(iter))

#     for result in results:
#         sum_probs += result

#     return sum_probs/iter

def process_seed_chunk_MCp(seed_chunk, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid):
    """Accumulate posterior draws for a chunk of seeds. Passed to joblib `delayed`."""
    
    chunk_sum_probs = 0
    
    for seed in seed_chunk:
        result = process_sample_MCp(max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid, seed)
        chunk_sum_probs += result
        
    gc.collect()
    return chunk_sum_probs

def process_seed_chunk_MCp_bands(seed_chunk, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid):
    """As process_seed_chunk_MCp, but keeps every draw instead of accumulating a sum."""
    chunk_draws = []

    for seed in seed_chunk:
        result = process_sample_MCp(max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid, seed)
        chunk_draws.append(np.asarray(result))

    gc.collect()
    return chunk_draws


def sampling_MCp_bands(iter, x_grid, y_grid, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, base_seed, n_jobs, lower_q=0.025, upper_q=0.975):
    """Posterior mean and pointwise credible bands, drawing a probability
    allocation conditional on each sampled tree.

    The band counterpart of sampling_MCp, in the same way sampling_MCt_bands is
    the counterpart of sampling_MCt. The difference between the two families is
    which quantity is averaged: sampling_MCt averages the posterior mean
    conditional on each sampled tree, while this one averages a Beta probability
    allocation drawn conditional on each sampled tree. They share a posterior
    mean but not a spread, so the bands differ.

    This is the path the flow-cytometry analysis used (`sample_tree_and_cond_prob`
    in the older dimension-generic implementation), which is why the bands it
    produced cannot be reproduced from sampling_MCt_bands.

    Returns
    -------
    dict with "mean", "lower_band", "upper_band" (each shaped like x_grid).
    """
    seeds = [base_seed + i for i in range(iter)]

    chunk_size = iter // n_jobs
    remainder = iter % n_jobs

    seed_chunks = []
    start = 0
    for i in range(n_jobs):
        current_chunk_size = chunk_size + 1 if i < remainder else chunk_size
        seed_chunks.append(seeds[start:start + current_chunk_size])
        start += current_chunk_size

    chunk_results = Parallel(n_jobs=n_jobs)(delayed(process_seed_chunk_MCp_bands)(seed_chunk, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid) for seed_chunk in seed_chunks)

    draws = np.stack([draw for chunk in chunk_results for draw in chunk])

    return {
        "mean": np.mean(draws, axis=0),
        "lower_band": np.quantile(draws, lower_q, axis=0),
        "upper_band": np.quantile(draws, upper_q, axis=0),
    }


def sampling_MCp(iter, x_grid, y_grid, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, base_seed, n_jobs):
    """Posterior mean density by Monte Carlo over sampled trees.

    Averages `n_mc_samples` draws, parallelised over seed chunks.
    """

    seeds = [base_seed + i for i in range(iter)]

    sum_probs = np.zeros(x_grid.shape)

    total_seeds = iter

    chunk_size = total_seeds // n_jobs
    remainder = total_seeds % n_jobs

    seed_chunks = []
    start = 0

    # Distribute chunks across CPUs
    for i in range(n_jobs):
        # Each CPU gets the base chunk size, plus one extra iteration if there's a remainder
        current_chunk_size = chunk_size + 1 if i < remainder else chunk_size
        seed_chunks.append(seeds[start:start + current_chunk_size])
        start += current_chunk_size
    
    chunk_results = Parallel(n_jobs=n_jobs)(delayed(process_seed_chunk_MCp)(seed_chunk, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid) for seed_chunk in seed_chunks)


    for chunk_sum in chunk_results:
        sum_probs += chunk_sum

    return sum_probs/iter

#################################################################################################
############################### PPD APPROACH ####################################################


def xpred_node_children(x_pred, node_index, part_vec):
    """Children of the node containing a prediction point."""

    node = part_vec[node_index]
    length_x = node[0][1] - node[0][0]
    length_y = node[1][1] - node[1][0]

    lchild_i, rchild_i, bchild_i, tchild_i = children_index(node_index)
    
    if  node[0][0] <= x_pred[0] < (node[0][0] + length_x/2):
        node_x = lchild_i
    else:
        node_x = rchild_i

    if  node[1][0] <= x_pred[1] < (node[1][0] + length_y/2):
        node_y = bchild_i
    else:
        node_y = tchild_i

    return node_x, node_y
    

def xpred_update_setsizevec(x_pred, part_vec, set_size_vec, max_res):
    """Update node counts to include a prediction point."""
    # Start from the root node, assumed to be node 0
    nodes_contain_xpred = [0]

    l = 0

    while l < max_res:
        new_nodes = []  # To track the next set of nodes containing x_pred
        
        for node_index in nodes_contain_xpred:
            # Find the child nodes of the current node that contain x_pred
            node_x, node_y = xpred_node_children(x_pred, node_index, part_vec)
            
            # Update the set size counts for the child nodes
            set_size_vec[node_x] += 1
            set_size_vec[node_y] += 1
            
            # Add the children to the list of nodes to check further
            new_nodes.append(node_x)
            new_nodes.append(node_y)
        
        # Move to the next level of nodes that contain x_pred
        nodes_contain_xpred = new_nodes
        l = l+1

    return set_size_vec


def xpred_phi0_full(x_pred, max_res, p0, lx, alpha0, results_binary): 
    """Recompute phi0 along the path to a prediction point, with that point added.

    The posterior predictive at a point is the ratio of this to the phi0 of the
    observed sample alone.
    """
    
    part_vec = results_binary["part_vec"]
    ratio1 = results_binary["ratio1"]
    set_size = results_binary["set_size"]
    
    #res_level = results_binary["res_level"]
    eta0_x = results_binary["eta0_x"]
    eta0_y = results_binary["eta0_y"]
    eta1_x = results_binary["eta1_x"]
    eta1_y = results_binary["eta1_y"]
    phi0 = results_binary["phi0"]
    phi1 = results_binary["phi1"]
    rho01_x = results_binary["rho01_x"]
    rho00_x = results_binary["rho00_x"]
    rho01_y = results_binary["rho01_y"]
    rho00_y = results_binary["rho00_y"]
    alpha_vec = results_binary["alpha_vec"]
    
    RHO = [np.log(1-p0), np.log(p0)]

    lambda_x = np.log(lx)
    lambda_y = np.log(1 - lx)

    set_size_updated = xpred_update_setsizevec(x_pred, part_vec, set_size, max_res)
    
    start = 4**max_res +1 
    for j in reversed(range(0, max_res)):

        end = start + 4**j - 1
        for k in range(start, end +1):

            if set_size_updated[-k] > set_size[-k]:

                node      = part_vec[-k]
                lchild_i, rchild_i, bchild_i, tchild_i = children_index(len(part_vec) - k) 
                lchild = part_vec[lchild_i]
                rchild = part_vec[rchild_i]
                tchild = part_vec[tchild_i]
                bchild = part_vec[bchild_i]
            
                eta0_x[-k] = eta_f(0.5, 0.5, set_size_updated[lchild_i], set_size_updated[rchild_i], alpha_vec[-k])
                eta0_y[-k] = eta_f(0.5, 0.5, set_size_updated[bchild_i], set_size_updated[tchild_i], alpha_vec[-k])
            
                eta1_x[-k] = eta_f(0.5, 0.5, set_size_updated[lchild_i], set_size_updated[rchild_i], alpha0)
                eta1_y[-k] = eta_f(0.5, 0.5, set_size_updated[bchild_i], set_size_updated[tchild_i], alpha0)
            
            
                phi0[-k], phi1[-k] = phi_f(lambda_x, lambda_y, RHO, eta0_x[-k], eta1_x[-k], eta0_y[-k], eta1_y[-k], \
                                 phi0[lchild_i], phi0[rchild_i], phi1[lchild_i], phi1[rchild_i], \
                                 phi0[bchild_i], phi0[tchild_i], phi1[bchild_i], phi1[tchild_i])
                

                rho00_x[-k] = posterior_prob(lambda_x, RHO[0], eta0_x[-k], phi0[lchild_i], phi0[rchild_i], phi0[-k])
                rho01_x[-k] = posterior_prob(lambda_x, RHO[1], eta1_x[-k], phi1[lchild_i], phi1[rchild_i], phi0[-k])

                rho00_y[-k] = posterior_prob(lambda_y, RHO[0], eta0_y[-k], phi0[bchild_i], phi0[tchild_i], phi0[-k])
                rho01_y[-k] = posterior_prob(lambda_y, RHO[1], eta1_y[-k], phi1[bchild_i], phi1[tchild_i], phi0[-k]) 


        start = start + 4**j

   
    
    return {"phi0_xpred": np.longdouble(phi0[0])}


def process_chunk_posterior_predictive(chunk, max_res, p0, lx, alpha0, results_binary):
    """Posterior predictive over a chunk of points. Passed to joblib `delayed`."""
    # Process each prediction point in the chunk
    results = []
    for x_pred in chunk:
        result = xpred_phi0_full(x_pred, max_res, p0, lx, alpha0, results_binary)
        
        numeric_value = result.get('phi0_xpred')
        results.append(numeric_value)
    
    return np.array(results, dtype=np.longdouble)


def posterior_predictive_OPT(x0, y0, max_res, p0, lx, alpha0, results_binary, n_jobs=-1):
    """Posterior predictive density on a grid. Currently unused by the simulations."""
    
    # Generate meshgrid and coordinates
    #x0, y0 = np.meshgrid(np.linspace(0.00001, 0.99999, marginal_points), np.linspace(0.00001, 0.99999, marginal_points))
    X_pred = np.column_stack([x0.ravel(), y0.ravel()])

    phi0_root = np.longdouble(results_binary["phi0"][0])
    
    # Split the work into chunks
    num_jobs = len(X_pred)
    n_jobs = min(n_jobs if n_jobs > 0 else num_jobs, num_jobs)  # Use at most `n_jobs` or `num_jobs` if fewer
    chunk_size = num_jobs // n_jobs  # Base chunk size
    remainder = num_jobs % n_jobs  # The remainder to distribute among chunks
    
    # Create a list of indices for each chunk
    chunks = []
    start = 0
    for i in range(n_jobs):
        # Each CPU gets the base chunk size, plus one extra iteration if there's a remainder
        current_chunk_size = chunk_size + 1 if i < remainder else chunk_size
        chunks.append(X_pred[start:start + current_chunk_size])
        start += current_chunk_size
    

    # Process chunks in parallel
    phi0_xpred_chunks = Parallel(n_jobs=n_jobs)(delayed(process_chunk_posterior_predictive)(chunk, max_res, p0, lx, alpha0, results_binary) for chunk in chunks)

    # Flatten the result and subtract the root value
    phi0_xpred = np.array([item for sublist in phi0_xpred_chunks for item in sublist], dtype=np.longdouble)
    phi0_xpred -= phi0_root

    # Reshape result back to a grid
    return np.exp(phi0_xpred).reshape(x0.shape)

#################################################################################################
############################### MC - SAMPLE TREE ################################################


def sample_tree(max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0):
    """Draw one tree from the posterior, returning its nodes and their states."""

    tree = []           # this vector collects all the nodes in the sampled tree
    tree.append(part_vec[0])
    original_index = [0]    # this vector collects the indices in the original partition vector for the nodes in the sampled tree
    parent_index = [0]  # this vector collects the map index of the parent node for vectors in the map tree
    node_state = []        # this vector collects the state of a node in the map tree (0 if state 0, 1 if state 1)
    split_dim = []
    
    m = 0
    l = 1
    while m <= max_res-1:
        new = []
        
        for node in tree[-l:]:
            
            k = tree.index(node)
            node_pvec_idx = original_index[k]
            
            node_size = u1[node_pvec_idx]
        
            parent_idx = parent_index[k]
           
            
            if k == 0:
                parent_state = 0
            else: 
                parent_state = node_state[parent_idx]
           
            lchild_i, rchild_i, bchild_i, tchild_i = children_index(node_pvec_idx)  #define the node children
    
            if parent_state == 0:
                
                rho0_0 = posterior_rho_s_s1(rho0_0x[node_pvec_idx], rho0_0y[node_pvec_idx])   
                rho0_1 = posterior_rho_s_s1(rho0_1x[node_pvec_idx], rho0_1y[node_pvec_idx])
                rho0_1_exp = np.clip(np.exp(rho0_1), 0, 1)
                s = np.random.binomial(1, rho0_1_exp)
                    
                if s == 0: # no stopping
                    node_state.append(0)
                    cpar = alpha_vec[node_pvec_idx]
                    
                    rho00_x = rho0_0x[node_pvec_idx] - rho0_0 # splitting dimension
    
                    rho00_x_exp = np.clip(np.exp(rho00_x), 0, 1)
                    dx = np.random.binomial(1, rho00_x_exp)
                    
    
                    if dx == 1:
                        split_dim.append("x")
                        child1_i = lchild_i
                        child2_i = rchild_i            
                    
                    elif dx == 0:
                        split_dim.append("y")
                        child1_i = bchild_i
                        child2_i = tchild_i
                  
                elif s == 1: # stopping
                    node_state.append(1)
                    cpar = alpha0
                    
                    rho01_x = rho0_1x[node_pvec_idx] - rho0_1
    
                    rho01_x_exp = np.clip(np.exp(rho01_x), 0, 1)
                    dx = np.random.binomial(1, rho01_x_exp)
                    
    
                    if dx == 1:
                        split_dim.append("x")
                        child1_i = lchild_i
                        child2_i = rchild_i      
                    
                    elif dx == 0:
                        split_dim.append("y")
                        child1_i = bchild_i
                        child2_i = tchild_i
    
            elif parent_state == 1:
                node_state.append(1)
                cpar = alpha0
                
                rho11_x = np.log(lx) 
                rho11_x_exp = np.clip(np.exp(rho11_x), 0, 1)
                dx = np.random.binomial(1, rho11_x_exp)
    
                if dx == 1:
                        split_dim.append("x")
                        child1_i = lchild_i
                        child2_i = rchild_i
                       
                elif dx == 0:
                        split_dim.append("y")
                        child1_i = bchild_i
                        child2_i = tchild_i
                       
            child1 = part_vec[child1_i]
            child2 = part_vec[child2_i]
                
            new.append(child1)
            new.append(child2)
            original_index.append(child1_i)
            original_index.append(child2_i)
            parent_index.append(k)
            parent_index.append(k)          
    
        tree.extend(new)
        l = len(new)
         
        m = m + 1
    
    split_dim.extend(["t"]*len(new))

    return {"tree": tree, "node_states": node_state, "original_index": original_index,\
            "parent_index": parent_index, "split_dimension": split_dim}

def csi_f(RHO_post, c0, c1, base_child, n_set, n_child, csi0_child, csi1_child):
    """Propagate the posterior mean density from children up to a node.

    The 2D counterpart of the 1D `csi_f`, mixing the stop and split branches
    with the posterior state probabilities. Logs throughout.
    """
    
    csi1 = np.log(c1*base_child + n_child) - np.log(c1 + n_set - 1) - np.log(base_child) + csi1_child
    
    t1 = RHO_post[0] + np.log(c0*base_child + n_child) - np.log(c0 + n_set - 1) - np.log(base_child) + csi0_child
    t2 = RHO_post[1] + csi1
    
    csi0 = log_sum_exp_multiple([t1, t2])
    
    return csi0, csi1

def OPT_posterior_mean_cond_tree(tree, original_index, parent_index, max_res, rho0_0x, rho0_0y, rho0_1x, rho0_1y, alpha_vec, alpha0, ratio1, u1):
    """Posterior mean density given a fixed tree.

    `original_index` maps each node of the sampled tree back to its position in
    the full partition vector.
    """
    
    csi0 = np.zeros((2**max_res, max_res + 1))
    csi1 = np.zeros((2**max_res, max_res + 1))

    for j in range(0, 2**max_res):
        
        index_child_tree = 2**max_res+j-1 # h is the index of the child set in the sampled tree
        node_child = tree[index_child_tree] 

        for r in reversed(range(0, max_res)):

            index_node_tree = parent_index[index_child_tree] # index_node_tree is the index of the node in the sampled tree
            index_node_pvec = original_index[index_node_tree] # index_node_pvec is the index of the node in part_vec 
            index_child_pvec = original_index[index_child_tree] # index_child_pvec is the index of the child in part_vec
            node = tree[index_node_tree] 

            rho0_0 = posterior_rho_s_s1(rho0_0x[index_node_pvec], rho0_0y[index_node_pvec])   
            rho0_1 = posterior_rho_s_s1(rho0_1x[index_node_pvec], rho0_1y[index_node_pvec])
         
            csi0[j][r], csi1[j][r]  = csi_f([rho0_0, rho0_1], alpha_vec[index_child_pvec], alpha0, \
                                         ratio1[index_child_pvec], u1[index_node_pvec], u1[index_child_pvec], csi0[j][r+1], csi1[j][r+1])
    # ALPHA OF THE CHILD OR OF THE NODE FOR THE NON-STOPPING CASE???
            index_child_tree = index_node_tree

    tp = np.exp(csi0[:, 0]) 

    last_sets =tree[-2**max_res:]

    return{"sets": last_sets, "prob": tp}
            
def sample_tree_and_mean(seed, x_grid, y_grid, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0):
    """Draw one tree and return the posterior mean density conditional on it."""

    random.seed(seed)
    np.random.seed(seed)
    
    tree = sample_tree(max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0)
    posterior_mean_cond_tree = OPT_posterior_mean_cond_tree(tree["tree"], tree["original_index"], tree["parent_index"], max_res, rho0_0x, rho0_0y, rho0_1x, rho0_1y, alpha_vec, alpha0, ratio1, u1)
    sets = posterior_mean_cond_tree["sets"]
    prob = posterior_mean_cond_tree["prob"]

    posterior_mean = interpolate_step_function_2d(sets, prob, x_grid, y_grid)

    return posterior_mean


#def process_sample_MCt(max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid, seed):
#    
#    sample = sample_tree_and_mean(seed, x_grid, y_grid, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0)
#    return interpolate_step_function_2d(sample["sets"], sample["prob"], x_grid, y_grid)


def process_seed_chunk_MCt(seed_chunk, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid):
    """Accumulate tree-conditional means for a chunk of seeds. joblib `delayed` worker."""
    
    chunk_sum_probs = 0
    
    for seed in seed_chunk:
        result = sample_tree_and_mean(seed, x_grid, y_grid, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0)
        chunk_sum_probs += result
        
        
    gc.collect()
    return chunk_sum_probs

def sampling_MCt(iter, x_grid, y_grid, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, base_seed, n_jobs):
    """Posterior mean density by Monte Carlo over trees.

    Averages `sample_tree_and_mean` over `n_mc_samples` draws, parallelised
    over seed chunks. This is what the 2D simulations report.
    """

    seeds = [base_seed + i for i in range(iter)]

    sum_probs = np.zeros(x_grid.shape)

    total_seeds = iter

    chunk_size = total_seeds // n_jobs
    remainder = total_seeds % n_jobs

    seed_chunks = []
    start = 0

    # Distribute chunks across CPUs
    for i in range(n_jobs):
        # Each CPU gets the base chunk size, plus one extra iteration if there's a remainder
        current_chunk_size = chunk_size + 1 if i < remainder else chunk_size
        seed_chunks.append(seeds[start:start + current_chunk_size])
        start += current_chunk_size
    
    chunk_results = Parallel(n_jobs=n_jobs)(delayed(process_seed_chunk_MCt)(seed_chunk, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid) for seed_chunk in seed_chunks)


    for chunk_sum in chunk_results:
        sum_probs += chunk_sum

    return sum_probs/iter


def process_seed_chunk_MCt_bands(seed_chunk, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid):
    """As process_seed_chunk_MCt, but keeps every draw instead of accumulating a sum.

    Retaining the draws is what makes posterior quantiles available; the running
    sum in process_seed_chunk_MCt discards the information they need.
    """
    chunk_draws = []

    for seed in seed_chunk:
        result = sample_tree_and_mean(seed, x_grid, y_grid, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0)
        chunk_draws.append(np.asarray(result))

    gc.collect()
    return chunk_draws


def sampling_MCt_bands(iter, x_grid, y_grid, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, base_seed, n_jobs, lower_q=0.025, upper_q=0.975):
    """Posterior mean and pointwise credible bands from the same draws.

    sampling_MCt accumulates a running sum and can therefore only return the
    posterior mean. This variant keeps the individual draws so the pointwise
    quantiles can be taken as well -- what the flow-cytometry figure plots as its
    2.5% and 97.5% panels.

    The mean matches sampling_MCt's for the same arguments. Both draw with seeds
    base_seed + i for i in range(iter), so the per-draw seed does not depend on
    how the work is chunked across n_jobs, and the same draws are averaged either
    way. With n_jobs=1 the two means agree bit for bit; with n_jobs>1 they can
    differ by around one unit in the last place (~1e-16 here), because summing a
    chunk and averaging retained draws accumulate floating-point error in a
    different order.

    Returns
    -------
    dict with "mean", "lower_band", "upper_band" (each shaped like x_grid).
    """
    seeds = [base_seed + i for i in range(iter)]

    chunk_size = iter // n_jobs
    remainder = iter % n_jobs

    seed_chunks = []
    start = 0
    for i in range(n_jobs):
        current_chunk_size = chunk_size + 1 if i < remainder else chunk_size
        seed_chunks.append(seeds[start:start + current_chunk_size])
        start += current_chunk_size

    chunk_results = Parallel(n_jobs=n_jobs)(delayed(process_seed_chunk_MCt_bands)(seed_chunk, max_res, part_vec, rho0_0x, rho0_0y, rho0_1x, rho0_1y, lx, u1, ratio1, alpha_vec, alpha0, x_grid, y_grid) for seed_chunk in seed_chunks)

    draws = np.stack([draw for chunk in chunk_results for draw in chunk])

    return {
        "mean": np.mean(draws, axis=0),
        "lower_band": np.quantile(draws, lower_q, axis=0),
        "upper_band": np.quantile(draws, upper_q, axis=0),
    }

#################################################################################################
############################### NUMERICAL ANALYSIS ##############################################

def stopping_depth(sample_size, threshold): 
    """Depth at which a node of `sample_size` points falls below `threshold`."""

    s = sample_size
    d = 0
    while s > threshold:
        s = np.floor((s-2)/2)
        d = d + 1

    return d - 1
        
def interpolate_step_function_2d(intervals, heights, x_grid, y_grid):
    """Evaluate a piecewise-constant 2D density on a grid."""
    if len(intervals) != len(heights):
        raise ValueError("Number of intervals must match the number of heights")

    interpolated_heights = np.zeros(x_grid.shape)

    for idx, (interval, height) in enumerate(zip(intervals, heights)):
        x_start, x_end = interval[0]
        y_start, y_end = interval[1]
        
        # Create a mask for the current interval
        mask = (x_grid >= x_start) & (x_grid < x_end) & (y_grid >= y_start) & (y_grid < y_end)
        interpolated_heights[mask] = height

    return interpolated_heights


def l1_distance(true_density, estimated_density, dx, dy):
    """
    Compute the L1 distance between the true density and the estimated density.
    
    Parameters:
    - true_density: 1D array representing the true density values at sample points
    - estimated_density: 1D array representing the estimated density values at the same sample points
    - dx: The grid spacing in the x-dimension
    - dy: The grid spacing in the y-dimension
    
    Returns:
    - L1 distance as a float
    """
    # Ensure true_density and estimated_density are numpy arrays
    true_density = np.array(true_density)
    estimated_density = np.array(estimated_density)
    
    # Compute the absolute difference between the true and estimated densities
    diff = np.abs(true_density - estimated_density)
    
    # Compute the L1 distance (numerical integration over the grid)
    l1_dist = np.sum(diff * dx * dy) # NOTE THAT THIS FORMULA IS NOT GENERAL!! WE ARE ASSUMING AN EQUALLY SPACED GRID, THUS CAN STILL ACCEPT THIS FORMULA HERE. 
    
    return l1_dist

def l2_distance(true_density, estimated_density, dx, dy):
    """
    Compute the L2 distance between the true density and the estimated density.
    
    Parameters:
    - true_density: 1D array representing the true density values at sample points
    - estimated_density: 1D array representing the estimated density values at the same sample points
    - dx: The grid spacing in the x-dimension
    - dy: The grid spacing in the y-dimension
    
    Returns:
    - L2 distance as a float
    """
    # Ensure true_density and estimated_density are numpy arrays
    true_density = np.array(true_density)
    estimated_density = np.array(estimated_density)
    
    # Compute the absolute difference between the true and estimated densities
    square_diff = (true_density - estimated_density)**2
    
    # Compute the L1 distance (numerical integration over the grid)
    l2_dist = np.sum(square_diff * dx * dy)
    
    return np.sqrt(l2_dist)

def linfty_distance(true_density, estimated_density, dx, dy):
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

def distance_metric(x0, y0, partial_prob, full_prob, true_den, distance_func): #x0, y0 in meshgrid format
    """Distance from the true density to each of the two fits, on a common grid.

    Returns (partial, full).
    """

    partial_prob = partial_prob.ravel()
    full_prob = full_prob.ravel()
    true_den = true_den.ravel()

    dx = x0[0, 1] - x0[0, 0]  # EQUALLY SPACED GRIDPOINTS
    dy = y0[1, 0] - y0[0, 0] 
     
    dist_partial = distance_func(true_den, partial_prob, dx, dy)
    dist_full = distance_func(true_den, full_prob, dx, dy)
    
    return dist_partial, dist_full #first one is partial, second one is full



def sample_from_mixture(components, weights, sample_size, seed=None):
    """Draw `size` points from a 2D mixture with the given weights."""

    if seed is not None:
        np.random.seed(seed)
    
    weights = np.array(weights)
    weights /= np.sum(weights)

    component_indices = np.random.choice(len(weights), size=sample_size, p=weights)
    
    # Generate samples by selecting the appropriate component function
    samples_list = []
    for i in range(len(weights)):
        num_samples = np.sum(component_indices == i)
        if num_samples > 0:
            component_samples = components[i](size=num_samples, seed = seed)
            samples_list.append(component_samples)
    
    # Concatenate all samples into a single array
    samples = np.vstack(samples_list)
    
    return samples



def pdf_for_mixture_2D(x0, y0, pdf_functions, weights):
    """
    Compute the PDF of a mixture distribution at given 2D grid points.

    Parameters:
    - x0: 2D array of x coordinates (output of meshgrid)
    - y0: 2D array of y coordinates (output of meshgrid)
    - pdf_functions: list of functions, where each function computes the PDF of a 2D component
    - weights: list or array of mixture weights (should sum to 1)

    Returns:
    - mixture_pdf: 2D array of PDF values for the mixture distribution over the grid
    """
    # Ensure x0 and y0 are the same shape
    if x0.shape != y0.shape:
        raise ValueError("x0 and y0 must have the same shape")

    samples = np.stack([x0.ravel(), y0.ravel()], axis=-1)

    # Initialize the mixture PDF
    mixture_pdf = np.zeros(samples.shape[0], dtype=float)

    # Add the weighted PDF of each component directly on the grid
    for weight, pdf_func in zip(weights, pdf_functions):
        mixture_pdf += weight * pdf_func(samples)

    mixture_pdf = mixture_pdf.reshape(x0.shape)

    return mixture_pdf
    
    



    




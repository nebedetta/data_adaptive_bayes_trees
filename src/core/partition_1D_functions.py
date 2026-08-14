import numpy as np
import random
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.special import beta as bf
import timeit

# Function for the number of observations, efficient
def set_count_partial(node):
    """Number of observations strictly inside the node `[lo, hi]`."""
    sc = node[1] - node[0] - 1
    return sc

# Define percentile function with nearest-rank method, efficient 
def order_stat(n, p):
    """Rank of the `p`-quantile of `n` points, by the nearest-rank method."""
    return int(np.ceil(n * p))

# Function for the children of a set, efficient 
def child_partial(node, p):
    """Split `node` at its `p`-quantile, returning the two child intervals."""
    s = set_count_partial(node)
    k = order_stat(s, p)
    lnode = [node[0], node[0] + k] 
    rnode = [node[0] + k, node[1]]
    return [lnode, rnode]


# Function to generate the partition process according to the statistic of order k, embedded into the child function
def partition_partial(n, threshold, p):
    """Build the dyadic partition, splitting until a child holds fewer than
    `threshold` points.

    Returns (m, part_vec): the depth reached, and the nodes in breadth-first
    order -- the layout every other function here indexes by position.
    """
    m = 0
    prev = [[0, n+1]]
    sc = [n]
    l = 1
    while True:
        new = []
        for node in prev[-l:]:
            sub = child_partial(node, p)
            new.append(sub[0])
            new.append(sub[1])
            sc.append(set_count_partial(sub[0]))
            sc.append(set_count_partial(sub[1]))
        if min(set_count_partial(sub[0]), set_count_partial(sub[1])) < threshold:
            break 
        m = m + 1
        prev.extend(new)
        l = len(new)

    return m, prev


# For simplicity, just apply the partition function and write the parent and set size functions retrieving 
# from the application of the partition function
# Here obtain the recursion level of a node from the position in the partitioning vector
def rec_level(node, part_vec):
    """Depth of `node`, recovered from its position in `part_vec`."""
    position = part_vec.index(node) 
    
    power = 0
    sum_power = 2**power
    while sum_power <= position:
        power += 1
        sum_power += 2**power 
    
    level = power
    
    return level
    
# Here obtain the parent of a node from the position in the partitioning vector 
def parent(node, part_vec):
    """The parent of `node`, found by position in `part_vec`."""
    
    level = rec_level(node, part_vec)
    position = part_vec.index(node) 
    sum = 0
    for i in range(1, level) :
        sum += 2**i
    
    parent_index = int(sum - 2**(level-1)+ np.ceil(0.5*(position - sum)))
    parent = part_vec[parent_index]

    return parent
    
    
def base_ratio(part_vec, a, b, X_aug): 
    """Base-measure mass of each node relative to its parent.

    Returns (beta1, beta2): each node's own share, and its sibling's, under the
    Beta(a, b) base measure evaluated on the augmented sample `X_aug`.
    """
    beta1 = []
    beta2 = []
    for l in range(0, len(part_vec)):
        p_node = parent(part_vec[l], part_vec)
        
        p_parent = stats.beta.cdf(X_aug[p_node[1]], a, b) - stats.beta.cdf(X_aug[p_node[0]], a, b)
        p_set = stats.beta.cdf(X_aug[part_vec[l][1]], a, b) - stats.beta.cdf(X_aug[part_vec[l][0]], a, b)
        
        p_other = p_parent - p_set
        beta1.append(p_set/p_parent)
        beta2.append(p_other/p_parent)

    return beta1, beta2

def beta_prior(beta1, beta2, c): 
    """Scale the base-measure ratios by a single concentration `c`."""
    beta1_c = [c * element for element in beta1]
    beta2_c = [c * element for element in beta2]
    
    return beta1_c, beta2_c
    

def beta_prior_vec(beta1, beta2, c): # beta1 and beta2 not in log
    """Scale the base-measure ratios by a per-node concentration vector `c`."""
    if not (len(beta1) == len(beta2) == len(c)):
        raise ValueError("beta1, beta2, and c must have the same length.")
    
    beta1_c = [ci*b1 for ci, b1 in zip(c, beta1)]
    beta2_c = [ci*b2 for ci, b2 in zip(c, beta2)]
    
    return beta1_c, beta2_c
    
    
def vec_count(part_vec):
    """Observation counts per node and sibling, for the Beta likelihood update."""
    beta1_up = []
    beta2_up = []
    for l in range(0, len(part_vec)):
        
        p_node = parent(part_vec[l], part_vec) 
        s_parent = set_count_partial(p_node)
        
        s_set = set_count_partial(part_vec[l])
        s_other = s_parent - s_set - 1
        
        beta1_up.append(s_set)
        beta2_up.append(s_other)

    return beta1_up, beta2_up 

def vec_to_mat(v, max_res):
    """Expand a breadth-first node vector to a (2**max_res, max_res) matrix.

    Column `r` holds each node's value at depth `r`, repeated across the
    finest-resolution cells it covers, so summing across columns accumulates
    the path from root to leaf.
    """
   
    M = np.zeros((2**max_res, max_res))

    starter = 1
    for res in range(1, max_res+1):
        nsets = 2**res
        setsize = int(2**max_res/nsets)
        sub_vec = v[(starter):(starter + nsets)]
        M[:, (res-1)] =   [x for x in sub_vec for _ in range(setsize)]
           
        starter = starter + nsets

    return M
        


def measure_execution_time(func, *args, **kwargs):
    """Call `func` and return (result, elapsed seconds)."""
    start_time = timeit.default_timer()
    result = func(*args, **kwargs)
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    return result, total_time

import numpy as np
import random
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from bisect import bisect_left, bisect_right
from scipy.special import beta as bf
from scipy.special import comb
import timeit
from scipy.stats import beta
from joblib import Parallel, delayed
from numba import njit
import numpy as np
import gc


######################################################################################################
############################################# HELPERS ################################################

def order_stat(n, p):
    """Rank of the `p`-quantile of `n` points, by the nearest-rank method."""
    return int(np.ceil(n * p))


def sort_sample(sample, dimension):
    """Return `sample` sorted by one coordinate."""
    return sample[np.argsort(sample[:, dimension])]


def find_bounds(sorted_data, lower_bound, upper_bound):
    """Index range of a sorted array falling within an interval."""
    lower_idx = bisect_right(sorted_data, lower_bound)
    upper_idx = bisect_left(sorted_data, upper_bound)
    return lower_idx, upper_idx


def create_repeated_vector(max_value, dim):
    """Repeat each value to fill its resolution level. Currently unused."""
    split_factor = 2 ** dim
    vector = []
    for i in range(max_value + 1):
        vector.extend([i] * (split_factor ** i))
    return np.array(vector)


def log_sum_exp_multiple(log_values):
    """log-sum-exp over a list of log values."""
    if not log_values:
        return float('-inf')
    max_log_value = max(log_values)
    if math.isinf(max_log_value):
        if all(math.isinf(v) and v < 0 for v in log_values):
            return float('-inf')
        if all(math.isinf(v) and v > 0 for v in log_values):
            return float('inf')
        return float('inf')
    sum_exp = 0
    for log_val in log_values:
        if max_log_value - log_val < 50:
            sum_exp += math.exp(log_val - max_log_value)
        elif log_val == max_log_value:
            sum_exp += 1
    return max_log_value + math.log(sum_exp)


def exchange_adjacent_elements_with_offset(vector):
    """Swap adjacent entries at a fixed offset. Currently unused."""
    if len(vector) % 2 == 0:
        raise ValueError("Vector length must be odd.")
    new_vec = [0] * len(vector)
    new_vec[0] = vector[0]
    for i in range(1, len(vector) + 1, 2):
        if i + 1 < len(vector):
            new_vec[i] = vector[i + 1]
            new_vec[i + 1] = vector[i]
    return new_vec


def round_element(element, decimals):
    """Round one value for display."""
    if isinstance(element, list):
        return [round_element(sub_element, decimals) for sub_element in element]
    elif isinstance(element, (int, float)):
        return round(element, decimals)
    else:
        return element


def round_dataframe(df, decimals):
    """Round a table for display. Currently unused."""
    return df.apply(lambda col: col.apply(lambda x: round_element(x, decimals)))


def remove_points_from_sample(sorted_arrays, removed_points):
    """Drop the order-statistic points from each dimension's sorted copy."""
    removed = np.array(list(removed_points))
    result = []
    for arr in sorted_arrays:
        arr = np.array(arr)
        mask = ~np.isin(arr, removed).all(axis=1)
        result.append(arr[mask])
    return result




######################################################################################################
########### PARTITION GENERATION ####################################################################

def rec_level(node_index, d):
    """Depth of a node, from its index in the breadth-first node list."""
    n_children = 2 * d
    power = 0
    sum_power = n_children ** power
    while sum_power <= node_index:
        power += 1
        sum_power += n_children ** power
    return power


def children_index(node_index, d):
    """Indices of a node's 2*d children in the breadth-first node list."""
    n_children = 2 * d
    level = rec_level(node_index, d)
    sum_nodes = 0
    for i in range(level + 1):
        sum_nodes += n_children ** i
    first_child_i = int(sum_nodes + n_children * (node_index - (sum_nodes - n_children ** level)))
    return [first_child_i + c for c in range(n_children)]


def precompute_child_indices(n_nodes, d):
    """Child indices for every node, computed once and reused by the recursion."""
    n_children = 2 * d
    child_idx = np.zeros((n_nodes, n_children), dtype=np.int64)
    for node_idx in range(n_nodes):
        child_idx[node_idx] = children_index(node_idx, d)
    return child_idx


def order_stat_node_2(subsets, p=0.5):
    """Order statistic and remaining counts per dimension for one node.

    Returns the per-dimension order statistics, the points realising them, and
    how many points fall either side once those points are removed.
    """
    d = subsets[0].shape[1]
    set_size = len(subsets[0])
    os = order_stat(set_size, p)
    os_index = os - 1

    order_stats = []
    os_points = []
    for i in range(d):
        os_val = subsets[i][os_index, i]
        os_idx = np.searchsorted(subsets[i][:, i], os_val)
        order_stats.append(os_val)
        os_points.append(subsets[i][os_idx])

    # --- lo_sizes[i] / hi_sizes[i]: how many points remain in subsets[i] on
    # each side of order_stats[i], after removing all d order-stat points. ---
    #
    # Counted directly rather than by the cheaper arithmetic (subtract, from
    # the counts either side of the median, the other dimensions' order-stat
    # points that fall on each side). That shortcut assumes the d order-stat
    # points are distinct, which fails deep in the tree: with few points left,
    # one point can be the median-ranked point in several dimensions at once
    # -- a coincidence of rank, not of coordinate values -- and gets subtracted
    # twice. It produced negative sizes in over 10% of random trials at small
    # set_size. Deduplicating first and counting what remains cannot miscount
    # however the order-stat points coincide.
    unique_os_points = np.unique(np.array(os_points), axis=0)

    lo_sizes = []
    hi_sizes = []
    for i in range(d):
        pts = subsets[i]
        mask_keep = np.ones(len(pts), dtype=bool)
        for op in unique_os_points:
            mask_keep &= ~np.all(pts == op, axis=1)
        remaining = pts[mask_keep]
        lo_sizes.append(int(np.sum(remaining[:, i] < order_stats[i])))
        hi_sizes.append(int(np.sum(remaining[:, i] > order_stats[i])))

    return {
        "order_stats": order_stats,
        "os_points":   os_points,
        "lo_sizes":    lo_sizes,
        "hi_sizes":    hi_sizes
    }


def node_children_2(node, sorted_arrays, removed_points, p=0.5):
    """The 2*d children of a node, splitting at the order statistic per dimension."""
    d = len(node)

    subsets = []
    for i in range(d):
        arr = sorted_arrays[i]
        lower_idx, upper_idx = find_bounds(arr[:, i], node[i][0], node[i][1])
        sliced = arr[lower_idx:upper_idx]
        mask = np.ones(len(sliced), dtype=bool)
        for j in range(d):
            if j != i:
                mask &= (sliced[:, j] > node[j][0]) & (sliced[:, j] < node[j][1])
        subsets.append(sliced[mask])

    subsets = remove_points_from_sample(subsets, removed_points)

    if len(subsets[0]) == 0:
        # No data left in this node: order_stat_node_2 would compute
        # os_index = ceil(0*0.5) - 1 = -1 and index an empty array, raising
        # an IndexError. Fall back to a fixed midpoint split so the tree
        # shape (2d children per node) is preserved; every child inherits
        # size 0.
        children, children_sizes = [], []
        for i in range(d):
            midpoint = (node[i][0] + node[i][1]) / 2
            lo_child = [node[j][:] for j in range(d)]
            lo_child[i] = [node[i][0], midpoint]
            hi_child = [node[j][:] for j in range(d)]
            hi_child[i] = [midpoint, node[i][1]]
            children.append(lo_child)
            children.append(hi_child)
            children_sizes.append(0)
            children_sizes.append(0)
        return {
            "children":            children,
            "children_sizes":      children_sizes,
            "new_removed_points":  removed_points,
        }

    t = order_stat_node_2(subsets, p)
    order_stats = t["order_stats"]
    os_points   = t["os_points"]
    lo_sizes    = t["lo_sizes"]
    hi_sizes    = t["hi_sizes"]

    children       = []
    children_sizes = []
    for i in range(d):
        lo_child = [node[j][:] for j in range(d)]
        lo_child[i] = [node[i][0], order_stats[i]]

        hi_child = [node[j][:] for j in range(d)]
        hi_child[i] = [order_stats[i], node[i][1]]

        children.append(lo_child)
        children.append(hi_child)
        children_sizes.append(lo_sizes[i])
        children_sizes.append(hi_sizes[i])

    new_removed_points = removed_points.copy()
    for pt in os_points:
        new_removed_points.add(tuple(pt))

    return {
        "children":            children,
        "children_sizes":      children_sizes,
        "new_removed_points":  new_removed_points
    }


def generate_node_list_2(sample, max_res, depth_parallelization, chunk_number):
    """
    Generate a list of nodes representing a spatial partitioning of a d-dimensional
    sample, recursively excluding d marginal order statistic points per node.

    Parameters:
        sample (numpy.ndarray): Sample points as an array of shape (n, d).
        max_res (int): Maximum resolution level for partitioning.
        depth_parallelization (int): Resolution level below which to use parallelization.
        chunk_number (int): Number of parallel chunks.

    Returns:
        dict:
            - "part_vec": numpy array of shape (n_nodes, d, 2) of node bounds.
            - "set_size_vec": numpy array of sizes of the sets associated with each node.
            - "ratio1": numpy array of lo child volume ratios relative to parent.
            - "ratio2": numpy array of hi child volume ratios relative to parent.
            - "parent_index_list": numpy array of indices of parent nodes for each node.
            - "res_level": numpy array of resolution levels for each node.
    """
    n, d = sample.shape
    n_children = 2 * d
    n_nodes = sum(n_children ** i for i in range(max_res + 1))

    sorted_arrays = [np.array(sort_sample(sample, i)) for i in range(d)]

    root = [[0, 1]] * d

    # part_vec_arr, ratio1/ratio2, and the eta0/eta1/rho00/rho01/phi0/phi1
    # arrays in bottom_up_recursion_2 below are the largest per-node
    # allocations (O(n_nodes) or O(d * n_nodes) floats each) and dominate
    # memory at depth >= 7. Using float32 for these halves that footprint;
    # small, one-off arrays (log_lambdas, non_stopping_c's output, the
    # per-level phi0_terms/phi1_terms scratch arrays) are left at float64,
    # since they're negligible in size and some feed numerically sensitive
    # gammaln/exp(alpha0)-scale computations where float32's reduced range
    # and precision could introduce real error.
    part_vec_arr = np.zeros((n_nodes, d, 2), dtype=np.float32)
    part_vec_arr[0] = [[0, 1]] * d
    node_counter = 1

    prev = [root]
    set_size_list = [n]
    ratio1 = [np.log(1)]
    ratio2 = [np.log(1)]
    parent_index_list = [0]
    res_level = [0]
    removed_points_per_node = [set()]

    for m in range(max_res):
        l = n_children ** m
        new = []
        new_removed_points_per_node = []
        current_nodes = prev[-l:]

        if m <= depth_parallelization:
            for idx, node in enumerate(current_nodes):
                node_index = len(prev) - l + idx
                removed_points = removed_points_per_node[node_index]

                sub = node_children_2(node, sorted_arrays, removed_points, p=0.5)
                children = sub["children"]
                children_sizes = sub["children_sizes"]

                node_vol = sum(np.log(node[i][1] - node[i][0]) for i in range(d))
                child_vols = [sum(np.log(c[i][1] - c[i][0]) for i in range(d)) for c in children]

                new.extend(children)

                for c in children:
                    part_vec_arr[node_counter] = c
                    node_counter += 1

                for i in range(d):
                    lo_vol = child_vols[2 * i]
                    hi_vol = child_vols[2 * i + 1]
                    ratio1.append(lo_vol - node_vol)
                    ratio2.append(hi_vol - node_vol)
                    ratio1.append(hi_vol - node_vol)
                    ratio2.append(lo_vol - node_vol)

                set_size_list.extend(children_sizes)
                new_removed_points_per_node.extend([sub["new_removed_points"]] * n_children)
                parent_index_list.extend([node_index] * n_children)
                res_level.extend([m + 1] * n_children)

        else:
            chunk_size = max(1, len(current_nodes) // chunk_number)
            chunks = [current_nodes[i:i + chunk_size] for i in range(0, len(current_nodes), chunk_size)]
            start_indices = [i * chunk_size for i in range(len(chunks))]

            def process_nodes_chunk_local(chunk, prev, start_idx, l, sorted_arrays, removed_points_per_node):
                """Split a chunk of nodes in one parallel task."""
                results = []
                for idx, node in enumerate(chunk):
                    node_index = len(prev) - l + start_idx + idx
                    removed_points = removed_points_per_node[node_index]
                    sub = node_children_2(node, sorted_arrays, removed_points)
                    children = sub["children"]
                    children_sizes = sub["children_sizes"]
                    node_vol = sum(np.log(node[i][1] - node[i][0]) for i in range(d))
                    child_vols = [sum(np.log(c[i][1] - c[i][0]) for i in range(d)) for c in children]
                    results.append((node_index, sub["new_removed_points"], children, child_vols, node_vol, children_sizes))
                return results

            all_results = Parallel(n_jobs=chunk_number)(
                delayed(process_nodes_chunk_local)(
                    chunk, prev, start_idx, l, sorted_arrays, removed_points_per_node
                )
                for chunk, start_idx in zip(chunks, start_indices)
            )

            all_results_flat = [r for chunk_results in all_results for r in chunk_results]
            sorted_results = sorted(all_results_flat, key=lambda x: x[0])

            for result in sorted_results:
                node_index, new_removed_points, children, child_vols, node_vol, children_sizes = result

                new.extend(children)

                for c in children:
                    part_vec_arr[node_counter] = c
                    node_counter += 1

                for i in range(d):
                    lo_vol = child_vols[2 * i]
                    hi_vol = child_vols[2 * i + 1]
                    ratio1.append(lo_vol - node_vol)
                    ratio2.append(hi_vol - node_vol)
                    ratio1.append(hi_vol - node_vol)
                    ratio2.append(lo_vol - node_vol)

                set_size_list.extend(children_sizes)
                new_removed_points_per_node.extend([new_removed_points] * n_children)
                parent_index_list.extend([node_index] * n_children)
                res_level.extend([m + 1] * n_children)

        prev.extend(new)
        removed_points_per_node.extend(new_removed_points_per_node)

    return {
        "part_vec":          part_vec_arr,
        "set_size_vec":      np.array(set_size_list,     dtype=np.int64),
        "ratio1":            np.exp(ratio1).astype(np.float32),
        "ratio2":            np.exp(ratio2).astype(np.float32),
        "parent_index_list": np.array(parent_index_list, dtype=np.int64),
        "res_level":         np.array(res_level,         dtype=np.int64)
    }



######################################################################################################
######################################### BOTTOM UP REC ##############################################

def eta_f(base_al, base_ar, n_al, n_ar, c):
    """Log marginal likelihood of a node's split, per dimension.

    The nD counterpart of the 1D `eta_f`. Written via lgamma for numerical
    stability.
    """
    from scipy.special import gammaln
    t = (- n_al * np.log(base_al)
         - n_ar * np.log(base_ar)
         + math.lgamma(c * base_al + n_al)
         + math.lgamma(c * base_ar + n_ar)
         - math.lgamma(c + n_al + n_ar)
         - math.lgamma(c * base_al)
         - math.lgamma(c * base_ar)
         + math.lgamma(c))
    return t


def vectorized_eta_f(base_al, base_ar, n_al, n_ar, c):
    """`eta_f` over a whole depth level at once, using gammaln on arrays."""
    from scipy.special import gammaln
    t = (- n_al * np.log(base_al)
         - n_ar * np.log(base_ar)
         + gammaln(c * base_al + n_al)
         + gammaln(c * base_ar + n_ar)
         - gammaln(c + n_al + n_ar)
         - gammaln(c * base_al)
         - gammaln(c * base_ar)
         + gammaln(c))
    return t


def phi_f(lambdas, RHO, eta0_list, eta1_list, PHI0_children, PHI1_children):
    """Combine children's phi values into a node's, over all 2*d split directions."""
    d = len(lambdas)
    rho00 = RHO[0]
    rho01 = RHO[1]
    phi0_terms = []
    phi1_terms = []
    for i in range(d):
        lo_i = 2 * i
        hi_i = 2 * i + 1
        t_i0 = lambdas[i] + rho00 + eta0_list[i] + PHI0_children[lo_i] + PHI0_children[hi_i]
        t_i1 = lambdas[i] + rho01 + eta1_list[i] + PHI1_children[lo_i] + PHI1_children[hi_i]
        phi0_terms.append(t_i0)
        phi0_terms.append(t_i1)
        phi1_terms.append(lambdas[i] + eta1_list[i] + PHI1_children[lo_i] + PHI1_children[hi_i])
    phi0 = log_sum_exp_multiple(phi0_terms)
    phi1 = log_sum_exp_multiple(phi1_terms)
    return phi0, phi1


def vectorized_phi_f(lambdas_arr, rho00, rho01, eta0_level, eta1_level,
                     phi0_lo, phi0_hi, phi1_lo, phi1_hi):
    """`phi_f` over a whole depth level at once, using logsumexp on arrays.

    `lambdas_arr` holds the prior weight on splitting along each dimension.
    """
    from scipy.special import logsumexp
    d = len(lambdas_arr)
    n_level = eta0_level.shape[0]
    phi0_terms = np.empty((n_level, 2 * d), dtype=np.float64)
    phi1_terms = np.empty((n_level, d),     dtype=np.float64)
    for i in range(d):
        phi0_terms[:, 2*i]   = lambdas_arr[i] + rho00 + eta0_level[:, i] + phi0_lo[:, i] + phi0_hi[:, i]
        phi0_terms[:, 2*i+1] = lambdas_arr[i] + rho01 + eta1_level[:, i] + phi1_lo[:, i] + phi1_hi[:, i]
        phi1_terms[:, i]     = lambdas_arr[i]           + eta1_level[:, i] + phi1_lo[:, i] + phi1_hi[:, i]
    phi0_nodes = logsumexp(phi0_terms, axis=1)
    phi1_nodes = logsumexp(phi1_terms, axis=1)
    return phi0_nodes, phi1_nodes


def posterior_prob(lambda_i, rho_j, eta_i, phi_lo, phi_hi, phi_node):
    """Posterior log-probability of each state at a node. Currently unused."""
    return lambda_i + rho_j + eta_i + phi_lo + phi_hi - phi_node


def beta_prior(beta1, beta2, c):
    """Scale the base-measure ratios by the per-depth pseudocount. Currently unused."""
    beta1_c = [c * element for element in beta1]
    beta2_c = [c * element for element in beta2]
    return beta1_c, beta2_c


def non_stopping_c(res_level, d):
    """Pseudocount per node by depth: 2 for the root and its children, depth**2 below."""
    # The "first 5" cutoff in OPT_2D_functions.py (n_children=2*d=4) covers
    # exactly the root plus all of its depth-1 children (1 + 4 = 5 nodes). For
    # general d, that same "root + all depth-1 children" set has 1 + 2*d nodes,
    # not 5 -- ported unchanged, the fixed cutoff of 5 would apply c=2 to only
    # a fraction of the depth-1 nodes once d > 2, then switch to depth^2 mid-way
    # through depth 1 instead of starting cleanly at depth 2.
    nsc = np.array(res_level, dtype=np.float64)
    n_first_level = 1 + 2 * d
    nsc[0:n_first_level] = 2
    nsc[n_first_level:] = nsc[n_first_level:] ** 2
    return nsc


def bottom_up_recursion_2(X, max_res, p0, lambdas, alpha, alpha0, depth_parallelization=7, chunk_number=4):
    """
    Bottom-up recursion for a d-dimensional Optional Polya Tree.
    Vectorized implementation — processes all nodes at each level simultaneously.

    Parameters:
        X (numpy.ndarray): Sample of shape (n, d).
        max_res (int): Maximum resolution level.
        p0 (float): Prior stopping probability.
        lambdas (list): Split probabilities per dimension, length d, summing to 1.
        alpha (float): Alpha parameter.
        alpha0 (float): Alpha0 parameter.

    Returns:
        dict: eta0, eta1, phi0, phi1, rho00, rho01 per dimension, plus partition info.
    """
    n, d = X.shape
    n_children = 2 * d

    si        = generate_node_list_2(X, max_res, depth_parallelization, chunk_number)
    part_vec  = si["part_vec"]
    u1        = si["set_size_vec"]
    ratio1    = si["ratio1"]
    ratio2    = si["ratio2"]
    res_level = si["res_level"]
    del si
    gc.collect()

    alpha_vec   = non_stopping_c(res_level, d)
    n_nodes     = len(u1)
    log_lambdas = np.array([np.log(lam) for lam in lambdas], dtype=np.float64)

    rho00_val   = np.log(1 - p0)
    rho01_val   = np.log(p0)

    eta0  = np.zeros((d, n_nodes), dtype=np.float64)
    eta1  = np.zeros((d, n_nodes), dtype=np.float64)
    rho00 = np.zeros((d, n_nodes), dtype=np.float64)
    rho01 = np.zeros((d, n_nodes), dtype=np.float64)
    phi0  = np.zeros(n_nodes,      dtype=np.float64)
    phi1  = np.zeros(n_nodes,      dtype=np.float64)

    child_idx_matrix = precompute_child_indices(n_nodes, d)

    level_starts = {}
    level_ends   = {}
    cumsum = 0
    for m in range(max_res + 1):
        level_starts[m] = cumsum
        cumsum += n_children ** m
        level_ends[m]   = cumsum - 1

    for m in reversed(range(max_res)):
        node_indices = np.arange(level_starts[m], level_ends[m] + 1, dtype=np.int64)

        cidx   = child_idx_matrix[node_indices]
        lo_idx = cidx[:, 0::2]
        hi_idx = cidx[:, 1::2]

        r1_lo = ratio1[lo_idx]
        r1_hi = ratio1[hi_idx]
        u_lo  = u1[lo_idx].astype(np.float64)
        u_hi  = u1[hi_idx].astype(np.float64)

        alpha_level = alpha_vec[node_indices].reshape(-1, 1)

        eta0_level = vectorized_eta_f(r1_lo, r1_hi, u_lo, u_hi, alpha_level)
        eta1_level = vectorized_eta_f(r1_lo, r1_hi, u_lo, u_hi, alpha0)

        phi0_lo = phi0[lo_idx]
        phi0_hi = phi0[hi_idx]
        phi1_lo = phi1[lo_idx]
        phi1_hi = phi1[hi_idx]

        phi0_nodes, phi1_nodes = vectorized_phi_f(
            log_lambdas, rho00_val, rho01_val,
            eta0_level, eta1_level,
            phi0_lo, phi0_hi, phi1_lo, phi1_hi
        )

        phi0[node_indices] = phi0_nodes
        phi1[node_indices] = phi1_nodes

        phi0_nodes_col = phi0_nodes.reshape(-1, 1)

        rho00_level = log_lambdas + rho00_val + eta0_level + phi0_lo + phi0_hi - phi0_nodes_col
        rho01_level = log_lambdas + rho01_val + eta1_level + phi1_lo + phi1_hi - phi0_nodes_col

        eta0[:, node_indices]  = eta0_level.T
        eta1[:, node_indices]  = eta1_level.T
        rho00[:, node_indices] = rho00_level.T
        rho01[:, node_indices] = rho01_level.T

    return {
        "part_vec": part_vec,
        "ratio1":   ratio1,
        "ratio2":   ratio2,
        "set_size": u1,
        "eta0":  [np.longdouble(eta0[i])  for i in range(d)],
        "eta1":  [np.longdouble(eta1[i])  for i in range(d)],
        "phi0":  np.longdouble(phi0),
        "phi1":  np.longdouble(phi1),
        "rho00": [np.longdouble(rho00[i]) for i in range(d)],
        "rho01": [np.longdouble(rho01[i]) for i in range(d)],
        "alpha_vec": alpha_vec
    }


def posterior_rho_s_s1(rho_s_s1_list):
    """Aggregate posterior rho across all d dimensions."""
    return log_sum_exp_multiple(rho_s_s1_list)


def posterior_ss1_dim(rho_s_s1_dim, rho_s_s1):
    """Log-probability of splitting on one dimension, given a split. Currently unused."""
    return rho_s_s1_dim - rho_s_s1


def map_parent_index(i):
    """Index of each node's parent in the node list. Currently unused."""
    if i == 0:
        return 0
    return (i - 1) // 2


######################################################################################################
################################# OPT RECURSION ######################################################

def find_bounds_binary(sorted_data, lower_bound, upper_bound):
    """Index range of a sorted array within an interval, by binary search."""
    lower_idx = bisect_left(sorted_data, lower_bound)
    upper_idx = bisect_left(sorted_data, upper_bound)
    return lower_idx, upper_idx


def binary_set_count(node, sorted_sample, d):
    """Number of sample points inside a node, from its sorted-index bounds."""
    lower_idx, upper_idx = find_bounds_binary(sorted_sample[:, 0], node[0][0], node[0][1])
    filtered = sorted_sample[lower_idx:upper_idx]
    mask = np.ones(len(filtered), dtype=bool)
    for j in range(1, d):
        mask &= (filtered[:, j] >= node[j][0]) & (filtered[:, j] < node[j][1])
    return mask.sum()


def binary_node_children(node, node_size, sorted_sample, d):
    """The 2*d children of a node under the binary-search partition."""
    children       = []
    children_sizes = []
    for i in range(d):
        midpoint = np.mean(node[i])
        lo_child = [node[j][:] for j in range(d)]
        lo_child[i] = [node[i][0], midpoint]
        hi_child = [node[j][:] for j in range(d)]
        hi_child[i] = [midpoint, node[i][1]]
        lo_size = binary_set_count(lo_child, sorted_sample, d)
        hi_size = node_size - lo_size
        children.append(lo_child)
        children.append(hi_child)
        children_sizes.append(lo_size)
        children_sizes.append(hi_size)
    return {"children": children, "children_sizes": children_sizes}


def binary_generate_node_list(sample, max_res):
    """
    Generate a list of nodes representing a binary spatial partitioning of a
    d-dimensional sample, splitting each dimension at its midpoint.

    Parameters:
        sample (numpy.ndarray): Sample points of shape (n, d).
        max_res (int): Maximum resolution level.

    Returns:
        dict:
            - "part_vec": numpy array of shape (n_nodes, d, 2) of node bounds.
            - "set_size_vec": numpy array of sizes of the sets associated with each node.
            - "parent_index_list": numpy array of indices of parent nodes for each node.
            - "res_level": numpy array of resolution levels for each node.
    """
    n, d = sample.shape
    n_children = 2 * d
    n_nodes = sum(n_children ** i for i in range(max_res + 1))

    sorted_sample = sort_sample(sample, 0)

    # float32 for memory (see bottom_up_recursion_2's part_vec_arr for rationale);
    # part_vec is a geometric quantity in [0,1], safe at reduced precision.
    part_vec_arr = np.zeros((n_nodes, d, 2), dtype=np.float32)
    part_vec_arr[0] = [[0, 1]] * d
    node_counter = 1

    prev = [part_vec_arr[0].tolist()]
    set_size_list     = [n]
    parent_index_list = [0]
    res_level         = [0]

    for m in range(max_res):
        l   = n_children ** m
        new = []

        for idx, node in enumerate(prev[-l:]):
            node_index = len(prev) - l + idx

            sub            = binary_node_children(node, set_size_list[node_index], sorted_sample, d)
            children       = sub["children"]
            children_sizes = sub["children_sizes"]

            new.extend(children)

            for c in children:
                part_vec_arr[node_counter] = c
                node_counter += 1

            set_size_list.extend(children_sizes)
            parent_index_list.extend([node_index] * n_children)
            res_level.extend([m + 1] * n_children)

        prev.extend(new)

    return {
        "part_vec":          part_vec_arr,
        "set_size_vec":      np.array(set_size_list,     dtype=np.int64),
        "parent_index_list": np.array(parent_index_list, dtype=np.int64),
        "res_level":         np.array(res_level,         dtype=np.int64)
    }


def binary_bottom_up_recursion(X, max_res, p0, lambdas, alpha, alpha0):
    """
    Bottom-up recursion for a d-dimensional binary Optional Polya Tree.
    Vectorized implementation — processes all nodes at each level simultaneously.

    Parameters:
        X (numpy.ndarray): Sample of shape (n, d).
        max_res (int): Maximum resolution level.
        p0 (float): Prior stopping probability.
        lambdas (list): Split probabilities per dimension, length d, summing to 1.
        alpha (float): Alpha parameter.
        alpha0 (float): Alpha0 parameter.

    Returns:
        dict: eta0, eta1, phi0, phi1, rho00, rho01 per dimension, plus partition info.
    """
    n, d = X.shape
    n_children = 2 * d

    si        = binary_generate_node_list(X, max_res)
    part_vec  = si["part_vec"]
    u1        = si["set_size_vec"]
    res_level = si["res_level"]
    del si
    gc.collect()

    alpha_vec   = non_stopping_c(res_level, d)
    n_nodes     = len(u1)
    log_lambdas = np.array([np.log(lam) for lam in lambdas], dtype=np.float64)

    rho00_val   = np.log(1 - p0)
    rho01_val   = np.log(p0)

    eta0  = np.zeros((d, n_nodes), dtype=np.float64)
    eta1  = np.zeros((d, n_nodes), dtype=np.float64)
    rho00 = np.zeros((d, n_nodes), dtype=np.float64)
    rho01 = np.zeros((d, n_nodes), dtype=np.float64)
    phi0  = np.zeros(n_nodes,      dtype=np.float64)
    phi1  = np.zeros(n_nodes,      dtype=np.float64)

    child_idx_matrix = precompute_child_indices(n_nodes, d)

    level_starts = {}
    level_ends   = {}
    cumsum = 0
    for m in range(max_res + 1):
        level_starts[m] = cumsum
        cumsum += n_children ** m
        level_ends[m]   = cumsum - 1

    r1_fixed = np.float64(0.5)

    for m in reversed(range(max_res)):
        node_indices = np.arange(level_starts[m], level_ends[m] + 1, dtype=np.int64)

        cidx   = child_idx_matrix[node_indices]
        lo_idx = cidx[:, 0::2]
        hi_idx = cidx[:, 1::2]

        u_lo = u1[lo_idx].astype(np.float64)
        u_hi = u1[hi_idx].astype(np.float64)

        alpha_level = alpha_vec[node_indices].reshape(-1, 1)

        eta0_level = vectorized_eta_f(r1_fixed, r1_fixed, u_lo, u_hi, alpha_level)
        eta1_level = vectorized_eta_f(r1_fixed, r1_fixed, u_lo, u_hi, alpha0)

        phi0_lo = phi0[lo_idx]
        phi0_hi = phi0[hi_idx]
        phi1_lo = phi1[lo_idx]
        phi1_hi = phi1[hi_idx]

        phi0_nodes, phi1_nodes = vectorized_phi_f(
            log_lambdas, rho00_val, rho01_val,
            eta0_level, eta1_level,
            phi0_lo, phi0_hi, phi1_lo, phi1_hi
        )

        phi0[node_indices] = phi0_nodes
        phi1[node_indices] = phi1_nodes

        phi0_nodes_col = phi0_nodes.reshape(-1, 1)

        rho00_level = log_lambdas + rho00_val + eta0_level + phi0_lo + phi0_hi - phi0_nodes_col
        rho01_level = log_lambdas + rho01_val + eta1_level + phi1_lo + phi1_hi - phi0_nodes_col

        eta0[:, node_indices]  = eta0_level.T
        eta1[:, node_indices]  = eta1_level.T
        rho00[:, node_indices] = rho00_level.T
        rho01[:, node_indices] = rho01_level.T

    return {
        "part_vec": part_vec,
        "ratio1":   [0.5] * n_nodes,
        "set_size": u1,
        "eta0":  [np.longdouble(eta0[i])  for i in range(d)],
        "eta1":  [np.longdouble(eta1[i])  for i in range(d)],
        "phi0":  np.longdouble(phi0),
        "phi1":  np.longdouble(phi1),
        "rho00": [np.longdouble(rho00[i]) for i in range(d)],
        "rho01": [np.longdouble(rho01[i]) for i in range(d)],
        "alpha_vec": alpha_vec
    }


def log_diff_exp(log_a, log_b):
    """log(exp(x) - exp(y)), computed stably. Currently unused."""
    if log_a < log_b:
        raise ValueError("log_a must be greater than or equal to log_b to ensure the result is real and positive.")
    if log_a == log_b:
        return float('-inf')
    return log_a + math.log1p(-math.exp(log_b - log_a))


#################################################################################################
############################### MC - SAMPLE TREE ################################################

def sample_tree(max_res, part_vec, rho00, rho01, lambdas, u1, ratio1, alpha_vec, alpha0, d):
    """
    Sample a tree from the d-dimensional Optional Polya Tree posterior.

    Parameters:
        max_res (int): Maximum resolution level.
        part_vec (np.ndarray): Partition vector of shape (n_nodes, d, 2).
        rho00 (list of np.ndarray): List of d arrays of rho00 values.
        rho01 (list of np.ndarray): List of d arrays of rho01 values.
        lambdas (list): Split probabilities per dimension.
        u1 (np.ndarray): Set sizes.
        ratio1 (np.ndarray): Lo child volume ratios.
        alpha_vec (np.ndarray): Alpha values per node.
        alpha0 (float): Global alpha0.
        d (int): Dimension.

    Returns:
        dict:
            - "tree": list of nodes in the sampled tree.
            - "node_states": list of node states (0 non-stopping, 1 stopping).
            - "original_index": list of indices in part_vec for each tree node.
            - "parent_index": list of parent indices in the tree.
            - "split_dimension": list of split dimension indices per node.
    """
    tree          = [part_vec[0]]
    original_index = [0]
    parent_index  = [0]
    node_state    = []
    split_dim     = []

    m = 0
    l = 1
    while m <= max_res - 1:
        new = []

        for idx, node in enumerate(tree[-l:]):
            k             = len(tree) - l + idx
            node_pvec_idx = original_index[k]

            if k == 0:
                parent_state = 0
            else:
                parent_state = node_state[parent_index[k]]

            child_indices = children_index(node_pvec_idx, d)

            if parent_state == 0:
                rho0_0 = posterior_rho_s_s1([rho00[i][node_pvec_idx] for i in range(d)])
                rho0_1 = posterior_rho_s_s1([rho01[i][node_pvec_idx] for i in range(d)])

                # rho00/rho01 are np.longdouble (see bottom_up_recursion_2 /
                # binary_bottom_up_recursion), which is float128 on Linux x86_64
                # but silently aliases to float64 on macOS ARM. np.random.binomial's
                # p= and np.random.choice's p= both require strict float64 and
                # refuse to auto-downcast from float128 -- cast explicitly here,
                # right before sampling, so this only affects platforms where
                # longdouble is genuinely wider than float64.
                rho0_1_exp = np.clip(np.exp(rho0_1), 0, 1).astype(np.float64)
                s = np.random.binomial(1, rho0_1_exp)

                if s == 0:
                    node_state.append(0)
                    rho00_dims = np.array([rho00[i][node_pvec_idx] - rho0_0 for i in range(d)])
                    dim_probs  = np.clip(np.exp(rho00_dims), 0, 1).astype(np.float64)
                    dim_probs  = dim_probs / dim_probs.sum()
                    chosen_dim = np.random.choice(d, p=dim_probs)
                else:
                    node_state.append(1)
                    rho01_dims = np.array([rho01[i][node_pvec_idx] - rho0_1 for i in range(d)])
                    dim_probs  = np.clip(np.exp(rho01_dims), 0, 1).astype(np.float64)
                    dim_probs  = dim_probs / dim_probs.sum()
                    chosen_dim = np.random.choice(d, p=dim_probs)

            else:
                node_state.append(1)
                dim_probs  = np.array(lambdas)
                chosen_dim = np.random.choice(d, p=dim_probs)

            split_dim.append(chosen_dim)
            child1_i = child_indices[2 * chosen_dim]
            child2_i = child_indices[2 * chosen_dim + 1]

            new.append(part_vec[child1_i])
            new.append(part_vec[child2_i])
            original_index.extend([child1_i, child2_i])
            parent_index.extend([k, k])

        tree.extend(new)
        l = len(new)
        m += 1

    split_dim.extend([-1] * len(new))

    return {
        "tree":            tree,
        "node_states":     node_state,
        "original_index":  original_index,
        "parent_index":    parent_index,
        "split_dimension": split_dim
    }


def csi_f(RHO_post, c0, c1, base_child, n_set, n_child, csi0_child, csi1_child):
    """Propagate the posterior mean density from children up to a node.

    The nD counterpart of the 1D `csi_f`. Logs throughout.
    """
    csi1 = np.log(c1 * base_child + n_child) - np.log(c1 + n_set - 1) - np.log(base_child) + csi1_child
    t1   = RHO_post[0] + np.log(c0 * base_child + n_child) - np.log(c0 + n_set - 1) - np.log(base_child) + csi0_child
    t2   = RHO_post[1] + csi1
    csi0 = log_sum_exp_multiple([t1, t2])
    return csi0, csi1


def OPT_posterior_mean_cond_tree(tree, original_index, parent_index, max_res, rho00, rho01, alpha_vec, alpha0, ratio1, u1, d):
    """
    Compute posterior mean conditional on a sampled tree for a d-dimensional
    Optional Polya Tree, integrating out Beta weights analytically via csi_f.

    Parameters:
        tree (list): Sampled tree nodes.
        original_index (list): Indices in part_vec for each tree node.
        parent_index (list): Parent indices in the tree.
        max_res (int): Maximum resolution level.
        rho00 (list of np.ndarray): List of d arrays of rho00 values.
        rho01 (list of np.ndarray): List of d arrays of rho01 values.
        alpha_vec (np.ndarray): Alpha values per node.
        alpha0 (float): Global alpha0.
        ratio1 (np.ndarray): Lo child volume ratios.
        u1 (np.ndarray): Set sizes.
        d (int): Dimension.

    Returns:
        dict:
            - "sets": list of leaf nodes.
            - "prob": array of posterior mean probabilities for each leaf.
    """
    n_leaves = 2 ** max_res
    csi0 = np.zeros((n_leaves, max_res + 1))
    csi1 = np.zeros((n_leaves, max_res + 1))

    for j in range(n_leaves):
        index_child_tree = n_leaves + j - 1
        for r in reversed(range(max_res)):
            index_node_tree  = parent_index[index_child_tree]
            index_node_pvec  = original_index[index_node_tree]
            index_child_pvec = original_index[index_child_tree]

            rho0_0 = posterior_rho_s_s1([rho00[i][index_node_pvec] for i in range(d)])
            rho0_1 = posterior_rho_s_s1([rho01[i][index_node_pvec] for i in range(d)])

            csi0[j][r], csi1[j][r] = csi_f(
                [rho0_0, rho0_1],
                alpha_vec[index_child_pvec],
                alpha0,
                ratio1[index_child_pvec],
                u1[index_node_pvec],
                u1[index_child_pvec],
                csi0[j][r + 1],
                csi1[j][r + 1]
            )
            index_child_tree = index_node_tree

    tp        = np.exp(csi0[:, 0])
    last_sets = tree[-n_leaves:]
    return {"sets": last_sets, "prob": tp}


def sample_tree_and_mean(seed, x_grid, max_res, part_vec, rho00, rho01, lambdas, u1, ratio1, alpha_vec, alpha0, d, interpolate_fn):
    """Draw one tree from the posterior and return the mean density given it."""
    random.seed(seed)
    np.random.seed(seed)
    tree = sample_tree(max_res, part_vec, rho00, rho01, lambdas, u1, ratio1, alpha_vec, alpha0, d)
    pm   = OPT_posterior_mean_cond_tree(
        tree["tree"], tree["original_index"], tree["parent_index"],
        max_res, rho00, rho01, alpha_vec, alpha0, ratio1, u1, d
    )
    return interpolate_fn(pm["sets"], pm["prob"], x_grid)


def process_seed_chunk_MCt(seed_chunk, max_res, part_vec, rho00, rho01, lambdas, u1, ratio1, alpha_vec, alpha0, x_grid, d, interpolate_fn):
    """Accumulate tree-conditional means for a chunk of seeds. joblib `delayed` worker."""
    chunk_sum_probs = 0
    for seed in seed_chunk:
        chunk_sum_probs += sample_tree_and_mean(seed, x_grid, max_res, part_vec, rho00, rho01,
                                                 lambdas, u1, ratio1, alpha_vec, alpha0, d, interpolate_fn)
    gc.collect()
    return chunk_sum_probs


def sampling_MCt(iter, x_grid, max_res, part_vec, rho00, rho01, lambdas, u1, ratio1, alpha_vec, alpha0, base_seed, n_jobs, d, interpolate_fn):
    """
    Monte Carlo estimation of the posterior mean density via Rao-Blackwellized
    tree sampling for a d-dimensional Optional Polya Tree.

    Parameters:
        iter (int): Number of MC iterations.
        x_grid (list of np.ndarray): List of d meshgrid arrays.
        max_res (int): Maximum resolution level.
        part_vec (np.ndarray): Partition vector of shape (n_nodes, d, 2).
        rho00 (list of np.ndarray): List of d arrays of rho00 values.
        rho01 (list of np.ndarray): List of d arrays of rho01 values.
        lambdas (list): Split probabilities per dimension.
        u1 (np.ndarray): Set sizes.
        ratio1 (np.ndarray): Lo child volume ratios.
        alpha_vec (np.ndarray): Alpha values per node.
        alpha0 (float): Global alpha0.
        base_seed (int): Base random seed.
        n_jobs (int): Number of parallel jobs.
        d (int): Dimension.
        interpolate_fn (callable): Interpolation function for d-dimensional grids.

    Returns:
        np.ndarray: Monte Carlo estimate of the posterior mean density on the grid.
    """
    seeds      = [base_seed + i for i in range(iter)]
    sum_probs  = np.zeros(x_grid[0].shape)
    chunk_size = iter // n_jobs
    remainder  = iter % n_jobs
    seed_chunks = []
    start = 0
    for i in range(n_jobs):
        current_chunk_size = chunk_size + 1 if i < remainder else chunk_size
        seed_chunks.append(seeds[start:start + current_chunk_size])
        start += current_chunk_size

    chunk_results = Parallel(n_jobs=n_jobs)(
        delayed(process_seed_chunk_MCt)(
            seed_chunk, max_res, part_vec, rho00, rho01, lambdas,
            u1, ratio1, alpha_vec, alpha0, x_grid, d, interpolate_fn
        )
        for seed_chunk in seed_chunks
    )

    for chunk_sum in chunk_results:
        sum_probs += chunk_sum
    return sum_probs / iter


#################################################################################################
############################### NUMERICAL ANALYSIS ##############################################

def stopping_depth(sample_size, threshold, d):
    """
    Compute the stopping depth for a d-dimensional Optional Polya Tree.

    Parameters:
        sample_size (int): Number of observations.
        threshold (int): Minimum sample size to continue splitting.
        d (int): Dimension.

    Returns:
        int: Stopping depth.
    """
    s     = sample_size
    depth = 0
    while s > threshold:
        s = np.floor((s - d) / 2)
        depth += 1
    return depth - 1


def interpolate_step_function(intervals, heights, grids):
    """
    Interpolate a d-dimensional step function over a grid.

    Parameters:
        intervals (list of list of list): Each interval is a list of d [min, max] pairs.
        heights (list or np.ndarray): Height values for each interval.
        grids (list of np.ndarray): List of d meshgrid arrays, one per dimension.

    Returns:
        np.ndarray: Interpolated heights on the grid.
    """
    if len(intervals) != len(heights):
        raise ValueError("Number of intervals must match the number of heights.")
    d = len(grids)
    interpolated_heights = np.zeros(grids[0].shape)
    for interval, height in zip(intervals, heights):
        mask = np.ones(grids[0].shape, dtype=bool)
        for i in range(d):
            mask &= (grids[i] >= interval[i][0]) & (grids[i] < interval[i][1])
        interpolated_heights[mask] = height
    return interpolated_heights


def l1_distance(true_density, estimated_density, n_eval):
    """L1 distance between two densities on a grid, scaled by the cell volume."""
    true_density      = np.array(true_density)
    estimated_density = np.array(estimated_density)
    cell_volume       = 1.0 / n_eval
    return np.sum(np.abs(true_density - estimated_density)) * cell_volume


def l2_distance(true_density, estimated_density, n_eval):
    """L2 distance between two densities on a grid, scaled by the cell volume."""
    true_density      = np.array(true_density)
    estimated_density = np.array(estimated_density)
    cell_volume       = 1.0 / n_eval
    return np.sqrt(np.sum((true_density - estimated_density) ** 2) * cell_volume)


def linfty_distance(true_density, estimated_density, n_eval=None):
    """Maximum absolute difference between two densities on a grid.

    Takes the cell volume for signature compatibility with l1/l2; it is unused.
    """
    true_density      = np.array(true_density)
    estimated_density = np.array(estimated_density)
    return np.max(np.abs(true_density - estimated_density))


def distance_metric(grids, partial_prob, full_prob, true_den, distance_func):
    """Distance from the true density to each of the two fits. Returns (partial, full)."""
    d            = len(grids)
    n_eval       = len(grids[0])
    partial_prob = partial_prob.ravel()
    full_prob    = full_prob.ravel()
    true_den     = true_den.ravel()

    dist_partial = distance_func(true_den, partial_prob, n_eval)
    dist_full    = distance_func(true_den, full_prob,    n_eval)
    return dist_partial, dist_full


def sample_from_mixture(components, weights, sample_size, seed=None):
    """Draw `size` points from an nD mixture with the given weights."""
    if seed is not None:
        np.random.seed(seed)
    weights            = np.array(weights)
    weights           /= np.sum(weights)
    component_indices  = np.random.choice(len(weights), size=sample_size, p=weights)
    samples_list       = []
    for i in range(len(weights)):
        num_samples = np.sum(component_indices == i)
        if num_samples > 0:
            samples_list.append(components[i](size=num_samples, seed=seed))
    return np.vstack(samples_list)


def pdf_for_mixture(grids, pdf_functions, weights):
    """
    Compute the PDF of a mixture distribution over a d-dimensional grid.

    Parameters:
        grids (list of np.ndarray): List of d meshgrid arrays, one per dimension.
        pdf_functions (list of callable): Each function computes the PDF of one component.
        weights (list or np.ndarray): Mixture weights, summing to 1.

    Returns:
        np.ndarray: Mixture PDF values on the grid, same shape as each grid array.
    """
    samples     = np.stack([g.ravel() for g in grids], axis=-1)
    mixture_pdf = np.zeros(samples.shape[0], dtype=float)
    for weight, pdf_func in zip(weights, pdf_functions):
        mixture_pdf += weight * pdf_func(samples)
    return mixture_pdf.reshape(grids[0].shape)
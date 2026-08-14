import numpy as np
import random
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.special import beta as bf

from src.core import partition_1D_functions as basic
from src.core import OPT_1D_partial_functions as latent


def set_count_full(X, node):
    """Number of points of `X` in the half-open interval `node`."""
    count = 0
    for element in X:
        if node[0] < element <= node[1]:
            count += 1
    return count
    
def child_full(node):
    """Split `node` at its midpoint, returning the two child intervals.

    The full model splits on the interval rather than on an order statistic,
    which is what distinguishes it from `basic.child_partial`.
    """
    mid = (node[0] + node[1])/2
    
    lnode = [node[0], mid] 
    rnode = [mid, node[1]]
    
    return lnode, rnode

def parent_full(node):
    """The interval `node` was split from."""
    
    length = node[1] - node[0]
    dist_0 = node[0] 
    
    if (dist_0/length)% 2 == 0:
        parent = [node[0], (node[0]+2*length)]
    else:
        parent = [(node[1]-2*length), node[1]]    
    
    return parent

def partition_full(depth):
    """Build the dyadic partition of [0, 1] to a fixed `depth`.

    Nodes come out in breadth-first order, the layout the rest of this module
    indexes by position.
    """
    
    prev = [[0, 1]]

    for m in range(0, depth):
        l = 2**m
        new = []
        for node in prev[-l:]:
            sub = child_full(node)
            new.append(sub[0])
            new.append(sub[1])
        prev.extend(new)
        l = len(new)

    return prev

# This partition function reaches a minimum depth and then stops by a set count threshold
def partition_size(X, n, threshold, min_depth):
    """Partition until a node holds fewer than `threshold` points.

    Stops no earlier than `min_depth`. Currently unused: the simulations grow
    the partition to a fixed depth instead.
    """
    
    prev = [[0, 1]]
    sc = [n]
    l = 1
    m = 0
    
    while True:
        new = []
        current_min_count = float('inf')
        for node in prev[-l:]:
            sub = child_full(node)
            new.append(sub[0])
            new.append(sub[1])
            c1 = set_count_full(X, sub[0])
            c2 = set_count_full(X, sub[1])
            sc.append(c1)
            sc.append(c2)
            current_min_count = min(current_min_count, min(c1, c2))
        if m >= min_depth and current_min_count < threshold:
            break
        m += 1
        prev.extend(new)
        l = len(new)
        
    return prev, m


# This partition function reaches a minimum depth and then stops by a set length threshold
def partition_length(X, n, length, min_depth):
    """Partition until a node is narrower than `length`.

    Stops no earlier than `min_depth`. Currently unused: the simulations grow
    the partition to a fixed depth instead.
    """
    prev = [[0, 1]]
    sl = [n] 
    l = 1
    m = 0

    while True:
        new = []
        current_min_length = float('inf')
        for node in prev[-l:]:
            sub = child_full(node)
            new.append(sub[0])
            new.append(sub[1])
            sl.append(sub[0][1] - sub[0][0])
            sl.append(sub[1][1] - sub[1][0])
            current_min_length = min(current_min_length, min(sl[-2], sl[-1]))
        if m >= min_depth and current_min_length < length:
            break
        m += 1
        prev.extend(new)
        l = len(new)

    return prev, m


    

def base_ratio(part_vec): 
    """Base-measure mass of each node relative to its parent.

    Always 1/2 under the midpoint split, so this needs neither the sample nor
    the Beta parameters that `basic.base_ratio` takes.
    """
    length = len(part_vec)
    beta1 = [1] + [0.5] * (length - 1)
    beta2 = [0] + [0.5] * (length - 1)

    return beta1, beta2

    
def vec_count(part_vec, X):
    """Observation counts per node and sibling, for the Beta likelihood update."""
    beta1_up = []
    beta2_up = []
    for l in range(0, len(part_vec)):
        
        p_node = parent_full(part_vec[l]) 
        s_parent = set_count_full(X, p_node)
        
        s_set = set_count_full(X, part_vec[l])
        s_other = s_parent - s_set
        
        beta1_up.append(s_set)
        beta2_up.append(s_other)

    return beta1_up, beta2_up 

# OPT(X, depth, alpha, a, b, alpha0, p0)
# X: is the sample
# depth: is maximum level of resolution chosen
# alpha: is the concentration parameter
# a, b: the parameters of the Beta distribution of the base measure
# alpha0: the concentration parameter in the stopping case
# p0: the prior  stopping probability 

def PT_full_model(X, depth, a, b):
    """Fit the fixed-depth Polya tree on the midpoint partition.

    The full-model counterpart of `latent.PT_partial_model`: same accumulation
    of log Beta means along root-to-leaf paths, but on intervals split at their
    midpoints rather than at order statistics.

    Returns (intervals, density).
    """

    part_vec = partition_full(depth)
    s1, s2 = base_ratio(part_vec)
    
    alpha_vec = [1]
    
    for m in range(1, depth+1):
        l = 2**m
        new = [m**4]*l
        alpha_vec.extend(new)
    
    b1, b2 = basic.beta_prior_vec(s1, s2, alpha_vec)
    #b1, b2 = basic.beta_prior(s1, s2, alpha)
    u1, u2 = vec_count(part_vec, X) 
    b1_u = [x + y for x, y in zip(b1, u1)]
    b2_u = [x + y for x, y in zip(b2, u2)]

    mean = []
    for l in range(0, len(b1_u)):
        mean.append(np.log(b1_u[l])-np.log(b1_u[l]+b2_u[l]))
    
    mean_M = basic.vec_to_mat(mean, depth)
    post_mean = np.sum(mean_M, axis=1)

    for k in reversed(range(1, 2**depth+1)):
        post_mean[-k] =  post_mean[-k] - np.log(stats.beta.cdf([part_vec[-k][1]], a, b) - stats.beta.cdf([part_vec[-k][0]], a, b))

    last_sets = list(set(np.concatenate(part_vec[-2**depth:])))
    last_sets = np.sort(last_sets)
     
    return last_sets, np.exp(post_mean)





    

def OPT_full_model(X, part_vec, depth, alpha, a, b, alpha0, p0):
    """Run the optional-stopping recursion on the midpoint partition.

    The full-model counterpart of `latent.BT_Optional_Stopping_fdepth`, reusing
    that module's eta/phi/rho/csi steps. `alpha0` is the concentration in the
    stopping state and `p0` the prior stopping probability.
    """
    
    X = np.sort(X)
    n = len(X)
 
  #  part_vec = partition_full(depth)
    s1, s2 = base_ratio(part_vec)
    b1, b2 = basic.beta_prior(s1, s2, alpha)
    u1, u2 = vec_count(part_vec, X)
    b1_u = [x + y for x, y in zip(b1, u1)]
    b2_u = [x + y for x, y in zip(b2, u2)]

    eta0 = [0]*(len(part_vec))
    eta1 = [0]*(len(part_vec))
    phi0 = [0]*(len(part_vec))
    phi1 = [0]*(len(part_vec))
    rho01  = [0]*(len(part_vec))
    rho00  = [0]*(len(part_vec))

    csi0_l = [0]*(len(part_vec))
    csi0_r = [0]*(len(part_vec))
    csi1_l = [0]*(len(part_vec))
    csi1_r = [0]*(len(part_vec))

    RHO = [np.log(1-p0), np.log(p0)]
    csi0 = np.zeros((2**depth, depth + 1))
    csi1 = np.zeros((2**depth, depth + 1))

    data = []

    start = 2**depth +1 
    for j in reversed(range(0, depth)):

        end = start + 2**j - 1
        for k in range(start, end +1):

            node      = part_vec[-k]
            lchild, rchild = child_full(node)
            lchild_i  = part_vec.index(lchild)
            rchild_i  = part_vec.index(rchild)
            
            if u1[-k] <= 1:
                
                eta0[-k] = 0
                eta1[-k] = 0

                phi0[-k] = 0
                phi1[-k] = 0

                rho01[-k] = RHO[1]
                rho00[-k] = RHO[0]
                
            else:

                eta0[-k]  = latent.eta_f(s1[lchild_i], s1[rchild_i], u1[lchild_i], u1[rchild_i], alpha)
                eta1[-k]  = latent.eta_f(s1[lchild_i], s1[rchild_i], u1[lchild_i], u1[rchild_i], alpha0)

                phi0[-k], phi1[-k]  = latent.phi_f(phi0[lchild_i], phi1[lchild_i], phi0[rchild_i], phi1[rchild_i], RHO, eta0[-k], eta1[-k])

                rho01[-k] = latent.rho0s_f(RHO[1], phi0[-k], phi1[lchild_i], phi1[rchild_i], eta1[-k])
                rho00[-k] = latent.rho0s_f(RHO[0], phi0[-k], phi0[lchild_i], phi0[rchild_i], eta0[-k])

            #print(j, node, eta0[-k], eta1[-k], phi0[-k], phi1[-k], rho01[-k], rho00[-k])
            data.append([j, node, s1[-k], s1[lchild_i], s1[rchild_i], u1[-k], u1[lchild_i], u1[rchild_i], eta0[-k], eta1[-k], phi0[-k], phi1[-k], rho01[-k], rho00[-k]])

        start = start + 2**j

    matrix = np.zeros((2**depth, depth + 1))

    for j in range(0, 2**depth):
        h = 2**depth+j-1 
        node_child = part_vec[h]

        for r in reversed(range(0, depth)):
            node = parent_full(node_child)

            node_i  = part_vec.index(node)
            child_i = part_vec.index(node_child)

            csi0[j][r], csi1[j][r]  = latent.csi_f([rho00[node_i], rho01[node_i]], alpha, alpha0, \
                                         s1[child_i], u1[node_i], u1[child_i], csi0[j][r+1], csi1[j][r+1])
            matrix[j][r] = node_i

            node_child = node
        
    final = csi0[:, 0] 

    tp = np.exp(csi0[:, 0]) 

    last_sets = list(set(np.concatenate(part_vec[-2**depth:])))
    last_sets = np.sort(last_sets)


    # return np.exp(np.float128(eta0)), np.exp(np.float128(eta1)), \
    #        np.exp(np.float128(phi0)), np.exp(np.float128(phi1)), \
    #        np.exp(np.float128(rho01)), tp, last_sets, part_vec, s1, s2, b1, b2, u1, u2  

    return np.longdouble(eta0), np.longdouble(eta1), \
            np.longdouble(phi0), np.longdouble(phi1), \
            np.exp(np.longdouble(rho01)), tp, last_sets, part_vec, s1, s2, b1, b2, u1, u2  

   

def posterior_sampler_full(part_vec, s1, s2, v1, v2, alpha, max_res, rho01_vec, X_aug, a, b, p):
    """Draw one posterior sample of the log-density under the full model.

    The counterpart of `latent.posterior_sampler`, for credible bands on the
    full model. Currently unused: the shipped pipeline reports posterior means
    rather than bands.
    """
    
    b1_n, b2_n = basic.beta_prior(s1, s2, alpha) # Generating the prior for the non-stopping case
   
    b1_n_u = [x + y for x, y in zip(b1_n, v1)]  # Updating the priors for all sets for the stopping and non-stopping cases
    b2_n_u = [x + y for x, y in zip(b2_n, v2)]
     
    # Stores the sampling probabilities for each set of the partition
    sample_p = [0]*(len(part_vec))
    sample_p[0] = 1
    
    # Stores the stopping variables for each set of the partition
    stopping = [0]*(len(part_vec))
    
    # Obtain a stopping variable for the omega set
    stopping[0] = int(np.random.binomial(n=1, p= rho01_vec[0], size=1))    
    
    if stopping[0] == 0:
        sample_p[1] = float(np.random.beta(b1_n_u[1], b2_n_u[1], size=1))
        sample_p[2] = 1 - sample_p[1]
    
    elif stopping[0] == 1:
        sample_p[1] = s1[1]
        sample_p[2] = 1 - sample_p[1]
    
    
    for index in range(1, 2**max_res - 1):
        
        # Consider every node generated in the partitioning process 
        node = part_vec[index]                 
        prt = parent_full(node)
        parent_index = part_vec.index(prt)
        
        lchild, rchild = child_full(node) 
        lchild_index = part_vec.index(lchild)
        rchild_index = lchild_index + 1
        
        if stopping[parent_index] == 1:
            stopping[index] = 1
        elif stopping[parent_index] == 0:
            stopping[index] = int(np.random.binomial(n = 1, p = rho01_vec[index], size = 1))
            
        if stopping[index] == 1:
            sample_p[lchild_index] = s1[lchild_index]
        
        elif stopping[index] == 0:
            sample_p[lchild_index] = float(np.random.beta(b1_n_u[lchild_index], b2_n_u[lchild_index], size=1))
            
        
        sample_p[rchild_index] = 1 - sample_p[lchild_index]
        
        
    SAMPLE = basic.vec_to_mat(np.log(sample_p), max_res)
    post_sample = np.sum(SAMPLE, axis=1)
    
    post_sample_scaled = [0]*len(post_sample)
    for k in reversed(range(1, 2**max_res+1)):
        post_sample_scaled[-k] =  post_sample[-k] - np.log(stats.beta.cdf(part_vec[-k][1], a, b) - stats.beta.cdf(part_vec[-k][0], a, b))
    
    return np.exp(post_sample_scaled), stopping

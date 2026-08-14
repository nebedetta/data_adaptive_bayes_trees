import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import scipy as sp
import scipy.stats as stats
from scipy.special import beta as bf

from src.core import partition_1D_functions as basic

# For numerical stability, the beta function will be expressed in terms of the gamma function
# Arguments not in log, output in log
def eta_f(base_al, base_ar, n_al, n_ar, c):
    """Log marginal likelihood of a node's split, under a Beta-Binomial.

    Written via lgamma rather than the Beta function directly, for numerical
    stability. Arguments are on the natural scale; the result is a log.
    """
    
    t = - n_al * np.log(base_al) - n_ar * np.log(base_ar) + \
         math.lgamma(c * base_al + n_al) + math.lgamma(c * base_ar + n_ar) - \
         math.lgamma(c + n_al + n_ar) - \
         math.lgamma(c * base_al) - math.lgamma(c * base_ar) + math.lgamma(c)
  
    return t         


def log_exp_x_plus_exp_y(x, y):
    """log(exp(x) + exp(y)), computed stably.

    Returns the larger argument outright once they differ by more than 50 in
    log space, and handles two infinities of the same sign.
    """
    if math.isinf(x) and math.isinf(y) and x < 0 and y < 0:
        return float('-inf')
    elif math.isinf(x) and math.isinf(y) and x > 0 and y > 0:
        return float('inf')
    elif x - y >= 50:
        return x
    elif x - y <= -50:
        return y
    else:
        if x > y:
            return y + math.log(1 + math.exp(x - y))
        else:
            return x + math.log(1 + math.exp(y - x))


# Arguments in log, output in log
def phi_f(PHI_l0, PHI_l1, PHI_r0, PHI_r1, RHO, eta0, eta1):
    """Combine children's phi values into a node's, over both stopping states.

    phi0 mixes the split and stop branches with prior log-weights `RHO`; phi1
    is the stop branch alone. All arguments and returns are logs.
    """
    
    t1_0 =  RHO[0] + eta0 + PHI_l0 + PHI_r0 
    t2_0 =  RHO[1] + eta1 + PHI_l1 + PHI_r1 
    
    phi0 = log_exp_x_plus_exp_y(t1_0, t2_0)
    
    phi1 =  PHI_l1 + PHI_r1 + eta1
    
    return phi0, phi1

# Arguments in log, output in log
def rho0s_f(rho0s, phi0, PHI_ls, PHI_rs, etas):
    """Posterior log-probability that a node is in the stopping state.

    The prior `rho0s` reweighted by the node's own evidence and its children's
    phi values, normalised by phi0. Logs throughout.
    """
    
    rho0s_post = rho0s + etas + PHI_ls + PHI_rs - phi0
    
    return rho0s_post


def csi_f(RHO_post, c0, c1, base_child, n_set, n_child, csi0_child, csi1_child):
    """Propagate the posterior mean density from a child up to its parent.

    csi1 is the stopping branch; csi0 mixes it with the splitting branch using
    the posterior state probabilities `RHO_post`. Logs throughout.
    """
    
    csi1 = np.log(c1*base_child + n_child) - np.log(c1 + n_set - 1) - np.log(base_child) + csi1_child
    
    t1 = RHO_post[0] + np.log(c0*base_child + n_child) - np.log(c0 + n_set - 1) - np.log(base_child) + csi0_child
    t2 = RHO_post[1] + csi1
    
    csi0 = log_exp_x_plus_exp_y(t1, t2)
    
    return csi0, csi1


def PT_partial_model(X, depth, a, b, p = 0.5):
    """Fit the fixed-depth Polya tree and return its posterior mean density.

    Builds the depth-`depth` partition of the sorted sample, forms Beta priors
    from the base measure's mass ratios scaled by a depth-dependent
    pseudocount (m**4 at depth m), adds the observed counts, and accumulates
    the resulting log means along each root-to-leaf path.

    Returns (intervals, density): the leaf boundaries, and the posterior mean
    density on them.
    """

    X = np.sort(X)
    n = len(X)
    X_aug = list(X)
    X_aug.insert(0, 0)
    X_aug.append(1)

    max_res, part_vec = partition_fdepth(n, depth, p)
    s1, s2 = basic.base_ratio(part_vec, a, b, X_aug)
    
    alpha_vec = [1]
    
    for m in range(1, depth+1):
        l = 2**m
        new = [m**4]*l
        alpha_vec.extend(new)
    
    b1, b2 = basic.beta_prior_vec(s1, s2, alpha_vec)
    
    #b1, b2 = basic.beta_prior(s1, s2, alpha)
    u1, u2 =  basic.vec_count(part_vec)
    b1_u = [x + y for x, y in zip(b1, u1)]
    b2_u = [x + y for x, y in zip(b2, u2)]

    mean = []
    for l in range(0, len(b1_u)):
        mean.append(np.log(b1_u[l])-np.log(b1_u[l]+b2_u[l]))
    
    mean_M = basic.vec_to_mat(mean, depth)
    post_mean = np.sum(mean_M, axis=1)

    for k in reversed(range(1, 2**depth+1)):
        post_mean[-k] =  post_mean[-k] - np.log(stats.beta.cdf(X_aug[part_vec[-k][1]], a, b) - stats.beta.cdf(X_aug[part_vec[-k][0]], a, b))

    last_sets = list(set(np.concatenate(part_vec[-2**max_res:])))
    last_sets = np.sort(last_sets)
    intervals = []
    for i in last_sets:
        intervals.append(X_aug[i])
     
    return intervals, np.exp(post_mean)

# BT_Optional_Stopping(X, threshold, p, alpha, a, b, alpha0, a0, b0, p0)
# X: is the sample
# threshold: is the number of observations in the leaves for which the recursion is stopped
# p: the order statistic of choice 
# alpha: is the concentration parameter
# a, b: the parameters of the Beta distribution of the base measure
# alpha0: the concentration parameter in the stopping case
# p0: the prior  stopping probability 


def BT_Optional_Stopping(X, threshold, p, alpha, a, b, alpha0, p0):
    """Run the optional-stopping recursion on a partition grown to a count threshold.

    Splits until a node holds fewer than `threshold` points, then passes
    bottom-up: eta_f for each node's split evidence, phi_f to combine children,
    rho0s_f for the posterior stopping probabilities, and csi_f to carry the
    posterior mean back down. `alpha0` is the concentration in the stopping
    state, `p0` the prior stopping probability.
    """
    
    X = np.sort(X)
    n = len(X)
    X_aug = list(X)
    X_aug.insert(0, 0)
    X_aug.append(1)

    max_res, part_vec = basic.partition_partial(n, threshold, p)
    s1, s2 = basic.base_ratio(part_vec, a, b, X_aug)
    b1, b2 = basic.beta_prior(s1, s2, alpha)
    u1, u2 =  basic.vec_count(part_vec)
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
    csi0 = np.zeros((2**max_res, max_res + 1))
    csi1 = np.zeros((2**max_res, max_res + 1))

    data = []

    start = 2**max_res +1 
    for j in reversed(range(0, max_res)):

        end = start + 2**j - 1
        for k in range(start, end +1):

            node      = part_vec[-k]
            lchild, rchild = basic.child_partial(node, p)
            lchild_i  = part_vec.index(lchild)
            rchild_i  = part_vec.index(rchild)
            
            
       #     if u1[-k] <= 1:
                
       #         eta0[-k] = 0
       #         eta1[-k] = 0

        #        phi0[-k] = 0
        #        phi1[-k] = 0

        #        rho01[-k] = RHO[1]
        #        rho00[-k] = RHO[0]
        #        
         #   else:
                
            eta0[-k]  = eta_f(s1[lchild_i], s1[rchild_i], u1[lchild_i], u1[rchild_i], alpha)
            eta1[-k]  = eta_f(s1[lchild_i], s1[rchild_i], u1[lchild_i], u1[rchild_i], alpha0)

            phi0[-k], phi1[-k]  = phi_f(phi0[lchild_i], phi1[lchild_i], phi0[rchild_i], phi1[rchild_i], RHO, eta0[-k], eta1[-k])

            rho01[-k] = rho0s_f(RHO[1], phi0[-k], phi1[lchild_i], phi1[rchild_i], eta1[-k])
            rho00[-k] = rho0s_f(RHO[0], phi0[-k], phi0[lchild_i], phi0[rchild_i], eta0[-k])

            #print(j, node, eta0[-k], eta1[-k], phi0[-k], phi1[-k], rho01[-k], rho00[-k])
            data.append([j, node, s1[-k], s1[lchild_i], s1[rchild_i], u1[-k], u1[lchild_i], u1[rchild_i], eta0[-k], eta1[-k], phi0[-k], phi1[-k], rho01[-k], rho00[-k]])

        start = start + 2**j

    matrix = np.zeros((2**max_res, max_res + 1))

    for j in range(0, 2**max_res):
        h = 2**max_res+j-1 
        node_child = part_vec[h]

        for r in reversed(range(0, max_res)):
            node = basic.parent(node_child, part_vec)

            node_i  = part_vec.index(node)
            child_i = part_vec.index(node_child)

            csi0[j][r], csi1[j][r]  = csi_f([rho00[node_i], rho01[node_i]], alpha, alpha0, \
                                         s1[child_i], u1[node_i], u1[child_i], csi0[j][r+1], csi1[j][r+1])
            matrix[j][r] = node_i

            node_child = node

    final = csi0[:, 0] 

    tp = np.exp(csi0[:, 0]) 

    last_sets = list(set(np.concatenate(part_vec[-2**max_res:])))
    last_sets = np.sort(last_sets)
    intervals = []
    for i in last_sets:
        intervals.append(X_aug[i])

    # return np.exp(np.float128(eta0)), np.exp(np.float128(eta1)), \
    #        np.exp(np.float128(phi0)), np.exp(np.float128(phi1)), \
    #        np.exp(np.float128(rho01)), tp, intervals, max_res, part_vec, s1, s2, b1, b2, u1, u2
    
    return np.longdouble(eta0), np.longdouble(eta1), \
           np.longdouble(phi0), np.longdouble(phi1), \
           np.exp(np.longdouble(rho01)), tp, intervals, max_res, part_vec, s1, s2, b1, b2, u1, u2  
    


def sample_from_mixture(components, weights, size=1):
    """Draw `size` values from a mixture of `components` with the given `weights`."""
  
    weights = np.array(weights)
    weights /= np.sum(weights)

    component_indices = np.random.choice(len(components), size=size, p=weights)

    samples = np.array([components[i]() for i in component_indices])

    return samples



# (Truncated) Posterior sampling function 
# Taking as argument the base ratio

# s1, s2 = basic.base_ratio(part_vec, a, b, X_aug)
# v1, v2 = basic.vec_count(part_vec)

def posterior_sampler(part_vec, s1, s2, v1, v2, alpha, max_res, rho01_vec, X_aug, a, b, p):
    """Draw one posterior sample of the log-density on the finest partition.

    Samples each node's stopping state from its posterior `rho01_vec`, then the
    split fraction from the corresponding Beta, and accumulates along each
    root-to-leaf path. Passed to `numerical_analysis_functions.confidence_bands`
    to build credible bands.
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
    stopping[0] = int(np.random.binomial(n=1, p = rho01_vec[0], size=1))    
    
    if stopping[0] == 0:
        sample_p[1] = float(np.random.beta(b1_n_u[1], b2_n_u[1], size=1))
        sample_p[2] = 1 - sample_p[1]
    
    elif stopping[0] == 1:
        sample_p[1] = s1[1]
        sample_p[2] = 1 - sample_p[1]
    
    
    for index in range(1, 2**max_res - 1):
        
        # Consider every node generated in the partitioning process 
        node = part_vec[index]                 
        parent = basic.parent(node, part_vec)
        parent_index = part_vec.index(parent)
        
        lchild, rchild = basic.child_partial(node, p) 
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
        post_sample_scaled[-k] =  post_sample[-k] - np.log(stats.beta.cdf(X_aug[part_vec[-k][1]], a, b) - stats.beta.cdf(X_aug[part_vec[-k][0]], a, b))
    
    return np.exp(post_sample_scaled), stopping



def partition_fdepth(n, depth, p):
    """Build the partition to a fixed `depth` rather than to a count threshold.

    Returns (depth, part_vec), matching `basic.partition_partial`'s shape so the
    two are interchangeable downstream.
    """
    
    prev = [[0, n+1]]

    for m in range(0, depth):
        l = 2**m
        new = []
        for node in prev[-l:]:
            sub = basic.child_partial(node, p)
            new.append(sub[0])
            new.append(sub[1])
        prev.extend(new)
        l = len(new)

    return depth, prev


def BT_Optional_Stopping_fdepth(X, p, alpha, a, b, alpha0, p0, depth):
    """As `BT_Optional_Stopping`, but on a partition of fixed depth.

    Same bottom-up recursion; the partition is grown to `depth` regardless of
    how many points reach each leaf, which is what the simulations sweep over.
    """
    
    X = np.sort(X)
    n = len(X)
    X_aug = list(X)
    X_aug.insert(0, 0)
    X_aug.append(1)

    max_res, part_vec = partition_fdepth(n, depth, p)
    s1, s2 = basic.base_ratio(part_vec, a, b, X_aug)
    b1, b2 = basic.beta_prior(s1, s2, alpha)
    u1, u2 =  basic.vec_count(part_vec)
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
    csi0 = np.zeros((2**max_res, max_res + 1))
    csi1 = np.zeros((2**max_res, max_res + 1))

    data = []

    start = 2**max_res +1 
    for j in reversed(range(0, max_res)):

        end = start + 2**j - 1
        for k in range(start, end +1):

            node      = part_vec[-k]
            lchild, rchild = basic.child_partial(node, p)
            lchild_i  = part_vec.index(lchild)
            rchild_i  = part_vec.index(rchild)
    
            
            eta0[-k]  = eta_f(s1[lchild_i], s1[rchild_i], u1[lchild_i], u1[rchild_i], alpha)
            eta1[-k]  = eta_f(s1[lchild_i], s1[rchild_i], u1[lchild_i], u1[rchild_i], alpha0)

            phi0[-k], phi1[-k]  = phi_f(phi0[lchild_i], phi1[lchild_i], phi0[rchild_i], phi1[rchild_i], RHO, eta0[-k], eta1[-k])

            rho01[-k] = rho0s_f(RHO[1], phi0[-k], phi1[lchild_i], phi1[rchild_i], eta1[-k])
            rho00[-k] = rho0s_f(RHO[0], phi0[-k], phi0[lchild_i], phi0[rchild_i], eta0[-k])

            #print(j, node, eta0[-k], eta1[-k], phi0[-k], phi1[-k], rho01[-k], rho00[-k])
            data.append([j, node, s1[-k], s1[lchild_i], s1[rchild_i], u1[-k], u1[lchild_i], u1[rchild_i], eta0[-k], eta1[-k], phi0[-k], phi1[-k], rho01[-k], rho00[-k]])

        start = start + 2**j

    matrix = np.zeros((2**max_res, max_res + 1))

    for j in range(0, 2**max_res):
        h = 2**max_res+j-1 
        node_child = part_vec[h]

        for r in reversed(range(0, max_res)):
            node = basic.parent(node_child, part_vec)

            node_i  = part_vec.index(node)
            child_i = part_vec.index(node_child)

            csi0[j][r], csi1[j][r]  = csi_f([rho00[node_i], rho01[node_i]], alpha, alpha0, \
                                         s1[child_i], u1[node_i], u1[child_i], csi0[j][r+1], csi1[j][r+1])
            matrix[j][r] = node_i

            node_child = node

    final = csi0[:, 0] 

    tp = np.exp(csi0[:, 0]) 

    last_sets = list(set(np.concatenate(part_vec[-2**max_res:])))
    last_sets = np.sort(last_sets)
    intervals = []
    for i in last_sets:
        intervals.append(X_aug[i])

    return np.longdouble(eta0), np.longdouble(eta1), \
           np.longdouble(phi0), np.longdouble(phi1), \
           np.exp(np.longdouble(rho01)), tp, intervals, max_res, part_vec, s1, s2, b1, b2, u1, u2  


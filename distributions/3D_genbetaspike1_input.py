import numpy as np
from scipy.special import gammaln

prefix = "3D_genbetaspike1"
description = "A single sharp, off-center, off-dyadic 3D generalized Beta spike"
dim = 3

# Olkin & Liu (2003) gamma-ratio construction, same family as 5D_genbetaspike1.
# Target means (0.71, 0.29, 0.79), verified off-dyadic through depth 9, with
# genuinely different per-dimension sharpness (aj = 110, 88, 143).
D  = 3
A0 = 150.0
B0 = 1.0

TARGET_MEANS = np.array([0.71, 0.29, 0.79])
AJ_VEC = np.array([110.0, 88.0, 143.0])
BJ_VEC = AJ_VEC / (TARGET_MEANS / (1 - TARGET_MEANS) * A0 / B0)


def extract_samples(size):
    rng = np.random.default_rng()
    g0 = rng.gamma(A0, 1.0 / B0, size=size)
    samples = np.zeros((size, D))
    for j in range(D):
        gj = rng.gamma(AJ_VEC[j], 1.0 / BJ_VEC[j], size=size)
        samples[:, j] = gj / (gj + g0)
    return samples


def compute_pdf(points):
    points = np.asarray(points)
    lam = BJ_VEC / B0
    a_sum = A0 + AJ_VEC.sum()
    logB = np.sum(gammaln(AJ_VEC)) + gammaln(A0) - gammaln(a_sum)

    log_num = np.zeros(len(points))
    ratio_sum = np.zeros(len(points))
    for j in range(D):
        xj = points[:, j]
        aj = AJ_VEC[j]
        log_num += aj * np.log(lam[j]) + (aj - 1) * np.log(xj) - (aj + 1) * np.log(1 - xj)
        ratio_sum += lam[j] * xj / (1 - xj)
    log_den = a_sum * np.log(1 + ratio_sum)

    return np.exp(log_num - log_den - logB)

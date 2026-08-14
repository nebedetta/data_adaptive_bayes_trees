import numpy as np
from scipy.stats import beta as beta_dist

prefix = "3D_smooth1"
description = "Smooth, near-symmetric, mildly-concentrated independent Beta product (baseline case)"
dim = 3

D = 3
SHAPES = [(4.0, 4.0), (5.0, 5.0), (4.5, 4.5)]


def extract_samples(size):
    rng = np.random.default_rng()
    samples = np.zeros((size, D))
    for j, (a, b) in enumerate(SHAPES):
        samples[:, j] = rng.beta(a, b, size=size)
    return samples


def compute_pdf(points):
    points = np.asarray(points)
    density = np.ones(len(points))
    for j, (a, b) in enumerate(SHAPES):
        density *= beta_dist.pdf(points[:, j], a, b)
    return density

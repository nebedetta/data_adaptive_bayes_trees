import numpy as np
from scipy.stats import uniform
from scipy.stats import beta
from scipy.stats import truncnorm

# `sample_from_mixture` is imported inside extract_samples: it pulls in numba,
# joblib and seaborn, and computing the density needs none of them.

# Distribution name and prefix
prefix = "1D_mix4"
description = "Mixture of 5 with Spike Beta(1200, 800)"


# Component 1
def component1_sample(size, seed=None):
    return np.random.uniform(0, 1, size=size).reshape(-1, 1)

def component1_pdf(X):
    return uniform.pdf(X, 0, 1)


# Component 2
def component2_sample(size, seed=None):
    return np.random.beta(2, 5, size=size).reshape(-1, 1)

def component2_pdf(X):
    return beta.pdf(X, 2, 5)


# Component 3
def component3_sample(size, seed=None):
    return np.random.beta(1200, 800, size=size).reshape(-1, 1)

def component3_pdf(X):
    return beta.pdf(X, 1200, 800)


# Component 4
def component4_sample(size, seed=None):
    return truncnorm.rvs((0.1 - 0.5) / 0.1, (0.9 - 0.5) / 0.1, loc=0.5, scale=0.1,
                          size=size).reshape(-1, 1)

def component4_pdf(X):
    return truncnorm.pdf(X, (0.1 - 0.5) / 0.1, (0.9 - 0.5) / 0.1, loc=0.5, scale=0.1)


# Component 5
def component5_sample(size, seed=None):
    return truncnorm.rvs((0.3 - 0.7) / 0.05, (0.8 - 0.7) / 0.05, loc=0.7, scale=0.05,
                          size=size).reshape(-1, 1)

def component5_pdf(X):
    return truncnorm.pdf(X, (0.3 - 0.7) / 0.05, (0.8 - 0.7) / 0.05, loc=0.7, scale=0.05)


# Component weights
weights = [0.1, 0.2, 0.2, 0.3, 0.2]


# Function to generate X values
def extract_samples(size):
    from src.core import OPT_nD_functions as f

    samples = f.sample_from_mixture(
        [component1_sample, component2_sample, component3_sample,
         component4_sample, component5_sample],
        weights=weights, sample_size=size,
    )
    return samples.ravel()


# Function to calculate y values (PDF)
def compute_pdf(X):
    X = np.asarray(X)
    y = (weights[0] * component1_pdf(X) + weights[1] * component2_pdf(X)
         + weights[2] * component3_pdf(X) + weights[3] * component4_pdf(X)
         + weights[4] * component5_pdf(X))

    return y

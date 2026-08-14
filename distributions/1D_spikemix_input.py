import numpy as np
from scipy.stats import uniform
from scipy.stats import beta

# `sample_from_mixture` is imported inside extract_samples: it pulls in numba,
# joblib and seaborn, and computing the density needs none of them.

# Distribution name and prefix
prefix = "1D_spikemix"
description = "Mixture with a sharp Beta(6000, 4000) spike"


# Component 1: U(0, 1)
def component1_sample(size, seed=None):
    return np.random.uniform(0, 1, size=size).reshape(-1, 1)

def component1_pdf(X):
    return uniform.pdf(X, 0, 1)


# Component 2: U(0.25, 0.5)
def component2_sample(size, seed=None):
    return np.random.uniform(0.25, 0.5, size=size).reshape(-1, 1)

def component2_pdf(X):
    return uniform.pdf(X, 0.25, 0.25)


# Component 3: Beta(2, 2) rescaled to [0.25, 0.5]
def component3_sample(size, seed=None):
    return (0.25 + 0.25 * np.random.beta(2, 2, size=size)).reshape(-1, 1)

def component3_pdf(X):
    return beta.pdf((X - 0.25) / 0.25, 2, 2) / 0.25


# Component 4: Beta(6000, 4000), sharp spike centered at x=0.6
def component4_sample(size, seed=None):
    return np.random.beta(6000, 4000, size=size).reshape(-1, 1)

def component4_pdf(X):
    return beta.pdf(X, 6000, 4000)


# Component weights
weights = [0.1, 0.3, 0.4, 0.2]


# Function to generate X values
def extract_samples(size):
    from src.core import OPT_nD_functions as f

    samples = f.sample_from_mixture(
        [component1_sample, component2_sample, component3_sample, component4_sample],
        weights=weights, sample_size=size,
    )
    return samples.ravel()


# Function to calculate y values (PDF)
def compute_pdf(X):
    X = np.asarray(X)
    y = (weights[0] * component1_pdf(X) + weights[1] * component2_pdf(X)
         + weights[2] * component3_pdf(X) + weights[3] * component4_pdf(X))

    return y

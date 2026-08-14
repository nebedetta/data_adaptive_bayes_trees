import numpy as np
from scipy.stats import uniform

# `sample_from_mixture` is imported inside extract_samples: it pulls in numba,
# joblib and seaborn, and computing the density needs none of them.

# Distribution name and prefix
prefix = "1D_multispikevariedwidth"
description = "Uniform background with 5 narrow spikes of varying widths at dyadic-adjacent locations"

SPIKE_CENTERS    = [0.1, 0.3, 0.5, 0.7, 0.9]
SPIKE_HALF_WIDTH = [0.08, 0.04, 0.0025, 0.01, 0.0005]


# Component 1: U(0, 1) background
def component1_sample(size, seed=None):
    return np.random.uniform(0, 1, size=size).reshape(-1, 1)

def component1_pdf(X):
    return uniform.pdf(X, 0, 1)


# Components 2-6: narrow uniform spikes at each center, each with its own width
def _spike_sample(center, half_width, size):
    return np.random.uniform(center - half_width, center + half_width, size=size).reshape(-1, 1)

def _spike_pdf(center, half_width, X):
    return uniform.pdf(X, center - half_width, 2 * half_width)


def component2_sample(size, seed=None):
    return _spike_sample(SPIKE_CENTERS[0], SPIKE_HALF_WIDTH[0], size)

def component2_pdf(X):
    return _spike_pdf(SPIKE_CENTERS[0], SPIKE_HALF_WIDTH[0], X)


def component3_sample(size, seed=None):
    return _spike_sample(SPIKE_CENTERS[1], SPIKE_HALF_WIDTH[1], size)

def component3_pdf(X):
    return _spike_pdf(SPIKE_CENTERS[1], SPIKE_HALF_WIDTH[1], X)


def component4_sample(size, seed=None):
    return _spike_sample(SPIKE_CENTERS[2], SPIKE_HALF_WIDTH[2], size)

def component4_pdf(X):
    return _spike_pdf(SPIKE_CENTERS[2], SPIKE_HALF_WIDTH[2], X)


def component5_sample(size, seed=None):
    return _spike_sample(SPIKE_CENTERS[3], SPIKE_HALF_WIDTH[3], size)

def component5_pdf(X):
    return _spike_pdf(SPIKE_CENTERS[3], SPIKE_HALF_WIDTH[3], X)


def component6_sample(size, seed=None):
    return _spike_sample(SPIKE_CENTERS[4], SPIKE_HALF_WIDTH[4], size)

def component6_pdf(X):
    return _spike_pdf(SPIKE_CENTERS[4], SPIKE_HALF_WIDTH[4], X)


# Component weights: background + one per spike
# wider spikes get extra weight so they read as taller, not flatter;
# weight pulled off the narrowest spike (component6) caps its height at 75
_w = [16, 39, 24, 16, 16, 9]
weights = [x / sum(_w) for x in _w]


# Function to generate X values
def extract_samples(size):
    from src.core import OPT_nD_functions as f

    samples = f.sample_from_mixture(
        [component1_sample, component2_sample, component3_sample,
         component4_sample, component5_sample, component6_sample],
        weights=weights, sample_size=size,
    )
    return samples.ravel()


# Function to calculate y values (PDF)
def compute_pdf(X):
    X = np.asarray(X)
    y = (weights[0] * component1_pdf(X) + weights[1] * component2_pdf(X)
         + weights[2] * component3_pdf(X) + weights[3] * component4_pdf(X)
         + weights[4] * component5_pdf(X) + weights[5] * component6_pdf(X))

    return y

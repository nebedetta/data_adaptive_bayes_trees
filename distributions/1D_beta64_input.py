import numpy as np
from scipy.stats import beta

# Distribution name and prefix
prefix = "1D_beta64"
description = "Beta(6, 4)"

a = 6
b = 4

# Function to generate X values
def extract_samples(size):
    return np.random.beta(a, b, size=size)

# Function to calculate y values (PDF)
def compute_pdf(X):
    return beta.pdf(np.sort(X), a, b)
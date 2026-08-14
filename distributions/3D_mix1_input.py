import numpy as np
from scipy.special import gammaln

prefix = "3D_mix1"
description = "Two generalized-Beta modes at different scales (broad vs sharp), equal weight"
dim = 3

D = 3

COMPONENTS = {
    "broad": {"a0": 8.0,   "aj_vec": np.array([6.0, 6.0, 6.0]),       "target_means": np.array([0.21, 0.46, 0.62])},
    "sharp": {"a0": 150.0, "aj_vec": np.array([110.0, 110.0, 110.0]), "target_means": np.array([0.79, 0.71, 0.29])},
}
for _c in COMPONENTS.values():
    _c["b0"] = 1.0
    _c["bj_vec"] = _c["aj_vec"] / (_c["target_means"] / (1 - _c["target_means"]) * _c["a0"])

WEIGHTS = np.array([0.5, 0.5])


def _gen_beta_sample(size, comp, rng):
    g0 = rng.gamma(comp["a0"], 1.0 / comp["b0"], size=size)
    samples = np.zeros((size, D))
    for j in range(D):
        gj = rng.gamma(comp["aj_vec"][j], 1.0 / comp["bj_vec"][j], size=size)
        samples[:, j] = gj / (gj + g0)
    return samples


def _gen_beta_density(points, comp):
    a0, b0 = comp["a0"], comp["b0"]
    aj_vec, bj_vec = comp["aj_vec"], comp["bj_vec"]
    lam = bj_vec / b0
    a_sum = a0 + aj_vec.sum()
    logB = np.sum(gammaln(aj_vec)) + gammaln(a0) - gammaln(a_sum)

    log_num = np.zeros(len(points))
    ratio_sum = np.zeros(len(points))
    for j in range(D):
        xj = points[:, j]
        aj = aj_vec[j]
        log_num += aj * np.log(lam[j]) + (aj - 1) * np.log(xj) - (aj + 1) * np.log(1 - xj)
        ratio_sum += lam[j] * xj / (1 - xj)
    log_den = a_sum * np.log(1 + ratio_sum)
    return np.exp(log_num - log_den - logB)


def extract_samples(size):
    rng = np.random.default_rng()
    component_names = list(COMPONENTS.keys())
    counts = rng.multinomial(size, WEIGHTS)
    samples_list = []
    for name, count in zip(component_names, counts):
        if count > 0:
            samples_list.append(_gen_beta_sample(count, COMPONENTS[name], rng))
    samples = np.vstack(samples_list)
    rng.shuffle(samples)
    return samples


def compute_pdf(points):
    points = np.asarray(points)
    density = np.zeros(len(points))
    for w, comp in zip(WEIGHTS, COMPONENTS.values()):
        density += w * _gen_beta_density(points, comp)
    return density

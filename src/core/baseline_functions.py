"""
baseline_functions.py
---------------------
KDE and DPM (variational Bayes DPGMM) density estimators.
Mirrors the interface of OPT_functions.py so that distance metrics
are computed identically to sim_run_nD.py.
 
KDE implementation uses sklearn.neighbors.KernelDensity (Ball Tree),
which evaluates in O(n_eval * log n) instead of scipy's O(n * n_eval),
making it practical at n=50,000.
 
Two KDE variants:
  run_kde     — Scott's rule bandwidth (fast, no tuning)
  run_kde_cv  — cross-validated bandwidth (slower, data-adaptive)
 
DPM: variational Bayes DPGMM via BayesianGaussianMixture.
  Fixed recommended defaults: K=20, max_iter=200, covariance='full'.
  Use dpm_diagnostics() to check whether parameters are appropriate
  for a given dataset before running the full 200-iteration experiment.
 
Reference: Blei & Jordan (2006), Variational Inference for Dirichlet
Process Mixtures, Bayesian Analysis 1(1).
"""
 
import time
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture
 
 
# ── KDE ───────────────────────────────────────────────────────────────────────
 
def _scott_bandwidth(samples):
    """
    Scott's rule: h = n^{-1/(d+4)} * mean_std.
    Matches scipy.stats.gaussian_kde behaviour.

    Computed manually rather than via sklearn's bandwidth="scott" string:
    sklearn's built-in "scott"/"silverman" (>= 1.0) omit the std factor
    (they assume unit-variance data), giving badly oversmoothed bandwidths
    on unstandardized data.
    """
    n, d = samples.shape
    factor = n ** (-1.0 / (d + 4))
    return float(factor * np.std(samples, axis=0, ddof=1).mean())


def _silverman_bandwidth(samples):
    """
    Silverman's rule: h = (n * (d+2) / 4)^{-1/(d+4)} * mean_std.
    Computed manually for the same reason as _scott_bandwidth: sklearn's
    built-in "silverman" string omits the std factor.
    """
    n, d = samples.shape
    factor = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
    return float(factor * np.std(samples, axis=0, ddof=1).mean())
 
 
def fit_kde(samples, bw_method="scott", rtol=0.0):
    """
    Fit a Gaussian KDE using sklearn KernelDensity (Ball Tree algorithm).

    Parameters
    ----------
    samples   : (n, d) array
    bw_method : 'scott' (default) | 'silverman' | float
                'scott' and 'silverman' are always computed manually
                (see _scott_bandwidth / _silverman_bandwidth) rather than
                via sklearn's bandwidth="scott"/"silverman" strings, which
                (sklearn >= 1.0) omit the data's std and assume unit
                variance, giving badly oversmoothed bandwidths on
                unstandardized data.
    rtol      : float, relative tolerance for the Ball Tree approximate
                density evaluation (default 0.0 -- exact).

    Returns
    -------
    kde : fitted KernelDensity object
    """
    if isinstance(bw_method, float):
        bandwidth = bw_method
    elif bw_method == "scott":
        bandwidth = _scott_bandwidth(samples)
    elif bw_method == "silverman":
        bandwidth = _silverman_bandwidth(samples)
    else:
        raise ValueError(
            f"bw_method='{bw_method}' not supported. "
            "Use 'scott', 'silverman', or a float bandwidth."
        )

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth, algorithm="auto", rtol=rtol)
    kde.fit(samples)
    return kde
 
 
def evaluate_kde(kde, grid_points):
    """
    Evaluate a fitted KernelDensity at grid_points.
 
    Parameters
    ----------
    kde         : fitted KernelDensity object
    grid_points : (n_eval, d) array
 
    Returns
    -------
    density : (n_eval,) array of estimated density values
    """
    log_den = kde.score_samples(grid_points)
    return np.exp(log_den)
 
 
def run_kde(samples, grid_points, bw_method="scott", rtol=0.0):
    """
    Fit KDE with Scott's rule bandwidth and evaluate on grid_points.

    Parameters
    ----------
    rtol : float, relative tolerance for the Ball Tree approximate
           density evaluation (default 0.0 -- exact).

    Returns
    -------
    density   : (n_eval,) estimated density
    elapsed   : wall-clock fit+evaluate time in seconds
    bandwidth : float, the bandwidth actually used
    """
    t0      = time.time()
    kde     = fit_kde(samples, bw_method=bw_method, rtol=rtol)
    density = evaluate_kde(kde, grid_points)
    elapsed = time.time() - t0
    bw_used = kde.bandwidth_ if hasattr(kde, "bandwidth_") else kde.bandwidth
    return density, elapsed, float(bw_used)

def run_kde_cv(samples, grid_points, cv=5, n_bandwidths=60,
               bw_factor_range=(0.01, 10), rtol=0.0):
    from sklearn.model_selection import GridSearchCV

    t0 = time.time()

    h_ref = _scott_bandwidth(samples)

    bws = h_ref * np.logspace(
        np.log10(bw_factor_range[0]),
        np.log10(bw_factor_range[1]),
        n_bandwidths
    )

    gs = GridSearchCV(
        KernelDensity(kernel="gaussian", rtol=rtol),
        param_grid={"bandwidth": bws},
        cv=cv,
        n_jobs=1
    )

    gs.fit(samples)

    best_bw = gs.best_params_["bandwidth"]
    density = evaluate_kde(gs.best_estimator_, grid_points)

    info = {
        "h_ref": h_ref,
        "bw_search_min": bws[0],
        "bw_search_max": bws[-1],
        "n_bandwidths": n_bandwidths,
        "cv_folds": cv if isinstance(cv, int) else getattr(cv, "n_splits", None),
        "bw_at_lower_edge": bool(best_bw == bws[0]),
        "bw_at_upper_edge": bool(best_bw == bws[-1]),
    }

    return density, time.time() - t0, best_bw, gs.best_score_, info


# ── DPM (variational Bayes DPGMM via BayesianGaussianMixture) ─────────────────
 
# Recommended fixed defaults for all experiments (see discussion in notebook)
DPM_DEFAULT_K           = 50
DPM_DEFAULT_MAX_ITER    = 5000
DPM_DEFAULT_COV         = "full"
DPM_DEFAULT_TOL         = 1e-4
DPM_DEFAULT_ALPHA       = 1.0
DPM_DEFAULT_N_INIT      = 1
DPM_DEFAULT_INIT_PARAMS = "kmeans"


def fit_dpm(samples, max_components=DPM_DEFAULT_K, max_iter=DPM_DEFAULT_MAX_ITER,
            covariance_type=DPM_DEFAULT_COV, seed=42, tol=DPM_DEFAULT_TOL,
            alpha=DPM_DEFAULT_ALPHA, n_init=DPM_DEFAULT_N_INIT,
            init_params=DPM_DEFAULT_INIT_PARAMS):
    """
    Fit a variational Bayes Dirichlet Process Gaussian Mixture Model.

    Uses sklearn BayesianGaussianMixture with a Dirichlet process prior
    (truncated stick-breaking, K components).

    Parameters
    ----------
    samples         : (n, d) array
    max_components  : truncation level K (default 50)
    max_iter        : max EM iterations (default 5000)
    covariance_type : 'full' (default) | 'tied' | 'diag' | 'spherical'
    seed            : random state
    tol             : ELBO convergence tolerance (default 1e-4)
    alpha           : weight_concentration_prior, fixed (default 1.0)
    n_init          : number of initializations (default 1)
    init_params     : 'kmeans' (default) | 'k-means++' | 'random_from_data' | 'random'

    Returns
    -------
    model : fitted BayesianGaussianMixture object
    """
    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=alpha,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        init_params=init_params,
        random_state=seed,
        warm_start=False,
    )
    model.fit(samples)
    return model
 
 
def evaluate_dpm(model, grid_points):
    """
    Evaluate a fitted BayesianGaussianMixture density at grid_points.
 
    Parameters
    ----------
    model       : fitted BayesianGaussianMixture
    grid_points : (n_eval, d) array
 
    Returns
    -------
    density : (n_eval,) array of non-negative density values
    """
    log_den = model.score_samples(grid_points)
    return np.exp(log_den)
 
 
def run_dpm(samples, grid_points, max_components=DPM_DEFAULT_K,
            max_iter=DPM_DEFAULT_MAX_ITER, covariance_type=DPM_DEFAULT_COV,
            seed=42, tol=DPM_DEFAULT_TOL, alpha=DPM_DEFAULT_ALPHA,
            n_init=DPM_DEFAULT_N_INIT, init_params=DPM_DEFAULT_INIT_PARAMS):
    """
    Fit DPM and evaluate on grid_points.

    Returns
    -------
    density           : (n_eval,) estimated density
    elapsed           : wall-clock fit+evaluate time in seconds
    active_components : number of components with weight > 1e-3
    converged         : whether the EM algorithm converged
    n_iter_used       : actual number of EM iterations run
    """
    t0      = time.time()
    model   = fit_dpm(samples, max_components=max_components, max_iter=max_iter,
                      covariance_type=covariance_type, seed=seed, tol=tol,
                      alpha=alpha, n_init=n_init, init_params=init_params)
    density = evaluate_dpm(model, grid_points)
    elapsed = time.time() - t0

    active_components = int((model.weights_ > 1e-3).sum())
    return density, elapsed, active_components, model.converged_, model.n_iter_
 
 
def dpm_diagnostics(samples, grid_points, true_den, grids,
                    k_values=(5, 10, 20, 30),
                    covariance_types=("full", "diag"),
                    n_seeds=3,
                    max_iter=DPM_DEFAULT_MAX_ITER,
                    tol=DPM_DEFAULT_TOL,
                    distance_module=None):
    """
    Run a small diagnostic sweep to check whether DPM parameters are
    appropriate for a given dataset.
 
    Tests a grid of (K, covariance_type) combinations across n_seeds random
    seeds, reporting for each:
      - active_components  : how many components were actually used
      - converged          : whether EM converged
      - n_iter_used        : EM iterations actually run
      - integrates_to      : integral of estimated density on grid (should ≈ 1)
      - L1, L2, Linf       : distance to true density (requires distance_module)
 
    Parameters
    ----------
    samples         : (n, d) array — one dataset
    grid_points     : (n_eval, d) array
    true_den        : (n_eval,) array — true density on grid
    grids           : list of d flat (n_eval,) arrays — for distance_metric
    k_values        : tuple of K values to test (default (5, 10, 20, 30))
    covariance_types: tuple of covariance types to test
    n_seeds         : number of random seeds per (K, cov) combination
                      (checks sensitivity to initialisation)
    max_iter        : max EM iterations
    tol             : convergence tolerance
    distance_module : your OPT_functions module (for L1/L2/Linf).
                      If None, distance metrics are skipped.
 
    Returns
    -------
    df : pd.DataFrame with one row per (K, cov, seed) combination
    """
    import pandas as pd
 
    rows = []
    for cov in covariance_types:
        for K in k_values:
            for seed in range(n_seeds):
                t0    = time.time()
                model = fit_dpm(samples, max_components=K, max_iter=max_iter,
                                covariance_type=cov, seed=seed, tol=tol)
                den   = evaluate_dpm(model, grid_points)
                elapsed = time.time() - t0
 
                active = int((model.weights_ > 1e-3).sum())
                row = {
                    "K":                K,
                    "covariance_type":  cov,
                    "seed":             seed,
                    "active_K":         active,
                    "K_at_ceiling":     active >= K - 1,   # True = K too small
                    "converged":        model.converged_,
                    "n_iter":           model.n_iter_,
                    "integrates_to":    float(den.mean()),
                    "time_s":           round(elapsed, 2),
                }
 
                if distance_module is not None:
                    L1,   _ = distance_module.distance_metric(
                        *grids, den, den, true_den, distance_module.l1_distance)
                    L2,   _ = distance_module.distance_metric(
                        *grids, den, den, true_den, distance_module.l2_distance)
                    Linf, _ = distance_module.distance_metric(
                        *grids, den, den, true_den, distance_module.linfty_distance)
                    row.update({"log L1": np.log(L1), "log L2": np.log(L2), "log Linf": np.log(Linf)})
 
                rows.append(row)
 
    return pd.DataFrame(rows)
"""
fit_flow.py
-----------
Fit the flow-cytometry application and cache the result.

For one pair of markers and each requested maximum tree depth, this fits both
Optional Polya Tree estimators -- median-split and midpoint -- and summarises
each by Monte Carlo into a posterior mean and pointwise 2.5%/97.5% credible
bands. It also fits the Beta-tree histogram by calling R. Everything is written
to a single `.npz`, which `plot_flow.py` reads.

The two steps are separate because this one is slow: at the published settings
(10,000 Monte Carlo draws on a 400 x 400 grid, depths 8 and 10) it takes a bit
over an hour. Plotting from the cached fit takes seconds, so the figure can be
adjusted without refitting.

    THIS SCRIPT NEEDS DATA THAT IS NOT DISTRIBUTED WITH THIS REPOSITORY.

The flow-cytometry measurements are not ours to publish. See
`data/flow/README.md`. The published figure is committed under `results/figures/`
either way.

Requires R with the BetaTree package:
    devtools::install_github("zq00/BetaTree")

Usage (from anywhere):
    python src/data_analysis/fit_flow.py                       # published settings
    python src/data_analysis/fit_flow.py --n-mc-samples 200    # quick trial run
    python src/data_analysis/fit_flow.py --markers CD3 CD8 --depths 8
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import OPT_2D_functions as f2D  # noqa: E402
from src.core import io_utils  # noqa: E402

# --- Settings behind the published figure ------------------------------------
# The .Rdata file to read from data/flow/, and the name of the data frame inside
# it. The measurements themselves are not distributed;
# see data/flow/README.md.
FLOW_RDATA_NAME = "flow_normalized.Rdata"
FLOW_OBJECT_NAME = "flow.normalized"

N_STATES = 2
N_DIMS = 2
P0 = 0.5              # prior stopping probability; RHO = [[0.5, 0.5], [0, 1]]
LX = 0.5              # prior on which dimension a node splits
ALPHA0 = float(np.exp(20))   # concentration in the stopping state
BASE_SEED = 23

BT_METHOD = "weighted_bonferroni"

# Fits the Beta-tree histogram with its root box fixed to the unit square, so it
# covers the same support as the OPT estimators. Only the root node differs from
# BetaTree::BuildKDTree; the recursion, split rule, leaf criterion and confidence
# bounds are the package's own. BuildHist would otherwise take the support from
# the data's order statistics, which would not be comparable.
R_TEMPLATE = r"""
library(BetaTree)

BuildKDTree_unitbox <- function(X) {{
  n <- nrow(X); d <- ncol(X)
  rootnode <- list(leftchild = NULL, rightchild = NULL, ndat = n, depth = 0,
                   low = rep(0, d), up = rep(1, d),
                   lower = NA, upper = NA, leaf = FALSE, bounded = TRUE)
  nd <- NA
  add_node <- function(x, node, n, d) {{
    node$bounded = prod(sapply(c(node$low, node$up), function(t) !is.na(t)))
    if (is.na(nd[node$depth + 1])) nd[node$depth + 1] <<- 0
    if (node$bounded) nd[node$depth + 1] <<- nd[node$depth + 1] + 1
    if (node$ndat < 4 * log(n)) {{
      node$leaf = TRUE
    }} else {{
      leftnode = node; rightnode = node
      p = node$depth %% d + 1
      depth = node$depth + 1
      leftnode$depth = depth; rightnode$depth = depth
      x = x[order(x[, p]), , drop = F]
      m = node$ndat
      leftnode$ndat  = ceiling(m/2) - 1
      rightnode$ndat = m - ceiling(m/2)
      xleft  = x[1:(ceiling(m/2) - 1), , drop = F]
      xright = x[(ceiling(m/2) + 1):m, , drop = F]
      leftnode$up[p]   <- x[ceiling(m/2), p, drop = F]
      rightnode$low[p] <- x[ceiling(m/2), p, drop = F]
      node$leftchild  = add_node(xleft,  leftnode,  n, d)
      node$rightchild = add_node(xright, rightnode, n, d)
    }}
    return(node)
  }}
  kdtree <- add_node(X, rootnode, n, d)
  list(kdtree = kdtree, nd = nd)
}}

BuildHist_unitbox <- function(X, alpha, method) {{
  n <- nrow(X); d <- ncol(X)
  kdtree <- BuildKDTree_unitbox(X)
  ahat   <- BetaTree:::ConfLevel(kdtree$nd, alpha, method)
  tree   <- BetaTree:::SetBounds(kdtree$kdtree, ahat, n)
  B <- matrix(nrow = 0, ncol = (2 * d + 5))
  BetaTree:::SelectNodes(tree, B, ahat, n)
}}

X <- as.matrix(read.csv("{in_csv}"))
stopifnot(all(X >= 0 & X <= 1))
h <- BuildHist_unitbox(X, alpha = {alpha}, method = "{method}")
write.csv(as.data.frame(h), "{out_csv}", row.names = FALSE)
"""


def default_cache_path(markers):
    """Where the fit for this marker pair is cached.

    Beside the data it was fitted from, which is not distributed either.
    """
    return os.path.join(io_utils.DATA_DIR, "flow",
                        f"flow_{markers[0]}_{markers[1]}_fit.npz")


def load_markers(markers, data_file=FLOW_RDATA_NAME, data_object=FLOW_OBJECT_NAME):
    """The two marker columns, and the number of pairs the data allows.

    `data_file` is read from `data/flow/` unless given as an absolute path;
    `data_object` names the data frame inside it.
    """
    path = data_file if os.path.isabs(data_file) else os.path.join(
        io_utils.DATA_DIR, "flow", data_file)
    if not os.path.exists(path):
        raise SystemExit(
            f"Flow-cytometry data not found:\n  {path}\n\n"
            "This data is not distributed with the repository; see "
            "data/flow/README.md.\nPoint --data-file at it if it is stored "
            "elsewhere or under another name.\nThe published figure is "
            "committed at results/figures/2D_CD45RO_CD27_betatree.jpg."
        )
    try:
        import pyreadr
    except ImportError:
        raise SystemExit("pyreadr is required to read the .Rdata flow file.")

    contents = pyreadr.read_r(path)
    if data_object not in contents:
        raise SystemExit(f"No object named {data_object!r} in {path}\n"
                         f"Found: {list(contents)}\n"
                         "Name it with --data-object.")
    frame = contents[data_object]
    missing = [m for m in markers if m not in frame.columns]
    if missing:
        raise SystemExit(f"Markers not in the data: {missing}\n"
                         f"Available: {list(frame.columns)}")

    X = frame[list(markers)].to_numpy()
    # The estimator partitions the unit square.
    if not np.all((X >= 0) & (X <= 1)):
        raise SystemExit("Marker values must lie in [0, 1].")

    pairs = list(combinations(frame.columns, N_DIMS))
    index = pairs.index(tuple(markers)) if tuple(markers) in pairs else None
    return X, index, len(pairs)


def fit_opt(X, depth, estimator, grid, n_mc_samples, n_jobs):
    """One estimator at one depth, summarised by Monte Carlo."""
    x0, y0 = grid

    started = time.time()
    if estimator == "median":
        r = f2D.bottom_up_recursion_2(X, depth, P0, LX, ALPHA0, 99, n_jobs)
    elif estimator == "midpoint":
        r = f2D.binary_bottom_up_recursion(X, depth, P0, LX, ALPHA0)
    else:
        raise ValueError(estimator)
    recursion_time = time.time() - started

    started = time.time()
    # sampling_MCp_bands draws a probability allocation conditional on each
    # sampled tree -- the sampler the published run used. The MCt family
    # averages the posterior mean conditional on each tree instead: same
    # posterior mean, different bands.
    summary = f2D.sampling_MCp_bands(
        n_mc_samples, x0, y0, depth, r["part_vec"],
        r["rho00_x"], r["rho00_y"], r["rho01_x"], r["rho01_y"],
        LX, r["set_size"], r["ratio1"], r["alpha_vec"], ALPHA0,
        BASE_SEED, n_jobs,
    )
    summary["recursion_time"] = recursion_time
    summary["sampling_time"] = time.time() - started
    return summary


def find_rscript():
    """The Rscript binary, resolved rather than taken from PATH.

    A process launched from a desktop environment -- Jupyter started from the
    Anaconda launcher, an IDE -- inherits a minimal PATH that often omits
    /usr/local/bin, so `Rscript` is not found even though R is installed and
    works from a terminal. Falls back to the macOS framework location, which is
    where the symlink on PATH points anyway.
    """
    found = shutil.which("Rscript")
    if found:
        return found

    fallback = "/Library/Frameworks/R.framework/Resources/bin/Rscript"
    if os.path.exists(fallback):
        return fallback

    raise SystemExit(
        "Rscript not found. R is needed for the Beta-tree baseline:\n"
        "  devtools::install_github(\"zq00/BetaTree\")\n\n"
        "It was not on PATH, and not at\n"
        f"  {fallback}\n"
        "If R is installed elsewhere, add its bin directory to PATH."
    )


def fit_betatree(X, markers, alpha):
    """The Beta-tree histogram, one row per region."""
    with tempfile.TemporaryDirectory() as tmp:
        in_csv = os.path.join(tmp, "X.csv")
        out_csv = os.path.join(tmp, "hist.csv")
        pd.DataFrame(X, columns=list(markers)).to_csv(in_csv, index=False)

        r_code = R_TEMPLATE.format(in_csv=in_csv, out_csv=out_csv,
                                   alpha=alpha, method=BT_METHOD)
        result = subprocess.run([find_rscript(), "-e", r_code],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(
                "The Beta-tree fit failed. R with the BetaTree package is "
                "required:\n  devtools::install_github(\"zq00/BetaTree\")\n\n"
                + result.stderr
            )
        raw = pd.read_csv(out_csv).to_numpy()

    # BuildHist columns: per-dimension lower bounds, then upper bounds, then the
    # density, its two confidence bounds, the region's count, and its depth.
    columns = ([f"lo_{m}" for m in markers] + [f"hi_{m}" for m in markers]
               + ["density", "ci_lower", "ci_upper", "n_obs", "depth"])
    return pd.DataFrame(raw, columns=columns)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markers", nargs=2, default=["CD45RO", "CD27"],
                        metavar=("MARKER1", "MARKER2"))
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 10],
                        help="Maximum tree depths to fit (default: 8 10).")
    parser.add_argument("--n-mc-samples", type=int, default=10000,
                        help="Monte Carlo draws (default: 10000, the published "
                             "run's value). A few hundred is enough for a trial: "
                             "the posterior mean barely moves, but the credible "
                             "bands need the full count to be stable.")
    parser.add_argument("--grid-size", type=int, default=400,
                        help="Points per axis (default: 400, as published).")
    parser.add_argument("--bt-alpha", type=float, default=0.05,
                        help="Beta-tree simultaneous coverage is 1 - this "
                             "(default: 0.05).")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--data-file", default=FLOW_RDATA_NAME,
                        help="The .Rdata file to read, relative to data/flow/ "
                             "or absolute (default: %(default)s).")
    parser.add_argument("--data-object", default=FLOW_OBJECT_NAME,
                        help="Name of the data frame inside it "
                             "(default: %(default)s).")
    parser.add_argument("--out", default=None,
                        help="Destination .npz (default: data/2D/fits/).")
    args = parser.parse_args()

    markers = list(args.markers)
    X, pair_index, n_pairs = load_markers(markers, args.data_file, args.data_object)
    print(f"markers : {markers}"
          + (f"  (pair {pair_index} of {n_pairs})" if pair_index is not None else ""))
    print(f"sample  : {X.shape[0]} points")
    print(f"settings: {args.n_mc_samples} draws, {args.grid_size} x "
          f"{args.grid_size} grid, depths {args.depths}\n")

    grid_axis = np.linspace(0.00001, 0.99999, args.grid_size)
    grid = np.meshgrid(grid_axis, grid_axis)

    payload = {}
    for depth in args.depths:
        for estimator in ("median", "midpoint"):
            print(f"fitting depth {depth}, OPT {estimator} ...", flush=True)
            summary = fit_opt(X, depth, estimator, grid,
                              args.n_mc_samples, args.n_jobs)
            for key in ("mean", "lower_band", "upper_band"):
                payload[f"d{depth}_{estimator}_{key}"] = summary[key].astype(np.float32)
            # A density on the unit square integrates to 1, so a grid-mean far
            # from 1 means the fit went wrong rather than the plot.
            print(f"   recursion {summary['recursion_time']:.1f}s, "
                  f"sampling {summary['sampling_time']:.1f}s, "
                  f"grid-mean {summary['mean'].mean():.4f}")

    print("\nfitting the Beta-tree histogram "
          f"({int((1 - args.bt_alpha) * 100)}% simultaneous coverage) ...", flush=True)
    hist = fit_betatree(X, markers, args.bt_alpha)
    print(f"   {len(hist)} regions, depth {int(hist['depth'].min())}"
          f"-{int(hist['depth'].max())}")

    payload["betatree_columns"] = np.array(list(hist.columns))
    payload["betatree_values"] = hist.to_numpy().astype(np.float64)

    # The plot needs the samples for its colour scale, and the settings for its
    # captions -- so the cache is self-contained.
    payload["samples"] = X.astype(np.float32)
    payload["markers"] = np.array(markers)
    payload["depths"] = np.array(args.depths)
    payload["n_mc_samples"] = np.array(args.n_mc_samples)
    payload["grid_size"] = np.array(args.grid_size)
    payload["bt_alpha"] = np.array(args.bt_alpha)

    out_path = args.out or default_cache_path(markers)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"\nwrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print("Draw the figure with:  python src/data_analysis/plot_flow.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

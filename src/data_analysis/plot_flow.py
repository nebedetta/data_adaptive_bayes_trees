"""
plot_flow.py
------------
Draw the flow-cytometry figure from a cached fit.

Three blocks, stacked: the Optional Polya Tree estimators at each maximum tree
depth, then the Beta-tree histogram. Each OPT block shows the posterior mean and
the pointwise 2.5%/97.5% quantiles, for the median-split and midpoint variants.
The Beta-tree block shows its density estimate with the matching lower and upper
simultaneous confidence bounds.

Reads the `.npz` written by `fit_flow.py`, so it runs in seconds and the figure
can be adjusted without repeating the hour-long fit. If the cache is absent it
says so and exits; the published figure is committed under `results/figures/`
either way.

Usage (from anywhere):
    python src/data_analysis/plot_flow.py
    python src/data_analysis/plot_flow.py --markers CD3 CD8
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import io_utils  # noqa: E402

from fit_flow import default_cache_path  # noqa: E402

LABEL_FONTSIZE = 28
HIST_BINS = 50


def load_cache(path):
    """The cached fit, as a dict of arrays."""
    if not os.path.exists(path):
        raise SystemExit(
            f"No cached fit at:\n  {path}\n\n"
            "Produce it with:  python src/data_analysis/fit_flow.py\n"
            "That needs the flow-cytometry data, which is not distributed with "
            "this repository (see data/flow/README.md).\n"
            "The published figure is committed at "
            "results/figures/2D_CD45RO_CD27_betatree.jpg."
        )
    return np.load(path, allow_pickle=False)


def betatree_patches(hist, column, markers, cmap, norm):
    """The tree's regions as coloured rectangles."""
    rects = [
        Rectangle((row[f"lo_{markers[0]}"], row[f"lo_{markers[1]}"]),
                  row[f"hi_{markers[0]}"] - row[f"lo_{markers[0]}"],
                  row[f"hi_{markers[1]}"] - row[f"lo_{markers[1]}"])
        for _, row in hist.iterrows()
    ]
    patches = PatchCollection(rects, cmap=cmap, norm=norm,
                              edgecolor="none", linewidth=0)
    patches.set_array(hist[column].to_numpy())
    return patches


def plot(cache):
    """Two OPT blocks and a Beta-tree block, sharing one colour scale."""
    markers = [str(m) for m in cache["markers"]]
    depths = [int(d) for d in cache["depths"]]
    samples = cache["samples"]
    grid_size = int(cache["grid_size"])

    hist = pd.DataFrame(cache["betatree_values"],
                        columns=[str(c) for c in cache["betatree_columns"]])

    n_rows = 3 * len(depths) + 1   # 2 OPT rows + 1 spacer per depth, + Beta tree
    n_cols = 3
    height_ratios = [1, 1, 0.35] * len(depths) + [1]

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows),
        gridspec_kw={"width_ratios": [0.8, 0.8, 0.8],
                     "height_ratios": height_ratios}, squeeze=False)
    fig.subplots_adjust(wspace=0.07, hspace=0.05)

    # One colour scale for every panel, centred on the 99th percentile of the
    # data histogram: the density is spike-dominated, and on a linear scale the
    # spike flattens everything else.
    heatmap, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=HIST_BINS,
                                   range=[[0, 1], [0, 1]], density=True)
    norm = TwoSlopeNorm(vmin=heatmap.min(),
                        vcenter=np.quantile(heatmap, 0.99),
                        vmax=heatmap.max())
    cmap = plt.get_cmap("RdYlBu_r")

    for block in range(len(depths)):
        for col in range(n_cols):
            axes[block * 3 + 2][col].set_visible(False)

    grid_axis = np.linspace(0.00001, 0.99999, grid_size)
    x0, y0 = np.meshgrid(grid_axis, grid_axis)

    titles = ["Posterior Mean", "Posterior 2.5% Quantile", "Posterior 97.5% Quantile"]
    summaries = ["mean", "lower_band", "upper_band"]

    image = None
    for block, depth in enumerate(depths):
        for offset, estimator in enumerate(("median", "midpoint")):
            row = block * 3 + offset
            for col, summary in enumerate(summaries):
                ax = axes[row][col]
                values = cache[f"d{depth}_{estimator}_{summary}"]
                image = ax.scatter(x0, y0, c=values, cmap=cmap, s=1, norm=norm)
                ax.set_aspect("equal", adjustable="box")
                ax.margins(0, 0)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.tick_params(labelsize=LABEL_FONTSIZE - 14)
                if block == 0 and offset == 0:
                    ax.set_title(titles[col], fontsize=LABEL_FONTSIZE - 2)

    beta_row = n_rows - 1
    for col, (column, title) in enumerate([("density", "Estimate"),
                                           ("ci_lower", "Lower Bound"),
                                           ("ci_upper", "Upper Bound")]):
        ax = axes[beta_row][col]
        ax.add_collection(betatree_patches(hist, column, markers, cmap, norm))
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0, 0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=LABEL_FONTSIZE - 14)
        ax.set_title(title, fontsize=LABEL_FONTSIZE - 2)

    for row in range(n_rows):
        axes[row][1].set_yticks([])
        axes[row][2].set_yticks([])
    for block in range(len(depths)):
        for col in range(n_cols):
            axes[block * 3][col].set_xticks([])

    for block in range(len(depths)):
        axes[block * 3][0].set_ylabel("OPT median", fontsize=LABEL_FONTSIZE)
        axes[block * 3 + 1][0].set_ylabel("OPT midpoint", fontsize=LABEL_FONTSIZE)
    axes[beta_row][0].set_ylabel("Beta-tree", fontsize=LABEL_FONTSIZE)

    # aspect="equal" shrinks the axes, so their positions are final only after a
    # draw; the section and axis labels below are placed from those positions.
    fig.canvas.draw()

    sections = [(block * 3, block * 3 + 1,
                 f"({chr(ord('a') + block)}) OPT - Maximum Tree Depth {depth}")
                for block, depth in enumerate(depths)]
    sections.append((beta_row, beta_row,
                     f"({chr(ord('a') + len(depths))}) Beta-tree"))

    for row_top, row_bottom, label in sections:
        pos_top = axes[row_top][2].get_position()
        pos_bottom = axes[row_bottom][2].get_position()
        pos_left = axes[row_bottom][0].get_position()

        fig.text(0.01, pos_top.y1 + 0.028, label, fontsize=LABEL_FONTSIZE + 2)
        fig.text((pos_left.x0 + pos_top.x1) / 2, pos_bottom.y0 - 0.022, markers[0],
                 fontsize=LABEL_FONTSIZE, fontweight="bold", ha="center")
        fig.text(pos_top.x1 + 0.055, (pos_bottom.y0 + pos_top.y1) / 2, markers[1],
                 fontsize=LABEL_FONTSIZE, fontweight="bold", rotation=90,
                 va="center")

    reference = axes[0][2].get_position()
    cax = fig.add_axes([reference.x1 + 0.015, reference.y0, 0.01, reference.height])
    colorbar = plt.colorbar(image, cax=cax)
    colorbar.ax.tick_params(labelsize=LABEL_FONTSIZE - 6)

    return fig, markers


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markers", nargs=2, default=["CD45RO", "CD27"],
                        metavar=("MARKER1", "MARKER2"))
    parser.add_argument("--cache", default=None,
                        help="Cached fit (default: the one for this marker pair).")
    args = parser.parse_args()

    cache_path = args.cache or default_cache_path(list(args.markers))
    cache = load_cache(cache_path)
    print(f"read {cache_path}  "
          f"({int(cache['n_mc_samples'])} draws, "
          f"depths {[int(d) for d in cache['depths']]})")

    fig, markers = plot(cache)

    output_dir = io_utils.ensure_figure_output_dir()
    output_path = os.path.join(output_dir,
                               f"2D_{markers[0]}_{markers[1]}_betatree.jpg")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

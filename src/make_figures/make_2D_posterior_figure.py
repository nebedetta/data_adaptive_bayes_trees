"""
make_2D_posterior_figure.py
---------------------------
Produce `results/figures/2D_itern15.jpg`.

One row per 2D scenario: the true density as a heatmap, then the posterior mean
density under the median-split (Partial) and midpoint (Full) estimators, for a
single representative run (iteration 15, n=5000, depth 9). All three panels in a
row share one colour scale, so the estimates can be compared against the truth
and each other directly.

This is the only figure that reads simulation output rather than an aggregated
metric table -- it plots the posterior mean surfaces themselves. The six
`.pkl.gz` files it needs are shipped under `data/2D/fits/<scenario>/`
(1.3 MB); the rest of the simulation output is not distributed.

The two mixture scenarios are spike-dominated, so their rows use a
`TwoSlopeNorm` centred on the 99th percentile rather than a linear scale.

Run from anywhere:
    python src/make_figures/make_2D_posterior_figure.py
"""

import gzip
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import io_utils  # noqa: E402

FIGURE_NAME = "2D_itern15.jpg"

SCENARIOS = ["2D_genbeta3", "2D_genbeta4", "2D_genbeta6",
             "2D_genbeta1", "2D_mix13", "2D_mix14"]

DISPLAY_NAMES = {
    "2D_genbeta3": "Gen Beta I",
    "2D_genbeta4": "Gen Beta II",
    "2D_genbeta6": "Gen Beta III",
    "2D_genbeta1": "Gen Beta IV",
    "2D_mix13": "Mixture I",
    "2D_mix14": "Mixture II",
}

SPIKE_SCENARIOS = {"2D_mix13", "2D_mix14"}

ITERATION = 15
DEPTH = 9
SAMPLE_SIZE = 5000

GRID_SIZE = 500
LABEL_FONTSIZE = 25

def load_posterior_means(prefix):
    """Read one scenario's Partial and Full posterior-mean surfaces."""
    filename = f"{prefix}_n{SAMPLE_SIZE}_depth{DEPTH}_itern{ITERATION}.pkl.gz"
    path = io_utils.fits_path("2D", prefix, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing figure input: {path}")
    with gzip.open(path, "rb") as handle:
        data = pickle.load(handle)
    return data["pm_mcmc_partial"], data["pm_mcmc_full"]


def colour_scale(z, prefix):
    """Shared colour scale for one scenario's row."""
    if prefix in SPIKE_SCENARIOS:
        return TwoSlopeNorm(vmin=z.min(), vcenter=np.quantile(z, 0.99), vmax=z.max())
    return Normalize(vmin=z.min(), vmax=z.max())


def plot():
    """Six scenario rows; true density, Partial estimate, Full estimate."""
    n_rows, n_cols = len(SCENARIOS), 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5.5 * n_rows), squeeze=False)
    fig.subplots_adjust(wspace=0.07, hspace=0.05)

    grid_axis = np.linspace(0.00001, 0.99999, GRID_SIZE)
    x0, y0 = np.meshgrid(grid_axis, grid_axis)
    cmap = plt.get_cmap("RdYlBu_r")

    for row, prefix in enumerate(SCENARIOS):
        density = io_utils.read_true_density(prefix)
        z = density["true_density"].to_numpy().reshape(GRID_SIZE, GRID_SIZE)
        norm = colour_scale(z, prefix)

        ax_density = axes[row][0]
        image = ax_density.imshow(z, extent=[0, 1, 0, 1], origin="lower",
                                  cmap=cmap, aspect="auto", norm=norm)
        ax_density.set_xlim(0, 1)
        ax_density.set_ylim(0, 1)

        divider = make_axes_locatable(ax_density)
        cax = divider.append_axes("left", size="5%", pad=0.05)
        cbar = plt.colorbar(image, cax=cax)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")

        if row == 0:
            ax_density.set_title("True Density\nHeatmap", fontsize=LABEL_FONTSIZE)
        ax_density.set_ylabel(DISPLAY_NAMES[prefix], fontsize=LABEL_FONTSIZE,
                              fontweight="bold")
        ax_density.yaxis.set_label_coords(-0.3, 0.5)

        partial_mean, full_mean = load_posterior_means(prefix)

        for col, (surface, name) in enumerate(
                [(partial_mean, "OPT Median"), (full_mean, "OPT Midpoint")], start=1):
            ax = axes[row][col]
            ax.scatter(x0, y0, c=surface, cmap=cmap, s=1, norm=norm)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if row == 0:
                ax.set_title(f"Posterior Mean\n{name}",
                             fontsize=LABEL_FONTSIZE)

    # Only the bottom row keeps x tick labels; y tick labels move to the right
    # of the last column, since all three panels in a row share a scale.
    for row in range(n_rows - 1):
        for col in range(n_cols):
            axes[row][col].tick_params(axis="x", labelbottom=False)
    for row in range(n_rows):
        for col in range(n_cols - 1):
            axes[row][col].tick_params(axis="y", labelleft=False)
        axes[row][n_cols - 1].yaxis.set_label_position("right")
        axes[row][n_cols - 1].yaxis.tick_right()

    return fig


def main():
    fig = plot()
    output_dir = io_utils.ensure_figure_output_dir()
    output_path = os.path.join(output_dir, FIGURE_NAME)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

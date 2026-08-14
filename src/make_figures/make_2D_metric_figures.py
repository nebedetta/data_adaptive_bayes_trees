"""
make_2D_metric_figures.py
-------------------------
Produce `results/figures/2D_L1.jpg` and `results/figures/2D_L2.jpg`.

One figure per metric. Rows are scenarios; the leftmost column shows the true
density as a heatmap and the remaining columns one sample size each. Within a
panel, points are the mean of log(distance) across iterations with +/- 1
standard deviation error bars, for the median-split (Partial) and midpoint
(Full) estimators.

The two mixture scenarios are dominated by a narrow spike, so a linear colour
scale renders everything else flat. Those rows use a `TwoSlopeNorm` centred on
the 99th percentile, which keeps the bulk of the density legible while still
showing the spike.

Reads only `results/aggregated_metric_tables/2D/` and `data/true_density/`. Run from anywhere:
    python src/make_figures/make_2D_metric_figures.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import io_utils  # noqa: E402

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

# Spike-dominated scenarios needing a percentile-centred colour scale.
SPIKE_SCENARIOS = {"2D_mix13", "2D_mix14"}

SAMPLE_SIZES = [500, 5000, 50000]

METRIC_LABELS = {"L1": r"$\mathbf{L_1}$", "L2": r"$\mathbf{L_2}$"}

# The density CSVs store a 500x500 grid flattened row-major.
GRID_SIZE = 500

MARKERSIZE = 5
ALPHA = 0.8
SHIFT = 0.1
TICK_FONTSIZE = 20
LABEL_FONTSIZE = 25


def plot_density_heatmap(ax, prefix, show_title):
    """Leftmost reference column: the true density as a heatmap."""
    density = io_utils.read_true_density(prefix)
    z = density["true_density"].to_numpy().reshape(GRID_SIZE, GRID_SIZE)

    cmap = plt.get_cmap("RdYlBu_r")
    if prefix in SPIKE_SCENARIOS:
        norm = TwoSlopeNorm(vmin=z.min(), vcenter=np.quantile(z, 0.99), vmax=z.max())
    else:
        norm = Normalize(vmin=z.min(), vmax=z.max())

    image = ax.imshow(z, extent=[0, 1, 0, 1], origin="lower", cmap=cmap,
                      aspect="auto", norm=norm)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.yaxis.tick_left()
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, labelleft=True)

    if show_title:
        ax.set_title("Sampling Density", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(f"{DISPLAY_NAMES[prefix]}\n", fontsize=LABEL_FONTSIZE,
                  fontweight="bold")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE - 5)


def plot_metric(metric):
    """One figure for `metric`: scenarios down the rows, sample sizes across."""
    n_scenarios = len(SCENARIOS)
    n_cols = len(SAMPLE_SIZES) + 2  # density column + spacer + one per sample size

    # Zero-height spacer rows separate scenarios without stretching the panels.
    n_rows = 2 * n_scenarios - 1
    height_ratios = [1, 0] * (n_scenarios - 1) + [1]
    width_ratios = [0.85, 0.3] + [1] * len(SAMPLE_SIZES)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows - 2),
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, prefix in enumerate(SCENARIOS):
        grid_row = 2 * i

        mean_df = io_utils.read_metric_table(
            "2D", prefix, f"{prefix}_metric_table_mean.csv")
        std_df = io_utils.read_metric_table(
            "2D", prefix, f"{prefix}_metric_table_std.csv")

        plot_density_heatmap(axes[grid_row][0], prefix, show_title=(i == 0))

        for col_i, sample_size in enumerate(SAMPLE_SIZES):
            col = col_i + 2
            ax = axes[grid_row][col]

            partial_mean = mean_df[f"{sample_size}_{metric}P"]
            full_mean = mean_df[f"{sample_size}_{metric}F"]
            partial_sd = std_df[f"{sample_size}_{metric}P"]
            full_sd = std_df[f"{sample_size}_{metric}F"]

            depths = range(1, len(partial_mean) + 1)

            ax.plot([t - SHIFT for t in depths], partial_mean, "o",
                    color="red", markersize=MARKERSIZE)
            ax.errorbar([t - SHIFT for t in depths], partial_mean, yerr=partial_sd,
                        fmt="none", ecolor="red", alpha=ALPHA, capsize=3.5)

            ax.plot([t + SHIFT for t in depths], full_mean, "s",
                    color="blue", markersize=MARKERSIZE)
            ax.errorbar([t + SHIFT for t in depths], full_mean, yerr=full_sd,
                        fmt="none", ecolor="blue", alpha=ALPHA, capsize=3.5)

            x_min, x_max = ax.get_xlim()
            ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 1, 2))
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, labelleft=True)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.sharey(axes[grid_row][n_cols - 1])

            if col == 2:
                ax.set_ylabel(f"log {METRIC_LABELS[metric]} Risk",
                              fontsize=LABEL_FONTSIZE, fontweight="bold")
            if grid_row == 0:
                ax.set_title(f"Sample Size {sample_size}", fontsize=LABEL_FONTSIZE)

    for row in range(0, n_rows, 2):
        axes[row][1].axis("off")
    for row in range(1, n_rows, 2):
        for col in range(n_cols):
            axes[row][col].axis("off")

    # Only the bottom row keeps x tick labels; only the first metric column
    # keeps y tick labels, since columns share a y-scale.
    for row in range(n_rows - 1):
        for col in range(n_cols):
            axes[row][col].tick_params(axis="x", labelbottom=False)
    for col in range(3, n_cols):
        for row in range(n_rows):
            axes[row][col].tick_params(axis="y", labelleft=False)

    axes[n_rows - 1][3].set_xlabel("\n Maximum Tree Depth", va="center",
                                   ha="center", labelpad=30, fontsize=22)

    handles = [
        plt.Line2D([], [], color="red", marker="o", linestyle="none",
                   label="OPT median"),
        plt.Line2D([], [], color="blue", marker="s", linestyle="none",
                   label="OPT midpoint"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.78, 0.07),
               ncol=2, fontsize=LABEL_FONTSIZE)

    return fig


def main():
    output_dir = io_utils.ensure_figure_output_dir()
    for metric in ("L1", "L2"):
        fig = plot_metric(metric)
        output_path = os.path.join(output_dir, f"2D_{metric}.jpg")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

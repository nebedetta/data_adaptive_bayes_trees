"""
make_3D_figure.py
-----------------
Produce `results/figures/3D_L1_L2.jpg`.

One block per 3D scenario; within a block, a log-L1 row above a log-L2 row, and
one column per sample size. Points are the mean of log(distance) across
iterations, error bars +/- 1 standard deviation, with the median-split (Partial)
and midpoint (Full) estimators offset slightly left and right of each depth so
their error bars stay legible.

Unlike the 1D and 2D figures there is no true-density column: a 3D density has
no direct 2D-image analogue. There is no benchmark overlay either, as the
KDE/DPM/DRBART baselines were not run in 3D.

Reads only `results/aggregated_metric_tables/3D/`. Run from anywhere:
    python src/make_figures/make_3D_figure.py
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import io_utils  # noqa: E402

FIGURE_NAME = "3D_L1_L2.jpg"

SCENARIOS = ["3D_smooth1", "3D_mix1", "3D_genbetaspike1"]

DISPLAY_NAMES = {
    "3D_smooth1": "Smooth",
    "3D_mix1": "Mixture",
    "3D_genbetaspike1": "Gen Beta Spike",
}

SAMPLE_SIZES = [500, 5000, 50000]

ROW_METRICS = [("L1", r"$\mathbf{L_1}$"), ("L2", r"$\mathbf{L_2}$")]

MARKERSIZE = 5
ALPHA = 0.8
SHIFT = 0.1
TICK_FONTSIZE = 20
LABEL_FONTSIZE = 25

# Fixed axes-fraction x-position, so "log L1 Risk" and "log L2 Risk" sit at the
# same distance from their axes despite differing tick-label widths.
YLABEL_X = -0.16


def load_tables(scenarios):
    """Read each scenario's mean/std tables from `results/aggregated_metric_tables/3D/`."""
    tables = {}
    for prefix in scenarios:
        tables[prefix] = {
            stat: io_utils.read_metric_table(
                "3D", prefix, f"{prefix}_metric_table_{stat}.csv")
            for stat in ("mean", "std")
        }
    return tables


def plot(tables, scenarios, sample_sizes):
    """Alternating L1/L2 rows per scenario, one column per sample size.

    Each scenario gets a short blank row carrying its name, so every data row
    keeps the same height regardless of where the titles fall.
    """
    rows_per_scenario = 1 + len(ROW_METRICS)  # title row + one row per metric
    n_rows = len(scenarios) * rows_per_scenario
    n_cols = len(sample_sizes)
    height_ratios = [0.6, 1, 1] * len(scenarios)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8 * n_cols, 5 * (n_rows - 0.4 * len(scenarios))),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.1, hspace=0.15)

    for scenario_idx, prefix in enumerate(scenarios):
        mean_df = tables[prefix]["mean"]
        std_df = tables[prefix]["std"]
        title_row = scenario_idx * rows_per_scenario

        for col in range(n_cols):
            axes[title_row][col].set_axis_off()
        centre = axes[title_row][n_cols // 2]
        centre.text(0.5, 0.15, DISPLAY_NAMES[prefix],
                    fontsize=LABEL_FONTSIZE + 2, fontweight="bold",
                    ha="center", va="center", transform=centre.transAxes)

        for metric_offset, (metric, metric_label) in enumerate(ROW_METRICS):
            row = title_row + 1 + metric_offset
            is_last_data_row = metric_offset == len(ROW_METRICS) - 1

            for col, sample_size in enumerate(sample_sizes):
                ax = axes[row][col]

                col_p = f"{sample_size}_{metric}P"
                col_f = f"{sample_size}_{metric}F"
                if col_p not in mean_df.columns or col_f not in mean_df.columns:
                    ax.set_axis_off()
                    continue

                partial_mean, full_mean = mean_df[col_p], mean_df[col_f]
                partial_sd, full_sd = std_df[col_p], std_df[col_f]

                # Row i of the table is depth i+1; depths a sample size never
                # reached are NaN-padded and simply skipped.
                depth = pd.Series(range(1, len(partial_mean) + 1))
                mask_p, mask_f = partial_mean.notna(), full_mean.notna()

                ax.plot(depth[mask_p] - SHIFT, partial_mean[mask_p], "o",
                        color="red", markersize=MARKERSIZE)
                ax.errorbar(depth[mask_p] - SHIFT, partial_mean[mask_p],
                            yerr=partial_sd[mask_p], fmt="none", ecolor="red",
                            alpha=ALPHA, capsize=3.5)

                ax.plot(depth[mask_f] + SHIFT, full_mean[mask_f], "s",
                        color="blue", markersize=MARKERSIZE)
                ax.errorbar(depth[mask_f] + SHIFT, full_mean[mask_f],
                            yerr=full_sd[mask_f], fmt="none", ecolor="blue",
                            alpha=ALPHA, capsize=3.5)

                ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE,
                               labelbottom=is_last_data_row, labelleft=(col == 0))
                ax.sharey(axes[row][0])

                if metric_offset == 0:
                    ax.set_title(f"n = {sample_size}", fontsize=LABEL_FONTSIZE)
                if col == 0:
                    ax.set_ylabel(f"log {metric_label} Risk",
                                  fontsize=LABEL_FONTSIZE, fontweight="bold")
                    ax.yaxis.set_label_coords(YLABEL_X, 0.5)
                if is_last_data_row and col == n_cols // 2:
                    ax.set_xlabel("Maximum Tree Depth", fontsize=LABEL_FONTSIZE - 3)

    partial_handle = plt.Line2D([], [], color="red", marker="o",
                                linestyle="none", label="OPT median")
    full_handle = plt.Line2D([], [], color="blue", marker="s",
                             linestyle="none", label="OPT midpoint")
    # Anchor the legend below the bottom row's real position: on a figure this
    # tall a fixed offset from y=0 would drift far from the axes.
    bottom_axes_y = axes[n_rows - 1][n_cols // 2].get_position().y0
    fig.legend(handles=[partial_handle, full_handle], loc="upper center",
               bbox_to_anchor=(0.5, bottom_axes_y - 0.06), ncol=2,
               fontsize=LABEL_FONTSIZE)

    return fig


def main():
    tables = load_tables(SCENARIOS)
    fig = plot(tables, SCENARIOS, SAMPLE_SIZES)

    output_dir = io_utils.ensure_figure_output_dir()
    output_path = os.path.join(output_dir, FIGURE_NAME)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

"""
make_1D_figures.py
------------------
Produce `results/figures/1D_L1_complete.jpg` and
`results/figures/1D_L2_complete.jpg`.

One figure per metric. Rows are scenarios; the leftmost column shows the
scenario's sampling density and the remaining columns one sample size each.
Within a panel, points are the mean of log(distance) across iterations with
+/- 1 standard deviation error bars, for the median-split (Partial) and
midpoint (Full) estimators, offset left and right of each depth. Dashed
horizontal lines overlay the KDE/DPM/DRBART benchmarks, which have no notion of
tree depth and so are drawn flat across each panel.

Two aggregated table layouts are read, sliced differently:

* `by_sample_size/` one file per (scenario, n), with a `depth` column and
                    `{metric}{P|F}_{mean,sd}` columns.
* `by_statistic/`   one file per statistic, with `{n}_{metric}{P|F}` columns
                    and depth implied by row order.

They are complementary rather than redundant: `by_statistic/` covers all seven
scenarios, `by_sample_size/` supplies the 500/5000 runs for three of them.
`BY_SAMPLE_SIZE_RUNS` records which (scenario, sample size) pairs come from the
latter; everything else is read from the former.

The benchmark lines come from `baselines/`, one table per scenario holding all
four methods -- KDE Scott, KDE CV, DPM and DRBART. `KDE_scott` is fitted with
the bandwidth computed in `src/core/baseline_functions.py`, which scales by the
data's standard deviation; sklearn's own `bandwidth="scott"` string omits it and
oversmooths badly.

Reads only `results/aggregated_metric_tables/1D/` and `data/true_density/`. Run from anywhere:
    python src/make_figures/make_1D_figures.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core import io_utils  # noqa: E402

SCENARIOS = [
    "1D_beta64", "1D_beta91", "1D_beta50020", "1D_mix4",
    "1D_multispikevariedwidth", "1D_spikemix", "1D_overlappingscales",
]

DISPLAY_NAMES = {
    "1D_beta64": "Beta(6,4)",
    "1D_beta91": "Beta(9,1)",
    "1D_beta50020": "Beta(500,20)",
    "1D_mix4": "Mixture I",
    "1D_multispikevariedwidth": "Mixture II",
    "1D_spikemix": "Mixture III",
    "1D_overlappingscales": "Mixture IV",
}

SAMPLE_SIZES = [500, 1000, 5000]

# Scenarios whose 500/5000 runs live in `by_sample_size/`; their n=1000 run is
# in `by_statistic/`, so a single row of the figure mixes both layouts.
BY_SAMPLE_SIZE_RUNS = {
    "1D_beta64": (500, 5000),
    "1D_beta50020": (500, 5000),
    "1D_mix4": (500, 5000),
}

BENCHMARK_COLORS = {
    "KDE_scott": "tab:orange",
    "KDE_cv": "gold",
    "DPM": "tab:green",
    "DRBART": "tab:purple",
}

DISPLAY_METHOD_NAMES = {
    "KDE_scott": "KDE scott",
    "KDE_cv": "KDE CV",
    "DPM": "DPM",
    "DRBART": "DRBART",
    "Partial": "OPT median",
    "Full": "OPT midpoint",
}

METRIC_LABELS = {"L1": r"$L_1$", "L2": r"$L_2$"}

MARKERSIZE = 5
ALPHA = 0.5
SHIFT = 0.1
TICK_FONTSIZE = 20
LABEL_FONTSIZE = 25


def read_by_statistic_panel(prefix, n, metric):
    """One panel's data from the `by_statistic/` layout.

    Depth is implied by row order (row i is depth i+1). Sample sizes that
    stopped short of the maximum depth are NaN-padded, so the panel is
    restricted to the depths that actually have data.
    """
    mean_df = io_utils.read_metric_table(
        "1D", "by_statistic", prefix, f"{prefix}_metric_table_mean.csv")
    std_df = io_utils.read_metric_table(
        "1D", "by_statistic", prefix, f"{prefix}_metric_table_std.csv")

    partial_col, full_col = f"{n}_{metric}P", f"{n}_{metric}F"
    depths = np.arange(1, len(mean_df) + 1)
    valid = mean_df[partial_col].notna() | mean_df[full_col].notna()

    return (depths[valid],
            mean_df[partial_col][valid], std_df[partial_col][valid],
            mean_df[full_col][valid], std_df[full_col][valid])


def read_by_sample_size_panel(prefix, n, metric):
    """One panel's data from the per-(scenario, n) layout."""
    df = io_utils.read_metric_table(
        "1D", "by_sample_size", prefix, f"{prefix}_n{n}_metric_table.csv")
    return (df["depth"],
            df[f"{metric}P_mean"], df[f"{metric}P_sd"],
            df[f"{metric}F_mean"], df[f"{metric}F_sd"])


def read_panel(prefix, n, metric):
    """Dispatch to whichever layout holds this (scenario, sample size)."""
    if n in BY_SAMPLE_SIZE_RUNS.get(prefix, ()):
        return read_by_sample_size_panel(prefix, n, metric)
    return read_by_statistic_panel(prefix, n, metric)


def read_benchmarks(prefix):
    """The benchmark mean table, indexed by method.

    All four methods come from one table, built by `src/aggregate_tables.py`
    from a single per-iteration directory. Returns None if the scenario has no
    benchmark table, in which case the panels simply carry no benchmark lines.
    """
    try:
        return io_utils.read_metric_table(
            "1D", "baselines", prefix,
            f"{prefix}_baseline_metric_table_mean.csv")
    except FileNotFoundError:
        return None


def plot_true_density(ax, prefix, row_label, show_title):
    """Leftmost reference column: the scenario's sampling density."""
    density = io_utils.read_true_density(prefix)
    ax.plot(density["y"], density["true_density"], color="black", linewidth=1.2)
    if show_title:
        ax.set_title("Sampling Density", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("y", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(row_label, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def add_benchmark_overlay(ax, benchmark_means, n, metric):
    """One flat dashed line per benchmark method.

    Benchmarks have no depth parameter, so each is drawn as a constant
    reference across the panel. Uncertainty bands are omitted: shaded
    intervals for four methods made the panels unreadable.
    """
    if benchmark_means is None:
        return
    col = f"{n}_{metric}"
    for method, color in BENCHMARK_COLORS.items():
        row = benchmark_means[benchmark_means["method"] == method]
        if row.empty or col not in row.columns or pd.isna(row[col].iloc[0]):
            continue
        ax.axhline(row[col].iloc[0], color=color, linestyle="--",
                   linewidth=1.5, alpha=0.9)


def plot_metric(metric):
    """One figure for `metric`: scenarios down the rows, sample sizes across."""
    n_scenarios = len(SCENARIOS)
    n_cols = len(SAMPLE_SIZES) + 2  # density column + spacer + one per sample size

    # A zero-height spacer row between scenarios is what gives this layout its
    # vertical separation; hspace alone would also stretch the panels.
    n_rows = 2 * n_scenarios - 1
    height_ratios = [1, 0] * (n_scenarios - 1) + [1]
    width_ratios = [0.85, 0.3] + [1] * len(SAMPLE_SIZES)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows - 2),
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for grid_row in range(0, n_rows, 2):
        axes[grid_row][1].axis("off")
    for grid_row in range(1, n_rows, 2):
        for col in range(n_cols):
            axes[grid_row][col].axis("off")

    for i, prefix in enumerate(SCENARIOS):
        grid_row = 2 * i
        plot_true_density(axes[grid_row][0], prefix,
                          DISPLAY_NAMES.get(prefix, str(i + 1)), show_title=(i == 0))

        benchmark_df = read_benchmarks(prefix)

        for col_i, n in enumerate(SAMPLE_SIZES):
            col = col_i + 2
            ax = axes[grid_row][col]

            depth, partial_mean, partial_sd, full_mean, full_sd = read_panel(
                prefix, n, metric)

            ax.plot([t - SHIFT for t in depth], partial_mean, "o",
                    color="red", markersize=MARKERSIZE)
            ax.errorbar([t - SHIFT for t in depth], partial_mean, yerr=partial_sd,
                        fmt="none", ecolor="red", alpha=ALPHA, capsize=3.5)

            ax.plot([t + SHIFT for t in depth], full_mean, "s",
                    color="blue", markersize=MARKERSIZE)
            ax.errorbar([t + SHIFT for t in depth], full_mean, yerr=full_sd,
                        fmt="none", ecolor="blue", alpha=ALPHA, capsize=3.5)

            add_benchmark_overlay(ax, benchmark_df, n, metric)

            ax.set_xticks(depth)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE,
                           labelbottom=(i == n_scenarios - 1), labelleft=(col == 2))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.sharey(axes[grid_row][n_cols - 1])

            if col == 2:
                ax.set_ylabel(f"log {METRIC_LABELS[metric]} Risk",
                              fontsize=LABEL_FONTSIZE, fontweight="bold")
            if i == 0:
                ax.set_title(f"Sample Size {n}", fontsize=LABEL_FONTSIZE)

    axes[n_rows - 1][3].set_xlabel("\n Maximum Tree Depth", va="center",
                                   ha="center", labelpad=30, fontsize=LABEL_FONTSIZE)

    drawn_methods = read_benchmarks(SCENARIOS[0])
    handles = [
        plt.Line2D([], [], color="red", marker="o", linestyle="none",
                   label=DISPLAY_METHOD_NAMES["Partial"]),
        plt.Line2D([], [], color="blue", marker="s", linestyle="none",
                   label=DISPLAY_METHOD_NAMES["Full"]),
    ] + [
        # Only the benchmarks with a row in the table, so the legend cannot
        # advertise a line that was never drawn.
        plt.Line2D([], [], color=color, linestyle="--",
                   label=DISPLAY_METHOD_NAMES[method])
        for method, color in BENCHMARK_COLORS.items()
        if drawn_methods is not None and method in set(drawn_methods["method"])
    ]
    # Anchor to the bottom row's real figure-fraction position: on a figure this
    # tall, a small fixed offset from y=0 would land inches away from the axes.
    bottom_axes_y = axes[n_rows - 1][3].get_position().y0
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, bottom_axes_y - 0.03),
               ncol=len(handles), fontsize=LABEL_FONTSIZE)

    return fig


def main():
    output_dir = io_utils.ensure_figure_output_dir()
    for metric in ("L1", "L2"):
        fig = plot_metric(metric)
        output_path = os.path.join(output_dir, f"1D_{metric}_complete.jpg")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

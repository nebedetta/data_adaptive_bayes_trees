"""
aggregate.py
------------
Aggregate per-iteration metric CSVs into the mean/std/lower/upper tables that
every metric figure reads.

Each simulation iteration writes one CSV of raw (unlogged) distances, indexed by
tree depth down the rows and by `{sample_size}_{metric}{P|F}` across the columns.
This module stacks those files across iterations, takes logs, and reduces to four
summary tables. Figures plot log-risk, so the log is applied before averaging --
the mean of the logs, not the log of the mean.

Two guards run before anything is aggregated:

* Iteration files must agree on row count and column names, so a partial or
  stale run fails loudly instead of being silently stacked.
* Non-positive values are rejected before the log. A single zero would become
  `-inf`, propagate to `-inf`/`nan` in the summaries, and then render as a
  missing marker -- indistinguishable from a depth that was never run. No such
  value occurs in the shipped results; the guard keeps a future run from
  producing one unnoticed.
"""

import glob
import os

import numpy as np
import pandas as pd

STAT_NAMES = ("mean", "std", "lower", "upper")

LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975

# The benchmark methods are written by three separate simulation scripts, each
# producing its own per-iteration file family, and one family can carry more than
# one method. Order here fixes the row order of the aggregated table.
BASELINE_FILE_FAMILIES = (
    ("kde", ("KDE_scott", "KDE_cv")),
    ("dpm", ("DPM",)),
    ("drbart", ("DRBART",)),
)

BASELINE_CORE_METRICS = ("L1", "L2", "Linf", "time")


def _load_iteration_files(input_folder, pattern, ordered_columns=True):
    """Read every per-iteration CSV in `input_folder`, checking they agree.

    `ordered_columns=False` compares the column *set* rather than the sequence,
    for callers that select columns by name. The DRBART runs emit two column
    orderings of the same 13 columns -- 500/5000/1000 in most iterations,
    500/1000/5000 in a few -- which is harmless under name-based selection but
    would fail an order-sensitive comparison.
    """
    paths = sorted(glob.glob(os.path.join(input_folder, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {input_folder}"
        )

    frames = [pd.read_csv(path) for path in paths]

    shapes = {os.path.basename(p): len(df) for p, df in zip(paths, frames)}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Iteration files disagree on row count: {shapes}")

    key = (lambda df: tuple(df.columns)) if ordered_columns else \
          (lambda df: tuple(sorted(df.columns)))
    columns = {os.path.basename(p): key(df) for p, df in zip(paths, frames)}
    if len(set(columns.values())) != 1:
        raise ValueError(
            f"Iteration files disagree on columns: {sorted(columns)[:5]} ..."
        )

    return paths, frames


def _check_positive(stacked, paths):
    """Reject non-positive metrics, which `np.log` would turn into `-inf`.

    NaN is expected and allowed: sample sizes that stopped short of the maximum
    depth are NaN-padded, and the figures mask those panels.
    """
    bad = (stacked <= 0) & ~np.isnan(stacked)
    if not bad.any():
        return
    iteration_idx, row, col = (int(i[0]) for i in np.nonzero(bad))
    raise ValueError(
        f"Non-positive metric in {os.path.basename(paths[iteration_idx])} "
        f"at row {row}, column {col} (value {stacked[iteration_idx, row, col]!r}). "
        "Taking logs would silently produce -inf and render as a missing point."
    )


def aggregate_metric_tables(input_folder, output_folder, prefix,
                            pattern=None, write=True):
    """Aggregate one scenario's per-iteration CSVs into four summary tables.

    Parameters
    ----------
    input_folder  : directory of per-iteration CSVs for a single scenario
    output_folder : destination for `{prefix}_metric_table_{stat}.csv`
    prefix        : scenario prefix, e.g. `3D_smooth1`
    pattern       : glob for iteration files (default `{prefix}_iter*_metric.csv`)
    write         : if False, compute and return without writing to disk

    Returns
    -------
    dict of stat name -> DataFrame (of log-scale summaries)
    """
    if pattern is None:
        pattern = f"{prefix}_iter*_metric.csv"

    paths, frames = _load_iteration_files(input_folder, pattern)

    stacked = np.stack([df.values.astype(float) for df in frames])
    _check_positive(stacked, paths)

    log_data = np.log(stacked)
    columns = frames[0].columns

    tables = {
        "mean": np.mean(log_data, axis=0),
        "std": np.std(log_data, axis=0),
        "lower": np.quantile(log_data, LOWER_QUANTILE, axis=0),
        "upper": np.quantile(log_data, UPPER_QUANTILE, axis=0),
    }
    tables = {name: pd.DataFrame(values, columns=columns)
              for name, values in tables.items()}

    if write:
        os.makedirs(output_folder, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(
                os.path.join(output_folder, f"{prefix}_metric_table_{name}.csv"),
                index=False,
            )

    return tables


def aggregate_baseline_tables(input_folder, output_folder, prefix, write=True):
    """Aggregate one scenario's per-iteration benchmark CSVs into four tables.

    Unlike the OPT tables, benchmarks have no notion of tree depth: each
    per-iteration file holds one row per method, with metrics across the columns
    as `{sample_size}_{metric}`. The aggregated tables are therefore one row per
    method rather than one row per depth.

    The methods arrive in three file families (`kde`, `dpm`, `drbart`) written by
    separate simulation scripts; see `BASELINE_FILE_FAMILIES`. A family with no
    files is skipped rather than raising, so a scenario that was never run
    through DRBART still aggregates -- with the methods it does have.

    Columns are selected by name, never by position. The DRBART files order their
    sample sizes 500/5000/1000 while the KDE and DPM files use 500/1000/5000;
    selecting positionally would transpose two of DRBART's columns silently.

    Returns
    -------
    dict of stat name -> DataFrame, each with a leading `method` column
    """
    frames = {}
    for family, _ in BASELINE_FILE_FAMILIES:
        try:
            paths, family_frames = _load_iteration_files(
                input_folder, f"{prefix}_iter*_baseline_{family}_metric.csv",
                ordered_columns=False)
        except FileNotFoundError:
            # Skipping is deliberate -- see the docstring -- but silence is not:
            # a family whose files are simply not where they were looked for
            # produces a table quietly missing its methods, and under --check a
            # mismatch that does not say why. Name it.
            print(f"    note: no {family} iteration files for {prefix} in "
                  f"{input_folder}; its methods "
                  f"({', '.join(dict(BASELINE_FILE_FAMILIES)[family])}) "
                  "are absent from the aggregated table")
            continue
        frames[family] = (paths, family_frames)

    if "kde" not in frames:
        raise FileNotFoundError(
            f"No KDE iteration files in {input_folder}; the sample sizes are "
            "read from them, so they are required."
        )

    # Sample sizes come from the KDE files, which cover every size that was run.
    # A method missing one of those columns raises on lookup below rather than
    # being quietly dropped.
    kde_columns = frames["kde"][1][0].columns
    sample_sizes = sorted({int(c.split("_")[0]) for c in kde_columns
                           if c.split("_")[0].isdigit()})
    metric_columns = [f"{n}_{m}" for n in sample_sizes
                      for m in BASELINE_CORE_METRICS]

    methods, stacked_rows = [], []
    for family, family_methods in BASELINE_FILE_FAMILIES:
        if family not in frames:
            continue
        paths, family_frames = frames[family]
        for method in family_methods:
            rows = [df.loc[df["method"] == method, metric_columns]
                      .iloc[0].to_numpy(dtype=float)
                    for df in family_frames]
            stacked = np.stack(rows)
            # (iterations, 1, metrics) to match _check_positive's expectations.
            _check_positive(stacked[:, None, :], paths)
            methods.append(method)
            stacked_rows.append(stacked)

    log_data = {method: np.log(stacked)
                for method, stacked in zip(methods, stacked_rows)}

    def build(stat_fn):
        table = pd.DataFrame([stat_fn(log_data[m]) for m in methods],
                             columns=metric_columns)
        table.insert(0, "method", methods)
        return table

    tables = {
        "mean": build(lambda a: np.mean(a, axis=0)),
        "std": build(lambda a: np.std(a, axis=0)),
        "lower": build(lambda a: np.quantile(a, LOWER_QUANTILE, axis=0)),
        "upper": build(lambda a: np.quantile(a, UPPER_QUANTILE, axis=0)),
    }

    if write:
        os.makedirs(output_folder, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(
                os.path.join(output_folder,
                             f"{prefix}_baseline_metric_table_{name}.csv"),
                index=False,
            )

    return tables

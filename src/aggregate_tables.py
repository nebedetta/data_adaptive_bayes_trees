"""
aggregate_tables.py
-------------------
Build the metric tables under `results/aggregated_metric_tables/` from per-iteration
simulation output.

This is the step between the simulation and the figures. Each iteration of a
simulation writes one CSV of raw distances, indexed by tree depth down the rows
and by `{sample_size}_{metric}{P|F}` across the columns. This script stacks
those files across iterations, takes logs, and reduces them to the mean, standard
deviation and 2.5%/97.5% quantiles the metric figures plot.

The per-iteration CSVs are not distributed -- producing them means re-running the
simulation on a cluster (see the README). This script is here so the derivation
of the shipped tables is inspectable and repeatable given that output, in the
same way `src/data_analysis/fit_flow.py` records how the flow-cytometry fits were
made.

Usage (from anywhere):
    # everything, from the standard output layout
    python src/aggregate_tables.py --input-root /path/to/output

    # one dimension, or one scenario
    python src/aggregate_tables.py --input-root /path/to/output --only 3D
    python src/aggregate_tables.py --input-root /path/to/output --only 3D_smooth1

    # check the shipped tables can be reproduced, without writing
    python src/aggregate_tables.py --input-root /path/to/output --check
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import io_utils  # noqa: E402
from src.core.aggregate import (STAT_NAMES, aggregate_baseline_tables,  # noqa: E402
                                aggregate_metric_tables)

# A simulation run writes two things per iteration, under --input-root:
#
#     <dim>/fits/<scenario>/     the fitted trees, as .pkl.gz
#     <dim>/metrics/<scenario>/  the distances computed from them, as .csv
#
# This script reads the second. The baseline jobs are a separate family, with
# no trees to write, so their densities go to `fits_baselines/` and their
# distances to `metrics_baselines/`. `scripts/` declares the same paths.
METRICS_SUBDIR = "metrics"

SCENARIOS_1D_SPEED = ["1D_beta64", "1D_beta91", "1D_beta50020", "1D_mix4",
                      "1D_multispikevariedwidth", "1D_spikemix",
                      "1D_overlappingscales"]

SCENARIOS_2D = ["2D_genbeta1", "2D_genbeta3", "2D_genbeta4", "2D_genbeta6",
                "2D_mix13", "2D_mix14"]

SCENARIOS_3D = ["3D_smooth1", "3D_mix1", "3D_genbetaspike1"]

# The benchmark tables the 1D figures overlay. Three simulation scripts write
# three per-iteration file families into this one directory -- kde (carrying
# both KDE_scott and KDE_cv), dpm, and drbart -- and `aggregate_baseline_tables`
# merges them on `method` into a single table per scenario. See
# `src/core/aggregate.py` for the families and how a missing one is handled.
BASELINE_INPUT_SUBDIR = os.path.join("1D", "metrics_baselines")
BASELINE_OUTPUT_SUBDIR = os.path.join("1D", "baselines")


def targets():
    """Every (prefix, list of input subdirectories, output subdirectory).

    A scenario can have more than one input directory. The 1D sweep was run
    first without n=1000 and extended later, and the extension went to its own
    directory; which sample sizes landed where varies by scenario, so both are
    read and merged by column. Four scenarios have all their sizes in
    `metrics_n1000/` and are unaffected by the merge.
    """
    out = []
    for prefix in SCENARIOS_1D_SPEED:
        out.append((prefix,
                    [os.path.join("1D", METRICS_SUBDIR, prefix),
                     os.path.join("1D", "metrics_n1000", prefix)],
                    os.path.join("1D", "by_statistic", prefix)))
    for prefix in SCENARIOS_2D:
        out.append((prefix,
                    [os.path.join("2D", METRICS_SUBDIR, prefix)],
                    os.path.join("2D", prefix)))
    for prefix in SCENARIOS_3D:
        out.append((prefix,
                    [os.path.join("3D", METRICS_SUBDIR, prefix)],
                    os.path.join("3D", prefix)))
    return out


# The order sim_run_1D_speed.py writes its metrics in, per sample size:
# Partial before Full, for each of L1, L2, Linfty and time.
METRIC_SUFFIXES = ["L1P", "L1F", "L2P", "L2F", "LIP", "LIF", "TP", "TF"]


def merge_by_column(frames):
    """Combine per-directory tables that cover disjoint sample sizes.

    Rows are aligned by position -- every table is indexed by depth 1, 2, 3, ...
    -- and the shorter ones are NaN-padded to the longest, which is the existing
    convention for a sample size that stopped short of the maximum depth.

    A column seen in more than one directory keeps its first occurrence. The
    directories are meant to cover disjoint sample sizes, so an overlap means the
    same run was aggregated twice; taking the first copy makes that harmless
    rather than producing the duplicate columns some of the shipped tables carry
    (`1000_L1P`, `1000_L1P.1`, ... -- pandas' marker for a repeated name).

    Columns come out in the order the simulation writes them: ascending sample
    size, then METRIC_SUFFIXES within each.
    """
    n_rows = max(len(f) for f in frames)

    merged = {}
    for frame in frames:
        padded = frame.reindex(range(n_rows))
        for column in padded.columns:
            merged.setdefault(column, padded[column])

    sizes = sorted({int(c.split("_")[0]) for c in merged
                    if c.split("_")[0].isdigit()})
    ordered = [f"{n}_{suffix}" for n in sizes for suffix in METRIC_SUFFIXES
               if f"{n}_{suffix}" in merged]
    ordered += [c for c in merged if c not in ordered]

    return pd.DataFrame({c: merged[c] for c in ordered})


def compare_to_shipped(tables, prefix, output_subdir):
    """Largest difference between freshly aggregated tables and the shipped ones.

    Compares column by column, by name, rather than as whole arrays. Some shipped
    tables carry duplicate columns from an aggregation that merged the same run
    repeatedly, so their layout differs even where every value agrees. Matching
    by name checks what actually matters -- the figures also select columns by
    name, and pandas hands them the first occurrence.

    Returns (worst difference, note). The note records a layout difference; it is
    not a failure on its own.
    """
    worst = 0.0
    note = None
    for stat in STAT_NAMES:
        path = os.path.join(io_utils.METRIC_TABLE_DIR, output_subdir,
                            f"{prefix}_metric_table_{stat}.csv")
        if not os.path.exists(path):
            return None, f"no shipped table at {os.path.relpath(path, io_utils.REPO_ROOT)}"
        shipped = pd.read_csv(path)
        fresh = tables[stat]

        missing = [c for c in fresh.columns if c not in shipped.columns]
        if missing:
            return None, f"shipped table is missing {missing[:3]}"

        if list(shipped.columns) != list(fresh.columns) and note is None:
            extra = len(shipped.columns) - len(fresh.columns)
            note = (f"layout differs ({extra} extra columns in the shipped table)"
                    if extra > 0 else "column order differs")

        for column in fresh.columns:
            # Timing columns are wall-clock seconds, so two runs of the same
            # configuration differ legitimately. They are recorded but never
            # plotted; comparing them would flag re-runs as disagreements.
            if column.split("_")[-1].startswith("T"):
                continue
            a = fresh[column].to_numpy(dtype=float)
            b = shipped[column]
            # A duplicated name gives a DataFrame; the figures see the first.
            b = b.iloc[:, 0] if isinstance(b, pd.DataFrame) else b
            b = b.to_numpy(dtype=float)
            length = min(len(a), len(b))
            a, b = a[:length], b[:length]
            # NaN marks depths a sample size never reached; it must line up.
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                return None, f"NaN pattern differs in {column}"
            both_nan = np.isnan(a) & np.isnan(b)
            worst = max(worst, float(np.max(np.where(both_nan, 0.0, np.abs(a - b)))))
    return worst, note


def compare_baselines_to_shipped(tables, prefix):
    """`compare_to_shipped`, for the method-indexed benchmark tables.

    These are indexed by method rather than depth, so rows are matched on the
    `method` column instead of by position, and a missing method is a failure
    rather than a length mismatch.
    """
    worst = 0.0
    for stat in STAT_NAMES:
        path = os.path.join(io_utils.METRIC_TABLE_DIR, BASELINE_OUTPUT_SUBDIR,
                            prefix, f"{prefix}_baseline_metric_table_{stat}.csv")
        if not os.path.exists(path):
            return None, f"no shipped table at {os.path.relpath(path, io_utils.REPO_ROOT)}"
        shipped = pd.read_csv(path).set_index("method")
        fresh = tables[stat].set_index("method")

        missing = [m for m in fresh.index if m not in shipped.index]
        if missing:
            return None, f"shipped table is missing method {missing}"

        for column in fresh.columns:
            # Wall-clock seconds; recorded but never plotted.
            if column.endswith("_time"):
                continue
            if column not in shipped.columns:
                return None, f"shipped table is missing {column}"
            a = fresh[column].to_numpy(dtype=float)
            b = shipped.loc[fresh.index, column].to_numpy(dtype=float)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                return None, f"NaN pattern differs in {column}"
            both_nan = np.isnan(a) & np.isnan(b)
            worst = max(worst, float(np.max(np.where(both_nan, 0.0, np.abs(a - b)))))
    return worst, None


def run_baselines(input_root, check):
    """Aggregate the benchmark tables. Returns (worst difference, failures)."""
    worst_overall = 0.0
    failures = []
    for prefix in SCENARIOS_1D_SPEED:
        input_folder = os.path.join(input_root, BASELINE_INPUT_SUBDIR, prefix)
        if not os.path.isdir(input_folder):
            print(f"  skipped {prefix}: no {BASELINE_INPUT_SUBDIR}/{prefix} "
                  "under --input-root")
            continue

        output_folder = os.path.join(io_utils.METRIC_TABLE_DIR,
                                     BASELINE_OUTPUT_SUBDIR, prefix)
        try:
            tables = aggregate_baseline_tables(input_folder, output_folder,
                                               prefix, write=not check)
        except (FileNotFoundError, ValueError) as error:
            failures.append(f"{prefix}: {error}")
            print(f"  FAILED  {prefix}: {error}")
            continue

        if check:
            worst, note = compare_baselines_to_shipped(tables, prefix)
            if worst is None:
                failures.append(f"{prefix}: {note}")
                print(f"  FAILED  {prefix}: {note}")
            else:
                worst_overall = max(worst_overall, worst)
                methods = ", ".join(tables["mean"]["method"])
                print(f"  ok      {prefix}  max|diff| = {worst:.3e}   [{methods}]")
        else:
            methods = ", ".join(tables["mean"]["method"])
            print(f"  wrote   {prefix} -> "
                  f"{os.path.relpath(output_folder, io_utils.REPO_ROOT)}"
                  f"   [{methods}]")
    return worst_overall, failures


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", required=True,
                        help="Root of the per-iteration simulation output.")
    parser.add_argument("--only", default=None,
                        help="Restrict to a dimension (1D/2D/3D), one scenario, "
                             "or 'baselines' for the benchmark tables.")
    parser.add_argument("--check", action="store_true",
                        help="Compare against the shipped tables instead of "
                             "overwriting them.")
    args = parser.parse_args()

    # The benchmark tables are indexed by method rather than depth and are
    # aggregated on their own path; `--only baselines` runs just those.
    want_baselines = args.only in (None, "baselines", "1D")
    want_opt = args.only != "baselines"

    selected = [t for t in targets()
                if args.only is None or t[0] == args.only
                or t[0].startswith(args.only)] if want_opt else []
    if not selected and not want_baselines:
        raise SystemExit(f"Nothing matches --only {args.only!r}")

    worst_overall = 0.0
    failures = []
    for prefix, input_subdirs, output_subdir in selected:
        present = [os.path.join(args.input_root, d) for d in input_subdirs]
        present = [d for d in present if os.path.isdir(d)]
        if not present:
            print(f"  skipped {prefix}: none of "
                  f"{', '.join(input_subdirs)} under --input-root")
            continue

        output_folder = os.path.join(io_utils.METRIC_TABLE_DIR, output_subdir)
        try:
            # Aggregate each directory separately, then merge by column: they
            # cover disjoint sample sizes, not disjoint iterations.
            per_directory = [
                aggregate_metric_tables(folder, output_folder, prefix, write=False)
                for folder in present
            ]
            tables = {stat: merge_by_column([p[stat] for p in per_directory])
                      for stat in STAT_NAMES}
        except (FileNotFoundError, ValueError) as error:
            failures.append(f"{prefix}: {error}")
            print(f"  FAILED  {prefix}: {error}")
            continue

        if not args.check:
            os.makedirs(output_folder, exist_ok=True)
            for stat, frame in tables.items():
                frame.to_csv(os.path.join(
                    output_folder, f"{prefix}_metric_table_{stat}.csv"), index=False)

        if args.check:
            worst, note = compare_to_shipped(tables, prefix, output_subdir)
            if worst is None:
                failures.append(f"{prefix}: {note}")
                print(f"  FAILED  {prefix}: {note}")
            else:
                worst_overall = max(worst_overall, worst)
                suffix = f"   [{note}]" if note else ""
                print(f"  ok      {prefix}  max|diff| = {worst:.3e}{suffix}")
        else:
            print(f"  wrote   {prefix} -> "
                  f"{os.path.relpath(output_folder, io_utils.REPO_ROOT)}")

    if want_baselines:
        baseline_worst, baseline_failures = run_baselines(args.input_root, args.check)
        worst_overall = max(worst_overall, baseline_worst)
        failures.extend(baseline_failures)

    if args.check:
        print(f"\nlargest difference from the shipped tables: {worst_overall:.3e}")
        if failures:
            print("FAILED:\n  " + "\n  ".join(failures))
            return 1
        # Differences at this level are floating-point summation order, not a
        # change in the numbers.
        if worst_overall < 1e-9:
            print("PASS -- every metric column reproduces from the per-iteration "
                  "output. Timing columns are excluded: they are wall-clock "
                  "seconds, and are recorded but never plotted.")
            return 0
        print("Differences exceed rounding; investigate before relying on these.")
        return 1

    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

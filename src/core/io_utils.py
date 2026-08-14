"""
io_utils.py
-----------
Shared I/O helpers: repository-root resolution, distribution-spec loading, and
reading the density tables shipped under `data/`.

Every path in the repository resolves through here, relative to this file's
location, so the code runs unmodified on any machine and from any working
directory.
"""

import gzip
import importlib.util
import os

import numpy as np
import pandas as pd

# .../repo/src/core/io_utils.py -> .../repo
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# What the figures read.
DATA_DIR = os.path.join(REPO_ROOT, "data")
TRUE_DENSITY_DIR = os.path.join(DATA_DIR, "true_density")

# What the pipeline produced and this repository ships: the aggregated metric
# tables, and the figures built from them.
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
METRIC_TABLE_DIR = os.path.join(RESULTS_DIR, "aggregated_metric_tables")
FIGURE_OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")


def repo_path(*parts):
    """Absolute path from repository-root-relative components."""
    return os.path.join(REPO_ROOT, *parts)


def import_distribution_module(module_name, module_path):
    """Load a distribution spec (`distributions/*_input.py`) as a module.

    `module_path` may be absolute or relative to the repository root. Note that
    the 2D specs are metadata-only stubs: their `compute_pdf` lives in the
    matching `.R` file, which is why 2D true densities are shipped as data
    rather than regenerated in Python.
    """
    if not os.path.isabs(module_path):
        module_path = repo_path(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load distribution module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The grid the 1D densities are evaluated on. `grid_points_1D.csv` holds the
# same points rounded to six significant figures, for the R-driven scenarios;
# computing them here instead keeps full float64 precision.
GRID_1D = (0.0, 1.0, 100000)


def compute_true_density_1D(prefix):
    """Evaluate a 1D scenario's density from its `distributions/*_input.py`.

    The 1D densities are analytic -- `compute_pdf` is a handful of scipy calls
    -- so they are computed on demand rather than shipped. This costs a few
    milliseconds per scenario and avoids committing ~3.6 MB of derived data
    that would only ever be a lossy copy of what these specs already define.
    """
    module = import_distribution_module(
        f"{prefix}_input", os.path.join("distributions", f"{prefix}_input.py"))
    grid = np.linspace(*GRID_1D)
    return pd.DataFrame({"y": grid,
                         "true_density": np.asarray(module.compute_pdf(grid),
                                                    dtype=float).ravel()})


def read_true_density(prefix):
    """One scenario's true density, computed if it can be and read if it cannot.

    1D densities are computed from their Python spec (see
    `compute_true_density_1D`). The 2D specs are metadata-only stubs whose
    `compute_pdf` lives in the matching `.R` file, so those densities are
    shipped under `data/true_density/`, gzipped at six significant figures.

    A shipped file takes precedence for any scenario that has one, so a
    regenerated `.csv` can be dropped in to override.
    """
    gz_path = os.path.join(TRUE_DENSITY_DIR, f"{prefix}_true_density.csv.gz")
    csv_path = os.path.join(TRUE_DENSITY_DIR, f"{prefix}_true_density.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    if os.path.exists(gz_path):
        with gzip.open(gz_path, "rt") as handle:
            return pd.read_csv(handle)
    if prefix.startswith("1D_"):
        return compute_true_density_1D(prefix)
    raise FileNotFoundError(
        f"No true density for '{prefix}' in {TRUE_DENSITY_DIR} "
        f"(looked for {os.path.basename(gz_path)} and {os.path.basename(csv_path)})"
    )


def fits_path(dimension, prefix, filename):
    """Path to one per-iteration fit, laid out as the simulation writes them.

    A cluster run puts its pickles in `<output-root>/<dim>/fits/<scenario>/`,
    and `data/` mirrors that. Only the six files `2D_itern15` needs are shipped
    -- it plots posterior mean surfaces rather than a summary, so no aggregated
    table would do -- but they sit at the path a full run would produce, so a
    reader with their own output can point at it unchanged.
    """
    return os.path.join(DATA_DIR, dimension, "fits", prefix, filename)


def metric_table_path(dimension, *parts):
    """Path inside `results/aggregated_metric_tables/{1D,2D,3D}/`."""
    return os.path.join(METRIC_TABLE_DIR, dimension, *parts)


def read_metric_table(dimension, *parts):
    """Read one aggregated metric table CSV."""
    path = metric_table_path(dimension, *parts)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metric table: {path}")
    return pd.read_csv(path)


def ensure_figure_output_dir():
    """Create and return `results/figures/`, where every figure script writes."""
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
    return FIGURE_OUTPUT_DIR

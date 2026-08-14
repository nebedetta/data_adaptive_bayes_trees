"""
generate_true_density_1D.py
----------------------------
Evaluate compute_pdf() on the fixed grid distributions/grid_points_1D.csv
and write distributions/true_density/<prefix>_true_density.csv, matching
the output format of generating_1D_samples.R (columns "y", "true_density").

The grid is read from grid_points_1D.csv rather than regenerated, so the
densities are pinned to the grid shared with the R-driven scenarios.

Usage:
    python distributions/generate_true_density_1D.py
"""

import os
import sys
import csv
import importlib.util
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GRID_PATH = os.path.join(REPO_ROOT, "distributions", "grid_points_1D.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "distributions", "true_density")

DIST_MODULES = [
    "1D_beta64_input.py",
    "1D_beta50020_input.py",
    "1D_mix4_input.py",
]


def import_distribution_module(module_path):
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    grid = pd.read_csv(GRID_PATH)["y"].to_numpy()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in DIST_MODULES:
        module_path = os.path.join(REPO_ROOT, "distributions", filename)
        dist_module = import_distribution_module(module_path)
        prefix = dist_module.prefix

        true_density = dist_module.compute_pdf(grid)

        out_path = os.path.join(OUTPUT_DIR, f"{prefix}_true_density.csv")
        pd.DataFrame({"y": grid, "true_density": true_density}).to_csv(
            out_path, index=False, quoting=csv.QUOTE_NONNUMERIC
        )
        print(f"{prefix:15s} n_grid={len(grid)}  "
              f"mass={np.trapezoid(true_density, grid):.5f}  -> {out_path}")


if __name__ == "__main__":
    main()

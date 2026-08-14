"""
Generate distributions/grid_points_3D.csv.gz: a regular 50^3 = 125,000-point
evaluation grid over [1e-5, 1-1e-5]^3, matching the endpoint convention used
in grid_points_2D.csv.gz (avoids exact 0/1 to sidestep boundary singularities in
Beta-family densities).

Run once: python distributions/generate_grid_3D.py
"""
import numpy as np
import pandas as pd

N_PER_DIM = 50
LOW, HIGH = 1e-5, 1 - 1e-5

axis = np.linspace(LOW, HIGH, N_PER_DIM)
x0, x1, x2 = np.meshgrid(axis, axis, axis, indexing="ij")

grid = pd.DataFrame({
    "x0": x0.ravel(),
    "x1": x1.ravel(),
    "x2": x2.ravel(),
})

out_path = "distributions/grid_points_3D.csv.gz"
grid.to_csv(out_path, index=False)   # pandas gzips on the .gz suffix
print(f"Wrote {len(grid):,} points to {out_path}")

# The grid is committed, so this only needs re-running if the resolution or the
# endpoint convention changes.

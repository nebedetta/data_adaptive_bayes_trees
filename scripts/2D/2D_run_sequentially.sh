#!/bin/bash
# Submit the three 2D sample sizes as a dependent chain: 500, then 5000, then
# 50000, each starting only if the previous array finished cleanly. Run from a
# login node, not under sbatch -- it only submits.
#
#   ./2D_run_sequentially.sh 2D_mix14
#
# The scenario defaults to 2D_mix14, the one this was last run for.

set -eu

DIST_NAME="${1:-2D_mix14}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

dist_module_name="${DIST_NAME}_input"
dist_module_path="distributions/${DIST_NAME}_input.py"
sim_output_path="output/2D/fits"
metric_output_path="output/2D/metrics"

# --- Editable per-run knobs -------------------------------------------------
depth_parallelization="8"
ncores="17"
n_iter="1000"
# -----------------------------------------------------------------------------

if [ ! -f "$dist_module_path" ]; then
    echo "No distribution spec at $dist_module_path" >&2
    exit 1
fi

mkdir -p "$sim_output_path" "$metric_output_path"

submit() {
    sbatch --parsable "$@" \
        "$dist_module_name" "$dist_module_path" \
        "$sim_output_path" "$metric_output_path" \
        "$depth_parallelization" "$ncores" "$n_iter"
}

job500=$(submit scripts/2D/2D_sim_500.sh)
echo "Submitted n=500 as $job500"

job5000=$(submit --dependency=afterok:"$job500" scripts/2D/2D_sim_5000.sh)
echo "Submitted n=5000 as $job5000, after $job500"

job50000=$(submit --dependency=afterok:"$job5000" scripts/2D/2D_sim_50000.sh)
echo "Submitted n=50000 as $job50000, after $job5000"

#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=research_group
#SBATCH --job-name=sim_1D_speed
#SBATCH --array=0-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

source activate pt_sim

# --- Editable per-run knobs -------------------------------------------------
SAMPLE_SIZES="1000"
DEPTHS="1,2,3,4,5,6,7,8"
# -----------------------------------------------------------------------------

if [ -z "$DIST_NAME" ]; then
    echo "DIST_NAME not set. Submit with: sbatch --export=ALL,DIST_NAME=<name> $0" >&2
    exit 1
fi

iteration=$SLURM_ARRAY_TASK_ID
dist_module_name="${DIST_NAME}_input"
dist_module_path="distributions/${DIST_NAME}_input.py"

sim_output_path="output/1D/fits"
metric_output_path="output/1D/metrics"
density_folder="distributions/true_density"
samples_folder="distributions/samples"

mkdir -p "$sim_output_path" "$metric_output_path"

# sim_run_1D_speed.py uses an absolute package import (`from src.core import
# ...`), so it must be run as a module from the repo root, not invoked by
# file path (matches 3D_sim.sh's pattern).


python -m src.sim_run_1D_speed \
    "$iteration" \
    "$SAMPLE_SIZES" \
    "$DEPTHS" \
    "$dist_module_name" \
    "$dist_module_path" \
    "$sim_output_path" \
    "$metric_output_path" \
    "$density_folder" \
    "$samples_folder"

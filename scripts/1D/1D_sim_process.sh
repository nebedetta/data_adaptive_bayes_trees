#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=research_group
#SBATCH --job-name=1D_sim_process
#SBATCH --array=0-2
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Stage 2 of the two-stage 1D pipeline behind the by_sample_size tables: read
# the trees 1D_sim.sh wrote and reduce them to per-(scenario, n) metrics. One
# array task per sample size in SAMPLE_SIZE_VEC below, so --array must cover its
# indices -- 0-2 for the three sizes here.
#
#   sbatch --export=ALL,DIST_NAME=1D_mix4 scripts/1D/1D_sim_process.sh

source activate pt_sim

if [ -z "$DIST_NAME" ]; then
    echo "DIST_NAME not set. Submit with: sbatch --export=ALL,DIST_NAME=<name> $0" >&2
    exit 1
fi

# --- Editable per-run knobs -------------------------------------------------
# The sizes 1D_sim.sh fitted; must match sample_size_vec in src/sim_run_1D.py.
SAMPLE_SIZE_VEC=(500 5000 50000)
# -----------------------------------------------------------------------------

dist_module_name="${DIST_NAME}_input"
dist_module_path="distributions/${DIST_NAME}_input.py"

pickle_path="output/1D/output_raw"
output_path="output/1D/output_processed"
mkdir -p "$output_path"

title="processed_results"

SAMPLE_SIZE=${SAMPLE_SIZE_VEC[$SLURM_ARRAY_TASK_ID]}
if [ -z "$SAMPLE_SIZE" ]; then
    echo "No sample size at index $SLURM_ARRAY_TASK_ID; --array must cover" \
         "0-$(( ${#SAMPLE_SIZE_VEC[@]} - 1 ))" >&2
    exit 1
fi

# sim_process_1D.py uses an absolute package import (`from src.core import
# ...`), so it must be run as a module from the repo root, not invoked by file
# path (matches 1D_sim_speed.sh's pattern).
python -m src.sim_process_1D \
    "$SAMPLE_SIZE" \
    "$dist_module_name" \
    "$dist_module_path" \
    "$pickle_path" \
    "$output_path" \
    "$title"

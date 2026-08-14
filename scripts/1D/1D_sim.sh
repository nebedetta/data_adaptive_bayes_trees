#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=research_group
#SBATCH --job-name=1D_sim
#SBATCH --array=0-199
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Stage 1 of the two-stage 1D pipeline behind the by_sample_size tables: fit
# every (sample size, depth) for one iteration and write the trees. Stage 2 is
# 1D_sim_process.sh, which reduces them to metrics. The later by_statistic runs
# used 1D_sim_speed.sh instead, which does both in one job.
#
# Sample sizes and depths are set inside src/sim_run_1D.py, not here.
#
#   sbatch --export=ALL,DIST_NAME=1D_mix4 scripts/1D/1D_sim.sh

source activate pt_sim

if [ -z "$DIST_NAME" ]; then
    echo "DIST_NAME not set. Submit with: sbatch --export=ALL,DIST_NAME=<name> $0" >&2
    exit 1
fi

iteration=$SLURM_ARRAY_TASK_ID
dist_module_name="${DIST_NAME}_input"
dist_module_path="distributions/${DIST_NAME}_input.py"

output_path="output/1D/output_raw"
mkdir -p "$output_path"

echo "Running iteration $iteration with module $dist_module_name"

# sim_run_1D.py uses an absolute package import (`from src.core import ...`), so
# it must be run as a module from the repo root, not invoked by file path
# (matches 1D_sim_speed.sh's pattern).
python -m src.sim_run_1D \
    "$iteration" \
    "$dist_module_name" \
    "$dist_module_path" \
    "$output_path"

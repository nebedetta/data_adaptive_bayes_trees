#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=research_group
#SBATCH --job-name=baselines_1D_kde
#SBATCH --array=0-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

source activate pt_sim

# --- Editable per-run knobs -------------------------------------------------
KDE_CV_FOLDS=5
KDE_CV_N_BANDWIDTHS=60 #30 
KDE_CV_MAX_N=50000
KDE_RTOL=0 #1e-5 
# -----------------------------------------------------------------------------

if [ -z "$DIST_NAME" ]; then
    echo "DIST_NAME not set. Submit with: sbatch --export=ALL,DIST_NAME=<name>,N=<n> $0" >&2
    exit 1
fi
if [ -z "$N" ]; then
    echo "N not set. Submit with: sbatch --export=ALL,DIST_NAME=<name>,N=<n> $0" >&2
    exit 1
fi

iteration=$SLURM_ARRAY_TASK_ID
dist_module_name="${DIST_NAME}_input"
dist_module_path="distributions/${DIST_NAME}_input.py"

metric_output_path="output/1D/metrics_baselines"
sim_output_path="output/1D/fits_baselines"
density_folder="distributions/true_density"
samples_folder="distributions/samples"

mkdir -p "$metric_output_path" "$sim_output_path"

# sim_run_baselines_1D_kde.py uses an absolute package import (`from src.core
# import ...`), so it must be run as a module from the repo root, not invoked
# by file path (matches sim_run_1D_speed.py / 1D_sim_speed.sh's pattern).
# Submit this job from the repo root so relative paths above resolve correctly.

python -m src.sim_run_baselines_1D_kde \
    "$iteration" "$N" \
    "$dist_module_name" "$dist_module_path" \
    "$metric_output_path" "$sim_output_path" \
    "$density_folder" "$samples_folder" \
    "$KDE_CV_FOLDS" "$KDE_CV_N_BANDWIDTHS" "$KDE_CV_MAX_N" "$KDE_RTOL"

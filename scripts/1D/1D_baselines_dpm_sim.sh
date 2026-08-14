#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=research_group
#SBATCH --job-name=baselines_1D_dpm
#SBATCH --array=0-199
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00

source activate pt_sim

# --- Editable per-run knobs -------------------------------------------------
# Matches src/core/baseline_functions.py's DPM_DEFAULT_* -- override here only
# to deviate from those defaults for a specific run.
DPM_MAX_COMPONENTS=50
DPM_MAX_ITER=5000
DPM_COVARIANCE=full
DPM_ALPHA=1.0
DPM_N_INIT=1
DPM_INIT_PARAMS=kmeans
DPM_TOL=1e-4
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

# sim_run_baselines_1D_dpm.py uses an absolute package import (`from src.core
# import ...`), so it must be run as a module from the repo root, not invoked
# by file path (matches sim_run_baselines_1D_kde.py / 1D_baselines_kde_sim.sh's
# pattern). Submit this job from the repo root so relative paths above resolve
# correctly.

python -m src.sim_run_baselines_1D_dpm \
    "$iteration" "$N" \
    "$dist_module_name" "$dist_module_path" \
    "$metric_output_path" "$sim_output_path" \
    "$density_folder" "$samples_folder" \
    "$DPM_MAX_COMPONENTS" "$DPM_MAX_ITER" "$DPM_COVARIANCE" \
    "$DPM_ALPHA" "$DPM_N_INIT" "$DPM_INIT_PARAMS" "$DPM_TOL"

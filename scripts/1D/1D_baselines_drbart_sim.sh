#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=research_group
#SBATCH --job-name=baselines_1D_drbart
#SBATCH --array=0-199
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00

source activate pt_sim

# Rscript is not on PATH by default in pt_sim or in a batch job's environment
# (module loads aren't inherited the way they are in an interactive shell) --
# drbart is installed under this R module's personal library
# (~/R/x86_64-pc-linux-gnu-library/4.3), so this module must match.
module load R/4.3.1-rhel8

# --- Editable per-run knobs -------------------------------------------------
# Defaults mirror run_one()'s own defaults in src/sim_run_baselines_1D_drbart.py
# (the single source of truth) -- override here only if this run should differ.
DRBART_NBURN=10000
DRBART_NSIM=1000
DRBART_NTHIN=10
DRBART_VARIANCE=ux
DRBART_MAX_N=5000   # DR-BART is slow -- see experiments/tune_1D_baselines.ipynb /
                     # experiments/tune_1D_drbart.ipynb for the timing/feasibility
                     # tradeoff behind this cap. Set to 0 to skip DR-BART entirely.
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

# sim_run_baselines_1D_drbart.py uses an absolute package import (`from src.core
# import ...`), so it must be run as a module from the repo root, not invoked
# by file path (matches sim_run_baselines_1D_kde.py / 1D_baselines_kde_sim.sh's
# pattern). Submit this job from the repo root so relative paths above resolve
# correctly.

python -m src.sim_run_baselines_1D_drbart \
    "$iteration" "$N" \
    "$dist_module_name" "$dist_module_path" \
    "$metric_output_path" "$sim_output_path" \
    "$density_folder" "$samples_folder" \
    "$DRBART_NBURN" "$DRBART_NSIM" "$DRBART_NTHIN" "$DRBART_VARIANCE" "$DRBART_MAX_N"

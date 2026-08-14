#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=research_group
#SBATCH --job-name=3D_sim
#SBATCH --array=0-199
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --time=02:00:00

source activate pt_sim

# --- Editable per-run knobs -------------------------------------------------
# Depths and iterations are independent of the array range below: rerunning
# this script later with a different DEPTHS (e.g. just "10") or a narrower
# --array range (e.g. new seeds only) extends the existing per-iteration
# metric CSVs in place rather than overwriting other depths/seeds -- see
# sim_run.main()'s get_or_create_column/extend_dataframe logic.
DEPTHS="1,2,3,4,5,6,7,8,9"
SAMPLE_SIZES="5000"
D=3
# -----------------------------------------------------------------------------

iteration=$SLURM_ARRAY_TASK_ID
dist_module_name=$1
dist_module_path=$2
sim_output_path=$3
metric_output_path=$4
depth_parallelization=$5
ncores=$6
n_iter=$7
density_folder=$8
samples_folder=$9

mkdir -p "$sim_output_path"

# sim_run.py uses an absolute package import (`from src import ...`), so it
# must be run as a module from the repo root, not invoked by file path.
cd "$(dirname "$0")/../.."

python -m src.sim_run \
    "$iteration" \
    "$SAMPLE_SIZES" \
    "$DEPTHS" \
    "$D" \
    "$dist_module_name" \
    "$dist_module_path" \
    "$sim_output_path" \
    "$metric_output_path" \
    "$depth_parallelization" \
    "$ncores" \
    "$n_iter" \
    "$density_folder" \
    "$samples_folder"

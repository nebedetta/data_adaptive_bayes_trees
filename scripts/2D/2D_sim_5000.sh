#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=research_group
#SBATCH --job-name=2D_n5000
#SBATCH --array=0-199
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17


source activate pt_sim

iteration=$SLURM_ARRAY_TASK_ID
sample_sizes="5000"
depths="1,2,3,4,5,6,7,8,9"
dist_module_name=$1
dist_module_path=$2

sim_output_path=$3
metric_output_path=$4
depth_parallelization=$5
ncores=$6
n_iter=$7
mkdir -p "$sim_output_path"

# sim_run_2D.py uses an absolute package import (`from src.core import
# ...`), so it must be run as a module from the repo root, not invoked by
# file path (matches 3D_sim.sh's pattern).
cd "$(dirname "$0")/../.."

python -m src.sim_run_2D "$iteration" "$sample_sizes" "$depths" "$dist_module_name" "$dist_module_path" "$sim_output_path" "$metric_output_path" "$depth_parallelization" "$ncores" "$n_iter"



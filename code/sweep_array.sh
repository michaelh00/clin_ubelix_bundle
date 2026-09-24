#!/bin/bash
# Final reconstruction sweep as a job array. Submit from the code folder:
#   sbatch sweep_array.sh
# Change --array to the number of chunks you want (0-19 = 20 chunks). More chunks =
# shorter tasks = less lost work if one fails. %4 limits how many run at once
# (check `sqos` for your actual GPU limit).
#SBATCH --job-name=clin-sweep
#SBATCH --nodes=1
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=FAIL

source "$SLURM_SUBMIT_DIR/env.sh"
cd "$PROJECT/code"
python run_sweep_chunk.py "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT"

#!/bin/bash
# Training job.  Submit from the code folder:  sbatch train.sh
# The script resumes automatically from the latest checkpoint in $CLIN_MODEL_DIR,
# so to continue after the time limit just submit it again (or chain, see README steps).
#SBATCH --job-name=clin-train
#SBATCH --nodes=1
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL

source "$SLURM_SUBMIT_DIR/env.sh"
cd "$PROJECT/code"

echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python clin_era5_reconstruction.py train
echo "End: $(date)"

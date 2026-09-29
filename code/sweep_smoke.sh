#!/bin/bash
# Smoke test for the sweep pipeline (8 days, small batch, 50 sampling steps).
# Submit from the code folder:  sbatch sweep_smoke.sh
# Runs target_period_sweep.py's __main__ block directly (NOT run_sweep_chunk.py,
# which is for the real sweep_array.sh job array).
#SBATCH --job-name=clin-sweep-smoke
#SBATCH --nodes=1
#SBATCH --account=gratis
#SBATCH --partition=gpu-invest
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=job_gpu_preemptable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out

source "$SLURM_SUBMIT_DIR/env.sh"
cd "$PROJECT/code"

echo "Host: $(hostname)  Job: $SLURM_JOB_ID  Start: $(date)"
python target_period_sweep.py
echo "End: $(date)"

#!/bin/bash
# Quick check that PyTorch sees the GPU and that data files are where they should be.
# Submit from the code folder:  sbatch test_gpu.sh
#SBATCH --job-name=clin-gputest
#SBATCH --nodes=1
#SBATCH --account=gratis
#SBATCH --qos=job_debug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

source "$SLURM_SUBMIT_DIR/env.sh"

nvidia-smi
python - << 'PY'
import os, torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
d = os.environ["CLIN_DATA_DIR"]
print("DATA_DIR:", d)
print("n T_ERA5 files:", len([f for f in os.listdir(d) if f.startswith("T_ERA5_")]), "(expect 86)")
for f in ["ERA5_elevation.nc", "ERA5_lsm.nc", "sin_doy_1667_2025.txt",
          "cos_doy_1667_2025.txt", "hist_obs.nc", "hist_obs_metadata.csv"]:
    print(f, "OK" if os.path.exists(os.path.join(d, f)) else "MISSING")
PY

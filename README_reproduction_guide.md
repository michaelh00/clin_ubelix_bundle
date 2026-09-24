# Running the CLIN diffusion model on UBELIX – reproduction guide

Goal: train the CLIN-style diffusion model (unconditional U-Net on de-warmed ERA5 daily temperature, 128×192 grid) **from scratch** on UBELIX, then evaluate it on the 1940s test decade with historical observation masks.
This guide reproduces the setup exactly as it was done, lists every file, and documents every change made to the original scripts.

## 0. Status – what has and has not been verified

| Part | Status |
|---|---|
| Environment setup, GPU job, data upload + loading | **Verified on UBELIX** |
| 15-min training rehearsal on real data (loading, training, validation, checkpoint) | **Verified on UBELIX** |
| Checkpoint resume / pruning, sweep logic incl. bug fix, array wrapper, merge | Verified only on small **synthetic data** (CPU sandbox), not yet on UBELIX |
| Full-length training, chained jobs, GPU sweep with a trained model | **Not yet run** |
| Real 1667–1939 reconstruction from observations | **Not yet written** (needs decisions, see §6) |

## 1. Requirements

- UBELIX account (staff/student/faculty campus account), university VPN when off campus.
- The 92 data files (§3, Step 4) and the bundle of scripts (§4).
- Roughly 4 GB for data, a few GB for the Python environment, ~190 MB per checkpoint. Check your quota with `quota`.
- Free tier (`--account=gratis`) is sufficient: batch jobs up to 96 h; `--qos=job_debug` for 20-minute tests.

Everything lives in one project folder (`$HOME/CLIN` below; any path works, incl. a Workspace path):

```
~/CLIN/
  code/      all scripts (job scripts are submitted from here; logs/ is created here)
  data/      the 92 input files
  model/     checkpoints, norm_stats.npy, train.log      (created by setup)
  results/   sweep output                                (created by setup)
  env/       Python venv                                 (created by setup)
```

## 2. Step-by-step

**Step 1 – Log in**
```bash
ssh <campus_account>@submit.unibe.ch
```

**Step 2 – Upload the scripts** (manually, e.g. WinSCP/scp) into `~/CLIN/code/`: all files listed in §4. Then:
```bash
mkdir -p ~/CLIN/code/logs
```

**Step 3 – Create the environment** (once)
```bash
conda deactivate 2>/dev/null        # only if a conda "(base)" env is active
cd ~/CLIN/code
bash setup_env.sh $HOME/CLIN
```
This creates the folders, loads `module load Python` (was Python 3.12.3-GCCcore-13.3.0), builds a venv in `~/CLIN/env`, `pip install`s torch, accelerate, netCDF4, numpy, pandas, tqdm, matplotlib (torch was 2.14.0+cu130), and writes the project path and Python module name into `env.sh`.
It must end with a line printing the package versions. Record exact versions for reproducibility: `source env.sh && pip freeze > ~/CLIN/requirements_lock.txt`.
To use another Python module: `PYTHON_MODULE=<name> bash setup_env.sh $HOME/CLIN` (see `module avail Python`).

**Step 4 – Upload the data** (manually) into `~/CLIN/data/` – all 92 files directly in that folder, no subfolders:

| File | Content |
|---|---|
| `T_ERA5_1940.nc` … `T_ERA5_2025.nc` (86 files) | var `T_ERA5` [°C, de-warmed]; `day_index` = days since 1667-01-01, **1-based** |
| `ERA5_elevation.nc` (var `elev`), `ERA5_lsm.nc` (var `lsm`) | static fields on the same grid |
| `sin_doy_1667_2025.txt`, `cos_doy_1667_2025.txt` | one value per line, 131122 lines; line i ↔ day_index i+1 |
| `hist_obs.nc` (var `hist_obs`), `hist_obs_metadata.csv` | historical station obs 1667–1939 (NaN = missing); csv has no header: id, lat, lon |

Grid: 128 lat (66.25→34.5, descending) × 192 lon (−12.25→35.5, ascending). Our files are stored as `(lon, lat, day_index)` / `(lon, lat)` / `(id, day_index)`; the scripts read variables **by dimension name** (`lat`, `lon`, `day_index`, `id`), so axis order does not matter, but these names must exist.
Verify:
```bash
cd ~/CLIN/code && source env.sh && python check_data.py
```
Expected: 92 files; T_ERA5_1940 `(366,128,192) float32`, ≈ −43…+38 °C, day_index 99711–100076; elevation ≈ −80…2430 m; `hist_obs (99710, 634)`, observed fraction ≈ 0.11.

**Step 5 – GPU smoke test** (always submit from `~/CLIN/code`)
```bash
cd ~/CLIN/code
sbatch test_gpu.sh
squeue --me
cat logs/clin-gputest_<jobid>.out      # after it finished
```
Expect: `cuda available: True`, `GPU: NVIDIA GeForce RTX 4090`, 86 T_ERA5 files, no `MISSING`. State `COMPLETED` in `sacct -X -S today`.

**Step 6 – 15-minute training rehearsal** (real data, free of cost)
```bash
sbatch --qos=job_debug --time=00:15:00 train.sh
tail -c 1000 logs/clin-train_<jobid>.out | tr '\r' '\n' | tail -4     # progress bar, it/s
grep -E "Temperature|Train samples|Device|params" logs/clin-train_<jobid>.out
srun --overlap --jobid <jobid> nvidia-smi
```
Expect (our run): `Train samples: 24107  Val samples: 3652`, `Model params: 16,462,657`, `Device: cuda`, ≈ 2 it/s, first validation + `best` checkpoint at step 1000. Final state `TIMEOUT` is normal. **Then clean up:** `rm -f ~/CLIN/model/*`.

**Step 7 – Real training**
```bash
sbatch train.sh                                   # up to 96 h
# to go beyond 96 h, chain jobs; each resumes from the latest checkpoint:
J1=$(sbatch --parsable train.sh)
J2=$(sbatch --parsable --dependency=afterany:$J1 train.sh)
```
Monitor:
```bash
squeue --me
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -f ~/CLIN/model/train.log                    # step, train loss, val loss every 1000 steps
srun --overlap --jobid <jobid> nvidia-smi
scancel <jobid>                                   # stop (also cancel dependent jobs)
```
Notes: the script has **no stopping criterion** (`EPOCHS = 1000` ≈ 753,000 steps ≈ 100+ h on an RTX 4090). Stop it yourself when validation loss plateaus. After a restart the epoch counter starts again at 1, but the step count, weights, optimizer and best validation loss are restored. Best model = the single `clin_era5_*_best_*.pt`; `norm_stats.npy` must always accompany it.

**Step 8 – Test-set evaluation (1940s masks) – needs a trained best checkpoint**
Smoke test (8 days, 50 steps):
```bash
sbatch --qos=job_debug --time=00:15:00 --account=gratis --partition=gpu --gres=gpu:rtx4090:1 \
  --nodes=1 --cpus-per-task=4 --mem=32G --output=logs/sweep_smoke_%j.out \
  --wrap="source $PWD/env.sh && cd \$PROJECT/code && python target_period_sweep.py"
```
Full sweep as job array (20 tasks, 4 at a time; task k handles every 20th day starting at k):
```bash
sbatch sweep_array.sh
python merge_sweep.py        # after all tasks finished (source env.sh first)
```
Resubmitting only redoes tasks whose CSV is missing. Adjust `--array`, `--time` after timing one task (its log prints `k/M days done`). **Rough cost estimate** (extrapolated from training speed, unverified): ~10⁵ days × 1000 steps ≈ 3·10⁶ batched forward passes ≈ 100–200 GPU-hours on one RTX 4090 → that is why it is split into an array.

**Step 9 – Real reconstruction of 1667–1939 from observations: not implemented yet.**

## 3. Job settings used (`train.sh`)

Account `gratis`, partition `gpu`, `--gres=gpu:rtx4090:1`, `--nodes=1` (UBELIX rejects `--mem` without `--nodes`), 8 CPUs, 24 GB RAM, 96 h. Other GPU types available in the partition: rtx3090 (max 4 CPUs/GPU), a100, h100, h200. `sqos` shows your limits.

## 4. File reference (bundle)

| File | Purpose |
|---|---|
| `setup_env.sh` | One-time: creates project folders + venv, records project path / Python module in `env.sh`. Run from `code/`. |
| `env.sh` | Sourced by every job: sets `PROJECT`, `CLIN_DATA_DIR`, `CLIN_MODEL_DIR`, `CLIN_RESULTS_DIR`, loads the Python module, activates the venv, extends `PYTHONPATH`. Must sit in the folder you submit from. |
| `check_data.py` | Quick sanity check of the uploaded data (Step 4). |
| `test_gpu.sh` | 10-min debug job: `nvidia-smi`, torch sees GPU, data files present. |
| `train.sh` | Training job; runs `python clin_era5_reconstruction.py train`; auto-resumes. |
| `clin_era5_reconstruction.py` | **Main module** (original + patches): schedule, U-Net, data loading, training, samplers, evaluation helper. Imported by all other Python scripts – keep this name. |
| `target_period_sweep.py` | Original sweep script (patched): applies each historical day's mask to a season-matched 1940s ERA5 field, reconstructs, records RMSE/MAE. |
| `run_sweep_chunk.py` | One array task of the sweep (every N-th day, offset k); writes `results/sweep_taskKKK_ofNNN.csv`, skips finished tasks. |
| `sweep_array.sh` | Slurm job array running `run_sweep_chunk.py`. |
| `merge_sweep.py` | Concatenates chunk CSVs → `target_period_reconstruction_results.csv` + error-vs-year plot. |
| `example_usage.py` | Original example (patched): unconditional sample + one reconstruction with a historical mask. |

## 5. Changes to the original Python scripts – and why

**`clin_era5_reconstruction.py`**

1. **Paths from environment** (`CLIN_DATA_DIR`, `CLIN_MODEL_DIR`; log file inside the model dir; dir created before logging starts). *Why:* original hard-coded `/data/era5_clin`; the log file is opened at import time, so every script importing the module crashed on UBELIX.
2. **`NUM_WORKERS = SLURM_CPUS_PER_TASK`** (default 8). *Why:* worker count must match the CPUs requested from Slurm.
3. **netCDF reader by dimension name** (`_read_var`, used in `load_era5_stack`, `load_static_field`, `load_historical_obs`); masked values → NaN; extra checks (fields vs day_index count, station count vs metadata, warning if station ids differ). *Why:* real files are stored `(lon, lat, day_index)` etc., not the documented order; the first job failed with a shape error. The old masked-array handling in `load_historical_obs` never had an effect (`np.asarray` drops the mask) and was replaced.
4. **Checkpointing**: `best_loss` stored in every checkpoint and restored on resume (`maybe_resume` now returns `(step, best_loss)`); only the newest `*_best_*.pt` is kept; `prune_checkpoints` never deletes it. *Why:* with chained 96-h jobs the original forgot the best loss after each restart (first validation counted as "best") and could delete the best checkpoint once 20 newer ones existed.

**`target_period_sweep.py`**

1. **Loads the best checkpoint** (`*_best_*.pt`) instead of the latest. *Why:* final evaluation should use the best-validation model.
2. **`start_offset` argument** (`target_days[start_offset::stride_days]`). *Why:* enables splitting the sweep into job-array tasks; the original could only stride from day 0 and saved nothing until the very end.
3. **Bug fix: known values.** Original fed the *historical station values* as known values but scored against a *modern* ERA5 field (unrelated weather). Now: mask from the historical day, values = modern ERA5 at those masked cells (as the docstring and `evaluate_with_historical_mask` describe). *This changes results – please confirm it is the intended design.*
4. Progress print uses `flush=True` (so Slurm logs update live).

**`example_usage.py`**: loads the best checkpoint (same reason as above). Nothing else changed.

Everything else (model, noise schedule, sampler, loss, splits, normalisation, hyper-parameters) is **unchanged**. `MAX_CHECKPOINTS` is still 20 (≈ 4 GB at 190 MB each); reduce if quota is tight: `sed -i 's/^MAX_CHECKPOINTS = 20/MAX_CHECKPOINTS = 5/' clin_era5_reconstruction.py`.

## 6. Measured numbers and open points

| Measurement (RTX 4090, batch 32) | Value |
|---|---|
| Parameters | 16.46 M |
| Speed | ≈ 2 it/s (753 steps/epoch ≈ 6 min) |
| Full run (753k steps) | ≈ 100–115 h ≈ two chained jobs |
| GPU memory | ≈ 22.5 of 23 GB (includes PyTorch cache; no OOM at first validation) |
| Host RAM (MaxRSS) | ≈ 8 GB (24 GB requested) |
| Checkpoint size | 189 MB |

Open points to decide (not changed in the code):
- **Hyper-parameter tuning:** full runs cost ~4 days each; use short runs (e.g. 5–10 % of steps) to rank settings, then one full run. Needs a config table / per-run output folders / step limit (not yet built).
- **Validation loss is noisy** (random noise levels and noise, shuffled batches, 50 batches) – a fixed seed and fixed noise levels would make model selection more reliable.
- **Normalisation statistics** are computed on 1940–2025, i.e. including validation and test years (small leakage).
- **Architecture:** the comment says /16 downsampling; the code only downsamples twice (/4), no attention, no EMA, fp32.
- **Real reconstruction:** confirm that `hist_obs` is de-warmed like the training data; decide single reconstruction vs ensemble.

## 7. Errors we hit and their fixes

| Symptom | Cause / fix |
|---|---|
| `sbatch: You must request --nodes with --mem` | add `#SBATCH --nodes=1` (already in the scripts) |
| Job fails after 1 s, exit code 139 (segfault at `import torch`) | venv Python needs its module: `env.sh` must `module load Python` before activating the venv |
| `ValueError: expected spatial shape (128, 192), got (128, 366)` | file axis order; fixed by the dimension-name reader |
| Job `PD` with reason "Nodes required … DOWN, DRAINED or reserved" | temporary; check `sinfo -p gpu`, or request another GPU type via `--gres=gpu:<type>:1` |
| Shell shows `>` after pasting a heredoc | closing `EOF` must start at column 1; use `check_data.py` instead |
| `TIMEOUT` on the 15-min run | expected for the rehearsal |

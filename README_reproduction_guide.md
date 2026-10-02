# Running the CLIN diffusion model on UBELIX – reproduction guide

Goal: train the CLIN-style diffusion model (unconditional U-Net on de-warmed ERA5 daily temperature, 128×192 grid) **from scratch** on UBELIX, then evaluate it on the 1940s test decade with historical observation masks.
This guide reproduces the setup exactly as it was done, lists every file, and documents every change made to the original scripts.

For the narrative of how we got here (bugs found, results, open questions) see `CLIN_PROJECT_STATUS.md` — that file is the living status log; this one is the step-by-step how-to.

## 0. Status – what has and has not been verified

| Part | Status |
|---|---|
| Environment setup, GPU job, data upload + loading | **Verified on UBELIX** |
| Full training run (1000 epochs / 753,000 steps) | **Done.** Best checkpoint `clin_era5_202609260333_best_00236.pt`, val_loss 0.0236 |
| Checkpoint resume / pruning across chained, preempted jobs | **Verified on UBELIX** across many preemption/resume cycles during the real run |
| Test-set evaluation sweep (1667–1939, full period) | **Done.** 99,710 days, see §8 and `CLIN_PROJECT_STATUS.md` §5 for results |
| Hyperparameter tuning | **Not started** |
| Real 1667–1939 reconstruction from observations | **Not yet written** (needs decisions, see §9) |

## 1. Requirements

- UBELIX account (staff/student/faculty campus account), university VPN when off campus.
- The 92 data files (§3, Step 4) and the bundle of scripts (§4).
- Roughly 4 GB for data, a few GB for the Python environment, ~190 MB per checkpoint. Check your quota with `quota`.
- Free tier (`--account=gratis`) is sufficient for everything done so far, but **read §3 before trusting GPU type / time-limit combinations** — several look valid but are silently rejected or unschedulable.

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
ssh <campus_account>@submit01.unibe.ch
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
**Important:** `env.sh` must `module load Python` *before* activating the venv — the venv's Python binary needs its originating environment module loaded for compiled dependencies (e.g. torch) to resolve. Without it, `import torch` segfaults (exit code 139) with no useful error.

**Step 4 – Upload the data** (manually) into `~/CLIN/data/` – all 92 files directly in that folder, no subfolders:

| File | Content |
|---|---|
| `T_ERA5_1940.nc` … `T_ERA5_2025.nc` (86 files) | var `T_ERA5` [°C, de-warmed]; `day_index` = days since 1667-01-01, **1-based** |
| `ERA5_elevation.nc` (var `elev`), `ERA5_lsm.nc` (var `lsm`) | static fields on the same grid |
| `sin_doy_1667_2025.txt`, `cos_doy_1667_2025.txt` | one value per line, 131122 lines; line i ↔ day_index i+1 |
| `hist_obs.nc` (var `hist_obs`), `hist_obs_metadata.csv` | historical station obs 1667–1939 (NaN = missing); csv has **no header: `id, lon, lat`** (not `id, lat, lon` — see §6 for how this was found; getting this wrong silently puts every station at the wrong grid cell) |

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
Expect: `cuda available: True`, a GPU name, 86 T_ERA5 files, no `MISSING`. State `COMPLETED` in `sacct -X -S today`.

**Step 6 – 15-minute training rehearsal** (real data, free of cost)
```bash
sbatch --qos=job_debug --time=00:15:00 train.sh
tail -c 1000 logs/clin-train_<jobid>.out | tr '\r' '\n' | tail -4     # progress bar, it/s
grep -E "Temperature|Train samples|Device|params" logs/clin-train_<jobid>.out
srun --overlap --jobid <jobid> nvidia-smi
```
Expect (our run): `Train samples: 24107  Val samples: 3652`, `Model params: 16,462,657`, `Device: cuda`, first validation + `best` checkpoint at step 1000. Final state `TIMEOUT` is normal. **Then clean up:** `rm -f ~/CLIN/model/*`.

**Step 7 – Real training**

Submit from `~/CLIN/code`, using the job config in §3 (H100 on the preemptible investor partition):
```bash
sbatch train.sh
```
`train.sh` auto-resumes from the latest checkpoint, so re-running it after a time-limit kill or preemption just continues. To chain several jobs so the run survives being killed/preempted without you watching it:
```bash
bash submit_chain.sh         # default: chains 3 jobs via --dependency=afterany; see file header for usage
```
Monitor:
```bash
squeue --me
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -f ~/CLIN/model/train.log                    # step, train loss, val loss every 1000 steps
srun --overlap --jobid <jobid> nvidia-smi
scancel <jobid>                                   # stop (also cancel dependent jobs)
```
**Resume correctness:** the epoch counter, step count, weights, optimizer, and best validation loss are all restored on resume, so `EPOCHS = 1000` (≈ 753,000 steps) is enforced as a true cumulative target across however many chained/preempted jobs it takes — a job that resumes after training is already complete exits in seconds rather than repeating it. Best model = the single `clin_era5_*_best_*.pt`; `norm_stats.npy` must always accompany it.
Our actual run took **~46 hours of wall time**, across a chain of jobs with multiple preemption/resume cycles (see §3 — the H100/preemptible config is schedulable but *will* be preempted by investor-priority jobs; this is expected and is why the resume correctness matters). Validation loss plateaued after ~230,000 of the 753,000 steps (~30% through) — worth considering for how long individual hyperparameter-tuning runs actually need to be (§9).

**Step 8 – Test-set evaluation (1940s masks) – needs a trained best checkpoint**

Smoke test first (8 days, 50 sampling steps, pinned to a historically dense decade so the station mask actually does something — see §6):
```bash
sbatch sweep_smoke.sh
squeue --me
cat logs/clin-sweep-smoke_<jobid>.out
```
Check: `n_obs` in the tens (not single digits), no `station(s) did not match` warning, RMSE/MAE in a plausible few-°C range, no crash.

Full sweep as a job array. **Before launching at full scale, read §3 for the resource request** — the values in `sweep_array.sh` as shipped here are the ones that actually worked (right-sizing this took a few iterations: see §7):
```bash
sbatch sweep_array.sh
squeue --me                                        # watch progress; %2 = 2 chunks run at a time
ls results/ | grep sweep_task | wc -l              # should reach 20
source env.sh && python merge_sweep.py             # once all 20 chunks exist
```
Resubmitting only redoes tasks whose CSV is missing (useful if a chunk's job fails or is lost). **Our full run:** 20 chunks, 2 concurrent (GRES cap under `gratis`), ~7–8 h/chunk → **~70–80 h (~3 days) wall time**. See §6 for the final results and `CLIN_PROJECT_STATUS.md` §5 for the full discussion.

**Step 9 – Real reconstruction of 1667–1939 from observations: not implemented yet.** See §9.

## 3. Job settings actually used

**`train.sh`** (training): `--account=gratis --partition=gpu-invest --gres=gpu:h100:1 --qos=job_gpu_preemptable --nodes=1 --cpus-per-task=8 --mem=24G --time=24:00:00`.
This was not the first thing tried — see §7 for why A100 and plain-partition H100 don't work under `gratis`. The preemptible queue can hand you an A100 80GB or H100 NVL 96GB instead of a plain H100; that's fine (more headroom). Jobs on this queue **can be preempted and requeued at any time** by investor-priority jobs — expected behavior, not a bug, and the reason the checkpoint/resume logic in `clin_era5_reconstruction.py` had to be made correct across restarts.

**`sweep_array.sh`** (evaluation sweep): `--account=gratis --partition=gpu --gres=gpu:rtx4090:1 --nodes=1 --cpus-per-task=4 --mem=8G --time=10:00:00 --array=0-19%2`.
The `gratis` QoS caps `rtx4090` at 2 concurrent per user (`sqos gpu` shows this under `job_gratis`'s Resource Limits), hence `%2`; it also caps total running memory×time (`MaxTRESRunMinsPerUser`, `mem=86400G`-minutes) and CPU-minutes — the shipped `--mem=8G --time=10:00:00` fits comfortably under both for 2 concurrent chunks. These numbers came from timing one real chunk (§7), not guesswork — if you change `batch_size`, `n_steps`, or the GPU type, re-time before trusting the resource request.

Run `sqos` / `sqos <partition>` / `sqos <qos_name>` to see your own current limits; they can differ by account.

## 4. File reference (bundle)

| File | Purpose |
|---|---|
| `setup_env.sh` | One-time: creates project folders + venv, records project path / Python module in `env.sh`. Run from `code/`. |
| `env.sh` | Sourced by every job: sets `PROJECT`, `CLIN_DATA_DIR`, `CLIN_MODEL_DIR`, `CLIN_RESULTS_DIR`, loads the Python module, activates the venv, extends `PYTHONPATH`. Must sit in the folder you submit from. |
| `check_data.py` | Quick sanity check of the uploaded data (Step 4). |
| `test_gpu.sh` | Debug job: `nvidia-smi`, torch sees GPU, data files present. |
| `train.sh` | Training job; runs `python clin_era5_reconstruction.py train`; auto-resumes. |
| `submit_chain.sh` | Submits a dependency chain of `train.sh` jobs (`--dependency=afterany`) so training survives being killed/preempted without manual resubmission. |
| `clin_era5_reconstruction.py` | **Main module** (original + patches): schedule, U-Net, data loading, training, samplers, evaluation helper. Imported by all other Python scripts – keep this name. |
| `target_period_sweep.py` | Original sweep script (patched): applies each historical day's mask to a season-matched 1940s ERA5 field, reconstructs, records RMSE/MAE. Its `__main__` block runs an 8-day smoke test pinned to a dense decade (1800–1809). |
| `sweep_smoke.sh` | Slurm job running `target_period_sweep.py`'s smoke test directly (not the array). |
| `run_sweep_chunk.py` | One array task of the sweep (every N-th day, offset k); writes `results/sweep_taskKKK_ofNNN.csv`, skips finished tasks. |
| `sweep_array.sh` | Slurm job array running `run_sweep_chunk.py`. See §3 for the resource request. |
| `merge_sweep.py` | Concatenates chunk CSVs → `target_period_reconstruction_results.csv` + error-vs-year plot. |
| `example_usage.py` | Original example (patched): unconditional sample + one reconstruction with a historical mask. |

## 5. Changes to the original Python scripts – and why

**`clin_era5_reconstruction.py`**

1. **Paths from environment** (`CLIN_DATA_DIR`, `CLIN_MODEL_DIR`; log file inside the model dir; dir created before logging starts). *Why:* original hard-coded `/data/era5_clin`; the log file is opened at import time, so every script importing the module crashed on UBELIX.
2. **`NUM_WORKERS = SLURM_CPUS_PER_TASK`** (default 8). *Why:* worker count must match the CPUs requested from Slurm.
3. **netCDF reader by dimension name** (`_read_var`, used in `load_era5_stack`, `load_static_field`, `load_historical_obs`); masked values → NaN; extra checks (fields vs day_index count, station count vs metadata, warning if station ids differ). *Why:* real files are stored `(lon, lat, day_index)` etc., not the documented order; the first job failed with a shape error.
4. **`hist_obs_metadata.csv` read as `(id, lon, lat)`, not documented `(id, lat, lon)`**, plus a hard validity check that raises if any station lands outside physically valid lat/lon ranges. *Why:* see §6 — every one of 634 stations landed outside the ERA5 domain when read in the documented order.
5. **Checkpointing**: `best_loss` stored in every checkpoint and restored on resume (`maybe_resume` now returns `(step, best_loss)`); only the newest `*_best_*.pt` is kept; `prune_checkpoints` never deletes it; an extra `torch.cuda.empty_cache()` after the checkpoint-save block (not just before validation). *Why:* with chained 24h jobs the original forgot the best loss after each restart (first validation counted as "best") and could delete the best checkpoint once 20 newer ones existed; the missing `empty_cache()` caused memory-fragmentation OOMs right after the first validation+checkpoint cycle.
6. **Epoch counter computed from `global_step` on resume** (`start_epoch = global_step // steps_per_epoch + 1`), with an early-return once `EPOCHS` is already reached. *Why:* the original always restarted the epoch loop at 1 regardless of how many steps had already run, so `EPOCHS` was never actually enforced across a chain of jobs — every resumed job silently redid the full target length.

**`target_period_sweep.py`**

1. **Loads the best checkpoint** (`*_best_*.pt`) instead of the latest. *Why:* final evaluation should use the best-validation model.
2. **`start_offset` argument** (`target_days[start_offset::stride_days]`). *Why:* enables splitting the sweep into job-array tasks; the original could only stride from day 0 and saved nothing until the very end.
3. **Bug fix: known values.** Original fed the *historical station values* as known values but scored against a *modern* ERA5 field (unrelated weather). Now: mask from the historical day, values = modern ERA5 at those masked cells (as the docstring and `evaluate_with_historical_mask` describe).
4. **Smoke test pinned to a dense decade** (1800–1809) instead of the start of the period. *Why:* 1667 has ~2 stations/day (see §6); a smoke test there barely exercises the mask.
5. Progress print uses `flush=True` (so Slurm logs update live).

**`example_usage.py`**: loads the best checkpoint (same reason as above). Nothing else changed.

Everything else (model, noise schedule, sampler, loss, splits, normalisation, hyper-parameters) is **unchanged**. `MAX_CHECKPOINTS` is still 20 (≈ 4 GB at 190 MB each); reduce if quota is tight: `sed -i 's/^MAX_CHECKPOINTS = 20/MAX_CHECKPOINTS = 5/' clin_era5_reconstruction.py`.

## 6. Measured numbers and results

**Training** (H100/preemptible, see §3):

| Measurement | Value |
|---|---|
| Parameters | 16.46 M |
| Full run | 1000 epochs = 753,000 steps, **~46 h wall time** (chained, preempted jobs) |
| Best validation loss | **0.0236**, reached at step 230,000 (~30% through training) — loss plateaued for the remaining ~70% of compute |
| Checkpoint size | 189 MB |

**`hist_obs_metadata.csv` column-order bug:** found via a domain-bounds sanity check. Station lat/lon read in the documented `(id, lat, lon)` order put **0 of 634** stations inside the valid ERA5 domain (34.5–66.25°N × 12.25°W–35.5°E); reading `(id, lon, lat)` put **all 634** inside it. Confirmed not to affect the completed training run — station metadata is only used by the sweep and `example_usage.py`, never during training.

**Station density across the target period** (from `hist_obs`/`hist_obs_metadata.csv`, decade-by-decade):

| Period | Mean stations/day | Days with ≥10 stations |
|---|---|---|
| 1667–1746 | 1.1 – 7.3 | ~0% |
| 1747–1766 | 12.8 | 78% |
| **1767 onward** | **25.6 → 362.8** (by 1927–1946) | **~100%** |

**Full test-set evaluation sweep** (99,710 days, 1667–1939, each day's real station mask applied to a season-matched 1940s ERA5 field):

| | RMSE °C | MAE °C |
|---|---|---|
| Overall mean | 3.93 | 2.99 |
| 1667–1746 (sparse, ~3 stations/day) | 3.99 | 3.04 |
| 1847–1939 (dense, ~171 stations/day) | 3.83 | 2.91 |

**Headline finding:** reconstruction error is only weakly sensitive to station density — a ~50× increase in stations per day (1667–1746 → 1847–1939) buys only a ~4% reduction in RMSE. The yearly-mean error drifts gently from ~4.1°C (1667) to ~3.6°C (1939) with no visible inflection around 1747–1767 where density actually jumps. Possible explanations (not yet distinguished): the model may lean mainly on its learned unconditional prior rather than strongly propagating information from nearby masked cells; or far-from-any-station cells (still common even at high density, given a 128×192 grid) may be washing out a real but localized improvement near stations. A follow-up stratifying error by distance-to-nearest-station (rather than just era/station-count) would help tell these apart — not yet done. Full results: `results/target_period_reconstruction_results.csv` and `results/reconstruction_error_vs_target_period.png`.

Other notes:
- **Hyper-parameter tuning:** not yet started; the early validation-loss plateau above suggests short runs (10–20% of full length) may be enough to rank configs — worth confirming with the original author before committing a tuning budget.
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
| GPU OOM right after the first validation+checkpoint save, iteration speed collapsing from ~2 it/s to seconds/it | Memory fragmentation on a near-full GPU | moved to a larger-memory GPU + added `torch.cuda.empty_cache()` after checkpoint save |
| `sbatch: Requested time limit is invalid (missing or exceeds some limit)` requesting A100 | A100 fully banned under `gratis` (`a100=0`); Slurm surfaces this as a confusing time-limit error rather than a clear rejection | switch to H100 |
| `QOSMaxGRESPerUser` requesting H100 under the plain `gpu` partition, even with nothing else running | H100 only actually schedulable via `--partition=gpu-invest --qos=job_gpu_preemptable` | switch partition/QoS (§3) |
| Every resumed training job silently redid the full 1000-epoch target | epoch counter always reset to 1 on resume, ignoring `global_step` | compute `start_epoch` from `global_step`; early-exit if already complete |
| `[WARNING] 634 station(s) did not match an ERA5 grid cell within 0.001 deg` (all 634/634) | `hist_obs_metadata.csv` columns are `(id, lon, lat)`, not documented `(id, lat, lon)` | diagnosed via domain-bounds check (§6); fixed column order + added a hard validity check |
| Sweep job seemed to vanish from `squeue` almost instantly | two distinct causes seen on different occasions: (1) genuinely failed — wrong working directory at submit time, so relative paths resolved wrong; (2) genuinely fine — the smoke test legitimately finishes in well under a minute | always check the log, don't assume from `squeue` alone; `cd` into the code folder before `sbatch` |
| `sweep_array.sh` tasks stuck `PD` with reason `MaxMemoryRunMinsPerUser` | per-task `--mem × --time` request (originally 32G/24h) too generous for several concurrent tasks under the `gratis` memory-time budget | timed one real chunk, right-sized to `--mem=8G --time=10:00:00` (§3) |
| Even after fixing memory budget, still only 2 (not 4) tasks ran concurrently, reason `JobArrayTaskLimit` | `gratis` caps `rtx4090` at 2 concurrent per user (`sqos gpu`) | changed `--array=0-19%4` to `%2` to match the real cap, rather than let Slurm keep failing to start a 3rd/4th |
| Shell shows `>` after pasting a heredoc | closing `EOF` must start at column 1; use `check_data.py` instead, or a `python - << 'EOF' ... EOF` block with no leading indentation |
| `TIMEOUT` on the 15-min rehearsal | expected |

## 8. Open points before Step 9 (real reconstruction)

Not yet decided or built:
- Hyperparameter tuning plan (config grid, per-run output folders, step-limited runs, comparison script).
- Whether `hist_obs` is de-warmed the same way as the training ERA5 data (needs confirming before real reconstruction).
- Single reconstruction vs. ensemble for the real 1667–1939 run.
- Follow-up: stratify sweep error by distance-to-nearest-station to understand the weak density-dependence finding in §6.

See `CLIN_PROJECT_STATUS.md` for the up-to-date narrative and next steps.

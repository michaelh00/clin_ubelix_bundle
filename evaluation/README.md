# CLIN: data analysis, baseline and diffusion re-evaluation

This folder gives you the data analysis, a regression baseline and a re-evaluation of the trained diffusion model, all on the same test cases.

The analyses were done on 2026-10-08. They use the training run documented in `CLIN_PROJECT_STATUS.md` (in the code repo) and the checkpoint `clin_era5_202609260333_best_00236.pt`.

**Start with `NEXT_STEPS.md`.** It has a one-page summary of what we found and what to do next, followed by a detailed task list written for Claude.

## What's here

| File | What it shows | Runtime / needs |
|---|---|---|
| `01_data_checks.ipynb` | Everything about the data that can go wrong for training: file formats, de-warming, normalisation, independent sample size, station coverage over time, station-vs-ERA5 consistency, mask sparsity. Also a first baseline comparison. Every finding is flagged `ISSUE`/`WARN`/`INFO` and summarised at the end. | ~10 min, ~8 GB RAM |
| `02_hist_obs_outliers.ipynb` | The 347 suspicious station values (\|z\| > 6 or jumps > 20 °C/day), each buddy-checked against stations within 300 km and plotted. | ~1 min |
| `03_ridge_baseline.ipynb` | The baseline every model should beat: per-cell seasonal normal plus linear regression of the anomaly field on the observed cells (optimal interpolation). Explained step by step, tuned on the 1950s, evaluated on the 1940s. | ~3 min, ~16 GB RAM (seasonal covariances) |
| `04_diffusion_reevaluation.ipynb` | The trained diffusion model re-run on the same cases: land/sea, distance to station, regions, 8-member ensembles, and an unconditional-prior and memorisation check. Reads precomputed results, so no GPU is needed. | ~3 min, ~8 GB RAM |
| `clin_common.py` | Shared loading, the fixed evaluation cases and scoring. All notebooks use it. | |
| `run_diffusion_eval.py` | GPU script that produced `results/diffusion_*.npz`. Rerun it for a new model or sampler. | ~16 s per reconstruction on a laptop GPU |
| `results/` | `eval_cases.csv` (the 240 shared test cases), the diffusion outputs, and the CSV/npz files written by notebooks 02 and 03. | |

The notebooks are saved with their outputs, so you can read them without running anything.

## Setup

```bash
python -m venv clin-env            # keep it outside OneDrive/Dropbox folders
clin-env/Scripts/activate          # Windows; on Linux/macOS: source clin-env/bin/activate
pip install numpy pandas xarray netCDF4 matplotlib openpyxl jupyter
# only for run_diffusion_eval.py:
pip install torch accelerate tqdm  # use the CUDA build matching your GPU, see pytorch.org
```

**Data.** Point `CLIN_DATA_DIR` to the folder holding `T_ERA5_*.nc`, `hist_obs.nc`, `hist_obs_metadata.csv`, `ERA5_*.nc` and the `sin/cos_doy` files. Alternatively, put that folder (named `Final CLIN variables`) next to this one or one level up.

**For `run_diffusion_eval.py` only:**
- Clone `https://github.com/michaelh00/clin_ubelix_bundle` into `./clin_ubelix_bundle`, or set `CLIN_CODE_DIR` to its `code/` folder.
- Put the checkpoint and `norm_stats.npy` in `./model/`, or set `CLIN_CKPT`.

**Run order.** Run 01–03 in any order. 04 needs `results/ridge_cases.npz`, which 03 writes.

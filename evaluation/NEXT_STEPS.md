# CLIN: findings and next steps

## Part A: summary

### What we found

**The data is in good shape:**
- complete files, consistent grids, no NaNs;
- no Julian-calendar dates left unconverted;
- station climatologies within ~0.2 °C of their ERA5 grid cells.

There are a few things to handle; see `01_data_checks.ipynb`:
- About 88% of the variance in the training fields is the fixed seasonal and spatial pattern. The model spends most of its effort re-learning that instead of the weather.
- The 24k training days amount to only about 3k independent fields.
- The de-warming leaves a small residual trend. It is about +0.05 °C/decade on average and up to 0.15 °C/decade in northern Norway, Russia and around the Baltic.
- There is a ~0.5 °C level step between the station data (1920–39) and ERA5 (1940–59).
- Station coverage changes enormously over time. Before 1750, ~3 stations/day cover ~20% of land within 500 km; after 1900 it's 99%.
- About 90 station values look like data errors (`02_hist_obs_outliers.ipynb`).

**The trained diffusion model is a good generator but ignores the stations** (`04_diffusion_reevaluation.ipynb`):
- Its unconditional samples are realistic, and it does **not** memorise training days.
- Its reconstructions are no better than a random field for the same calendar day.
- Averaging 8 samples only reaches the seasonal-normal ("climatology") level.
- The error is the same next to a station as 500 km away.
- The ensemble spread doesn't shrink as stations are added.

**The cause is how observations enter the model.** It is trained without any observations. At sampling time, the station values are pasted into 0.01–2% of the pixels at each denoising step ("replacement inpainting"), and that information doesn't spread to the neighbouring cells.

**A simple linear method already does much better on the same test cases** (`03_ridge_baseline.ipynb`): the seasonal normal plus a regression of the anomaly field on the station cells.

| Era | Stations/day | Climatology | Ridge baseline | Diffusion, 1 sample | Diffusion, 8-member mean* |
|---|---|---|---|---|---|
| 1667–1746 | ~3 | 2.66 | **2.24** | 3.89 | 2.84 |
| 1747–1766 | ~13 | 2.95 | **1.91** | 4.29 | 3.27 |
| 1767–1846 | ~30 | 2.84 | **1.63** | 3.78 | 3.61 |
| 1847–1939 | ~170 | 2.66 | **1.08** | 3.63 | 2.74 |

RMSE in °C over unobserved cells, land and sea, on 1940s ERA5 fields with real historical station masks. *The ensemble column covers 10 cases per era, not 60.

### What to do next (in order)

1. **Fix the conditioning before anything else.** Don't tune hyperparameters yet: the validation loss measures the unconditional model, which is already fine.
   - First try **RePaint resampling**. It's already in the code and needs no retraining.
   - Then try **guided sampling**, where each step is nudged toward the observations.
   - The most likely real fix is to **retrain or fine-tune the model with the station values and mask as extra inputs**. Train with masks taken from the real historical station network.
2. **Clean up the training data setup:**
   - train on anomalies (field minus seasonal normal);
   - compute normalisation statistics on training years only;
   - ask the data provider whether de-warming used one curve for the whole domain. If so, de-warm per cell.
3. **Evaluate every model the same way:** the shared 240 test cases, ensemble mean + spread, land/sea, distance to the nearest station. The bar to beat is the ridge baseline, not climatology.
4. **Prepare for the real reconstruction (1667–1939):**
   - filter the flagged outliers;
   - condition on station *anomalies*, so that station-vs-grid-cell offsets don't leak in;
   - settle the 0.5 °C de-warming step.
5. **Only then tune hyperparameters,** selecting checkpoints by reconstruction skill on the 1950s.

Part B below spells all of this out for Claude. Open Claude Code in this folder and point it at `NEXT_STEPS.md`, Part B.

---

## Part B: detailed tasks for Claude

> **Context for Claude.** You are helping a PhD student improve the CLIN diffusion model. CLIN reconstructs daily European 2 m temperature fields for 1667–1939 from sparse historical station observations; the model is trained on de-warmed ERA5 for 1940–2025. Read `README.md` and skim the summary cells at the end of the four notebooks before changing anything. Training code: `https://github.com/michaelh00/clin_ubelix_bundle`, mainly `code/clin_era5_reconstruction.py` and `code/target_period_sweep.py`; the repo also holds `CLIN_PROJECT_STATUS.md`, which records the UBELIX setup and earlier bug fixes. Work in small, verifiable steps, and report numbers against the baselines below. Ask the student before launching long cluster jobs.

### B.0 Facts you need (verified; don't re-derive)

**Data conventions** (all handled in `clin_common.py`; reuse it):
- `day_index` is **1-based**: 1 = 1667-01-01. Its netCDF attribute says "days since", so never let xarray decode it.
- The netCDF variables are stored as `(lon, lat, day_index)`, `(lon, lat)` and `(id, day_index)`. **Read them by dimension name.** `lat` is descending.
- `hist_obs_metadata.csv` has no header, and its columns are **(id, lon, lat)**, not (id, lat, lon) as `Summary_of_final_CLIN_variables.xlsx` says.
- There are 634 stations, each sitting exactly on its own ERA5 grid-cell centre.
- The sin/cos day-of-year files encode **day-of-leap-year**: 2π·doly/366, so Mar 1 is always day 61. Seasonal matching uses doly.
- Grid: 128 lat × 192 lon, 0.25°, 34.5–66.25 N, 12.25 W–35.5 E. 55% of cells are land (lsm > 0.5).

**Model and training** (`clin_era5_reconstruction.py`):
- **Network:** `CLINUNet`, 16.46M parameters (BASE_DIM=64, CHANNEL_MULTS=(1,2,4,4)), v-prediction, continuous log-SNR `LambdaSchedule` (λ ∈ [−20, 20]).
- **Schedule docstring is inverted:** it says u=0 is most noise, but `u_to_lambda(0)` = +20, i.e. the *least* noise. The sampler runs u from 1 to 0, which is correct.
- **Inputs:** `forward(x_t, lam, static_cond)` concatenates `x_t` (1 channel) with `static_cond` (4 channels: normalised elevation, lsm, sin_doy, cos_doy) and feeds them to a 1×1 `init_conv`. **New conditioning channels can be added by extending `static_cond`** and `N_STATIC_COND`.
- **Split:** train 1960–2025, val 1950–1959, test 1940–1949.
- **Normalisation:** one global z-score. `norm_stats.npy` = (9.449, 8.785) °C, computed over **1940–2025, including val and test years**: a small leak to fix.
- **Training:** unconditional (no masks), batch 32, lr 1e-4, 1000 epochs. The best val loss (0.0236) came at step 230k of 753k.
- **Sampling:** `clin_reconstruct()` is a deterministic DDIM-style update. Before each step it overwrites the masked cells with the noised known values. RePaint is available via `repaint_jump` / `repaint_resample` (default off). `SAMPLING_STEPS=1000`.

**Evaluation protocol** (`results/eval_cases.csv`; the same as `target_period_sweep.py`):
- There are 240 cases, 60 per era: 1667–1746, 1747–1766, 1767–1846, 1847–1939.
- Each case is a historical day's real station mask (`hist_row`) laid over a random 1940s ERA5 field with the same doly (`test_row` indexes the 1940–1949 stack).
- The model gets the true 1940s values at the masked cells. Scoring is over **unobserved cells only**, with `clin_common.score()` (all/land/sea).

**Reference numbers to beat** (RMSE °C, unobserved cells, from `results/ridge_scores.csv` and notebook 04):

| Era | Ridge, all | Ridge, land | Ridge, sea | Climatology, all | Diffusion now, 1 sample | Diffusion now, ens. mean |
|---|---|---|---|---|---|---|
| 1667–1746 | 2.24 | 2.68 | 1.50 | 2.66 | 3.89 | 2.84 |
| 1747–1766 | 1.91 | 2.28 | 1.29 | 2.95 | 4.29 | 3.27 |
| 1767–1846 | 1.63 | 1.88 | 1.21 | 2.84 | 3.78 | 3.61 |
| 1847–1939 | 1.08 | 1.22 | 0.87 | 2.66 | 3.63 | 2.74 |

By distance to the nearest observed cell (all eras, all cells):
- **Ridge:** 0.60 / 1.02 / 1.46 / 1.93 °C (0–100 / 100–250 / 250–500 / >500 km).
- **Diffusion now:** 3.88 / 4.08 / 4.09 / 3.50 °C.

Current diffusion ensemble spread is about 2.7 °C in every era; spread/skill is 0.76–1.0.

**Result files:**
- `results/diffusion_single.npz`: `recon` (240,1,128,192) float16 °C, `done`, `case`.
- `results/diffusion_ensemble.npz`: `recon` (40,8,…), `done`, `case` (indices into `eval_cases.csv`).
- `results/diffusion_uncond.npz`: `samples`, `dates`.
- `results/ridge_cases.npz`: `pred`, `post_sd`, `clim`, `truth` (240,128,192) °C.

### B.1 Reproduce first (½ day)

1. Set up the environment (`README.md`), clone the repo, and get the checkpoint.
2. Run `python run_diffusion_eval.py bench --batch 16` (expect ~16 ms per sample-step on a laptop GPU), then rerun ~16 cases.
3. Confirm the single-sample RMSE matches `results/diffusion_single.npz` within ±0.3 °C (the samples are stochastic).

**Done when:** a short note records the timing and the per-case RMSE agreement.

### B.2 Test RePaint resampling (no retraining; 1–2 days)

1. Add a `--repaint JUMP RESAMPLE` option to `run_diffusion_eval.py`, writing `results/diffusion_repaint_<jump>_<resample>.npz`. Pass the values through to `clin_reconstruct`.
2. Start with `jump=10, resample=10` (~10× the sampling cost). Use 10 cases per era × 4 members; the ensemble subset is the first 10 per era via `cases.groupby("era").head(10)`.
3. Then try `jump=20, resample=5`.
4. **Optional:** the current update is deterministic (η=0). RePaint was designed for stochastic DDPM sampling. Try a stochastic variant (η=1: add fresh noise in the update) with and without resampling.
5. Score each run like notebook 04 (copy its scoring cells or factor them into `clin_common.py`):
   - ensemble-mean RMSE all/land/sea per era;
   - RMSE by distance band;
   - spread vs number of observed cells.

**Success criteria:**
- error near stations (<100 km) drops clearly below the >500 km error;
- the ensemble spread shrinks as stations are added;
- the ensemble-mean RMSE moves from climatology toward ridge.

Report the result either way. If it doesn't help, that's an informative negative.

### B.3 Guided (posterior) sampling, DPS-style (2–4 days)

1. Implement a new sampler next to `clin_reconstruct` (don't modify the original). At each step:
   - run the network with gradients enabled;
   - compute `x0_hat` via `schedule.x0_eps_from_v`;
   - compute the misfit `L = ||M ⊙ (x0_hat − y)||² / (2 σ_obs²)`;
   - take the normal DDIM step, then subtract `ζ · ∇_{x_t} L`, normalised as in Chung et al. 2023 (DPS).
2. Memory roughly triples (backprop through the U-Net), so use a smaller batch.
3. Tune ζ (and σ_obs) on **validation** cases only:
   - build them with `clin_common.make_cases(...)` on 1950s fields, `seed=1, n_per_era=10`, as done in notebook 03;
   - never tune on the test cases.
4. Evaluate on the shared test cases with the same metrics as B.2.

### B.4 Conditional fine-tuning or retraining (the most likely real fix; 1–2 weeks incl. cluster time)

**Inputs.** Extend `static_cond` with 2 channels:
- `known`: normalised value at observed cells, 0 elsewhere;
- `mask`: 1 at observed cells.

Optionally add a third: the ridge estimate (see `03_ridge_baseline.ipynb`, `predict()`), giving the network a good linear first guess.

**Training masks** come from the real network:
- For each training example, draw a random historical day from `hist_obs`, **uniformly over the four eras** so that sparse days are not under-represented.
- Take its station cells, and fill `known` from the *training field itself* at those cells.
- Add **mask dropout** (p ≈ 0.1, empty mask) so the model still works unconditionally.
- Optionally add small Gaussian noise (σ ≈ 0.5 °C) to `known`, to mimic station-vs-cell representativeness error.

**Warm start.** Load the existing checkpoint and **zero-initialise the new input channels' weights in `init_conv`**. The model then starts exactly as the current unconditional model and only learns to use the new inputs. Do this first: it's cheap.

**Checkpoint selection** must use reconstruction skill, not v-loss:
- At each check, run a reduced sampler (e.g. 100 steps, 2 members) on ~40 fixed validation cases (1950s fields).
- Keep the checkpoint with the best ensemble-mean RMSE.
- Log CRPS too if cheap.

**Then the from-scratch variant (experiment B):**
- train on **anomalies**: subtract the per-cell harmonic climatology fitted on train years (`clin_common.harmonics`, as in notebooks 01/03), and optionally divide by the per-cell SD;
- compute normalisation statistics on train years only;
- consider a smaller network (BASE_DIM=32, ~4M params) and an EMA of the weights.

Compare A (warm start) and B on the shared test cases.

**UBELIX:**
- The working config is in `CLIN_PROJECT_STATUS.md` §4: `--partition=gpu-invest --qos=job_gpu_preemptable --gres=gpu:h100:1 --time=24:00:00`.
- Jobs get pre-empted, so keep the existing checkpoint/resume logic.
- Val loss plateaued at ~30% of the original run, so 100–250k steps should be enough to compare variants.

### B.5 Data hygiene (can run in parallel with B.2–B.4)

1. **Normalisation leak:** in `train()`, compute `t_mean, t_std` on `TRAIN_YEARS` only.
2. **De-warming:**
   - Draft a short question for the data provider: how was ERA5 de-warmed (one curve for the domain, or per cell)? Is `hist_obs` de-warmed the same way?
   - Evidence: residual trend +0.05 °C/decade on average, up to +0.15 °C/decade in northern Norway, western Russia and the southern Baltic; a 0.5 °C level step between stations (1920–39) and ERA5 (1940–59); and a diffusion warm bias of about +1 °C in the same northern and eastern regions.
   - If you prototype a fix, regress each cell's annual (or seasonal) means on a smoothed global-mean temperature curve and remove the scaled signal per cell. **Don't** fit a free linear trend per cell: it would also remove real multi-decadal variability.
3. **Outliers** (`results/hist_obs_outlier_candidates.csv`):
   - before any real-data reconstruction, drop `verdict == "isolated"` rows that are also `one_day_spike` or carry a transcription hint;
   - list the `no neighbours` cases for the data provider.

### B.6 Before the real 1667–1939 reconstruction (plan step 4 in STATUS)

- **Condition on station anomalies, not raw values:** station value − station's own seasonal normal + ERA5 cell normal. This removes the ~0.2 °C median station-vs-cell offset and part of the 0.5 °C step.
- **Treat observations as noisy:** use σ_obs in DPS, or noise augmentation in B.4. For the ridge baseline, set λ from the observation-error variance instead of 1e-3.
- **Report reconstructions as ensembles** (≥ 8 members) with spread.
- **Report skill as:**
  - a function of the year, from the test-set sweep;
  - land/sea separately;
  - stratified by distance to the nearest station.

### B.7 Evaluation standard (apply to every experiment)

- **Test cases:** always the 240 shared cases in `results/eval_cases.csv`, or the full sweep. **Never tune on them**; use 1950s validation cases for that.
- **Report per era:**
  - single-sample and ensemble-mean RMSE/MAE, for all, land and sea;
  - RMSE by distance band (0–100 / 100–250 / 250–500 / >500 km);
  - spread/skill.
- **Compare against the ridge row in the B.0 table.** A model that doesn't beat ridge on ensemble-mean RMSE in at least the dense eras isn't an improvement yet. The *added* value of diffusion should then show in calibrated spread and realistic extremes. Check the cold/warm tails, especially where anomalies are skewed (Adriatic, Baltic and eastern Mediterranean cold tails; Atlantic coasts of Iberia and Morocco warm tails).
- Keep every new result as an `.npz` in `results/` in the same layout, so notebook 04 can be pointed at it.

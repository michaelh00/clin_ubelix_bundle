# CLIN on UBELIX – Project Status

**Last updated:** 2026-09-29

This file tracks _where the project actually is_ — what's done, what broke and how it was fixed, and what's next. It complements `README_reproduction_guide.md`, which is the step-by-step setup guide (currently a bit behind — see "Known gaps in the setup guide" at the bottom). Append new dated sections as the project progresses; don't rewrite history above.

---

## 1. The plan (as given by the original author)

1. Train the CLIN diffusion model from scratch.
2. Repeat with different hyperparameters, pick the best by validation loss.
3. Apply the fully-tuned model to the test set (1940s ERA5 fields with historical-availability masks overlaid) to measure reconstruction accuracy.
4. Run real inference on the true historical period (1667–1939) using only the real observations.

## 2. Current status

| Step                              | Status                                                                                                                |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 1. Train from scratch             | **Done.** One full training run completed on UBELIX (see §4).                                                         |
| 2. Hyperparameter tuning          | **Not started.** Needs a tuning plan (config grid, short-run budget) — see §6.                                        |
| 3. Test-set evaluation sweep      | **In progress.** Pipeline verified end-to-end (bug fixed, smoke-tested); full sweep over 1667–1939 launched (see §5). |
| 4. Real historical reconstruction | **Not started.** Script doesn't exist yet; needs a few design decisions (see §6).                                     |

## 3. Environment & setup — what had to change vs. the original scripts

The original scripts assumed a different filesystem layout and made assumptions about the data files that didn't hold on UBELIX. All fixes live in `clin_era5_reconstruction.py` (the shared module everything else imports) unless noted.

1. **Paths from environment variables.** `CLIN_DATA_DIR`, `CLIN_MODEL_DIR` (and `CLIN_RESULTS_DIR` for the sweep) instead of hardcoded paths; `NUM_WORKERS` from `SLURM_CPUS_PER_TASK`. Set in `env.sh`, sourced by every job script.
2. **netCDF variables read by dimension name, not position** (`_read_var()` helper). The real data files are stored as `(lon, lat, day_index)` for the ERA5 stacks, `(lon, lat)` for the static fields, and `(id, day_index)` for `hist_obs` — all transposed relative to what the original code assumed. Reading by dimension name makes this irrelevant.
3. **`hist_obs_metadata.csv` columns are `(id, lon, lat)`, not `(id, lat, lon)` as originally documented.** Found via a domain-bounds sanity check (see §7) — every one of 634 stations landed outside the valid ERA5 lat/lon domain when read in the documented order, and all 634 landed inside it when the columns were swapped. Fixed in `load_historical_obs()`, plus a defensive check that now raises immediately if any station lands outside physically valid lat/lon ranges. **This does not affect the completed training run** — station metadata is only used by `target_period_sweep.py` and `example_usage.py`, never during training.
4. **Checkpointing hardened for long, interrupted runs:** `best_loss` is now stored in and restored from every checkpoint (the original forgot it on every job restart, so the first validation after a resume always counted as "best"); the single newest `*_best_*.pt` is protected from pruning.
5. **Epoch counter now correct across resumed jobs.** The original always restarted the epoch loop at 1 regardless of how many steps had already run, so `EPOCHS` was never actually enforced across a chain of jobs — every resumed job silently redid the full target length. Fixed: `start_epoch` is computed from `global_step`, and a job that resumes after training is already complete now exits in seconds instead of repeating it.
6. **GPU memory fragmentation fix:** added `torch.cuda.empty_cache()` after the checkpoint-save block (previously only called before validation), which fixed an OOM that appeared reliably right after the first validation+checkpoint cycle.

## 4. Training run (completed)

- **Checkpoint:** `clin_era5_202609260333_best_00236.pt` (+ `norm_stats.npy`, required alongside it).
- **Length:** 1000 epochs = 753,000 steps.
- **Wall time:** ~46 hours, across a chain of jobs (see "GPU/QoS reality" below) submitted via `submit_chain.sh`, surviving multiple preemptions and time-limit kills, each resuming from checkpoint.
- **Best validation loss:** 0.0236, reached at step 230,000 (~30% through training). **Validation loss plateaued for the remaining ~70% of compute** — worth relaying to the original author before deciding how long individual tuning runs need to be; it suggests much shorter runs may be enough to rank hyperparameter settings.
- **Model:** 16.46M parameters.

### GPU/QoS reality on UBELIX (`gratis` account)

This took some trial and error and is worth recording so the next person doesn't repeat it:

- `a100`, `h200`, `rtx3090` are **fully banned** for the `gratis` account (`=0` in every relevant QoS) — requesting them fails with a confusing "Requested time limit is invalid" error rather than a clear "not permitted" message.
- `rtx4090` (2 GPUs) and `h100` (1 GPU) are nominally allowed under `gratis`, but **H100 is not actually schedulable via the plain `gpu` partition** — it fails with `QOSMaxGRESPerUser` even with nothing else running.
- **Working config:** `--partition=gpu-invest --qos=job_gpu_preemptable --gres=gpu:h100:1 --time=24:00:00`. This allows up to h100=4/a100=1, max 24h, but jobs can be pre-empted and requeued at any time by investor-priority jobs — which is exactly why the checkpoint/resume correctness fixes in §3 mattered. In practice the allocator sometimes handed us an A100 80GB or an H100 NVL 96GB instead of a plain H100, which was fine (more headroom, not less).
- `train.sh` in the repo reflects this working configuration.

## 5. Test-set evaluation sweep (`target_period_sweep.py`)

**What it does:** for a given historical day, takes that day's _real_ station observation mask (which grid cells had a reporting station) and applies it to a randomly chosen 1940s ERA5 field (season-matched by day-of-leap-year) — a field where the true value is known everywhere. The model reconstructs the masked-out cells; error is measured against the true modern field. Run across the whole 1667–1939 period, this gives a reconstruction-accuracy estimate as a function of historical station density.

### Bug fixed

The original script fed the _raw historical station values_ as "known" inputs but scored the reconstruction against an _unrelated_ modern field — an apples-to-oranges comparison. Fixed: the mask comes from the historical day; the known values come from the chosen modern ERA5 field, restricted to the cells the historical mask marks as observed.

### Station density across the target period

Checked before committing to the full sweep, since a naive 8-day smoke test (Jan 1667) came back with only ~2 stations/day and RMSE numbers that didn't obviously mean anything:

| Period          | Mean stations/day               | Days with ≥10 stations |
| --------------- | ------------------------------- | ---------------------- |
| 1667–1746       | 1.1 – 7.3                       | ~0%                    |
| 1747–1766       | 12.8                            | 78%                    |
| **1767 onward** | **25.6 → 362.8** (by 1927–1946) | **~100%**              |

Conclusion: reconstruction quality for pre-1750 dates should be treated with real caution — the network is very sparse. 1767 onward is consistently well-observed.

### Smoke test (dense decade, 1800–1809, 8 days, 50 sampling steps)

Ran clean, no warnings, no crashes:

| Date       | RMSE °C | MAE °C |
| ---------- | ------- | ------ |
| 1800-01-01 | 4.04    | 2.91   |
| 1800-01-02 | 4.70    | 3.32   |
| 1800-01-03 | 5.07    | 3.81   |
| 1800-01-04 | 3.36    | 2.47   |
| 1800-01-05 | 4.05    | 3.24   |
| 1800-01-06 | 7.93    | 6.41   |
| 1800-01-07 | 4.40    | 3.22   |
| 1800-01-08 | 4.22    | 3.30   |

Most days: 3–5°C RMSE, a plausible range for winter-field reconstruction at this station density. Note the real full sweep uses `SAMPLING_STEPS = 1000` (not 50) per the module default, so per-day cost there is ~20x this smoke test.

### Full sweep — status

Launched via `sbatch sweep_array.sh` (job array, 20 chunks, 4 concurrent). First task ran fine; the array's default `--mem=32G --time=24:00:00` request turned out to be far more than needed and tripped `MaxMemoryRunMinsPerUser` (a memory×walltime budget cap) when 4 tasks tried to run at once under `gratis`. **Next step: get real per-batch timing from a running chunk and right-size `--mem`/`--time` in `sweep_array.sh`** so all 4 concurrent tasks fit under the cap. Not yet resolved as of this writing.

## 6. Open items / next steps

- Finish getting the full sweep running at full concurrency (§5).
- Once the sweep completes: `python merge_sweep.py` → `target_period_reconstruction_results.csv` + error-vs-year plot; review with the coworker, especially the pre-1750 sparse-network years.
- **Hyperparameter tuning (plan step 2):** not started. Given the val-loss plateau at ~30% of training (§4), short runs (e.g. 10–20% of full length) may be enough to rank configs — worth discussing with the original author before committing to a tuning budget. Needs: a config table, per-run output folders, a step-limited run mode, a comparison script.
- **Real historical reconstruction (plan step 4):** script not written. Needs answers to: (a) is `hist_obs` de-warmed the same way as the training ERA5 data, (b) single reconstruction vs. ensemble.
- Reconcile this status doc's findings into `README_reproduction_guide.md` (job config, hist_obs_metadata.csv column order, measured numbers, resume-correctness behavior) once the sweep work settles down.

## 7. Notable bugs/errors and how they were diagnosed (chronological, not already in the setup guide)

| Symptom                                                                                          | Root cause                                                                                                                                                                                                               | Fix                                                                                                                                   |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError: ... expected spatial shape (128,192), got (128,366)`                                | Data files transposed vs. documented axis order                                                                                                                                                                          | `_read_var()` reads by dimension name                                                                                                 |
| GPU OOM right after first validation+checkpoint save, previously-fine iteration speed collapsing | Memory fragmentation on a near-full GPU                                                                                                                                                                                  | Move to larger-memory GPU + extra `empty_cache()` call                                                                                |
| `sbatch: Requested time limit is invalid` when requesting A100                                   | A100 fully banned under `gratis` (`a100=0`); Slurm surfaces this as an odd time-limit error instead of a clear rejection                                                                                                 | Use H100 instead (see §4 GPU/QoS notes)                                                                                               |
| `QOSMaxGRESPerUser` requesting H100 under plain `gpu` partition, even with nothing else running  | H100 only actually schedulable via `gpu-invest` + `job_gpu_preemptable`                                                                                                                                                  | Switch partition/QoS (see §4)                                                                                                         |
| Every resumed training job silently redid the full 1000-epoch target                             | Epoch counter always reset to 1 on resume, ignoring `global_step`                                                                                                                                                        | Compute `start_epoch` from `global_step`; early-exit if already complete                                                              |
| `[WARNING] 634 station(s) did not match an ERA5 grid cell` (all 634/634)                         | `hist_obs_metadata.csv` columns are `(id, lon, lat)`, not documented `(id, lat, lon)`                                                                                                                                    | Diagnosed via domain-bounds check: 0/634 valid as documented, 634/634 valid swapped. Fixed column order + added a hard validity check |
| Sweep smoke-test job seemed to vanish from `squeue` almost instantly                             | Two distinct causes on two occasions: (1) genuinely failed — wrong working directory when submitting, so relative paths resolved wrong; (2) genuinely fine — the smoke test legitimately finishes in well under a minute | Always check the log, don't assume from `squeue` alone; `cd` into the code folder before `sbatch`                                     |
| `sweep_array.sh` tasks stuck `PD` with reason `MaxMemoryRunMinsPerUser`                          | Per-task `--mem × --time` request too generous for 4 concurrent tasks under the `gratis` memory-time budget                                                                                                              | Not yet fixed — needs real per-batch timing to right-size the request (see §5)                                                        |

---

## Known gaps in `README_reproduction_guide.md` (setup guide), not yet reconciled

- Job config in §3/Step 7 still shows RTX4090/plain `gpu` partition — should reflect §4 of this doc.
- Status table (§0) needs updating — training is done.
- Step 7 notes say the epoch counter resets on resume — no longer true (§3.5 of this doc).
- Step 4 data table says `hist_obs_metadata.csv` is `(id, lat, lon)` — wrong, it's `(id, lon, lat)` (§3.3 of this doc).
- No measured training numbers or sweep results are in the guide yet.

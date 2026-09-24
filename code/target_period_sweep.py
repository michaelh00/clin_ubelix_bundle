"""
Target-period reconstruction sweep (final test-set evaluation).

For EVERY day in the target period (1667-1939, or a sub-range you choose),
this applies that day's real historical observation mask to a RANDOMLY
CHOSEN 1940s ERA5 field that shares the same day-of-leap-year, runs CLIN
reconstruction, and records the error against that (masked-out) modern
truth field. The result is a per-day reconstruction-skill estimate spread
across the whole target period, matching what a real 1667-1939
reconstruction run will face day by day.

This intentionally does NOT run during tuning -- per the current plan,
tuning uses only the training-vs-validation denoising loss
(`evaluate_val_loss` in the main script). This sweep is for the ONE final
evaluation, on the held-out 1940s test decade, with your fully tuned
model and sampling settings.

Cost: reconstructing ~99,700 individual days (1667-1939, if none are
skipped for having zero observations) at SAMPLING_STEPS network passes
each is a substantial job. Sequential denoising steps cannot be
parallelized, but different days CAN be batched together on the GPU --
`batch_size` below controls this and is the main lever for wall-clock
cost. Start with `max_days` set to a small number to sanity-check the
pipeline before launching the full, unstrided run.

UBELIX PATCHES (compared with the original version):
  1. load_trained_model() loads the BEST-validation checkpoint
     (clin_era5_*_best_*.pt), not simply the latest one.
  2. run_target_period_sweep() has a new `start_offset` argument, so the
     sweep can be split into job-array tasks: task k of N uses
     stride_days=N, start_offset=k.
  3. BUG FIX: the "known values" fed to the sampler are now the ERA5
     values of the chosen modern field, restricted to the cells that the
     historical day's mask marks as observed (mask from history, values from
     the modern truth field, as described above). Previously the raw
     historical station values were used, which were then scored against an
     unrelated modern field.
"""

import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from clin_era5_reconstruction import (
    DATA_DIR, MODEL_DIR, MODEL_PREFIX, SAMPLING_STEPS, TEST_YEARS, DOMAIN_SHAPE,
    CLINUNet, LambdaSchedule, date_to_index,
    load_era5_stack, load_static_field, load_doy_fourier, load_historical_obs,
    station_to_grid_indices, build_grid_mask_for_day,
    day_index_to_day_of_leap_year, day_index_to_array_pos, index_to_date,
    clin_reconstruct,
)


def load_trained_model(device):
    files = sorted(glob(str(MODEL_DIR / f"{MODEL_PREFIX}_*_best_*.pt")))
    if not files:
        raise FileNotFoundError(f"No best checkpoint (*_best_*.pt) found under {MODEL_DIR}")
    ckpt = torch.load(files[-1], map_location=device, weights_only=True)
    model = CLINUNet().to(device)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    return model


def run_target_period_sweep(target_year_range=(1667, 1939), test_year_range=TEST_YEARS,
                             min_obs=1, stride_days=1, start_offset=0, max_days=None,
                             batch_size=32, repaint_jump=0, repaint_resample=1,
                             n_steps=SAMPLING_STEPS, seed=0, device=None):
    """One row per historical day in `target_year_range` (subsampled every
    `stride_days` starting at `start_offset`, optionally capped at
    `max_days`), each reconstructed using a randomly chosen
    `test_year_range` field of the same day-of-leap-year. Returns a pandas
    DataFrame."""
    rng = random.Random(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    schedule = LambdaSchedule(device=device)
    model = load_trained_model(device)
    t_mean, t_std = np.load(MODEL_DIR / "norm_stats.npy")

    # ---- Test-period (1940s) fields: the source of "random valid fields" ----
    era5_idx, era5_data, era5_lat, era5_lon = load_era5_stack(DATA_DIR, test_year_range[0], test_year_range[1])
    era5_data_norm = (era5_data - t_mean) / t_std
    elev, _, _ = load_static_field(DATA_DIR / "ERA5_elevation.nc", "elev", era5_lat, era5_lon)
    lsm, _, _ = load_static_field(DATA_DIR / "ERA5_lsm.nc", "lsm", era5_lat, era5_lon)
    elev = (elev - elev.mean()) / (elev.std() + 1e-6)
    sin_doy, cos_doy = load_doy_fourier(DATA_DIR)

    # Group test-period positions by day-of-leap-year, so for any
    # historical date we can quickly pick a matching-season modern field.
    test_by_doy = {}
    for pos, di in enumerate(era5_idx):
        doy = day_index_to_day_of_leap_year(int(di))
        test_by_doy.setdefault(doy, []).append(pos)

    # ---- Historical observations: define the target period + masks ----
    hist_day_index, hist_values, station_id, station_lat, station_lon = load_historical_obs(DATA_DIR)
    lat_idx, lon_idx = station_to_grid_indices(station_lat, station_lon, era5_lat, era5_lon)

    start_idx = date_to_index(target_year_range[0], 1, 1)
    end_idx = date_to_index(target_year_range[1], 12, 31)
    n_obs_per_day = np.sum(~np.isnan(hist_values), axis=1)
    keep = (hist_day_index >= start_idx) & (hist_day_index <= end_idx) & (n_obs_per_day >= min_obs)
    target_days = sorted(int(di) for di in hist_day_index[keep])
    target_days = target_days[start_offset::stride_days]
    if max_days is not None:
        target_days = target_days[:max_days]

    print(f"Sweeping {len(target_days)} historical days "
          f"({target_year_range[0]}-{target_year_range[1]}, stride={stride_days}, offset={start_offset})")

    records = []
    for batch_start in range(0, len(target_days), batch_size):
        batch_days = target_days[batch_start:batch_start + batch_size]

        known_values, known_masks, conds, truths, modern_dis, hist_dis = [], [], [], [], [], []
        for hist_di in batch_days:
            doy = day_index_to_day_of_leap_year(hist_di)
            candidates = test_by_doy.get(doy)
            if not candidates:
                continue  # no test-period field exists for this day-of-leap-year (shouldn't normally happen)
            pos = rng.choice(candidates)
            modern_di = int(era5_idx[pos])

            # Mask comes from the HISTORICAL day; the known VALUES come from the
            # chosen modern ERA5 field (the truth), observed only at masked cells.
            _, mask_grid = build_grid_mask_for_day(hist_di, hist_day_index, hist_values, lat_idx, lon_idx)
            known_value_norm = era5_data_norm[pos] * mask_grid  # (H, W); zero outside the mask

            arr_pos = day_index_to_array_pos(modern_di)
            sin_c = np.full(DOMAIN_SHAPE, sin_doy[arr_pos], dtype=np.float32)
            cos_c = np.full(DOMAIN_SHAPE, cos_doy[arr_pos], dtype=np.float32)

            known_values.append(known_value_norm[None].astype(np.float32))
            known_masks.append(mask_grid[None].astype(np.float32))
            conds.append(np.stack([elev, lsm, sin_c, cos_c], axis=0))
            truths.append(era5_data_norm[pos][None])
            modern_dis.append(modern_di)
            hist_dis.append(hist_di)

        if not known_values:
            continue

        known_value = torch.from_numpy(np.stack(known_values)).to(device)
        known_mask = torch.from_numpy(np.stack(known_masks)).to(device)
        cond = torch.from_numpy(np.stack(conds)).to(device)
        truth = torch.from_numpy(np.stack(truths)).to(device)

        with torch.no_grad():
            recon = clin_reconstruct(model, schedule, known_value, known_mask, cond,
                                      n_steps=n_steps, device=device,
                                      repaint_jump=repaint_jump, repaint_resample=repaint_resample)
            unobserved = known_mask < 0.5
            diff = recon - truth
            for b in range(len(hist_dis)):
                d = diff[b][unobserved[b]]
                if d.numel() == 0:
                    rmse = mae = float("nan")  # fully observed field (unusual, but handle gracefully)
                else:
                    rmse = torch.sqrt(torch.mean(d ** 2)).item() * t_std
                    mae = torch.mean(torch.abs(d)).item() * t_std
                records.append({
                    "historical_day_index": hist_dis[b],
                    "historical_date": str(index_to_date(hist_dis[b])),
                    "historical_year": index_to_date(hist_dis[b]).year,
                    "day_of_leap_year": day_index_to_day_of_leap_year(hist_dis[b]),
                    "modern_day_index": modern_dis[b],
                    "modern_date": str(index_to_date(modern_dis[b])),
                    "n_obs": int(known_mask[b].sum().item()),
                    "rmse_degC": rmse,
                    "mae_degC": mae,
                })

        print(f"  {batch_start + len(batch_days)}/{len(target_days)} days done", flush=True)

    return pd.DataFrame.from_records(records)


def plot_error_across_period(df, value_col="rmse_degC", out_path="reconstruction_error_vs_target_period.png"):
    """Yearly mean +/- std of reconstruction error across the full
    1667-1939 target period."""
    yearly = df.groupby("historical_year")[value_col].agg(["mean", "std", "count"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(yearly.index, yearly["mean"], color="tab:red", lw=1)
    ax.fill_between(yearly.index, yearly["mean"] - yearly["std"], yearly["mean"] + yearly["std"],
                     color="tab:red", alpha=0.2)
    ax.set_xlabel("Year (target period)")
    ax.set_ylabel(f"{value_col} (yearly mean +/- 1 std)")
    ax.set_title("CLIN reconstruction error across the 1667-1939 target period\n"
                 "(each day's real historical mask applied to a matched-season 1940s field)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    return yearly


if __name__ == "__main__":
    # Sanity-check pass first: a handful of days, small batch, reduced
    # sampling steps -- confirms the pipeline runs end to end cheaply.
    df_smoke = run_target_period_sweep(max_days=8, batch_size=4, n_steps=50)
    print(df_smoke)

    # Full run (uncomment once the smoke test above looks right):
    # df = run_target_period_sweep(stride_days=1, batch_size=32, repaint_jump=0)
    # df.to_csv("target_period_reconstruction_results.csv", index=False)
    # plot_error_across_period(df, value_col="rmse_degC")

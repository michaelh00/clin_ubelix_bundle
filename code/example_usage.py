"""
Example usage of the trained CLIN model for:
  1. Unconditional sampling (QC / visual sanity check of the learned prior)
  2. CLIN reconstruction conditioned on a real historical observation mask,
     evaluated against a held-out ERA5 field (1940s/1950s)

This is a template, not a finished pipeline -- in particular, the pairing
between a historical mask date and a modern evaluation date (see the
`# CHOOSE A PAIRING SCHEME` block below) is a scientific decision for you
to make; a few common options are sketched.

Run after training has produced at least one checkpoint under MODEL_DIR.

UBELIX PATCH: load_trained_model() now loads the BEST-validation checkpoint
(clin_era5_*_best_*.pt) instead of simply the latest one.
"""

import numpy as np
import torch

from clin_era5_reconstruction import (
    DATA_DIR, MODEL_DIR, DOMAIN_SHAPE, TEST_YEARS, MODEL_PREFIX, SAMPLING_STEPS,
    CLINUNet, LambdaSchedule, date_to_index, day_index_to_array_pos,
    load_era5_stack, load_static_field, load_doy_fourier, load_historical_obs,
    station_to_grid_indices, build_grid_mask_for_day,
    sample_unconditional, clin_reconstruct, evaluate_with_historical_mask,
)
from glob import glob
from pathlib import Path


def load_trained_model(device):
    files = sorted(glob(str(MODEL_DIR / f"{MODEL_PREFIX}_*_best_*.pt")))
    if not files:
        raise FileNotFoundError(f"No best checkpoint (*_best_*.pt) found under {MODEL_DIR}")
    ckpt_path = files[-1]
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = CLINUNet().to(device)
    state = ckpt["model"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schedule = LambdaSchedule(device=device)
    model = load_trained_model(device)

    t_mean, t_std = np.load(MODEL_DIR / "norm_stats.npy")

    # ---- Load static conditioning fields (same normalization as training) ----
    era5_idx, era5_data, era5_lat, era5_lon = load_era5_stack(DATA_DIR, TEST_YEARS[0], TEST_YEARS[1])
    era5_data = (era5_data - t_mean) / t_std
    elev, _, _ = load_static_field(DATA_DIR / "ERA5_elevation.nc", "elev", era5_lat, era5_lon)
    lsm, _, _ = load_static_field(DATA_DIR / "ERA5_lsm.nc", "lsm", era5_lat, era5_lon)
    elev = (elev - elev.mean()) / (elev.std() + 1e-6)
    sin_doy, cos_doy = load_doy_fourier(DATA_DIR)

    # ---- 1. Unconditional sample, for a chosen calendar date (QC check) ----
    example_di = date_to_index(1945, 7, 15)
    pos = day_index_to_array_pos(example_di)
    sin_c = np.full(DOMAIN_SHAPE, sin_doy[pos], dtype=np.float32)
    cos_c = np.full(DOMAIN_SHAPE, cos_doy[pos], dtype=np.float32)
    cond = np.stack([elev, lsm, sin_c, cos_c], axis=0)[None]  # (1, 4, H, W)
    cond_t = torch.from_numpy(cond).to(device)

    sample = sample_unconditional(model, schedule, cond_t, n_steps=SAMPLING_STEPS, device=device)
    sample_degC = sample.cpu().numpy()[0, 0] * t_std + t_mean
    print("Unconditional sample stats (degC):", sample_degC.min(), sample_degC.max())

    # ---- 2. CLIN reconstruction using a REAL historical observation mask ----
    hist_day_index, hist_values, station_id, station_lat, station_lon = load_historical_obs(DATA_DIR)
    lat_idx, lon_idx = station_to_grid_indices(station_lat, station_lon, era5_lat, era5_lon)

    # CHOOSE A PAIRING SCHEME between a historical mask date and a modern
    # evaluation date. Two simple examples:
    #   (a) match by day-of-leap-year: pick a historical date and a modern
    #       date that fall on (approximately) the same day of year, so the
    #       station network's seasonal characteristics roughly align.
    #   (b) cycle sequentially: iterate historical dates 1667-1939 in order,
    #       pairing each with modern test dates in a fixed rotation.
    # Below is a minimal single-pair example using (a).
    historical_date_idx = date_to_index(1750, 7, 15)   # example historical date
    modern_idx_in_test = np.where(era5_idx == example_di)[0]
    if len(modern_idx_in_test) == 0:
        raise ValueError("Chosen modern evaluation date not found in loaded ERA5 test stack")
    modern_field = era5_data[modern_idx_in_test[0]][None, None]  # (1,1,H,W)
    modern_field_t = torch.from_numpy(modern_field).to(device)

    recon, rmse, mae, n_obs = evaluate_with_historical_mask(
        model, schedule, modern_field_t, cond_t,
        hist_day_index, hist_values, lat_idx, lon_idx,
        historical_date_idx, device=device,
        repaint_jump=0, repaint_resample=1,  # set repaint_jump>0 to try RePaint
        n_steps=SAMPLING_STEPS,
    )
    print(f"n_obs={n_obs}  RMSE={rmse * t_std:.3f} degC  MAE={mae * t_std:.3f} degC")


if __name__ == "__main__":
    main()

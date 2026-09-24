"""Sanity check of the uploaded data (reads 1940 + static files + hist_obs; ~1 min).
Run from the code folder:   source env.sh && python check_data.py
"""
import os
import numpy as np
import clin_era5_reconstruction as m

files = sorted(os.listdir(m.DATA_DIR))
n_t = len([f for f in files if f.startswith("T_ERA5_")])
print(f"{len(files)} files in {m.DATA_DIR}  (expect 92; T_ERA5 files: {n_t}, expect 86)")
for f in ["ERA5_elevation.nc", "ERA5_lsm.nc", "sin_doy_1667_2025.txt",
          "cos_doy_1667_2025.txt", "hist_obs.nc", "hist_obs_metadata.csv"]:
    print(f"  {f}: {'OK' if f in files else 'MISSING'}")

i, d, lat, lon = m.load_era5_stack(m.DATA_DIR, 1940, 1940)
print("T_ERA5_1940:", d.shape, d.dtype, "min/max degC:", round(float(d.min()), 1), round(float(d.max()), 1),
      "day_index:", i[0], "-", i[-1], " (expect (366,128,192), 99711-100076)")
e, _, _ = m.load_static_field(m.DATA_DIR / "ERA5_elevation.nc", "elev", lat, lon)
print("elevation:", e.shape, round(float(e.min())), "to", round(float(e.max())), "m")
l, _, _ = m.load_static_field(m.DATA_DIR / "ERA5_lsm.nc", "lsm", lat, lon)
print("land-sea mask:", l.shape, "min/max:", float(l.min()), float(l.max()))
h = m.load_historical_obs(m.DATA_DIR)
print("hist_obs:", h[1].shape, "observed fraction:", round(float((~np.isnan(h[1])).mean()), 3),
      " (expect (99710, 634), ~0.11)")

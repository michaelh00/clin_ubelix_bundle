"""Shared helpers for the CLIN analysis notebooks and the diffusion re-evaluation.

Everything here mirrors the conventions of clin_era5_reconstruction.py:
day_index is 1-based (1 = 1667-01-01), fields are (day, lat, lon) with lat
descending, metadata columns are (id, lon, lat), seasons are matched by
day-of-leap-year.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent


def _find_data_dir():
    # CLIN_DATA_DIR wins; otherwise look for "Final CLIN variables" next to this folder or one level up
    if "CLIN_DATA_DIR" in os.environ:
        return Path(os.environ["CLIN_DATA_DIR"])
    for cand in (ROOT / "Final CLIN variables", ROOT.parent / "Final CLIN variables"):
        if cand.exists():
            return cand
    raise FileNotFoundError("Set CLIN_DATA_DIR to the folder holding T_ERA5_*.nc, hist_obs.nc, ...")


DATA_DIR = _find_data_dir()
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

T0 = pd.Timestamp("1667-01-01")
TRAIN_YEARS, VAL_YEARS, TEST_YEARS = (1960, 2025), (1950, 1959), (1940, 1949)
ERAS = [(1667, 1746, "sparse"), (1747, 1766, "growing"), (1767, 1846, "dense"), (1847, 1939, "very dense")]
CASES_FILE = RESULTS_DIR / "eval_cases.csv"


def day_of_leap_year(dates):
    d = pd.DatetimeIndex(dates)
    return np.asarray(d.dayofyear + ((~d.is_leap_year) & (d.month >= 3)).astype(int))


def harmonics(doly, k=3):
    """Design matrix [1, cos(j*th), sin(j*th)] for j = 1..k, th = 2*pi*doly/366."""
    th = 2 * np.pi * np.asarray(doly, dtype=np.float64) / 366
    cols = [np.ones_like(th)] + [f(j * th) for j in range(1, k + 1) for f in (np.cos, np.sin)]
    return np.stack(cols, 1).astype(np.float32)


def load_static():
    e = xr.open_dataset(DATA_DIR / "ERA5_elevation.nc")
    m = xr.open_dataset(DATA_DIR / "ERA5_lsm.nc")
    lat, lon = e.lat.values, e.lon.values
    elev = e.elev.transpose("lat", "lon").values.astype(np.float32)
    lsm = m.lsm.transpose("lat", "lon").values.astype(np.float32)
    return lat, lon, elev, lsm


def load_era5(year_start, year_end):
    """Returns T (n, lat, lon) float32 [C], dates, day-of-leap-year."""
    fields, idx = [], []
    for y in range(year_start, year_end + 1):
        with xr.open_dataset(DATA_DIR / f"T_ERA5_{y}.nc", decode_times=False) as ds:
            fields.append(ds.T_ERA5.transpose("day_index", "lat", "lon").values.astype(np.float32))
            idx.append(ds.day_index.values.astype(np.int64))
    idx = np.concatenate(idx)
    dates = T0 + pd.to_timedelta(idx - 1, unit="D")
    return np.concatenate(fields), dates, day_of_leap_year(dates)


def load_hist(lat, lon):
    """Returns H (ndays, nstation) float32 with NaN, dates, day-of-leap-year, metadata, flat grid cell per station."""
    with xr.open_dataset(DATA_DIR / "hist_obs.nc", decode_times=False) as ds:
        H = ds.hist_obs.transpose("day_index", "id").values.astype(np.float32)
        di = ds.day_index.values.astype(np.int64)
        ids = ds.id.values.astype(int)
    meta = pd.read_csv(DATA_DIR / "hist_obs_metadata.csv", header=None, names=["id", "lon", "lat"])
    assert np.array_equal(ids, meta.id.values)
    li = np.round((lat[0] - meta.lat.values) / 0.25).astype(int)
    lo = np.round((meta.lon.values - lon[0]) / 0.25).astype(int)
    cell = li * len(lon) + lo
    dates = T0 + pd.to_timedelta(di - 1, unit="D")
    return H, dates, day_of_leap_year(dates), meta, cell


def haversine_min(lat_c, lon_c, lat_s, lon_s, chunk=4000):
    """Distance [km] from each point (lat_c, lon_c) to the nearest of (lat_s, lon_s)."""
    out = np.empty(len(lat_c))
    la_s, lo_s = np.deg2rad(lat_s)[None], np.deg2rad(lon_s)[None]
    for i in range(0, len(lat_c), chunk):
        la, lo = np.deg2rad(lat_c[i:i + chunk])[:, None], np.deg2rad(lon_c[i:i + chunk])[:, None]
        d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((la - la_s) / 2) ** 2 +
                                         np.cos(la) * np.cos(la_s) * np.sin((lo - lo_s) / 2) ** 2))
        out[i:i + chunk] = d.min(1)
    return out


def make_cases(hist_dates, hist_doly, hist_obs_mask, test_dates, test_doly, n_per_era=60, seed=0):
    """Fixed evaluation cases, same protocol as target_period_sweep.py: a historical day's
    station mask laid over a random 1940s ERA5 field with the same day-of-leap-year."""
    rng = np.random.default_rng(seed)
    hyear = hist_dates.year.values
    by_doly = {d: np.where(test_doly == d)[0] for d in np.unique(test_doly)}
    rows = []
    for a, b, name in ERAS:
        cand = np.where((hyear >= a) & (hyear <= b))[0]
        for h in np.sort(rng.choice(cand, n_per_era, replace=False)):
            t = rng.choice(by_doly[hist_doly[h]])
            rows.append({"era": f"{a}-{b}", "hist_row": h, "hist_date": hist_dates[h].date(),
                         "test_row": t, "test_date": test_dates[t].date(), "n_obs": int(hist_obs_mask[h].sum())})
    return pd.DataFrame(rows)


def get_cases(hist_dates, hist_doly, hist_obs_mask, test_dates, test_doly, n_per_era=60):
    """Load the shared case list (so every method is scored on identical cases), or create it."""
    if CASES_FILE.exists():
        return pd.read_csv(CASES_FILE)
    cases = make_cases(hist_dates, hist_doly, hist_obs_mask, test_dates, test_doly, n_per_era)
    cases.to_csv(CASES_FILE, index=False)
    return cases


def score(pred, truth, known, land):
    """RMSE/MAE over unobserved cells, also split into land and sea. All arrays flat (ncell,)."""
    err = (pred - truth)
    un = ~known
    out = {}
    for name, m in [("all", un), ("land", un & land), ("sea", un & ~land)]:
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(err[m] ** 2)))
        out[f"mae_{name}"] = float(np.mean(np.abs(err[m])))
    return out

"""
CLIN-style diffusion model for reconstructing historical daily temperature
fields over Europe (1667-1939) from sparse point observations.

Adapted from:
  - Chao et al. (2024), https://doi.org/10.1029/2024JH000260 (CLIN model,
    original Wolfram Mathematica implementation)
  - A related PyTorch/accelerate v-prediction diffusion implementation
    (CMFD four-variable field generation)

Design summary
---------------
- Single-channel field: de-warmed (anthropogenic-trend-removed) ERA5 daily
  temperature, domain 34.5-66.25N x 12.25W-35.5E, 128 (lat) x 192 (lon),
  0.25 deg resolution.
- The diffusion model is trained UNCONDITIONALLY on complete ERA5 fields
  (no synthetic masking during training), following the original CLIN
  design: conditioning on sparse point observations happens only at
  SAMPLING time via masked reverse-diffusion ("inpainting").
- Static, always-available auxiliary conditioning channels (never masked,
  concatenated to the noisy field at the network input): elevation,
  land-sea mask, and sin/cos of day-of-leap-year (broadcast spatially).
- Noise schedule: continuous-time log-SNR (lambda) parameterization
  following Kingma et al. (2021) "Variational Diffusion Models", matching
  the original CLIN Mathematica implementation. Network predicts
  v = alpha*eps - sigma*x (v-prediction), matching the reference PyTorch
  implementation.
- Two usage modes at inference: (1) an ordinary unconditional sampler for
  QC/visual checks, and (2) a CLIN reconstruction sampler that conditions
  on a partial known field (real historical observation mask). The CLIN
  sampler supports an OPTIONAL RePaint-style resampling schedule
  (Lugmayr et al., 2022) as a configurable toggle -- default OFF, which
  reproduces the original single-pass CLIN masking exactly.

*** READ BEFORE RUNNING ***
This script assumes the following files exist under DATA_DIR (see CONFIG
section below) and follow exactly the formats confirmed in discussion:

  T_ERA5_1940.nc ... T_ERA5_2025.nc
      var 'T_ERA5' [degC], dims (day_index, lat, lon)
      day_index = days since 1667-01-01, 1-BASED: 1667-01-01 itself is
      day_index = 1 (not 0)
      lat descending, lon ascending

  ERA5_elevation.nc
      var 'elev' [m], dims (lat, lon), same grid as T_ERA5 files

  ERA5_lsm.nc
      var 'lsm', dims (lat, lon), same grid as T_ERA5 files

  sin_doy_1667_2025.txt, cos_doy_1667_2025.txt
      plain text, one value per line, length 131122 (1667-01-01 through
      2025-12-31 inclusive), 0-based array position i <-> day_index i+1
      (i.e. array position 0 is 1667-01-01 = day_index 1)

  hist_obs.nc
      var 'hist_obs' [degC], dims (day_index, id), day_index = days since
      1667-01-01, 1-based as above, covering 1667-1939, id in 1..634
      (missing = NaN)

  hist_obs_metadata.csv
      no header, columns: id, lat, lon (lat/lon pre-rounded to nearest
      ERA5 grid coordinate)

Edit the CONFIG section (paths, split years, hyperparameters, SLURM/GPU
particulars are handled at the job-submission level, not in this script)
before running. This script does not require any GPU-count-specific code;
`accelerate` handles single- or multi-GPU transparently.

UBELIX PATCHES (compared with the original version):
  1. DATA_DIR / MODEL_DIR are read from the environment variables
     CLIN_DATA_DIR / CLIN_MODEL_DIR (set by env.sh); NUM_WORKERS follows
     SLURM_CPUS_PER_TASK.
  2. netCDF variables are read by dimension NAME and reordered, because the
     actual files are stored as (lon, lat, day_index) for T_ERA5, (lon, lat)
     for elev/lsm and (id, day_index) for hist_obs.
  3. Checkpointing: best validation loss is stored in every checkpoint and
     restored on resume; only the newest "best" checkpoint is kept; the
     regular-checkpoint pruning never deletes the best checkpoint.
"""

import argparse
import calendar
import gc
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from glob import glob
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================
# CONFIG -- edit before running
# =============================================================================

# Paths come from environment variables (set by env.sh on UBELIX).
DATA_DIR = Path(os.environ["CLIN_DATA_DIR"])     # directory holding all input files
MODEL_DIR = Path(os.environ["CLIN_MODEL_DIR"])   # checkpoints written here
MODEL_DIR.mkdir(parents=True, exist_ok=True)     # must exist before logging starts
LOG_FILE = str(MODEL_DIR / "train.log")

REF_DATE = date(1667, 1, 1)                 # day_index is 1-BASED: REF_DATE itself is day_index 1

# Train/validation/test split (calendar years, inclusive)
TRAIN_YEARS = (1960, 2025)
VAL_YEARS = (1950, 1959)
TEST_YEARS = (1940, 1949)

DOMAIN_SHAPE = (128, 192)                   # (lat, height=128, lon, width=192)
CHANNEL = 1                                 # temperature only
N_STATIC_COND = 4                           # elevation, lsm, sin_doy, cos_doy

BASE_DIM = 64
CHANNEL_MULTS = (1, 2, 4, 4)                # depth = 4, matches 128/192 -> /16
RESNET_GROUPS = 4
LAMBDA_EMBED_DIM = 256                      # size of the lambda Fourier embedding
N_LAMBDA_FREQS = 128                        # matches original CLIN encoding size

LAMBDA_MIN = -20.0
LAMBDA_MAX = 20.0

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
EPOCHS = 1000
CHECK_STEPS = 1000
MAX_CHECKPOINTS = 20
GRAD_CLIP_NORM = 1.0
GRAD_ACCUM_STEPS = 1
MODEL_PREFIX = "clin_era5"

SAMPLING_STEPS = 1000                       # reverse-diffusion steps at inference


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Diffusion math: continuous-time log-SNR (lambda) schedule, v-prediction
# =============================================================================

class LambdaSchedule:
    """Continuous-time variational-diffusion-model noise schedule.

    u in [0, 1] indexes "time" (u=0 -> most noise, u=1 -> least noise),
    mapped to a log-SNR value lambda via the same arctan schedule used in
    the original CLIN implementation.
    """

    def __init__(self, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, device="cpu"):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.device = device
        self._b = math.atan(math.exp(-lambda_max / 2.0))
        self._a = math.atan(math.exp(-lambda_min / 2.0)) - self._b

    def u_to_lambda(self, u):
        return -2.0 * torch.log(torch.tan(self._a * u + self._b))

    def alpha(self, lam):
        return torch.sqrt(torch.sigmoid(lam))

    def sigma(self, lam):
        return torch.sqrt(torch.sigmoid(-lam))

    def sample_u(self, n):
        return torch.rand(n, device=self.device)

    def noisify(self, x, lam, eps=None):
        """x, lam: broadcastable. Returns (x_t, eps, v)."""
        if eps is None:
            eps = torch.randn_like(x)
        shape = [-1] + [1] * (x.dim() - 1)
        a = self.alpha(lam).view(shape)
        s = self.sigma(lam).view(shape)
        x_t = a * x + s * eps
        v = a * eps - s * x
        return x_t, eps, v

    def x0_eps_from_v(self, x_t, v, lam):
        """Recover predicted x0 and eps from a v-prediction."""
        shape = [-1] + [1] * (x_t.dim() - 1)
        a = self.alpha(lam).view(shape)
        s = self.sigma(lam).view(shape)
        x0 = a * x_t - s * v
        eps = s * x_t + a * v
        return x0, eps


class LambdaFourierEmbedding(nn.Module):
    """Fourier feature encoding of the scalar log-SNR value lambda,
    following the fixed-frequency encoding used in the original CLIN
    implementation (Kingma et al. 2021-style log-spaced frequencies with
    symmetric sign to cover a wide dynamic range)."""

    def __init__(self, n_freqs=N_LAMBDA_FREQS, f_min=0.01, f_max=3.0):
        super().__init__()
        half = n_freqs // 2
        freqs = torch.exp(torch.linspace(math.log(f_min), math.log(f_max), half))
        freqs = torch.cat([-freqs.flip(0), freqs])
        self.register_buffer("freqs", freqs)  # (n_freqs,)

    def forward(self, lam):
        # lam: (B,) -> (B, 2*n_freqs)
        arg = 2.0 * math.pi * lam.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)


# =============================================================================
# U-Net architecture (ResNet blocks + FiLM conditioning on lambda embedding)
# =============================================================================

def make_downsample(dim_in, dim_out, scale=2):
    return nn.Sequential(
        nn.MaxPool2d(scale),
        nn.Conv2d(dim_in, dim_out, kernel_size=1),
    )


def make_upsample(dim_in, dim_out, scale=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True),
        nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
    )


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, groups=RESNET_GROUPS):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, emb_dim, groups=RESNET_GROUPS):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim_out * 2))
        self.block1 = ConvBlock(dim_in, dim_out, groups=groups)
        self.block2 = ConvBlock(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, emb):
        scale_shift = self.mlp(emb)
        scale_shift = scale_shift.view(scale_shift.shape[0], -1, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.block1(x, scale_shift=(scale, shift))
        h = self.block2(h)
        return h + self.res_conv(x)


class CLINUNet(nn.Module):
    """U-Net for the CLIN diffusion model.

    Input channels = CHANNEL (noisy target field) + N_STATIC_COND (static
    conditioning: elevation, land-sea mask, sin_doy, cos_doy, all broadcast
    spatially). Output channels = CHANNEL (predicted v).
    """

    def __init__(self, channel=CHANNEL, n_static=N_STATIC_COND, c=BASE_DIM,
                 mults=CHANNEL_MULTS, groups=RESNET_GROUPS,
                 lambda_embed_dim=LAMBDA_EMBED_DIM, n_lambda_freqs=N_LAMBDA_FREQS):
        super().__init__()
        dim_in_total = channel + n_static
        self.init_conv = nn.Conv2d(dim_in_total, c, 1)

        self.lambda_fourier = LambdaFourierEmbedding(n_freqs=n_lambda_freqs)
        emb_dim = c * 4
        self.lambda_mlp = nn.Sequential(
            nn.Linear(2 * n_lambda_freqs, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        dims = [c * m for m in mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        n_res = len(in_out)

        self.downs = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            is_last = i >= n_res - 1
            self.downs.append(nn.ModuleList([
                ResnetBlock(d_in, d_in, emb_dim, groups),
                ResnetBlock(d_in, d_in, emb_dim, groups),
                make_downsample(d_in, d_out) if not is_last else nn.Conv2d(d_in, d_out, 3, padding=1),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, emb_dim, groups)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, emb_dim, groups)

        self.ups = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = i == n_res - 1
            self.ups.append(nn.ModuleList([
                ResnetBlock(d_out + d_in, d_out, emb_dim, groups),
                ResnetBlock(d_out + d_in, d_out, emb_dim, groups),
                make_upsample(d_out, d_in) if not is_last else nn.Conv2d(d_out, d_in, 3, padding=1),
            ]))

        self.final_res_block = ResnetBlock(c * 2, c, emb_dim, groups)
        self.final_conv = nn.Conv2d(c, channel, 1)

    def forward(self, x_t, lam, static_cond):
        """x_t: (B, CHANNEL, H, W); lam: (B,); static_cond: (B, N_STATIC_COND, H, W)."""
        x = torch.cat([x_t, static_cond], dim=1)
        x = self.init_conv(x)
        r = x.clone()

        emb = self.lambda_mlp(self.lambda_fourier(lam))

        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x, emb)
            h.append(x)
            x = block2(x, emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_block2(x, emb)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, emb)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, emb)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, emb)
        return self.final_conv(x)


# =============================================================================
# Data loading
# =============================================================================

def date_to_index(y, m, d):
    """Returns the 1-based day_index matching the data files' convention
    (REF_DATE itself, i.e. 1667-01-01, is day_index 1)."""
    return (date(y, m, d) - REF_DATE).days + 1


def day_index_to_array_pos(day_idx):
    """Converts a 1-based day_index into a 0-based array position, for
    indexing into arrays such as sin_doy/cos_doy that start at position 0
    for 1667-01-01."""
    return day_idx - 1


def index_to_date(day_idx):
    """Inverse of date_to_index: 1-based day_index -> calendar date."""
    return REF_DATE + timedelta(days=day_idx - 1)


def day_of_leap_year(y, m, d):
    """Day-of-leap-year on a fixed 1-366 scale: Dec 31 is always day 366;
    day 60 (Feb 29) exists only in leap years, and non-leap years jump
    from day 59 straight to day 61. This is computable directly from the
    calendar date -- no external file is needed to match dates by this
    quantity."""
    is_leap = calendar.isleap(y)
    doy = (date(y, m, d) - date(y, 1, 1)).days + 1
    if not is_leap and doy >= 60:
        doy += 1
    return doy


def day_index_to_day_of_leap_year(day_idx):
    d = index_to_date(day_idx)
    return day_of_leap_year(d.year, d.month, d.day)


def _read_var(ds, varname, target_dims):
    """Reads a netCDF variable as float32 and reorders its axes to
    `target_dims` (a tuple of dimension NAMES), so the files can be stored
    in any axis order (e.g. Mathematica exports (lon, lat, day_index)).
    Masked values are converted to NaN."""
    var = ds.variables[varname]
    dims = tuple(var.dimensions)
    if sorted(dims) != sorted(target_dims):
        raise ValueError(f"{varname}: dimensions {dims} do not match expected {tuple(target_dims)}")
    raw = var[:]
    if np.ma.isMaskedArray(raw):
        raw = raw.filled(np.nan)
    arr = np.asarray(raw, dtype=np.float32)
    perm = [dims.index(d) for d in target_dims]
    return np.ascontiguousarray(np.transpose(arr, perm))


def load_static_field(path, varname, ref_lat=None, ref_lon=None):
    with nc.Dataset(path) as ds:
        arr = _read_var(ds, varname, ("lat", "lon"))
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    if arr.shape != DOMAIN_SHAPE:
        raise ValueError(f"{path}: expected shape {DOMAIN_SHAPE}, got {arr.shape}")
    if ref_lat is not None and (not np.allclose(lat, ref_lat) or not np.allclose(lon, ref_lon)):
        raise ValueError(f"{path}: lat/lon grid does not match reference ERA5 grid")
    return arr, lat, lon


def load_era5_stack(data_dir, year_start, year_end):
    """Loads all yearly T_ERA5 files in [year_start, year_end] into one
    contiguous array. Returns (day_index (N,), data (N, H, W) float32,
    lat (H,), lon (W,))."""
    all_idx, all_data = [], []
    ref_lat = ref_lon = None
    for year in range(year_start, year_end + 1):
        path = data_dir / f"T_ERA5_{year}.nc"
        with nc.Dataset(path) as ds:
            day_index = np.asarray(ds.variables["day_index"][:], dtype=np.int64)
            data = _read_var(ds, "T_ERA5", ("day_index", "lat", "lon"))
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        if data.shape[0] != len(day_index):
            raise ValueError(f"{path}: {data.shape[0]} fields but {len(day_index)} day_index values")
        if data.shape[1:] != DOMAIN_SHAPE:
            raise ValueError(f"{path}: expected spatial shape {DOMAIN_SHAPE}, got {data.shape[1:]}")
        if ref_lat is None:
            ref_lat, ref_lon = lat, lon
        elif not np.allclose(lat, ref_lat) or not np.allclose(lon, ref_lon):
            raise ValueError(f"{path}: lat/lon grid inconsistent with earlier files")
        all_idx.append(day_index)
        all_data.append(data)
        logger.info(f"Loaded {path.name}: {data.shape[0]} days")
    day_index = np.concatenate(all_idx)
    data = np.concatenate(all_data, axis=0)
    order = np.argsort(day_index)
    return day_index[order], data[order], ref_lat, ref_lon


def load_doy_fourier(data_dir):
    sin_doy = np.loadtxt(data_dir / "sin_doy_1667_2025.txt", dtype=np.float32)
    cos_doy = np.loadtxt(data_dir / "cos_doy_1667_2025.txt", dtype=np.float32)
    if sin_doy.shape != cos_doy.shape:
        raise ValueError("sin_doy and cos_doy files have different lengths")
    return sin_doy, cos_doy  # index i <-> day_index i (both referenced to 1667-01-01)


def load_historical_obs(data_dir):
    """Returns (day_index (N,), values (N, n_station) float32 with NaN for
    missing, station_id (n_station,), station_lat (n_station,),
    station_lon (n_station,))."""
    with nc.Dataset(data_dir / "hist_obs.nc") as ds:
        day_index = np.asarray(ds.variables["day_index"][:], dtype=np.int64)
        values = _read_var(ds, "hist_obs", ("day_index", "id"))
        nc_ids = np.asarray(ds.variables["id"][:]) if "id" in ds.variables else None
    meta = pd.read_csv(data_dir / "hist_obs_metadata.csv", header=None,
                        names=["id", "lat", "lon"])
    if values.shape[1] != len(meta):
        raise ValueError(f"hist_obs has {values.shape[1]} stations but metadata lists {len(meta)}")
    if nc_ids is not None and not np.array_equal(nc_ids.astype(np.int64), meta["id"].to_numpy().astype(np.int64)):
        logger.warning("station ids in hist_obs.nc differ from hist_obs_metadata.csv (order/values) -- "
                       "check that columns of hist_obs correspond to the metadata rows")
    return day_index, values, meta["id"].to_numpy(), meta["lat"].to_numpy(), meta["lon"].to_numpy()


def station_to_grid_indices(station_lat, station_lon, era5_lat, era5_lon, tol=1e-3):
    """Maps each station's (pre-rounded) lat/lon to the nearest ERA5 grid
    cell (i, j). Warns if the match isn't exact to within `tol` degrees."""
    lat_idx = np.array([np.argmin(np.abs(era5_lat - lat)) for lat in station_lat])
    lon_idx = np.array([np.argmin(np.abs(era5_lon - lon)) for lon in station_lon])
    lat_err = np.abs(era5_lat[lat_idx] - station_lat)
    lon_err = np.abs(era5_lon[lon_idx] - station_lon)
    n_bad = np.sum((lat_err > tol) | (lon_err > tol))
    if n_bad > 0:
        logger.warning(f"{n_bad} station(s) did not match an ERA5 grid cell within {tol} deg")
    return lat_idx, lon_idx


def group_historical_dates_by_doy(hist_day_index, hist_values, min_obs=1):
    """Groups historical day_index values by day-of-leap-year, keeping
    only dates with at least `min_obs` reporting stations. Returns a dict
    {doy (1-366): list of day_index values}, used to find historical mask
    dates that are seasonally matched to a given modern evaluation date."""
    n_obs_per_day = np.sum(~np.isnan(hist_values), axis=1)
    by_doy = {}
    for di, n in zip(hist_day_index, n_obs_per_day):
        if n < min_obs:
            continue
        doy = day_index_to_day_of_leap_year(int(di))
        by_doy.setdefault(doy, []).append(int(di))
    return by_doy


def build_grid_mask_for_day(
        day_idx_value, hist_day_index, hist_values, lat_idx, lon_idx):
    """For a single historical day_index value, returns (value_grid (H, W)
    with NaN where unobserved, mask_grid (H, W) bool) by placing each
    reporting station's value at its mapped grid cell. If more than one
    station maps to the same grid cell, the mean of their values is used."""
    row = np.where(hist_day_index == day_idx_value)[0]
    value_grid = np.full(DOMAIN_SHAPE, np.nan, dtype=np.float32)
    if len(row) == 0:
        return value_grid, np.zeros(DOMAIN_SHAPE, dtype=bool)
    vals = hist_values[row[0]]
    valid = ~np.isnan(vals)
    sums = np.zeros(DOMAIN_SHAPE, dtype=np.float64)
    counts = np.zeros(DOMAIN_SHAPE, dtype=np.int32)
    for i, lo, v in zip(lat_idx[valid], lon_idx[valid], vals[valid]):
        sums[i, lo] += v
        counts[i, lo] += 1
    mask_grid = counts > 0
    value_grid[mask_grid] = (sums[mask_grid] / counts[mask_grid]).astype(np.float32)
    return value_grid, mask_grid


# =============================================================================
# Dataset
# =============================================================================

class ERA5FieldDataset(Dataset):
    def __init__(self, day_index, data, elev, lsm, sin_doy, cos_doy, year_range):
        start_idx = date_to_index(year_range[0], 1, 1)
        end_idx = date_to_index(year_range[1], 12, 31)
        keep = (day_index >= start_idx) & (day_index <= end_idx)
        self.day_index = day_index[keep]
        self.data = data[keep]
        self.elev = elev
        self.lsm = lsm
        self.sin_doy = sin_doy
        self.cos_doy = cos_doy

    def __len__(self):
        return len(self.day_index)

    def __getitem__(self, idx):
        di = int(self.day_index[idx])
        pos = day_index_to_array_pos(di)
        x = self.data[idx][None, :, :]  # (1, H, W)
        sin_c = np.full(DOMAIN_SHAPE, self.sin_doy[pos], dtype=np.float32)
        cos_c = np.full(DOMAIN_SHAPE, self.cos_doy[pos], dtype=np.float32)
        cond = np.stack([self.elev, self.lsm, sin_c, cos_c], axis=0)  # (4, H, W)
        return {
            "x": torch.from_numpy(x.copy()),
            "cond": torch.from_numpy(cond.copy()),
            "day_index": di,
        }


# =============================================================================
# Training
# =============================================================================

def maybe_resume(model, optimizer, accelerator):
    """Returns (global_step, best_loss) restored from the most recent
    checkpoint, or (0, inf) if there is none. Tries checkpoints newest-first
    and skips any that fail to load (e.g. left truncated by a job killed
    mid-write), so a single corrupted file cannot silently reset training
    to scratch or crash the run."""
    files = sorted(glob(str(MODEL_DIR / f"{MODEL_PREFIX}_*.pt")), key=os.path.getmtime, reverse=True)
    for path in files:
        try:
            ckpt = torch.load(path, map_location=accelerator.device, weights_only=True)
            state = ckpt["model"]
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            accelerator.unwrap_model(model).load_state_dict(state)
            optimizer.load_state_dict(ckpt["optimizer"])
            global_step = ckpt.get("global_step", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
            logger.info(f"Resumed from {path} (step {global_step}, best_val {best_loss:.4f})")
            return global_step, best_loss
        except Exception as e:
            logger.warning(f"Could not load checkpoint {path} ({e}); trying an older one.")
    logger.info("No usable checkpoint found, training from scratch.")
    return 0, float("inf")


def prune_checkpoints():
    """Keep at most MAX_CHECKPOINTS regular checkpoints. The "best"
    checkpoint is never pruned here (only one is kept, see save_checkpoint)."""
    files = sorted(
        (f for f in glob(str(MODEL_DIR / f"{MODEL_PREFIX}_*.pt")) if "_best_" not in f),
        key=os.path.getmtime)
    while len(files) > MAX_CHECKPOINTS:
        os.remove(files.pop(0))


def save_checkpoint(model, optimizer, global_step, loss_running, tag="", best_loss=float("inf")):
    suffix = f"_{tag}" if tag else ""
    ts = time.strftime("%Y%m%d%H%M")
    loss_str = f"{loss_running:.4f}".replace(".", "")
    path = MODEL_DIR / f"{MODEL_PREFIX}_{ts}{suffix}_{loss_str}.pt"
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "global_step": global_step, "best_loss": best_loss}, tmp_path)
    tmp_path.rename(path)  # atomic: a killed job can never leave a half-written .pt file
    logger.info(f"Saved checkpoint: {path.name}")
    if tag == "best":
        # keep only the newest best checkpoint
        for f in glob(str(MODEL_DIR / f"{MODEL_PREFIX}_*_best_*.pt")):
            if f != str(path):
                os.remove(f)


@torch.no_grad()
def evaluate_val_loss(model, val_loader, schedule, device, max_batches=50):
    """Held-out validation loss (same v-prediction MSE as training),
    used for checkpoint selection / early-stopping decisions. Capped at
    `max_batches` per call to keep this cheap relative to a training
    epoch; increase if you want a lower-variance estimate."""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True)
        u = schedule.sample_u(x.shape[0])
        lam = schedule.u_to_lambda(u)
        x_t, _eps, v_target = schedule.noisify(x, lam)
        v_pred = model(x_t, lam, cond)
        loss = nn.functional.mse_loss(v_pred, v_target)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading static fields and ERA5 stack...")
    era5_idx, era5_data, era5_lat, era5_lon = load_era5_stack(
        DATA_DIR, TRAIN_YEARS[0] if TEST_YEARS[0] > TRAIN_YEARS[0] else TEST_YEARS[0], TRAIN_YEARS[1])
    # NOTE: loads the full 1940-2025 span once; train/val/test datasets below
    # each filter this same in-memory array by year range.
    elev, _, _ = load_static_field(DATA_DIR / "ERA5_elevation.nc", "elev", era5_lat, era5_lon)
    lsm, _, _ = load_static_field(DATA_DIR / "ERA5_lsm.nc", "lsm", era5_lat, era5_lon)
    sin_doy, cos_doy = load_doy_fourier(DATA_DIR)

    # Normalize temperature and elevation (per-field z-score); lsm and DOY
    # trig values are already in sensible ranges and left unscaled.
    t_mean, t_std = float(np.nanmean(era5_data)), float(np.nanstd(era5_data))
    era5_data = (era5_data - t_mean) / t_std
    elev = (elev - float(elev.mean())) / (float(elev.std()) + 1e-6)
    logger.info(f"Temperature normalization: mean={t_mean:.3f}, std={t_std:.3f}")
    np.save(MODEL_DIR / "norm_stats.npy", np.array([t_mean, t_std]))

    train_ds = ERA5FieldDataset(era5_idx, era5_data, elev, lsm, sin_doy, cos_doy, TRAIN_YEARS)
    val_ds = ERA5FieldDataset(era5_idx, era5_data, elev, lsm, sin_doy, cos_doy, VAL_YEARS)
    logger.info(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=max(NUM_WORKERS // 2, 1), pin_memory=True)

    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    device = accelerator.device
    logger.info(f"Device: {device}")

    model = CLINUNet()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    schedule = LambdaSchedule(device=device)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    global_step, best_loss = maybe_resume(model, optimizer, accelerator)
    t_start = time.time()

    # Resume at the correct EPOCH, not epoch 1: steps_per_epoch is fixed (drop_last=True),
    # so global_step tells us exactly how many full epochs are already done. Without this,
    # every resumed job would silently restart the full EPOCHS-epoch target from scratch
    # (same weights, but training would never actually finish).
    steps_per_epoch = len(train_loader)
    start_epoch = global_step // steps_per_epoch + 1
    if start_epoch > EPOCHS:
        logger.info(f"Training already complete: global_step={global_step} >= "
                     f"{EPOCHS} epochs x {steps_per_epoch} steps/epoch. Nothing to do.")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        loss_running, n_seen = 0.0, 0
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)

            u = schedule.sample_u(x.shape[0])
            lam = schedule.u_to_lambda(u)
            x_t, _eps, v_target = schedule.noisify(x, lam)

            v_pred = model(x_t, lam, cond)
            loss = nn.functional.mse_loss(v_pred, v_target)

            loss_running = (loss_running * n_seen + loss.item()) / (n_seen + 1)
            n_seen += 1

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            pbar.set_postfix(mse=f"{loss.item():.4f}", running=f"{loss_running:.4f}", step=global_step)

            if global_step % CHECK_STEPS == 0:
                gc.collect()
                torch.cuda.empty_cache()
                val_loss = evaluate_val_loss(model, val_loader, schedule, device)
                logger.info(f"step={global_step}  train_running={loss_running:.4f}  val_loss={val_loss:.4f}")
                if accelerator.is_main_process:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        save_checkpoint(accelerator.unwrap_model(model), optimizer, global_step, val_loss,
                                        tag="best", best_loss=best_loss)
                    else:
                        save_checkpoint(accelerator.unwrap_model(model), optimizer, global_step, val_loss,
                                        best_loss=best_loss)
                    prune_checkpoints()
                gc.collect()
                torch.cuda.empty_cache()  # avoid fragmented memory right after eval + checkpoint save

        logger.info(f"Epoch {epoch} done  train_running={loss_running:.4f}  best_val={best_loss:.4f}  "
                    f"elapsed={(time.time()-t_start)/60:.1f}min")


# =============================================================================
# Sampling: unconditional (QC) and CLIN reconstruction (conditioned on a
# partial known field via masked reverse diffusion, with optional RePaint
# resampling)
# =============================================================================

def _step_sequence(n_steps, device):
    u_seq = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    return u_seq


@torch.no_grad()
def sample_unconditional(model, schedule, static_cond, n_steps=SAMPLING_STEPS, device="cuda"):
    """static_cond: (B, N_STATIC_COND, H, W). Returns (B, CHANNEL, H, W)."""
    B = static_cond.shape[0]
    x = torch.randn((B, CHANNEL) + DOMAIN_SHAPE, device=device)
    u_seq = _step_sequence(n_steps, device)

    for t in range(n_steps):
        u_now, u_next = u_seq[t], u_seq[t + 1]
        lam_now = schedule.u_to_lambda(torch.full((B,), u_now, device=device))
        lam_next = schedule.u_to_lambda(torch.full((B,), u_next, device=device))
        v_pred = model(x, lam_now, static_cond)
        x0, eps = schedule.x0_eps_from_v(x, v_pred, lam_now)
        a_next = schedule.alpha(lam_next).view(-1, 1, 1, 1)
        s_next = schedule.sigma(lam_next).view(-1, 1, 1, 1)
        x = a_next * x0 + s_next * eps
    return x


@torch.no_grad()
def clin_reconstruct(model, schedule, known_value, known_mask, static_cond,
                      n_steps=SAMPLING_STEPS, device="cuda",
                      repaint_jump=0, repaint_resample=1):
    """CLIN reconstruction: reverse-diffusion conditioned on a partial
    known field. Supports batch size B >= 1 -- pass B different
    (known_value, known_mask, static_cond) triples stacked along dim 0 to
    reconstruct many dates in parallel on the GPU (recommended: sequential
    denoising steps can't be parallelized, but different samples can, so
    batching is the main lever for making a large sweep tractable).

    known_value: (B, CHANNEL, H, W) normalized target values at observed
        cells (arbitrary elsewhere).
    known_mask: (B, 1, H, W) boolean/float, 1 where known_value is a real
        observation.
    static_cond: (B, N_STATIC_COND, H, W).
    repaint_jump / repaint_resample: set repaint_jump=0 (default) to
        reproduce the original single-pass CLIN masking exactly. Set
        repaint_jump > 0 to enable RePaint-style resampling (Lugmayr et
        al. 2022): after every `repaint_jump` steps, re-noise back up by
        that many steps and repeat the denoising `repaint_resample` times
        before continuing forward, to improve consistency between the
        known and unknown regions. This increases sampling cost by
        roughly a factor of `repaint_resample` and should be validated on
        held-out data before committing to it for the full reconstruction.
    """
    B = known_value.shape[0]
    x = torch.randn_like(known_value)
    u_seq = _step_sequence(n_steps, device)
    mask = known_mask.float()

    def lam_at(u_scalar):
        return schedule.u_to_lambda(torch.full((B,), u_scalar, device=device))

    t = 0
    while t < n_steps:
        u_now, u_next = u_seq[t], u_seq[t + 1]
        lam_now = lam_at(u_now)
        lam_next = lam_at(u_next)

        # Diffuse the known field to the current noise level and blend.
        known_t, _, _ = schedule.noisify(known_value, lam_now)
        x = mask * known_t + (1 - mask) * x

        v_pred = model(x, lam_now, static_cond)
        x0, eps = schedule.x0_eps_from_v(x, v_pred, lam_now)
        a_next = schedule.alpha(lam_next).view(-1, 1, 1, 1)
        s_next = schedule.sigma(lam_next).view(-1, 1, 1, 1)
        x = a_next * x0 + s_next * eps

        t += 1

        if repaint_jump > 0 and t % repaint_jump == 0 and t < n_steps:
            for _ in range(repaint_resample - 1):
                # Re-noise back up by `repaint_jump` steps then re-denoise,
                # per the RePaint algorithm.
                u_back = u_seq[max(t - repaint_jump, 0)]
                lam_back = lam_at(u_back)
                a_back = schedule.alpha(lam_back).view(-1, 1, 1, 1)
                s_back = schedule.sigma(lam_back).view(-1, 1, 1, 1)
                x = a_back * x0 + s_back * torch.randn_like(x)
                for tb in range(max(t - repaint_jump, 0), t):
                    u_b_now, u_b_next = u_seq[tb], u_seq[tb + 1]
                    lam_b_now = lam_at(u_b_now)
                    lam_b_next = lam_at(u_b_next)
                    known_bt, _, _ = schedule.noisify(known_value, lam_b_now)
                    x = mask * known_bt + (1 - mask) * x
                    v_pred = model(x, lam_b_now, static_cond)
                    x0, eps = schedule.x0_eps_from_v(x, v_pred, lam_b_now)
                    a_b_next = schedule.alpha(lam_b_next).view(-1, 1, 1, 1)
                    s_b_next = schedule.sigma(lam_b_next).view(-1, 1, 1, 1)
                    x = a_b_next * x0 + s_b_next * eps

    # Final hard-set of known cells to their exact observed values.
    x = mask * known_value + (1 - mask) * x
    return x


# =============================================================================
# Historical-mask evaluation harness
# =============================================================================

def evaluate_with_historical_mask(model, schedule, era5_field_norm, static_cond,
                                   hist_day_index, hist_values, lat_idx, lon_idx,
                                   historical_day_idx_value, device="cuda",
                                   repaint_jump=0, repaint_resample=1, n_steps=SAMPLING_STEPS):
    """Overlay the REAL observation mask from a given historical date
    (day_index relative to 1667-01-01) onto a chosen ERA5 field (e.g. from
    the 1940s/1950s), run CLIN reconstruction, and return (reconstructed,
    rmse, mae, n_obs) against the true ERA5 field at the unobserved cells.

    era5_field_norm: (1, CHANNEL, H, W) normalized true field for the
        evaluation date.
    NOTE: pairing a historical mask date with a modern evaluation date is
    a scientific choice (e.g. matched by day-of-year, cycled sequentially,
    or sampled at random) left for you to decide -- this function performs
    one such pairing given explicit inputs; wrap it in a loop to build a
    full skill-vs-time-period assessment.
    """
    _, mask_grid = build_grid_mask_for_day(historical_day_idx_value, hist_day_index,
                                            hist_values, lat_idx, lon_idx)
    known_mask = torch.from_numpy(mask_grid[None, None].astype(np.float32)).to(device)
    known_value = era5_field_norm * known_mask  # true value at observed cells only

    recon = clin_reconstruct(model, schedule, known_value, known_mask, static_cond,
                              n_steps=n_steps, device=device,
                              repaint_jump=repaint_jump, repaint_resample=repaint_resample)

    unobserved = (known_mask < 0.5)
    diff = (recon - era5_field_norm)[unobserved]
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    mae = torch.mean(torch.abs(diff)).item()
    n_obs = int(known_mask.sum().item())
    return recon, rmse, mae, n_obs


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train"], help="Currently supports 'train'; "
                         "call sample_unconditional()/clin_reconstruct()/"
                         "evaluate_with_historical_mask() directly from a separate "
                         "script or notebook for inference, after loading a checkpoint.")
    args = parser.parse_args()
    if args.mode == "train":
        train()

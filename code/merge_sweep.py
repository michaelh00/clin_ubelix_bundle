"""Merge all sweep chunk CSVs and make the error-vs-year plot.
Run on the login node or in a short job:  python merge_sweep.py
"""
import os
from glob import glob
from pathlib import Path

import pandas as pd

from target_period_sweep import plot_error_across_period

out_dir = Path(os.environ["CLIN_RESULTS_DIR"])
files = sorted(glob(str(out_dir / "sweep_task*_of*.csv")))
print(f"Found {len(files)} chunk files")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values("historical_day_index").reset_index(drop=True)
df.to_csv(out_dir / "target_period_reconstruction_results.csv", index=False)
print(f"{len(df)} days merged")
plot_error_across_period(df, out_path=str(out_dir / "reconstruction_error_vs_target_period.png"))

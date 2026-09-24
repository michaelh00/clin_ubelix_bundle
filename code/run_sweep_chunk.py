"""One task of the target-period sweep (used by sweep_array.sh).

Task k of N processes every N-th historical day starting at offset k, so each
task covers the whole 1667-1939 period and a partially finished sweep is still
a uniform sample. Results go to $CLIN_RESULTS_DIR/sweep_taskKKK_ofNNN.csv.
Finished tasks are skipped, so re-submitting only redoes the missing ones.

ASSUMES the sweep script is saved as target_period_sweep.py and was edited to
accept `start_offset` (see instructions). Adjust the import if you named it differently.
"""
import os
import sys
from pathlib import Path

from target_period_sweep import run_target_period_sweep

task, n_tasks = int(sys.argv[1]), int(sys.argv[2])
out_dir = Path(os.environ["CLIN_RESULTS_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / f"sweep_task{task:03d}_of{n_tasks:03d}.csv"

if csv_path.exists():
    print(f"{csv_path.name} already exists - skipping.")
    sys.exit(0)

df = run_target_period_sweep(
    stride_days=n_tasks,
    start_offset=task,
    batch_size=32,
    repaint_jump=0,          # set > 0 to test RePaint (much slower)
    seed=task,
)
tmp = csv_path.with_suffix(".tmp")
df.to_csv(tmp, index=False)
tmp.rename(csv_path)          # only appears once fully written
print(f"Wrote {csv_path} ({len(df)} rows)")

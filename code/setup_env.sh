#!/bin/bash
# One-time setup on the UBELIX login node (submit.unibe.ch).
# Creates the project folders (default: $HOME/clin_era5) and a Python venv.
#
# Usage:   bash setup_env.sh                 # project in $HOME/clin_era5
#          bash setup_env.sh /some/other/dir # project somewhere else
# If the default Python module is not what you want, check `module avail Python`
# and run e.g.  PYTHON_MODULE=Python/3.11.3-GCCcore-12.3.0 bash setup_env.sh
set -e

PROJECT=${1:-$HOME/clin_era5}
mkdir -p "$PROJECT"/{code,data,model,results,logs}
[ -w "$PROJECT" ] || { echo "ERROR: $PROJECT is not writable" >&2; exit 1; }
echo "Project folder: $PROJECT"

# Remember the choice for all job scripts (env.sh sits next to this script)
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -f "$HERE/env.sh" ]; then
  sed -i "s|^export PROJECT=.*|export PROJECT=$PROJECT|" "$HERE/env.sh"
  sed -i "s|^export PYTHON_MODULE=.*|export PYTHON_MODULE=${PYTHON_MODULE:-Python}|" "$HERE/env.sh"
  echo "Wrote PROJECT=$PROJECT and PYTHON_MODULE=${PYTHON_MODULE:-Python} into env.sh"
fi

module load "${PYTHON_MODULE:-Python}"

python3 -m venv "$PROJECT/env"
source "$PROJECT/env/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch accelerate netCDF4 numpy pandas tqdm matplotlib

python - << 'PY'
import torch, accelerate, netCDF4, numpy, pandas
print("torch", torch.__version__, "| accelerate", accelerate.__version__,
      "| netCDF4", netCDF4.__version__)
print("(CUDA is not visible on the login node - that is expected; test on a GPU node with test_gpu.sh)")
PY

du -sh "$PROJECT/env"
echo "Done. Next: upload data to $PROJECT/data and code to $PROJECT/code"

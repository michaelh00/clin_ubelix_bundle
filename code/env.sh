#!/bin/bash
# Sourced by all job scripts: defines where data / models / results live.
# The project path and Python module are written here by setup_env.sh (or edit by hand).
export PROJECT=$HOME/clin_era5
export PYTHON_MODULE=Python
export CLIN_DATA_DIR=$PROJECT/data
export CLIN_MODEL_DIR=$PROJECT/model
export CLIN_RESULTS_DIR=$PROJECT/results
# The venv was built with this module's Python, so it must be loaded before activating.
module load "$PYTHON_MODULE"
source "$PROJECT/env/bin/activate"
export PYTHONPATH=$PROJECT/code:$PYTHONPATH

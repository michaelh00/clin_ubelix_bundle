#!/bin/bash
# Submit a chain of N training jobs so that the run continues automatically
# whether a job is killed by the 96h time limit, crashes (e.g. a transient
# OOM), or finishes early -- afterany triggers the next job in all of these
# cases. Each job resumes from the latest checkpoint (train.sh -> maybe_resume),
# and the training script itself now detects when EPOCHS is reached and exits
# immediately instead of restarting, so a job that starts after training is
# already done just ends in a few seconds rather than repeating it.
#
# Usage:   bash submit_chain.sh [N]      (default N=3, i.e. up to 3*96 = 288h)
# To stop the chain before it completes: cancel the jobs that have not
# started yet (this script prints their IDs; queued/pending ones are safe to
# cancel -- the currently running one keeps its checkpoints either way):
#   scancel <jobid2> <jobid3> ...
set -e
N=${1:-3}

prev=""
ids=()
for i in $(seq 1 "$N"); do
  if [ -z "$prev" ]; then
    id=$(sbatch --parsable train.sh)
  else
    id=$(sbatch --parsable --dependency=afterany:"$prev" train.sh)
  fi
  ids+=("$id")
  prev=$id
done

echo "Submitted chain of $N jobs: ${ids[*]}"
echo "Monitor with:   squeue --me"
echo "Stop early with: scancel ${ids[*]:1}"

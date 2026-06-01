#!/bin/bash
#
# Submit the full experiment grid: 3 methods × 2 splits × 3 seeds = 18 jobs.
#
# Run from the project root on the cluster:
#   cd /tudelft.net/staff-umbrella/<project>/scalable-systems-ravan
#   bash jobs/sweep.sh
#
# Results are written to results/ in that same directory.
# Slurm logs go to logs/<jobname>_<jobid>.out/.err
#

set -e

SEEDS="0 1 2"
SPLITS="iid noniid"

# Slurm opens the log file BEFORE the job script runs, so the logs/ dir
# must exist at submission time, not inside the job script.
mkdir -p logs results

echo "=== FedIT ==="
for split in $SPLITS; do
  for seed in $SEEDS; do
    sbatch jobs/submit_fedit.sh --split $split --seed $seed
  done
done

echo "=== Ravan Gram-Schmidt ==="
for split in $SPLITS; do
  for seed in $SEEDS; do
    sbatch jobs/submit_ravan_gs.sh --split $split --seed $seed
  done
done

echo "=== Ravan SVD warm-up ==="
for split in $SPLITS; do
  for seed in $SEEDS; do
    sbatch jobs/submit_ravan_svd.sh --split $split --seed $seed
  done
done

echo ""
echo "Submitted 18 jobs.  Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/fedit_<jobid>.out"
echo ""
echo "After all jobs complete, download results:"
echo "  rsync --progress -avz --no-perms <NetID>@login.daic.tudelft.nl:\$(pwd)/results/ ./results/"
echo "  rsync --progress -avz --no-perms <NetID>@login.daic.tudelft.nl:\$(pwd)/results/all_results.csv ./results/"

#!/bin/bash
#
# Submit the full experiment grid: 3 methods × 2 splits × 3 seeds = 18 jobs.
# Run from the project root: bash jobs/sweep.sh
#

SEEDS="0 1 2"
SPLITS="iid noniid"

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

echo "Done — check queue with: squeue -u \$USER"

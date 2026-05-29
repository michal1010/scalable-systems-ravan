#!/bin/sh
#
# FedIT job — DAIC cluster (Slurm)
#
# Submit a single FedIT run:
#   sbatch jobs/submit_fedit.sh --split noniid --seed 0
#
# To sweep seeds / splits, loop externally:
#   for seed in 0 1 2; do for split in iid noniid; do
#       sbatch jobs/submit_fedit.sh --split $split --seed $seed
#   done; done
#
#SBATCH --job-name=fedit
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --gres=gpu:1
#SBATCH --output=logs/fedit_%j.out
#SBATCH --error=logs/fedit_%j.err

# ── environment ──────────────────────────────────────────────────────────────
module use /opt/insy/modulefiles
module load miniconda
conda activate ravan          # change to your env name

export HF_HOME=$HOME/.cache/huggingface

# ── run ──────────────────────────────────────────────────────────────────────
mkdir -p logs results

srun python -m federated.train_fedit \
    --split noniid \
    --seed 0 \
    --rounds 50 \
    --clients 20 \
    --clients_per_round 3 \
    --local_steps 50 \
    --rank 8 \
    --lr 1e-3 \
    --batch_size 16 \
    "$@"
#             ^^^ extra CLI args forwarded from sbatch arguments

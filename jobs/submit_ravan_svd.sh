#!/bin/sh
#
# Ravan (SVD warm-up init) job — DAIC cluster (Slurm)
#
# The SVD warm-up adds a short federated LoRA phase before the main
# Ravan training.  Extra time budget is included in the wall-clock limit.
#
# Usage:
#   sbatch jobs/submit_ravan_svd.sh
#   sbatch jobs/submit_ravan_svd.sh --split iid --seed 2
#
#SBATCH --job-name=ravan_svd
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ravan_svd_%j.out
#SBATCH --error=logs/ravan_svd_%j.err

module use /opt/insy/modulefiles
module load miniconda
conda activate ravan

mkdir -p logs results

srun python -m federated.train_ravan \
    --init svd \
    --split noniid \
    --seed 0 \
    --rounds 50 \
    --clients 20 \
    --clients_per_round 3 \
    --local_steps 50 \
    --heads 4 \
    --rank 55 \
    --lr 5e-4 \
    --batch_size 16 \
    --warmup_clients 5 \
    --warmup_steps 50 \
    "$@"

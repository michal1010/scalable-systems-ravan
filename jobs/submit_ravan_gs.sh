#!/bin/sh
#
# Ravan (Gram-Schmidt init) job — DAIC cluster (Slurm)
#
# Usage:
#   sbatch jobs/submit_ravan_gs.sh
#   sbatch jobs/submit_ravan_gs.sh --split iid --seed 1
#
#SBATCH --job-name=ravan_gs
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ravan_gs_%j.out
#SBATCH --error=logs/ravan_gs_%j.err

cd $SLURM_SUBMIT_DIR

module use /opt/insy/modulefiles
module load miniconda
conda activate ravan

export HF_HOME=$HOME/.cache/huggingface

mkdir -p results

srun python -m federated.train_ravan \
    --init gram_schmidt \
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
    "$@"

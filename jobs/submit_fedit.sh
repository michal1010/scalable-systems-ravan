#!/bin/bash
#
# FedIT job — DAIC cluster (Slurm)
#
# Prerequisites (run once from login node):
#   mkdir -p logs           # Slurm writes the log BEFORE the script runs
#
# Submit a single run (from project root on the cluster):
#   sbatch jobs/submit_fedit.sh
#   sbatch jobs/submit_fedit.sh --split iid --seed 1
#
# The full 18-job sweep is driven by jobs/sweep.sh
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

# ── working directory ─────────────────────────────────────────────────────────
# $SLURM_SUBMIT_DIR is the directory where sbatch was called from.
# Always submit from the project root on project storage, e.g.:
#   /tudelft.net/staff-umbrella/<project>/scalable-systems-ravan/
source "${SLURM_SUBMIT_DIR}/jobs/container/container_env.sh"

# ── run ───────────────────────────────────────────────────────────────────────
run_in_container python -m federated.train_fedit \
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
#             ^^^ extra CLI args forwarded when calling sbatch ... --split iid --seed 2

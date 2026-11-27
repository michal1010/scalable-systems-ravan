#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

check_cuda

"$PYTHON_BIN" -m federated.train_fedit \
    --split noniid \
    --seed 0 \
    --rounds 50 \
    --clients 20 \
    --clients_per_round 3 \
    --local_steps 50 \
    --rank 8 \
    --lr 1e-3 \
    --batch_size 16 \
    --device cuda \
    --output_dir "${RESULTS_ROOT}/single" \
    "$@"

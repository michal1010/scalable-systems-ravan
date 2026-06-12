#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

check_cuda

"$PYTHON_BIN" -m federated.train_ravan \
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
    --device cuda \
    --output_dir "${RESULTS_ROOT}/single" \
    "$@"

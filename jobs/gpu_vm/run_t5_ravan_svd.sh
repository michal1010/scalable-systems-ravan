#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

check_cuda

T5_ROUNDS="${T5_ROUNDS:-100}"
T5_CLIENTS="${T5_CLIENTS:-20}"
T5_CLIENTS_PER_ROUND="${T5_CLIENTS_PER_ROUND:-3}"
T5_LOCAL_STEPS="${T5_LOCAL_STEPS:-50}"
T5_BATCH_SIZE="${T5_BATCH_SIZE:-16}"
T5_GRAD_ACCUM_STEPS="${T5_GRAD_ACCUM_STEPS:-2}"
T5_MAX_LENGTH="${T5_MAX_LENGTH:-256}"
T5_ALPHA="${T5_ALPHA:-0.3}"
T5_RAVAN_HEADS="${T5_RAVAN_HEADS:-4}"
T5_RAVAN_RANK="${T5_RAVAN_RANK:-110}"
T5_RAVAN_LR="${T5_RAVAN_LR:-5e-4}"
T5_WARMUP_CLIENTS="${T5_WARMUP_CLIENTS:-5}"
T5_WARMUP_STEPS="${T5_WARMUP_STEPS:-50}"
T5_WARMUP_WEIGHTING="${T5_WARMUP_WEIGHTING:-uniform}"
T5_USE_AMP="${T5_USE_AMP:-1}"
T5_GRAD_CHECKPOINT="${T5_GRAD_CHECKPOINT:-1}"

EXTRA_ARGS=()
if [ "$T5_USE_AMP" != "0" ]; then
    EXTRA_ARGS+=(--use_amp)
fi
if [ "$T5_GRAD_CHECKPOINT" != "0" ]; then
    EXTRA_ARGS+=(--grad_checkpoint)
fi

"$PYTHON_BIN" -m federated.train_ravan \
    --model_type t5 \
    --init svd \
    --split noniid \
    --seed 0 \
    --rounds "$T5_ROUNDS" \
    --clients "$T5_CLIENTS" \
    --clients_per_round "$T5_CLIENTS_PER_ROUND" \
    --local_steps "$T5_LOCAL_STEPS" \
    --heads "$T5_RAVAN_HEADS" \
    --rank "$T5_RAVAN_RANK" \
    --lr "$T5_RAVAN_LR" \
    --batch_size "$T5_BATCH_SIZE" \
    --grad_accum_steps "$T5_GRAD_ACCUM_STEPS" \
    --max_length "$T5_MAX_LENGTH" \
    --dirichlet_alpha "$T5_ALPHA" \
    --warmup_clients "$T5_WARMUP_CLIENTS" \
    --warmup_steps "$T5_WARMUP_STEPS" \
    --warmup_weighting "$T5_WARMUP_WEIGHTING" \
    --device cuda \
    --output_dir "${RESULTS_ROOT}/t5_single" \
    "${EXTRA_ARGS[@]}" \
    "$@"

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
T5_NO_LORA_LR="${T5_NO_LORA_LR:-1e-3}"
T5_USE_AMP="${T5_USE_AMP:-1}"
T5_GRAD_CHECKPOINT="${T5_GRAD_CHECKPOINT:-1}"

EXTRA_ARGS=()
if [ "$T5_USE_AMP" != "0" ]; then
    EXTRA_ARGS+=(--use_amp)
fi
if [ "$T5_GRAD_CHECKPOINT" != "0" ]; then
    EXTRA_ARGS+=(--grad_checkpoint)
fi

"$PYTHON_BIN" -m federated.train_no_lora \
    --model_type t5 \
    --split noniid \
    --seed 0 \
    --rounds "$T5_ROUNDS" \
    --clients "$T5_CLIENTS" \
    --clients_per_round "$T5_CLIENTS_PER_ROUND" \
    --local_steps "$T5_LOCAL_STEPS" \
    --rank 0 \
    --lr "$T5_NO_LORA_LR" \
    --batch_size "$T5_BATCH_SIZE" \
    --grad_accum_steps "$T5_GRAD_ACCUM_STEPS" \
    --max_length "$T5_MAX_LENGTH" \
    --dirichlet_alpha "$T5_ALPHA" \
    --device cuda \
    --output_dir "${RESULTS_ROOT}/t5_single" \
    "${EXTRA_ARGS[@]}" \
    "$@"

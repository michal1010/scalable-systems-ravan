#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
GPU_VM_DIR="${GPU_VM_DIR:-${A100_VM_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"

SEEDS="${SEEDS:-0}"
SPLITS="${SPLITS:-iid noniid}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
POLL_SECONDS="${POLL_SECONDS:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

SWEEP_DIR="${SWEEP_DIR:-${RESULTS_ROOT}/t5_minimal_${RUN_ID}}"
PARTS_DIR="${SWEEP_DIR}/parts"
FINAL_DIR="${SWEEP_DIR}/final"
SWEEP_LOG_DIR="${LOG_ROOT}/t5_minimal_${RUN_ID}"

mkdir -p "$PARTS_DIR" "$FINAL_DIR" "$SWEEP_LOG_DIR"

check_cuda

echo "T5-Base minimal reproduction"
echo "Project root : $PROJECT_ROOT"
echo "Results     : $SWEEP_DIR"
echo "Logs        : $SWEEP_LOG_DIR"
echo "Seeds       : $SEEDS"
echo "Splits      : $SPLITS"
echo "Methods     : No-LoRA, FedIT, Ravan-GS, Ravan-SVD"
echo "Parallelism : $MAX_PARALLEL"
echo "Batch       : ${T5_BATCH_SIZE:-16} x grad_accum ${T5_GRAD_ACCUM_STEPS:-2}"
echo ""

declare -a PIDS=()
declare -a NAMES=()

active_jobs() {
    jobs -pr | wc -l | tr -d ' '
}

wait_for_slot() {
    while [ "$(active_jobs)" -ge "$MAX_PARALLEL" ]; do
        sleep "$POLL_SECONDS"
    done
}

launch_job() {
    local name="$1"
    shift

    local output_dir="${PARTS_DIR}/${name}"
    local log_path="${SWEEP_LOG_DIR}/${name}.log"

    mkdir -p "$output_dir"
    wait_for_slot

    echo "Launching $name"
    (
        set -euo pipefail
        "$@" --output_dir "$output_dir" >"$log_path" 2>&1
    ) &

    PIDS+=("$!")
    NAMES+=("$name")
}

for split in $SPLITS; do
    for seed in $SEEDS; do
        launch_job "t5_no_lora_${split}_seed${seed}" \
            "${GPU_VM_DIR}/run_t5_no_lora.sh" --split "$split" --seed "$seed" "$@"
    done
done

for split in $SPLITS; do
    for seed in $SEEDS; do
        launch_job "t5_fedit_${split}_seed${seed}" \
            "${GPU_VM_DIR}/run_t5_fedit.sh" --split "$split" --seed "$seed" "$@"
    done
done

for split in $SPLITS; do
    for seed in $SEEDS; do
        launch_job "t5_ravan_gs_${split}_seed${seed}" \
            "${GPU_VM_DIR}/run_t5_ravan_gs.sh" --split "$split" --seed "$seed" "$@"
    done
done

for split in $SPLITS; do
    for seed in $SEEDS; do
        launch_job "t5_ravan_svd_${split}_seed${seed}" \
            "${GPU_VM_DIR}/run_t5_ravan_svd.sh" --split "$split" --seed "$seed" "$@"
    done
done

FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "$pid"; then
        echo "Finished $name"
    else
        echo "FAILED $name; see ${SWEEP_LOG_DIR}/${name}.log" >&2
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "At least one run failed. Partial outputs are in $PARTS_DIR" >&2
    exit 1
fi

"$PYTHON_BIN" "${GPU_VM_DIR}/aggregate_results.py" \
    --parts-dir "$PARTS_DIR" \
    --final-dir "$FINAL_DIR"

echo ""
echo "T5 minimal sweep complete."
echo "Final results: $FINAL_DIR"
echo "Logs: $SWEEP_LOG_DIR"

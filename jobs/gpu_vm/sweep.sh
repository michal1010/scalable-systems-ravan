#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEEDS="${SEEDS:-0 1 2}"
SPLITS="${SPLITS:-iid noniid}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
POLL_SECONDS="${POLL_SECONDS:-10}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FEDIT_RANK="${FEDIT_RANK:-8}"
RAVAN_HEADS="${RAVAN_HEADS:-4}"
RAVAN_RANK="${RAVAN_RANK:-55}"

usage() {
    cat <<'EOF'
Usage: bash jobs/gpu_vm/sweep.sh [options]

Options:
  --heads N          Ravan heads; alias for --ravan-heads
  --rank N           Ravan per-head rank; alias for --ravan-rank
  --ravan-heads N    Ravan heads
  --ravan-rank N     Ravan per-head rank
  --fedit-rank N     FedIT LoRA rank
  -h, --help         Show this help

Environment defaults are also supported:
  BATCH_SIZE=16 FEDIT_RANK=8 RAVAN_HEADS=4 RAVAN_RANK=55
EOF
}

SWEEP_DIR="${SWEEP_DIR:-${RESULTS_ROOT}/sweep_${RUN_ID}}"
PARTS_DIR="${SWEEP_DIR}/parts"
FINAL_DIR="${SWEEP_DIR}/final"
SWEEP_LOG_DIR="${LOG_ROOT}/sweep_${RUN_ID}"

mkdir -p "$PARTS_DIR" "$FINAL_DIR" "$SWEEP_LOG_DIR"

check_cuda

echo "Project root : $PROJECT_ROOT"
echo "Results     : $SWEEP_DIR"
echo "Logs        : $SWEEP_LOG_DIR"
echo "Seeds       : $SEEDS"
echo "Splits      : $SPLITS"
echo "Parallelism : $MAX_PARALLEL"
echo "Batch size  : $BATCH_SIZE"
echo "FedIT rank  : $FEDIT_RANK"
echo "Ravan heads : $RAVAN_HEADS"
echo "Ravan rank  : $RAVAN_RANK"
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
        launch_job "fedit_${split}_seed${seed}" \
            "${A100_VM_DIR}/run_fedit.sh" --split "$split" --seed "$seed" --rank "$FEDIT_RANK" --batch_size "$BATCH_SIZE"
    done
done

for split in $SPLITS; do
    for seed in $SEEDS; do
        launch_job "ravan_gs_${split}_seed${seed}" \
            "${A100_VM_DIR}/run_ravan_gs.sh" --split "$split" --seed "$seed" --heads "$RAVAN_HEADS" --rank "$RAVAN_RANK" --batch_size "$BATCH_SIZE"
    done
done

for split in $SPLITS; do
    for seed in $SEEDS; do
        launch_job "ravan_svd_${split}_seed${seed}" \
            "${A100_VM_DIR}/run_ravan_svd.sh" --split "$split" --seed "$seed" --heads "$RAVAN_HEADS" --rank "$RAVAN_RANK" --batch_size "$BATCH_SIZE"
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

"$PYTHON_BIN" "${A100_VM_DIR}/aggregate_results.py" \
    --parts-dir "$PARTS_DIR" \
    --final-dir "$FINAL_DIR"

echo ""
echo "Sweep complete."
echo "Final results: $FINAL_DIR"
echo "Logs: $SWEEP_LOG_DIR"

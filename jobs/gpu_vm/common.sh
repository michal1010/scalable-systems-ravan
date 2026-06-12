#!/usr/bin/env bash

if [ "${A100_VM_COMMON_LOADED:-0}" = "1" ]; then
    return 0
fi
A100_VM_COMMON_LOADED=1

A100_VM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${A100_VM_DIR}/../.." && pwd)}"

cd "$PROJECT_ROOT"

if [ -n "${CONDA_ENV:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV"
    else
        echo "CONDA_ENV is set to '$CONDA_ENV', but conda is not available." >&2
        exit 1
    fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results_a100_vm}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs_a100_vm}"

mkdir -p "$RESULTS_ROOT" "$LOG_ROOT" "${PROJECT_ROOT}/.cache/huggingface" "${PROJECT_ROOT}/.cache/matplotlib"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.cache/matplotlib}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

check_cuda() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    fi

    "$PYTHON_BIN" - <<'PY'
import sys
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch. Check the VM driver, CUDA PyTorch install, and CUDA_VISIBLE_DEVICES.")

print("CUDA device:", torch.cuda.get_device_name(0))
PY
}

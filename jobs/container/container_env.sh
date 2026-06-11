#!/bin/bash

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-${PROJECT_ROOT}/jobs/container/ravan-experiments.sif}"

cd "$PROJECT_ROOT"
mkdir -p logs results .cache/huggingface

if [ ! -f "$APPTAINER_IMAGE" ]; then
    echo "Container image not found: $APPTAINER_IMAGE" >&2
    echo "Build it with: cd jobs/container && ./build.sh" >&2
    exit 1
fi

module use /opt/insy/modulefiles 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true

export APPTAINERENV_HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="${PROJECT_ROOT}"

ENV_FILE_ARGS=()
if [ -f "${HOME}/.env" ]; then
    ENV_FILE_ARGS=(--env-file "${HOME}/.env")
fi

run_in_container() {
    srun apptainer exec \
        --nv \
        "${ENV_FILE_ARGS[@]}" \
        -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
        --pwd "${PROJECT_ROOT}" \
        "${APPTAINER_IMAGE}" \
        "$@"
}

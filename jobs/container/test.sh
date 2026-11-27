#!/bin/bash
set -e

cd "$(dirname "$0")"
IMAGE="ravan-experiments.sif"
PROJECT_ROOT="$(cd ../.. && pwd)"

if [ ! -f "$IMAGE" ]; then
    echo "Container image not found: $IMAGE"
    echo "Run ./build.sh first."
    exit 1
fi

echo "Testing Python inside container..."
APPTAINERENV_PYTHONNOUSERSITE=1 apptainer exec "$IMAGE" which python
APPTAINERENV_PYTHONNOUSERSITE=1 apptainer exec "$IMAGE" python --version

echo "Testing RAVAN experiment dependencies..."
APPTAINERENV_PYTHONNOUSERSITE=1 \
APPTAINERENV_PYTHONPATH="$PROJECT_ROOT" \
apptainer exec \
    --nv \
    -B "$PROJECT_ROOT:$PROJECT_ROOT" \
    --pwd "$PROJECT_ROOT" \
    "$IMAGE" \
    python jobs/container/test-installation.py

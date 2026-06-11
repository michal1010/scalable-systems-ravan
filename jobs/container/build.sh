#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p .apptainer-tmp .apptainer-cache
export APPTAINER_TMPDIR="$PWD/.apptainer-tmp"
export APPTAINER_CACHEDIR="$PWD/.apptainer-cache"

apptainer build \
    --mksquashfs-args "-processors 1" \
    ravan-experiments.sif \
    Apptainer.def 2>&1 | tee build.log


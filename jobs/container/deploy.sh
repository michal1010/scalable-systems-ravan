#!/bin/bash
set -e

cd "$(dirname "$0")"

IMAGE="ravan-experiments.sif"

if [ ! -f "$IMAGE" ]; then
    echo "Container image not found: $IMAGE"
    echo "Run ./build.sh first."
    exit 1
fi

if [ "$#" -ne 1 ]; then
    echo "Usage: ./deploy.sh <destination>"
    echo "Example: ./deploy.sh daic:/tudelft.net/staff-umbrella/<project>/apptainer/ravan-experiments.sif"
    exit 1
fi

DEST="$1"
rsync -avc --progress "$IMAGE" "$DEST"

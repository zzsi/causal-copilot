#!/bin/bash
# Build and run the Causal-Copilot Docker container.
# Usage: ./build_and_run_docker.sh [cpu|gpu]

set -e

DEVICE=${1:-cpu}
IMAGE="causal-copilot:${DEVICE}"

if [[ "$DEVICE" == "gpu" ]]; then
    DOCKERFILE="Dockerfile.gpu"
    GPU_FLAG="--gpus all"
else
    DOCKERFILE="Dockerfile.cpu"
    GPU_FLAG=""
fi

echo "Building $IMAGE using $DOCKERFILE ..."
docker build -t "$IMAGE" -f "$DOCKERFILE" .

echo "Running $IMAGE ..."
docker run --rm -it $GPU_FLAG -v "$(pwd)":/app "$IMAGE"

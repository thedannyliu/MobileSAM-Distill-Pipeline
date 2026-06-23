#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-mobilesam-distill:latest}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PWD}/artifacts}"
mkdir -p "${ARTIFACT_ROOT}"/{data,features,checkpoints,outputs,cache}

docker run --rm --gpus all \
  -v "${ARTIFACT_ROOT}:/artifacts" \
  "${IMAGE_NAME}" \
  mobilesam-smoke-import

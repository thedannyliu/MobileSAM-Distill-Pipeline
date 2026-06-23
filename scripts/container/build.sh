#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-mobilesam-distill:latest}"
MOBILE_SAM_REPO="${MOBILE_SAM_REPO:-https://github.com/ChaoningZhang/MobileSAM.git}"
MOBILE_SAM_COMMIT="${MOBILE_SAM_COMMIT:-01ea8d0f5590082f0c1ceb0a3e2272593f20154b}"

docker build \
  --build-arg "MOBILE_SAM_REPO=${MOBILE_SAM_REPO}" \
  --build-arg "MOBILE_SAM_COMMIT=${MOBILE_SAM_COMMIT}" \
  -t "${IMAGE_NAME}" \
  .

#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PWD}/artifacts}"
COCO_ROOT="${COCO_ROOT:-}"
DOWNLOAD_FLAG=()
if [ -z "${COCO_ROOT}" ]; then
  DOWNLOAD_FLAG=(--download)
fi

python -m mobilesam_distill.data.coco10 \
  ${COCO_ROOT:+--coco_root "${COCO_ROOT}"} \
  --output_root "${ARTIFACT_ROOT}/data/coco_val5" \
  --num_images 5 \
  --seed "${COCO_VAL5_SEED:-1234}" \
  "${DOWNLOAD_FLAG[@]}"

#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PWD}/artifacts}"
COCO_ROOT="${COCO_ROOT:-}"
DOWNLOAD_FLAG=()
if [ -z "${COCO_ROOT}" ]; then
  DOWNLOAD_FLAG=(--download)
fi

PYTHON="${PYTHON:-}"
if [ -z "${PYTHON}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  else
    echo "ERROR: Python not found. Install Python 3 or set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

"${PYTHON}" -m mobilesam_distill.data.coco10 \
  ${COCO_ROOT:+--coco_root "${COCO_ROOT}"} \
  --output_root "${ARTIFACT_ROOT}/data/coco10" \
  --num_images "${COCO10_NUM_IMAGES:-10}" \
  --seed "${COCO10_SEED:-1234}" \
  "${DOWNLOAD_FLAG[@]}"

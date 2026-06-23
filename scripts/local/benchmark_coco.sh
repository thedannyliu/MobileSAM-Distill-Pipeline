#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PWD}/artifacts}"
COCO_ROOT="${COCO_ROOT:-${ARTIFACT_ROOT}/data/coco_val5}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/outputs/bench_coco_val5}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to an aggregated MobileSAM checkpoint.}"
STUDENT_ARCH="${STUDENT_ARCH:-tinyvit}"
MAX_IMAGES="${MAX_IMAGES:-5}"
DEVICE="${DEVICE:-cuda}"

mobilesam-benchmark-coco \
  --coco_root "${COCO_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --student_arch "${STUDENT_ARCH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_images "${MAX_IMAGES}" \
  --device "${DEVICE}"

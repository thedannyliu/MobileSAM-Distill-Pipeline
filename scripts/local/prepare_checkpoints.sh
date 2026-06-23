#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PWD}/artifacts}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ARTIFACT_ROOT}/checkpoints}"
SAM_URL="${SAM_URL:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth}"
SAM_CKPT="${SAM_CKPT:-${CHECKPOINT_ROOT}/sam_vit_h_4b8939.pth}"
MOBILE_SAM_CKPT="${MOBILE_SAM_CKPT:-${CHECKPOINT_ROOT}/mobile_sam.pt}"

mkdir -p "${CHECKPOINT_ROOT}"

if [ ! -s "${MOBILE_SAM_CKPT}" ]; then
  cp weights/distilled/mobile_sam.pt "${MOBILE_SAM_CKPT}"
fi

if [ "${DOWNLOAD_SAM_TEACHER:-0}" = "1" ] && [ ! -s "${SAM_CKPT}" ]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L "${SAM_URL}" -o "${SAM_CKPT}"
  else
    wget -O "${SAM_CKPT}" "${SAM_URL}"
  fi
fi

if [ ! -s "${SAM_CKPT}" ]; then
  echo "SAM teacher checkpoint is not present: ${SAM_CKPT}" >&2
  echo "Set SAM_CKPT to an existing file or rerun with DOWNLOAD_SAM_TEACHER=1." >&2
  exit 1
fi

echo "MobileSAM checkpoint: ${MOBILE_SAM_CKPT}"
echo "SAM teacher checkpoint: ${SAM_CKPT}"

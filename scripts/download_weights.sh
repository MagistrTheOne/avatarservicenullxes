#!/usr/bin/env bash
#
# Download ARACHNE-X-ULTRA-AVATAR weights into the RunPod Network Volume.
# Run once per volume; subsequent pod starts will reuse the cached weights.
#
# Usage:
#   WEIGHTS_DIR=/models ./scripts/download_weights.sh
#
# Requires: huggingface_hub[cli] installed on the pod.
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-/models}"
TARGET_NAME="${TARGET_NAME:-ARACHNE-X-ULTRA-AVATAR}"
HF_REPO="${HF_REPO:-MagistrTheOne/ARACHNE-X-ULTRA-AVATAR}"

mkdir -p "${WEIGHTS_DIR}/${TARGET_NAME}"

echo "[+] Downloading ${HF_REPO} -> ${WEIGHTS_DIR}/${TARGET_NAME}"
python -m pip install --quiet 'huggingface_hub[cli]>=0.26.0'

huggingface-cli download \
  "${HF_REPO}" \
  --local-dir "${WEIGHTS_DIR}/${TARGET_NAME}" \
  --local-dir-use-symlinks False

echo "[+] Done. Weight tree:"
du -sh "${WEIGHTS_DIR}/${TARGET_NAME}"/* | head -30

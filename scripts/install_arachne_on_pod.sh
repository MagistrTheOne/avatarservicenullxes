#!/usr/bin/env bash
#
# Install the upstream arachne_x package on the RunPod pod.
#
# The upstream repository (https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-)
# ships the Python package ``arachne_x`` that this service imports lazily when
# ``ARACHNE_MODE=real``. Keep this script idempotent.
set -euo pipefail

ARACHNE_REPO="${ARACHNE_REPO:-https://github.com/MagistrTheOne/ARACHNE-X-NULLXES-.git}"
ARACHNE_REV="${ARACHNE_REV:-main}"
TARGET_DIR="${TARGET_DIR:-/opt/arachne-x}"

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  echo "[+] Cloning ${ARACHNE_REPO} -> ${TARGET_DIR}"
  git clone --depth 1 --branch "${ARACHNE_REV}" "${ARACHNE_REPO}" "${TARGET_DIR}"
else
  echo "[+] Updating ${TARGET_DIR}"
  git -C "${TARGET_DIR}" fetch --depth 1 origin "${ARACHNE_REV}"
  git -C "${TARGET_DIR}" checkout "${ARACHNE_REV}"
fi

echo "[+] Installing avatar runtime requirements"
python -m pip install --no-cache-dir -r "${TARGET_DIR}/requirements.txt"
python -m pip install --no-cache-dir -r "${TARGET_DIR}/requirements_avatar.txt"

# Make `arachne_x` importable from anywhere on the pod.
PY_SITE=$(python -c "import site,sys;print(site.getsitepackages()[0])")
PTH_PATH="${PY_SITE}/arachne_x_nullxes.pth"
echo "${TARGET_DIR}" > "${PTH_PATH}"
echo "[+] Registered arachne_x import path: ${PTH_PATH}"

python -c "import arachne_x; from arachne_x.loader import load_avatar_pipeline; print('arachne_x import ok')"

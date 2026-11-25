#!/usr/bin/env bash
set -euo pipefail

# Setup isolated Python 3.11 venv and workspace caches strictly on /mnt/data0
# to avoid exhausting the small root filesystem ("/").

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${RSI_WORKSPACE_ROOT:-${REPO_ROOT}/.workspace}"
VENV_DIR="${RSI_VENV_DIR:-${REPO_ROOT}/.venv}"
WITH_GPU=0

for arg in "$@"; do
  case "$arg" in
    --with-gpu) WITH_GPU=1 ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--with-gpu]" >&2
      exit 1
      ;;
  esac
done

mkdir -p \
  "${WORKSPACE_ROOT}/hf_cache/datasets" \
  "${WORKSPACE_ROOT}/models" \
  "${WORKSPACE_ROOT}/datasets" \
  "${WORKSPACE_ROOT}/adapters" \
  "${WORKSPACE_ROOT}/runs" \
  "${WORKSPACE_ROOT}/pip_cache" \
  "${WORKSPACE_ROOT}/tmp"

export RSI_WORKSPACE_ROOT="${WORKSPACE_ROOT}"
export HF_HOME="${WORKSPACE_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${WORKSPACE_ROOT}/hf_cache/datasets"
export PIP_CACHE_DIR="${WORKSPACE_ROOT}/pip_cache"
export TMPDIR="${WORKSPACE_ROOT}/tmp"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[setup] Upgrading pip/setuptools/wheel (cache -> ${PIP_CACHE_DIR})"
pip install --upgrade pip setuptools wheel

echo "[setup] Installing rsi-agent-mvp core + dev dependencies"
pip install -e "${REPO_ROOT}[dev]"

if [[ "${WITH_GPU}" -eq 1 ]]; then
  echo "[setup] Installing CUDA 13 / Blackwell (sm_120) compatible PyTorch + data & GPU ML stack"
  TMPDIR="${WORKSPACE_ROOT}/tmp" pip install \
    --cache-dir "${PIP_CACHE_DIR}" \
    -e "${REPO_ROOT}[data,gpu]"
fi

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  echo "[setup] Created default ${REPO_ROOT}/.env"
fi

echo "[setup] Done. Activate with: source ${VENV_DIR}/bin/activate"
rsi doctor --no-gpu-matmul || true

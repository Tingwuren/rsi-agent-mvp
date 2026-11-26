#!/usr/bin/env bash
set -euo pipefail

# Self-contained single-GPU (RTX 5090) v0 baseline benchmark runner:
# starts vLLM with Qwen3-8B + LoRA support, evaluates MBPP+ & MATH-500 dev_mini,
# and gracefully shuts down vLLM to release GPU memory on exit.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv/bin"
MODEL_DIR="${1:-${REPO_ROOT}/.workspace/models/Qwen3-8B}"
RUN_TAG="${2:-v0_baseline_qwen3_8b}"
OUT_BASE="${REPO_ROOT}/.workspace/runs/${RUN_TAG}"
LOG_FILE="${OUT_BASE}/vllm_server.log"

mkdir -p "${OUT_BASE}"

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[baseline] Stopping vLLM server (pid=${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[baseline] Starting vLLM on GPU 1 with model=${MODEL_DIR}"
"${VENV}/rsi" serve \
  --config "${REPO_ROOT}/configs/code_mbpp_plus.yaml" \
  --model "${MODEL_DIR}" \
  --gpu-memory-utilization 0.85 \
  >"${LOG_FILE}" 2>&1 &
VLLM_PID=$!

echo "[baseline] Waiting for http://127.0.0.1:8000/v1/models ..."
READY=0
for _ in $(seq 1 90); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[baseline] ERROR: vLLM exited early. Tail of ${LOG_FILE}:" >&2
    tail -n 40 "${LOG_FILE}" >&2
    exit 1
  fi
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
  echo "[baseline] ERROR: vLLM did not become ready within 180s." >&2
  tail -n 40 "${LOG_FILE}" >&2
  exit 1
fi

echo "[baseline] vLLM ready! Running MBPP+ dev_mini (32 tasks)..."
"${VENV}/rsi" eval \
  --config "${REPO_ROOT}/configs/code_mbpp_plus.yaml" \
  --split dev \
  --max-samples 32 \
  --concurrency 8 \
  --output "${OUT_BASE}/mbpp_plus_dev32"

echo "[baseline] Running MATH-500 dev_mini (20 tasks)..."
"${VENV}/rsi" eval \
  --config "${REPO_ROOT}/configs/math_500.yaml" \
  --split dev \
  --max-samples 20 \
  --concurrency 8 \
  --output "${OUT_BASE}/math_500_dev20"

echo "[baseline] Finished! Artifacts saved to ${OUT_BASE}"

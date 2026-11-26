#!/usr/bin/env bash
set -euo pipefail

# Self-contained single-GPU (RTX 5090) 3-generation Scaffold RSI live experiment:
# starts vLLM (Qwen3-8B + LoRA), runs `rsi run` for 3 generations, prints lineage
# & best diff, and gracefully stops vLLM on exit.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv/bin"
MODEL_DIR="${1:-${REPO_ROOT}/.workspace/models/Qwen3-8B}"
CONFIG_PATH="${2:-${REPO_ROOT}/configs/code_mbpp_plus_m3_live.yaml}"
RUN_DIR="${3:-${REPO_ROOT}/.workspace/runs/m3_live_qwen3_8b_3gen}"
LOG_FILE="${RUN_DIR}/vllm_server.log"

mkdir -p "${RUN_DIR}"

VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[m3-live] Stopping vLLM server (pid=${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[m3-live] Starting vLLM on GPU 1 with model=${MODEL_DIR}"
"${VENV}/rsi" serve \
  --config "${CONFIG_PATH}" \
  --model "${MODEL_DIR}" \
  --gpu-memory-utilization 0.80 \
  >"${LOG_FILE}" 2>&1 &
VLLM_PID=$!

echo "[m3-live] Waiting for http://127.0.0.1:8000/v1/models ..."
READY=0
for _ in $(seq 1 90); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[m3-live] ERROR: vLLM exited early. Tail of ${LOG_FILE}:" >&2
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
  echo "[m3-live] ERROR: vLLM did not become ready within 180s." >&2
  tail -n 40 "${LOG_FILE}" >&2
  exit 1
fi

echo "[m3-live] vLLM ready! Launching 3-generation Scaffold RSI..."
# Copy config into RUN_DIR so we run cleanly in a deterministic directory
cp "${CONFIG_PATH}" "${RUN_DIR}/config.yaml"
"${VENV}/rsi" run --resume "${RUN_DIR}" --max-generations 3

echo "[m3-live] Final lineage inspection:"
"${VENV}/rsi" inspect --run "${RUN_DIR}"

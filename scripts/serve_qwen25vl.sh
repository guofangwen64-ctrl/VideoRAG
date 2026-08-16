#!/usr/bin/env bash
# Run this from the dedicated vLLM environment, not the project .venv.
set -euo pipefail

: "${VLM_HOST:=127.0.0.1}"
: "${VLM_PORT:=8000}"
: "${VLM_GPU_MEMORY_UTILIZATION:=0.90}"

exec vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host "$VLM_HOST" \
  --port "$VLM_PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len 8192 \
  --limit-mm-per-prompt image=16

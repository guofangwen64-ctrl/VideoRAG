#!/usr/bin/env bash
# Dedicated Qwen2.5-VL service for 64-frame VGent clip description.
# Keep this separate from serve_qwen25vl.sh, which preserves the original QA baseline.
set -euo pipefail

: "${VLM_HOST:=127.0.0.1}"
: "${VLM_PORT:=8001}"
: "${VLM_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VLM_MAX_MODEL_LEN:=32768}"
: "${VLM_MODEL:=Qwen/Qwen2.5-VL-7B-Instruct}"
: "${VLM_SERVED_MODEL_NAME:=Qwen/Qwen2.5-VL-7B-Instruct}"

exec vllm serve "$VLM_MODEL" \
  --served-model-name "$VLM_SERVED_MODEL_NAME" \
  --host "$VLM_HOST" \
  --port "$VLM_PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLM_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$VLM_MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --limit-mm-per-prompt image=64

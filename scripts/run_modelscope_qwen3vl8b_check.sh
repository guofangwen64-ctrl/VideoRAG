#!/usr/bin/env bash
# One-question API connectivity and image-payload validation.
set -euo pipefail

: "${MODELSCOPE_ACCESS_TOKEN:?Set MODELSCOPE_ACCESS_TOKEN to your ModelScope Token first}"

medrag answer \
  --config configs/reader_modelscope_qwen3vl8b.yaml \
  --annotations medhorizon_test.jsonl \
  --top-k 1 \
  --limit 1 \
  --output artifacts/qa_modelscope_qwen3vl8b_check.jsonl

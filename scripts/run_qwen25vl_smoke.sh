#!/usr/bin/env bash
# Run this from the project virtual environment after the vLLM server is ready.
set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

medrag answer \
  --config configs/reader_qwen25vl.yaml \
  --annotations medhorizon_test.jsonl \
  --top-k 1 \
  --limit 20 \
  --output artifacts/qa_qwen25vl_top1_20.jsonl

python experiments/evaluate_qa.py \
  --predictions artifacts/qa_qwen25vl_top1_20.jsonl \
  --output artifacts/qa_qwen25vl_top1_20_report.json

#!/usr/bin/env bash
# First remote-VLM experiment after run_modelscope_qwen25vl7b_check.sh succeeds.
set -euo pipefail

: "${MODELSCOPE_ACCESS_TOKEN:?Set MODELSCOPE_ACCESS_TOKEN to your ModelScope Token first}"

medrag answer \
  --config configs/reader_modelscope_qwen25vl7b.yaml \
  --annotations medhorizon_test.jsonl \
  --top-k 1 \
  --limit 20 \
  --output artifacts/qa_modelscope_qwen25vl7b_top1_20.jsonl

python experiments/evaluate_qa.py \
  --predictions artifacts/qa_modelscope_qwen25vl7b_top1_20.jsonl \
  --output artifacts/qa_modelscope_qwen25vl7b_top1_20_report.json

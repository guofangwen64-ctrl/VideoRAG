#!/usr/bin/env bash
# Controlled comparison against the Top-1 x 8-frame chunk Reader baseline.
set -euo pipefail

: "${MODELSCOPE_ACCESS_TOKEN:?Set MODELSCOPE_ACCESS_TOKEN to your ModelScope Token first}"

medrag answer \
  --config configs/reader_modelscope_qwen3vl8b_temporal_window.yaml \
  --annotations medhorizon_test.jsonl \
  --top-k 1 \
  --limit 20 \
  --output artifacts/qa_qwen3vl8b_temporal_window16_top1_20.jsonl

python experiments/evaluate_qa.py \
  --predictions artifacts/qa_qwen3vl8b_temporal_window16_top1_20.jsonl \
  --output artifacts/qa_qwen3vl8b_temporal_window16_top1_20_report.json

#!/usr/bin/env bash
# Strict no-video baseline: the API receives questions and answer choices only.
set -euo pipefail

: "${MODELSCOPE_ACCESS_TOKEN:?Set MODELSCOPE_ACCESS_TOKEN to your ModelScope Token first}"

medrag answer \
  --config configs/reader_modelscope_qwen3vl8b.yaml \
  --annotations medhorizon_test.jsonl \
  --question-only \
  --limit 20 \
  --output artifacts/qa_qwen3vl8b_question_only_20.jsonl

python experiments/evaluate_qa.py \
  --predictions artifacts/qa_qwen3vl8b_question_only_20.jsonl \
  --output artifacts/qa_qwen3vl8b_question_only_20_report.json

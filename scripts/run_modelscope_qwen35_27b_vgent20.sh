#!/usr/bin/env bash
set -euo pipefail

: "${MODELSCOPE_ACCESS_TOKEN:?Set MODELSCOPE_ACCESS_TOKEN first}"

MANIFEST="${1:?Usage: $0 MANIFEST [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-artifacts/vgent_baseline/modelscope_qwen35_27b_vgent20_087}"

# Stratified for temporal coverage and visible-event diversity in video 087.
# All selected clips contain exactly 64 cached 1 FPS frames; the partial tail is excluded.
CLIP_INDICES="0,5,9,10,19,21,22,26,31,33,39,46,48,50,55,64,67,75,79,86"

python experiments/describe_vgent_clips.py \
  --config configs/vgent_modelscope_qwen35_27b.yaml \
  --manifest "${MANIFEST}" \
  --clip-indices "${CLIP_INDICES}" \
  --fail-fast \
  --output "${OUTPUT_DIR}/descriptions.jsonl" \
  --errors "${OUTPUT_DIR}/errors.jsonl"

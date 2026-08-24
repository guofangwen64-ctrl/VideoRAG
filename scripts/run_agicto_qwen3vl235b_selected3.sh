#!/usr/bin/env bash
set -euo pipefail

: "${AGICTO_API_KEY:?Set AGICTO_API_KEY before running}"

VIDEO_ROOT="${1:-${MEDHORIZON_VIDEO_ROOT:-}}"
if [[ -z "${VIDEO_ROOT}" ]]; then
  echo "Usage: $0 VIDEO_ROOT [CACHE_ROOT] [OUTPUT_ROOT]" >&2
  echo "Alternatively set MEDHORIZON_VIDEO_ROOT." >&2
  exit 2
fi

CACHE_ROOT="${2:-artifacts/vgent_baseline/streaming_cache_selected3_qwen3vl235b}"
OUTPUT_ROOT="${3:-artifacts/vgent_baseline/agicto_qwen3vl235b_selected3_observation_first_v10}"
VIDEO_KEYS=("079" "047" "grasp_CASE003")
VIDEO_KEYS_CSV="$(IFS=,; echo "${VIDEO_KEYS[*]}")"
MANIFEST_DIR="${CACHE_ROOT}/video_manifests"

python experiments/extract_vgent_streaming.py \
  --config configs/vgent_baseline.yaml \
  --annotations medhorizon_test.jsonl \
  --video-root "${VIDEO_ROOT}" \
  --video-keys "${VIDEO_KEYS_CSV}" \
  --frame-root "${CACHE_ROOT}/frames" \
  --manifest-dir "${MANIFEST_DIR}" \
  --output "${CACHE_ROOT}/streaming_manifest.jsonl" \
  --report "${CACHE_ROOT}/streaming_report.json" \
  --errors "${CACHE_ROOT}/streaming_errors.jsonl" \
  --progress

for video_key in "${VIDEO_KEYS[@]}"; do
  shopt -s nullglob
  manifests=("${MANIFEST_DIR}/${video_key}_"*.json)
  shopt -u nullglob
  if (( ${#manifests[@]} != 1 )); then
    echo "Expected exactly one manifest for ${video_key}; found ${#manifests[@]}" >&2
    exit 1
  fi
  video_output="${OUTPUT_ROOT}/${video_key}"
  extra_args=()
  if [[ "${video_key}" == "079" ]]; then
    extra_args+=(--skip-clip-indices 32)
  fi
  python experiments/describe_vgent_clips.py \
    --config configs/vgent_agicto_qwen3vl235b.yaml \
    --manifest "${manifests[0]}" \
    --all-clips \
    "${extra_args[@]}" \
    --progress \
    --output "${video_output}/descriptions.jsonl" \
    --errors "${video_output}/errors.jsonl"
done

echo "All selected videos completed: ${OUTPUT_ROOT}"

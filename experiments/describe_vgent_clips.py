"""Describe evenly distributed complete VGent clips through a local Qwen2.5-VL service."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.core.config import load_config
from medhorizon_videorag.vgent_baseline import load_video_plan
from medhorizon_videorag.vgent_baseline.description import (
    DESCRIPTION_PROMPT_VERSION,
    OpenAICompatibleClipDescriber,
    select_even_full_clips,
)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _completed_clip_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            completed.add(str(json.loads(line)["clip_id"]))
    return completed


def _safe_error(error: Exception) -> str:
    message = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
        "<image-data>",
        str(error),
    )
    return message[:2000]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vgent_baseline.yaml")
    parser.add_argument(
        "--manifest", required=True, help="One extracted video-plan JSON"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--clip-count", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    description = config.vgent.get("description", {})
    frames_per_request = int(description.get("frames_per_request", 64))
    clip_count = args.clip_count or int(description.get("clip_count", 10))
    if frames_per_request != 64:
        raise ValueError("This VGent pilot requires exactly 64 frames per request")
    plan = load_video_plan(args.manifest)
    selected = select_even_full_clips(
        plan.clips,
        clip_count,
        frames_per_request=frames_per_request,
    )
    print(
        f"Selected clip indices: {[clip.clip_index for clip in selected]}", flush=True
    )

    describer = OpenAICompatibleClipDescriber(
        model=str(description.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        base_url=str(description.get("base_url", "http://127.0.0.1:8001/v1")),
        api_key_env=str(description.get("api_key_env", "OPENAI_API_KEY")),
        max_tokens=int(description.get("max_tokens", 1024)),
        timeout_seconds=float(description.get("timeout_seconds", 600)),
    )
    output = Path(args.output)
    errors = Path(args.errors)
    completed = _completed_clip_ids(output)
    succeeded = resumed = failed = 0
    for number, clip in enumerate(selected, start=1):
        if clip.id in completed:
            resumed += 1
            print(f"[{number}/{len(selected)}] {clip.id}: already complete", flush=True)
            continue
        started = time.monotonic()
        try:
            result = describer.describe(clip, frames_per_request=frames_per_request)
            _append_jsonl(
                output,
                {
                    "clip_id": clip.id,
                    "video_id": clip.video_id,
                    "clip_index": clip.clip_index,
                    "start_seconds": clip.start_seconds,
                    "end_seconds": clip.end_seconds,
                    "input_frames": frames_per_request,
                    "model": describer.model,
                    "prompt_version": DESCRIPTION_PROMPT_VERSION,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "description": result,
                },
            )
            succeeded += 1
            print(f"[{number}/{len(selected)}] {clip.id}: complete", flush=True)
        # Persist unpredictable client/server failures so the pilot can resume.
        except Exception as error:  # noqa: BLE001
            failed += 1
            _append_jsonl(
                errors,
                {
                    "clip_id": clip.id,
                    "clip_index": clip.clip_index,
                    "error_type": type(error).__name__,
                    "error": _safe_error(error),
                },
            )
            print(
                f"[{number}/{len(selected)}] {clip.id}: FAILED {_safe_error(error)}",
                flush=True,
            )
    print(
        f"Finished: {succeeded} described, {resumed} resumed, {failed} failed",
        flush=True,
    )


if __name__ == "__main__":
    main()

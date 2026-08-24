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
    find_summary_rule_violations,
    select_even_full_clips,
    select_full_clips_by_index,
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


def _progress(items, *, enabled: bool, description: str):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError("Install progress support: pip install tqdm") from error
    return tqdm(
        items,
        total=len(items),
        desc=description,
        unit="clip",
        dynamic_ncols=True,
    )


def _log(message: str, *, progress: bool) -> None:
    if progress:
        from tqdm.auto import tqdm

        tqdm.write(message)
    else:
        print(message, flush=True)


def _remove_resolved_errors(path: Path, completed: set[str]) -> None:
    if not path.is_file():
        return
    unresolved = [
        row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if str(row.get("clip_id", "")) not in completed
    ]
    if not unresolved:
        path.unlink()
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unresolved),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vgent_baseline.yaml")
    parser.add_argument(
        "--manifest", required=True, help="One extracted video-plan JSON"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--errors", required=True)
    parser.add_argument("--clip-count", type=int)
    parser.add_argument(
        "--clip-indices",
        help="Comma-separated exact complete clip indices; overrides --clip-count",
    )
    parser.add_argument(
        "--all-clips",
        action="store_true",
        help="Describe every clip; pad a valid partial tail by repeating its last frame",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed request instead of continuing the batch",
    )
    parser.add_argument(
        "--progress", action="store_true", help="Show a clip progress bar"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    description = config.vgent.get("description", {})
    frames_per_request = int(description.get("frames_per_request", 64))
    clip_count = args.clip_count or int(description.get("clip_count", 10))
    if frames_per_request != 64:
        raise ValueError("This VGent pilot requires exactly 64 frames per request")
    plan = load_video_plan(args.manifest)
    if args.all_clips and args.clip_indices:
        raise ValueError("--all-clips and --clip-indices cannot be used together")
    if args.clip_indices:
        indices = [int(value.strip()) for value in args.clip_indices.split(",")]
        selected = select_full_clips_by_index(
            plan.clips,
            indices,
            frames_per_request=frames_per_request,
        )
    elif args.all_clips:
        selected = list(plan.clips)
    else:
        selected = select_even_full_clips(
            plan.clips,
            clip_count,
            frames_per_request=frames_per_request,
        )
    _log(
        f"Selected {len(selected)} clips: "
        f"{selected[0].clip_index if selected else '-'}.."
        f"{selected[-1].clip_index if selected else '-'}",
        progress=args.progress,
    )

    describer = OpenAICompatibleClipDescriber(
        model=str(description.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")),
        base_url=str(description.get("base_url", "http://127.0.0.1:8001/v1")),
        api_key_env=str(description.get("api_key_env", "OPENAI_API_KEY")),
        max_tokens=int(description.get("max_tokens", 1024)),
        timeout_seconds=float(description.get("timeout_seconds", 600)),
        max_image_pixels=int(description.get("max_image_pixels", 200704)),
        rewrite_summary_violations=bool(
            description.get("rewrite_summary_violations", True)
        ),
        max_retries=int(description.get("max_retries", 2)),
        initial_retry_seconds=float(description.get("initial_retry_seconds", 10)),
        max_retry_seconds=float(description.get("max_retry_seconds", 120)),
        response_format_json=bool(description.get("response_format_json", True)),
        request_extra_body=description.get("request_extra_body"),
    )
    output = Path(args.output)
    errors = Path(args.errors)
    completed = _completed_clip_ids(output)
    succeeded = resumed = failed = 0
    clip_progress = _progress(
        selected,
        enabled=args.progress,
        description=f"{plan.video_id} Qwen3-VL descriptions",
    )
    for number, clip in enumerate(clip_progress, start=1):
        if clip.id in completed:
            resumed += 1
            _log(
                f"[{number}/{len(selected)}] {clip.id}: already complete",
                progress=args.progress,
            )
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
                    "source_frames": len(clip.frame_paths),
                    "input_frames": frames_per_request,
                    "padding_frames": frames_per_request - len(clip.frame_paths),
                    "model": describer.model,
                    "prompt_version": DESCRIPTION_PROMPT_VERSION,
                    "generation_attempts": describer.last_attempt_count,
                    "summary_rule_violations": find_summary_rule_violations(
                        str(result["summary"])
                    ),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "description": result,
                },
            )
            succeeded += 1
            _log(
                f"[{number}/{len(selected)}] {clip.id}: complete",
                progress=args.progress,
            )
        # Persist unpredictable client/server failures so the pilot can resume.
        except Exception as error:
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
            _log(
                f"[{number}/{len(selected)}] {clip.id}: FAILED {_safe_error(error)}",
                progress=args.progress,
            )
            if args.fail_fast:
                raise
    completed = _completed_clip_ids(output)
    _remove_resolved_errors(errors, completed)
    _log(
        f"Finished: {succeeded} described, {resumed} resumed, {failed} failed",
        progress=args.progress,
    )


if __name__ == "__main__":
    main()

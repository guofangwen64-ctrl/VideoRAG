"""Extract medical-streaming VGent clips at a true fixed FPS with per-video resume."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.core.config import load_config
from medhorizon_videorag.core.io import write_jsonl
from medhorizon_videorag.datasets import MedHorizonDataset
from medhorizon_videorag.vgent_baseline import (
    MedicalStreamingExtractor,
    VgentSlicingConfig,
    load_video_plan,
    save_video_plan,
    summarize_vgent_plans,
    video_manifest_path,
    video_plan_cache_complete,
)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _select_videos(
    videos: Sequence[Any], video_keys: str | None, limit: int | None
) -> list[Any]:
    if video_keys and limit is not None:
        raise ValueError("--video-keys and --limit cannot be used together")
    if not video_keys:
        return list(videos[:limit] if limit is not None else videos)
    requested = [item.strip() for item in video_keys.split(",") if item.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--video-keys must contain unique, non-empty keys")
    by_key = {str(video.key): video for video in videos}
    missing = [key for key in requested if key not in by_key]
    if missing:
        raise ValueError(f"Unknown video keys: {missing}")
    return [by_key[key] for key in requested]


def _progress(items: Sequence[Any], *, enabled: bool, description: str):
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
        unit="video",
        dynamic_ncols=True,
    )


def _log(message: str, *, progress: bool) -> None:
    if progress:
        from tqdm.auto import tqdm

        tqdm.write(message)
    else:
        print(message, flush=True)


def _remove_resolved_errors(path: Path, completed_video_keys: set[str]) -> None:
    if not path.is_file():
        return
    unresolved = [
        row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if str(row.get("video_key", "")) not in completed_video_keys
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
    parser.add_argument("--annotations", default="medhorizon_test.jsonl")
    parser.add_argument("--video-root", help="Override data.video_root")
    parser.add_argument("--frame-root", help="Persistent sampled-frame cache")
    parser.add_argument("--manifest-dir", help="Atomic per-video manifests")
    parser.add_argument("--output", help="Aggregated clip manifest JSONL")
    parser.add_argument("--report", help="Extraction report JSON")
    parser.add_argument("--errors", help="Per-video error JSONL")
    parser.add_argument(
        "--video-keys",
        help="Comma-separated exact video keys, processed in the given order",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N videos")
    parser.add_argument("--progress", action="store_true", help="Show a progress bar")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.pipeline.get("name") != "vgent_baseline":
        raise ValueError("Streaming extraction requires pipeline.name: vgent_baseline")
    slicing_config = VgentSlicingConfig.from_dict(config.vgent)
    if slicing_config.mode != "medical_streaming":
        raise ValueError("Streaming extraction requires vgent.mode: medical_streaming")
    extractor = MedicalStreamingExtractor(
        slicing_config,
        opencv_jpeg_quality=int(config.vgent.get("opencv_jpeg_quality", 95)),
        ffmpeg_jpeg_quality=int(config.vgent.get("ffmpeg_jpeg_quality", 2)),
        ffmpeg_fallback_min_incomplete_ratio=config.vgent.get(
            "ffmpeg_fallback_min_incomplete_ratio", 0.01
        ),
    )

    artifact_dir = Path(config.project.get("artifact_dir", "artifacts/vgent_baseline"))
    frame_root = Path(args.frame_root) if args.frame_root else artifact_dir / "frames"
    manifest_dir = (
        Path(args.manifest_dir)
        if args.manifest_dir
        else artifact_dir / "video_manifests"
    )
    output = (
        Path(args.output) if args.output else artifact_dir / "streaming_manifest.jsonl"
    )
    report_path = (
        Path(args.report) if args.report else artifact_dir / "streaming_report.json"
    )
    error_path = (
        Path(args.errors) if args.errors else artifact_dir / "streaming_errors.jsonl"
    )
    root_value = args.video_root or config.data.get("video_root")
    if not root_value:
        raise ValueError("Set --video-root or data.video_root before extracting frames")
    video_root = Path(root_value)

    dataset = MedHorizonDataset(args.annotations)
    videos = _select_videos(dataset.videos, args.video_keys, args.limit)
    plans = []
    skipped = processed = failed = 0
    video_progress = _progress(
        videos, enabled=args.progress, description="medical_streaming"
    )
    for number, video in enumerate(video_progress, start=1):
        manifest_path = video_manifest_path(manifest_dir, video.key)
        if manifest_path.is_file():
            existing = load_video_plan(manifest_path)
            if video_plan_cache_complete(existing):
                plans.append(existing)
                skipped += 1
                _log(
                    f"[{number}/{len(videos)}] {video.key}: already complete",
                    progress=args.progress,
                )
                continue
            _log(
                f"[{number}/{len(videos)}] {video.key}: retrying incomplete cache",
                progress=args.progress,
            )
        source = Path(video.video_path)
        if not source.is_absolute():
            source = video_root / source
        try:
            _log(
                f"[{number}/{len(videos)}] {video.key}: extracting 1 FPS frames",
                progress=args.progress,
            )
            plan = extractor.extract(
                video.key,
                source,
                frame_root,
                annotation_duration_seconds=video.duration_seconds,
            )
            save_video_plan(plan, manifest_path)
            plans.append(plan)
            processed += 1
            _log(
                f"[{number}/{len(videos)}] {video.key}: "
                f"{len(plan.clips)} clips, {plan.sampled_frames} sampled frames",
                progress=args.progress,
            )
        except (OSError, RuntimeError, ValueError) as error:
            failed += 1
            _append_jsonl(
                error_path,
                {
                    "video_id": video.key,
                    "video_path": str(source),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            _log(
                f"[{number}/{len(videos)}] FAILED {video.key}: {error}",
                progress=args.progress,
            )

    write_jsonl(output, (clip.to_dict() for plan in plans for clip in plan.clips))
    _remove_resolved_errors(
        error_path,
        {plan.video_id for plan in plans if video_plan_cache_complete(plan)},
    )
    report = summarize_vgent_plans(plans)
    report.update(
        {
            "annotations": args.annotations,
            "video_root": str(video_root),
            "frame_root": str(frame_root),
            "manifest_dir": str(manifest_dir),
            "manifest": str(output),
            "errors": str(error_path),
            "run": {"processed": processed, "resumed": skipped, "failed": failed},
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(
        f"Finished: {processed} processed, {skipped} resumed, {failed} failed. "
        f"Report: {report_path}",
        progress=args.progress,
    )


if __name__ == "__main__":
    main()

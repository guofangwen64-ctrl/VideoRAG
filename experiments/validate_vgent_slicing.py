"""Plan VGent-compatible clips and report long-video sampling deviations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhorizon_videorag.core.config import load_config
from medhorizon_videorag.core.io import write_jsonl
from medhorizon_videorag.datasets import MedHorizonDataset
from medhorizon_videorag.vgent_baseline import (
    VgentSlicingConfig,
    VgentSlicingPlanner,
    summarize_vgent_plans,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vgent_baseline.yaml")
    parser.add_argument("--annotations", default="medhorizon_test.jsonl")
    parser.add_argument("--output", help="Clip manifest JSONL")
    parser.add_argument("--report", help="Sampling diagnostics JSON")
    parser.add_argument("--limit", type=int, help="Plan only the first N videos")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.pipeline.get("name") != "vgent_baseline":
        raise ValueError("Slicing validation requires pipeline.name: vgent_baseline")
    slicing_config = VgentSlicingConfig.from_dict(config.vgent)
    planner = VgentSlicingPlanner(slicing_config)
    dataset = MedHorizonDataset(args.annotations)
    videos = dataset.videos[: args.limit] if args.limit is not None else dataset.videos
    video_root = (
        Path(config.data["video_root"]) if config.data.get("video_root") else None
    )

    plans = []
    missing_duration = []
    for video in videos:
        if video.duration_seconds is None or video.duration_seconds <= 0:
            missing_duration.append(video.key)
            continue
        video_path = Path(video.video_path)
        if video_root is not None and not video_path.is_absolute():
            video_path = video_root / video_path
        plans.append(planner.plan(video.key, str(video_path), video.duration_seconds))

    artifact_dir = Path(config.project.get("artifact_dir", "artifacts/vgent_baseline"))
    output = (
        Path(args.output) if args.output else artifact_dir / "slicing_manifest.jsonl"
    )
    report_path = (
        Path(args.report) if args.report else artifact_dir / "slicing_report.json"
    )
    write_jsonl(output, (clip.to_dict() for plan in plans for clip in plan.clips))
    report = summarize_vgent_plans(plans)
    report.update(
        {
            "config": {
                "mode": slicing_config.mode,
                "sample_fps": slicing_config.sample_fps,
                "frames_per_clip": slicing_config.frames_per_clip,
                "min_sampled_frames": slicing_config.min_sampled_frames,
                "max_sampled_frames": slicing_config.max_sampled_frames,
                "frame_factor": slicing_config.frame_factor,
                "n_retrieval": slicing_config.n_retrieval,
                "include_partial_clip": slicing_config.include_partial_clip,
            },
            "annotations": args.annotations,
            "missing_duration_video_ids": missing_duration,
            "manifest": str(output),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Planned {len(plans)} videos and {report['clips']['count']} clips.")
    print(f"Sampling mode: {slicing_config.mode}")
    print(f"Sampling cap applied to {report['videos']['sampling_capped']} videos.")
    print(f"Manifest: {output}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

"""Diagnostic red-region pixelation for rejected medical-video API probes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import VgentClipPlan, VgentVideoPlan


def pixelate_red_regions(
    image,
    *,
    min_red: int = 70,
    dominance_ratio: float = 1.18,
    dilation_size: int = 21,
    pixel_block_size: int = 18,
):
    """Pixelate red-dominant regions and return the image plus mask statistics.

    This is a diagnostic transformation, not a blood detector. It deliberately
    uses only pixel appearance and must not be interpreted as medical labeling.
    """

    try:
        from PIL import Image, ImageFilter
    except ImportError as error:  # pragma: no cover - dependency error
        raise RuntimeError("Install Pillow to prepare redaction probes") from error

    if not 0 <= min_red <= 255:
        raise ValueError("min_red must be between 0 and 255")
    if dominance_ratio <= 1:
        raise ValueError("dominance_ratio must be greater than 1")
    if dilation_size < 1 or dilation_size % 2 == 0:
        raise ValueError("dilation_size must be a positive odd integer")
    if pixel_block_size < 2:
        raise ValueError("pixel_block_size must be at least 2")

    rgb = image.convert("RGB")
    pixels = np.asarray(rgb, dtype=np.float32)
    red = pixels[..., 0]
    green = pixels[..., 1]
    blue = pixels[..., 2]
    raw_mask = (
        (red >= min_red)
        & (red >= green * dominance_ratio)
        & (red >= blue * dominance_ratio)
    )
    raw_fraction = float(raw_mask.mean())
    mask = Image.fromarray(np.uint8(raw_mask) * 255, mode="L")
    if dilation_size > 1:
        mask = mask.filter(ImageFilter.MaxFilter(dilation_size))
    mask_fraction = float((np.asarray(mask) > 0).mean())

    width, height = rgb.size
    small_width = max(1, math.ceil(width / pixel_block_size))
    small_height = max(1, math.ceil(height / pixel_block_size))
    pixelated = rgb.resize((small_width, small_height), Image.Resampling.BILINEAR)
    pixelated = pixelated.resize((width, height), Image.Resampling.NEAREST)
    redacted = Image.composite(pixelated, rgb, mask)
    return redacted, {
        "raw_red_fraction": round(raw_fraction, 6),
        "redacted_fraction": round(mask_fraction, 6),
    }


def prepare_redacted_video_plan(
    plan: VgentVideoPlan,
    *,
    clip_indices: Sequence[int],
    output_dir: str | Path,
    min_red: int = 70,
    dominance_ratio: float = 1.18,
    dilation_size: int = 21,
    pixel_block_size: int = 18,
    jpeg_quality: int = 90,
) -> tuple[VgentVideoPlan, dict[str, Any]]:
    """Create redacted frame copies and a compatible derived video plan."""

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency error
        raise RuntimeError("Install Pillow to prepare redaction probes") from error

    selected = [int(index) for index in clip_indices]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("clip_indices must contain unique clip indices")
    if any(index < 0 or index >= len(plan.clips) for index in selected):
        raise ValueError("clip_indices contains an index outside the video plan")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    destination_root = Path(output_dir)
    if destination_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing redaction output: {destination_root}"
        )
    destination_root.mkdir(parents=True)

    selected_set = set(selected)
    derived_clips: list[VgentClipPlan] = []
    clip_reports = []
    for clip in plan.clips:
        if clip.clip_index not in selected_set:
            derived_clips.append(clip)
            continue
        if len(clip.frame_paths) != clip.sampled_frame_count:
            raise ValueError(f"Clip {clip.id} does not have a complete frame cache")

        clip_dir = destination_root / f"clip_{clip.clip_index:05d}"
        clip_dir.mkdir()
        derived_paths = []
        frame_reports = []
        for frame_index, source_path_value in enumerate(clip.frame_paths):
            source_path = Path(source_path_value)
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing source frame: {source_path}")
            with Image.open(source_path) as source:
                redacted, statistics = pixelate_red_regions(
                    source,
                    min_red=min_red,
                    dominance_ratio=dominance_ratio,
                    dilation_size=dilation_size,
                    pixel_block_size=pixel_block_size,
                )
            destination = clip_dir / f"frame_{frame_index:03d}.jpg"
            redacted.save(destination, format="JPEG", quality=jpeg_quality)
            derived_paths.append(str(destination))
            frame_reports.append(
                {
                    "frame_index": frame_index,
                    "source_path": str(source_path),
                    "redacted_path": str(destination),
                    **statistics,
                }
            )

        ranked = sorted(
            frame_reports,
            key=lambda row: (-row["redacted_fraction"], row["frame_index"]),
        )
        clip_reports.append(
            {
                "clip_id": clip.id,
                "clip_index": clip.clip_index,
                "frame_count": len(frame_reports),
                "mean_raw_red_fraction": round(
                    sum(row["raw_red_fraction"] for row in frame_reports)
                    / len(frame_reports),
                    6,
                ),
                "mean_redacted_fraction": round(
                    sum(row["redacted_fraction"] for row in frame_reports)
                    / len(frame_reports),
                    6,
                ),
                "highest_redaction_frames": [
                    {
                        "frame_index": row["frame_index"],
                        "raw_red_fraction": row["raw_red_fraction"],
                        "redacted_fraction": row["redacted_fraction"],
                    }
                    for row in ranked[:10]
                ],
                "frames": frame_reports,
            }
        )
        derived_clips.append(
            replace(
                clip,
                frame_paths=derived_paths,
                metadata={
                    **clip.metadata,
                    "redaction_probe": {
                        "mode": "red_region_pixelation",
                        "medical_label": False,
                    },
                },
            )
        )

    report = {
        "video_id": plan.video_id,
        "mode": "red_region_pixelation",
        "diagnostic_only": True,
        "medical_label": False,
        "clip_indices": selected,
        "parameters": {
            "min_red": min_red,
            "dominance_ratio": dominance_ratio,
            "dilation_size": dilation_size,
            "pixel_block_size": pixel_block_size,
            "jpeg_quality": jpeg_quality,
        },
        "clips": clip_reports,
    }
    derived_plan = replace(
        plan,
        clips=derived_clips,
        metadata={
            **plan.metadata,
            "redaction_probe": {
                "mode": "red_region_pixelation",
                "clip_indices": selected,
                "diagnostic_only": True,
            },
        },
    )
    return derived_plan, report

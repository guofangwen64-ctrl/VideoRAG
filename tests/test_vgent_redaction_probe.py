from pathlib import Path

import numpy as np
import pytest

from medhorizon_videorag.vgent_baseline.redaction_probe import (
    pixelate_red_regions,
    prepare_redacted_video_plan,
)
from medhorizon_videorag.vgent_baseline.schemas import VgentClipPlan, VgentVideoPlan

PIL = pytest.importorskip("PIL")
from PIL import Image


def _plan(tmp_path: Path) -> VgentVideoPlan:
    frames = []
    for index, color in enumerate([(180, 20, 20), (20, 180, 20)]):
        path = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (24, 16), color).save(path)
        frames.append(str(path))
    clip = VgentClipPlan(
        id="video_vgent_00000",
        video_id="video",
        video_path="video.mp4",
        clip_index=0,
        start_seconds=0.0,
        end_seconds=2.0,
        sample_start_index=0,
        sample_end_index=2,
        sampled_frame_count=2,
        effective_fps=1.0,
        is_partial=True,
        frame_paths=frames,
    )
    return VgentVideoPlan(
        video_id="video",
        video_path="video.mp4",
        duration_seconds=2.0,
        sampling_mode="medical_streaming",
        target_fps=1.0,
        effective_fps=1.0,
        desired_sampled_frames=2,
        sampled_frames=2,
        frames_per_clip=64,
        sampling_capped=False,
        meets_official_min_frames=False,
        official_skip_reason=None,
        clips=[clip],
    )


def test_pixelates_only_red_dominant_region() -> None:
    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    pixels[:, :20] = (200, 20, 20)
    pixels[:, 20:] = (20, 200, 20)
    image = Image.fromarray(pixels)

    redacted, stats = pixelate_red_regions(
        image,
        dilation_size=1,
        pixel_block_size=4,
    )

    assert redacted.size == image.size
    assert stats["raw_red_fraction"] == pytest.approx(0.5)
    assert stats["redacted_fraction"] == pytest.approx(0.5)


def test_prepares_derived_plan_without_modifying_source_paths(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    original_paths = list(plan.clips[0].frame_paths)

    derived, report = prepare_redacted_video_plan(
        plan,
        clip_indices=[0],
        output_dir=tmp_path / "redacted",
        dilation_size=1,
        pixel_block_size=4,
    )

    assert plan.clips[0].frame_paths == original_paths
    assert derived.clips[0].frame_paths != original_paths
    assert all(Path(path).is_file() for path in derived.clips[0].frame_paths)
    assert report["diagnostic_only"] is True
    assert report["medical_label"] is False
    assert report["clips"][0]["highest_redaction_frames"][0]["frame_index"] == 0


def test_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "redacted"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_redacted_video_plan(plan, clip_indices=[0], output_dir=output)

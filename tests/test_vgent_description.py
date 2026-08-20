from pathlib import Path

import pytest

from medhorizon_videorag.vgent_baseline.description import (
    DESCRIPTION_PROMPT,
    DESCRIPTION_PROMPT_VERSION,
    prepare_request_frame_paths,
    select_even_full_clips,
)
from medhorizon_videorag.vgent_baseline.schemas import VgentClipPlan


def _clip(tmp_path: Path, index: int, frame_count: int = 64) -> VgentClipPlan:
    frame_paths = []
    for frame_index in range(frame_count):
        path = tmp_path / f"{index}_{frame_index}.jpg"
        path.touch()
        frame_paths.append(str(path))
    return VgentClipPlan(
        id=f"v_vgent_{index:05d}",
        video_id="v",
        video_path="v.mp4",
        clip_index=index,
        start_seconds=float(index * 64),
        end_seconds=float(index * 64 + frame_count),
        sample_start_index=index * 64,
        sample_end_index=index * 64 + frame_count,
        sampled_frame_count=frame_count,
        effective_fps=1.0,
        is_partial=frame_count < 64,
        frame_paths=frame_paths,
    )


def test_selects_ten_even_complete_clips_and_excludes_partial(tmp_path: Path) -> None:
    clips = [_clip(tmp_path, index) for index in range(87)]
    clips.append(_clip(tmp_path, 87, 60))

    selected = select_even_full_clips(clips, 10, frames_per_request=64)

    assert [clip.clip_index for clip in selected] == [
        0,
        10,
        19,
        29,
        38,
        48,
        57,
        67,
        76,
        86,
    ]
    assert all(len(clip.frame_paths) == 64 for clip in selected)


def test_rejects_insufficient_complete_clips(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only 1 are available"):
        select_even_full_clips([_clip(tmp_path, 0)], 2, frames_per_request=64)


def test_pads_partial_tail_to_exact_request_length(tmp_path: Path) -> None:
    clip = _clip(tmp_path, 87, 60)

    request_paths = prepare_request_frame_paths(clip, frames_per_request=64)

    assert len(request_paths) == 64
    assert request_paths[:60] == clip.frame_paths
    assert request_paths[60:] == [clip.frame_paths[-1]] * 4


def test_observation_first_prompt_contract() -> None:
    assert DESCRIPTION_PROMPT_VERSION == "medical_clip_observation_first_v4"
    assert "The summary MUST contain only directly visible information." in (
        DESCRIPTION_PROMPT
    )
    assert '"red fluid" instead of "active bleeding"' in DESCRIPTION_PROMPT
    assert "Do not use external context, video title, dataset metadata" in (
        DESCRIPTION_PROMPT
    )
    assert "return an empty medical_inferences array" in DESCRIPTION_PROMPT
    assert "The action may represent suturing." not in DESCRIPTION_PROMPT

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VgentClipPlan:
    """One contiguous group of uniformly sampled frames in the VGent baseline."""

    id: str
    video_id: str
    video_path: str
    clip_index: int
    start_seconds: float
    end_seconds: float
    sample_start_index: int
    sample_end_index: int
    sampled_frame_count: int
    effective_fps: float
    is_partial: bool
    frame_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.video_id or not self.video_path:
            raise ValueError("VGent clip identity and video path must not be empty")
        if self.clip_index < 0 or self.sample_start_index < 0:
            raise ValueError("VGent clip indices must be non-negative")
        if self.sample_end_index <= self.sample_start_index:
            raise ValueError(
                "VGent sample_end_index must be greater than sample_start_index"
            )
        if self.sampled_frame_count != self.sample_end_index - self.sample_start_index:
            raise ValueError(
                "VGent sampled_frame_count does not match its sample index range"
            )
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("VGent clip must have a positive temporal duration")
        if self.effective_fps <= 0:
            raise ValueError("VGent effective_fps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VgentClipPlan:
        return cls(**payload)


@dataclass(frozen=True)
class VgentVideoPlan:
    """Sampling diagnostics and planned clips for one source video."""

    video_id: str
    video_path: str
    duration_seconds: float
    sampling_mode: str
    target_fps: float
    effective_fps: float
    desired_sampled_frames: int
    sampled_frames: int
    frames_per_clip: int
    sampling_capped: bool
    meets_official_min_frames: bool
    official_skip_reason: str | None
    clips: list[VgentClipPlan]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.video_id or not self.video_path:
            raise ValueError("VGent video identity and path must not be empty")
        if (
            self.duration_seconds <= 0
            or self.target_fps <= 0
            or self.effective_fps <= 0
        ):
            raise ValueError("VGent duration and sampling rates must be positive")
        if self.sampled_frames <= 0 or self.frames_per_clip <= 0:
            raise ValueError("VGent sampled frame counts must be positive")
        if any(clip.video_id != self.video_id for clip in self.clips):
            raise ValueError("All VGent clips must belong to the planned video")

    def to_dict(self, *, include_clips: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_clips:
            payload.pop("clips")
            payload["clip_count"] = len(self.clips)
            payload["partial_clip_count"] = sum(clip.is_partial for clip in self.clips)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VgentVideoPlan:
        values = dict(payload)
        values["clips"] = [VgentClipPlan.from_dict(item) for item in values["clips"]]
        return cls(**values)

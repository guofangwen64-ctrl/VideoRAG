from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .schemas import VgentClipPlan, VgentVideoPlan


@dataclass(frozen=True)
class VgentSlicingConfig:
    """Sampling parameters for official-cap and medical-streaming protocols."""

    mode: str = "medical_streaming"
    sample_fps: float = 1.0
    frames_per_clip: int = 64
    min_sampled_frames: int = 128
    max_sampled_frames: int = 7200
    frame_factor: int = 2
    n_retrieval: int = 20
    include_partial_clip: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"medical_streaming", "official_cap"}:
            raise ValueError("mode must be medical_streaming or official_cap")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        for name in (
            "frames_per_clip",
            "min_sampled_frames",
            "max_sampled_frames",
            "frame_factor",
            "n_retrieval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_sampled_frames > self.max_sampled_frames:
            raise ValueError("min_sampled_frames must not exceed max_sampled_frames")
        if self.min_sampled_frames % self.frame_factor:
            raise ValueError("min_sampled_frames must be divisible by frame_factor")
        if self.max_sampled_frames % self.frame_factor:
            raise ValueError("max_sampled_frames must be divisible by frame_factor")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> VgentSlicingConfig:
        allowed = {
            name: values[name] for name in cls.__dataclass_fields__ if name in values
        }
        return cls(**allowed)


class VgentSlicingPlanner:
    """Plan VGent clips from duration metadata without decoding or writing frames."""

    def __init__(self, config: VgentSlicingConfig | None = None) -> None:
        self.config = config or VgentSlicingConfig()

    def plan(
        self, video_id: str, video_path: str, duration_seconds: float
    ) -> VgentVideoPlan:
        if not video_id or not video_path:
            raise ValueError("video_id and video_path must not be empty")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        if self.config.mode == "medical_streaming":
            desired = max(1, math.ceil(duration_seconds * self.config.sample_fps))
            sampled_frames = desired
        else:
            desired = max(
                self.config.frame_factor,
                _round_by_factor(
                    duration_seconds * self.config.sample_fps,
                    self.config.frame_factor,
                ),
            )
            sampled_frames = min(
                max(desired, self.config.min_sampled_frames),
                self.config.max_sampled_frames,
            )
        effective_fps = sampled_frames / duration_seconds
        minimum_frames = self.config.frames_per_clip * self.config.n_retrieval
        meets_minimum = sampled_frames >= minimum_frames
        skip_reason = (
            None
            if meets_minimum
            else (f"official_vgent_graph_requires_{minimum_frames}_sampled_frames")
        )

        clips: list[VgentClipPlan] = []
        seconds_per_sample = (
            1.0 / self.config.sample_fps
            if self.config.mode == "medical_streaming"
            else duration_seconds / sampled_frames
        )
        for start_index in range(0, sampled_frames, self.config.frames_per_clip):
            end_index = min(start_index + self.config.frames_per_clip, sampled_frames)
            frame_count = end_index - start_index
            partial = frame_count < self.config.frames_per_clip
            if partial and not self.config.include_partial_clip:
                break
            clip_index = len(clips)
            clips.append(
                VgentClipPlan(
                    id=f"{video_id}_vgent_{clip_index:05d}",
                    video_id=video_id,
                    video_path=video_path,
                    clip_index=clip_index,
                    start_seconds=round(start_index * seconds_per_sample, 3),
                    end_seconds=round(
                        min(duration_seconds, end_index * seconds_per_sample), 3
                    ),
                    sample_start_index=start_index,
                    sample_end_index=end_index,
                    sampled_frame_count=frame_count,
                    effective_fps=round(effective_fps, 6),
                    is_partial=partial,
                    metadata={"target_fps": self.config.sample_fps},
                )
            )

        return VgentVideoPlan(
            video_id=video_id,
            video_path=video_path,
            duration_seconds=duration_seconds,
            sampling_mode=self.config.mode,
            target_fps=self.config.sample_fps,
            effective_fps=effective_fps,
            desired_sampled_frames=desired,
            sampled_frames=sampled_frames,
            frames_per_clip=self.config.frames_per_clip,
            sampling_capped=(
                self.config.mode == "official_cap"
                and desired > self.config.max_sampled_frames
            ),
            meets_official_min_frames=meets_minimum,
            official_skip_reason=skip_reason,
            clips=clips,
            metadata={
                "planning_source": "annotation_duration_only",
                "sampling_mode": self.config.mode,
                "min_sampled_frames": self.config.min_sampled_frames,
                "max_sampled_frames": self.config.max_sampled_frames,
                "frame_factor": self.config.frame_factor,
                "n_retrieval": self.config.n_retrieval,
            },
        )


def summarize_vgent_plans(plans: Iterable[VgentVideoPlan]) -> dict[str, Any]:
    rows = list(plans)
    clip_rows = [clip for plan in rows for clip in plan.clips]
    effective_fps = [plan.effective_fps for plan in rows]
    clip_durations = [clip.end_seconds - clip.start_seconds for clip in clip_rows]
    planning_sources = sorted(
        {str(plan.metadata.get("planning_source", "unknown")) for plan in rows}
    )
    return {
        "protocol": {
            "sampling_modes": sorted({plan.sampling_mode for plan in rows}),
            "sampling": "fixed-rate streaming or official globally capped sampling",
            "planning_sources": planning_sources,
            "validation_level": (
                planning_sources[0] if len(planning_sources) == 1 else "mixed"
            )
            if planning_sources
            else "empty",
            "graph_construction_performed": False,
        },
        "videos": {
            "count": len(rows),
            "meets_official_min_frames": sum(
                plan.meets_official_min_frames for plan in rows
            ),
            "below_official_min_frames": sum(
                not plan.meets_official_min_frames for plan in rows
            ),
            "sampling_capped": sum(plan.sampling_capped for plan in rows),
        },
        "clips": {
            "count": len(clip_rows),
            "partial": sum(clip.is_partial for clip in clip_rows),
            "duration_seconds": _distribution(clip_durations),
        },
        "effective_fps": _distribution(effective_fps),
        "video_plans": [plan.to_dict(include_clips=False) for plan in rows],
    }


def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": round(min(values), 6),
        "mean": round(mean(values), 6),
        "max": round(max(values), 6),
    }

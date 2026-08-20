"""VGent-compatible preprocessing utilities kept separate from the existing baseline."""

from .description import (
    DESCRIPTION_PROMPT_VERSION,
    OpenAICompatibleClipDescriber,
    prepare_request_frame_paths,
    select_even_full_clips,
)
from .schemas import VgentClipPlan, VgentVideoPlan
from .slicing import VgentSlicingConfig, VgentSlicingPlanner, summarize_vgent_plans
from .streaming import (
    MedicalStreamingExtractor,
    load_video_plan,
    safe_video_key,
    save_video_plan,
    video_manifest_path,
    video_plan_cache_complete,
)

__all__ = [
    "DESCRIPTION_PROMPT_VERSION",
    "MedicalStreamingExtractor",
    "OpenAICompatibleClipDescriber",
    "VgentClipPlan",
    "VgentSlicingConfig",
    "VgentSlicingPlanner",
    "VgentVideoPlan",
    "load_video_plan",
    "prepare_request_frame_paths",
    "safe_video_key",
    "save_video_plan",
    "select_even_full_clips",
    "summarize_vgent_plans",
    "video_manifest_path",
    "video_plan_cache_complete",
]

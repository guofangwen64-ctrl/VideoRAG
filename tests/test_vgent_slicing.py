import pytest

from medhorizon_videorag.vgent_baseline import (
    VgentSlicingConfig,
    VgentSlicingPlanner,
    summarize_vgent_plans,
)


def test_vgent_planner_keeps_official_partial_tail_clip() -> None:
    planner = VgentSlicingPlanner(VgentSlicingConfig(min_sampled_frames=2))
    plan = planner.plan("case-1", "case-1.mp4", 130.0)

    assert plan.sampled_frames == 130
    assert len(plan.clips) == 3
    assert [clip.sampled_frame_count for clip in plan.clips] == [64, 64, 2]
    assert plan.clips[-1].is_partial is True
    assert plan.clips[-1].end_seconds == 130.0


def test_vgent_planner_reports_qwen_frame_cap_for_very_long_video() -> None:
    planner = VgentSlicingPlanner(VgentSlicingConfig(mode="official_cap"))
    plan = planner.plan("case-long", "case-long.mp4", 10_000.0)

    assert plan.desired_sampled_frames == 10_000
    assert plan.sampled_frames == 7200
    assert plan.sampling_capped is True
    assert plan.effective_fps == pytest.approx(0.72)
    assert len(plan.clips) == 113
    assert plan.clips[0].end_seconds == pytest.approx(88.889)
    assert plan.meets_official_min_frames is True


def test_medical_streaming_preserves_one_fps_without_global_cap() -> None:
    plan = VgentSlicingPlanner().plan("case-long", "case-long.mp4", 10_000.0)

    assert plan.sampling_mode == "medical_streaming"
    assert plan.sampled_frames == 10_000
    assert plan.sampling_capped is False
    assert plan.effective_fps == pytest.approx(1.0)
    assert len(plan.clips) == 157
    assert plan.clips[0].end_seconds == 64.0


def test_vgent_summary_is_explicitly_pre_graph_metadata_validation() -> None:
    plan = VgentSlicingPlanner().plan("case-1", "case-1.mp4", 2000.0)
    report = summarize_vgent_plans([plan])

    assert report["protocol"]["graph_construction_performed"] is False
    assert report["protocol"]["validation_level"] == "annotation_duration_only"
    assert report["videos"]["count"] == 1


def test_vgent_config_rejects_invalid_sampling_bounds() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        VgentSlicingConfig(min_sampled_frames=7202, max_sampled_frames=7200)

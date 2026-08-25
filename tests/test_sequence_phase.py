import pytest

from medhorizon_videorag.graph_rag import (
    EvidenceInterval,
    GraphNode,
    VideoEvidenceGraph,
    build_open_activity_segmentation_prompt,
    build_sequence_phase_prompt,
    build_strict_phase_mapping_prompt,
    compact_observation_sequence,
    normalize_open_activity_response,
    normalize_sequence_phase_response,
    normalize_strict_phase_mapping_response,
    project_sequence_phases_to_events,
)


def _rows() -> list[dict]:
    return [
        {
            "clip_id": f"case_vgent_{index:05d}",
            "video_id": "case",
            "clip_index": index,
            "start_seconds": index * 64.0,
            "end_seconds": (index + 1) * 64.0,
            "description": {
                "summary": f"tool moves in clip {index}",
                "observed_facts": {
                    "visible_instruments": ["metallic tool"],
                    "visible_objects": ["thread-like material"],
                    "actions": [
                        {
                            "subject": "tool",
                            "action": "pulls",
                            "target": "thread-like material",
                        }
                    ],
                    "state_changes": [],
                },
                "medical_inferences": [{"inference": "SECRET INFERENCE"}],
            },
        }
        for index in range(4)
    ]


def _graph() -> VideoEvidenceGraph:
    nodes = []
    for index, clip_ids in enumerate(
        [
            ["case_vgent_00000", "case_vgent_00001"],
            ["case_vgent_00002", "case_vgent_00003"],
        ]
    ):
        nodes.append(
            GraphNode(
                f"event:case:{index:05d}",
                "case",
                "temporal_event",
                f"event {index}",
                [EvidenceInterval("case", index * 128.0, (index + 1) * 128.0)],
                metadata={"supporting_clip_ids": clip_ids},
            )
        )
    return VideoEvidenceGraph("case", nodes, [], "test")


def _payload() -> dict:
    return {
        "phase_segments": [
            {
                "label": "Preparation",
                "start_clip_id": "case_vgent_00000",
                "end_clip_id": "case_vgent_00001",
                "confidence": "medium",
                "basis_clip_ids": ["case_vgent_00000"],
                "basis": "tool setup is visible",
            },
            {
                "label": "Left Atrium Suturing",
                "start_clip_id": "case_vgent_00002",
                "end_clip_id": "case_vgent_00003",
                "confidence": "high",
                "basis_clip_ids": ["case_vgent_00002"],
                "basis": "repeated thread manipulation is visible",
            },
        ]
    }


def test_sequence_prompt_uses_only_compact_observations() -> None:
    compact = compact_observation_sequence(_rows())
    prompt = build_sequence_phase_prompt(
        compact, ["Preparation", "Left Atrium Suturing"]
    )
    assert "tool moves in clip 0" in prompt
    assert "SECRET INFERENCE" not in prompt
    assert "medical_inferences" not in prompt


def test_normalize_sequence_phase_response_requires_exact_coverage() -> None:
    compact = compact_observation_sequence(_rows())
    segments = normalize_sequence_phase_response(
        _payload(), compact, ["Preparation", "Left Atrium Suturing"]
    )
    assert [item["label"] for item in segments] == [
        "Preparation",
        "Left Atrium Suturing",
    ]
    assert segments[1]["start_seconds"] == 128.0

    broken = _payload()
    broken["phase_segments"][1]["start_clip_id"] = "case_vgent_00003"
    with pytest.raises(ValueError, match="ordered, contiguous"):
        normalize_sequence_phase_response(
            broken, compact, ["Preparation", "Left Atrium Suturing"]
        )


def test_project_sequence_phases_to_temporal_events() -> None:
    compact = compact_observation_sequence(_rows())
    segments = normalize_sequence_phase_response(
        _payload(), compact, ["Preparation", "Left Atrium Suturing"]
    )
    rows = project_sequence_phases_to_events(_graph(), segments, source="test-model")
    assert [row["event_id"] for row in rows] == [
        "event:case:00000",
        "event:case:00001",
    ]
    assert rows[0]["phase_hypothesis"]["label"] == "Preparation"
    assert rows[1]["phase_hypothesis"]["label"] == "Left Atrium Suturing"
    assert rows[0]["instrument_hypotheses"] == []


def test_named_phase_is_not_outvoted_by_unknown_within_coarse_event() -> None:
    compact = compact_observation_sequence(_rows())
    payload = {
        "phase_segments": [
            {
                "label": "unknown",
                "start_clip_id": "case_vgent_00000",
                "end_clip_id": "case_vgent_00000",
                "confidence": "low",
                "basis_clip_ids": ["case_vgent_00000"],
                "basis": "insufficient evidence",
            },
            {
                "label": "Preparation",
                "start_clip_id": "case_vgent_00001",
                "end_clip_id": "case_vgent_00001",
                "confidence": "medium",
                "basis_clip_ids": ["case_vgent_00001"],
                "basis": "distinctive setup cue",
            },
            {
                "label": "unknown",
                "start_clip_id": "case_vgent_00002",
                "end_clip_id": "case_vgent_00003",
                "confidence": "low",
                "basis_clip_ids": ["case_vgent_00002"],
                "basis": "insufficient evidence",
            },
        ]
    }
    segments = normalize_sequence_phase_response(
        payload, compact, ["Preparation"]
    )
    rows = project_sequence_phases_to_events(_graph(), segments, source="test-model")
    assert rows[0]["phase_hypothesis"] == {
        "label": "Preparation",
        "confidence": "medium",
        "basis": (
            "Sequence-level phase segment vote covers 1/2 supporting clips; "
            "sources: sequence_phase:00001."
        ),
    }


def _activity_payload() -> dict:
    return {
        "activity_segments": [
            {
                "activity_label": "tool and thread manipulation",
                "start_clip_id": "case_vgent_00000",
                "end_clip_id": "case_vgent_00001",
                "confidence": "high",
                "basis_clip_ids": ["case_vgent_00000"],
                "observed_pattern": "a tool repeatedly pulls thread-like material",
                "boundary_reason": "video start",
            },
            {
                "activity_label": "mesh placement and thread tightening",
                "start_clip_id": "case_vgent_00002",
                "end_clip_id": "case_vgent_00003",
                "confidence": "medium",
                "basis_clip_ids": ["case_vgent_00002"],
                "observed_pattern": "thread is tightened around a mesh-like object",
                "boundary_reason": "a mesh-like object becomes visible",
            },
        ]
    }


def test_two_stage_prompts_separate_activity_from_phase_candidates() -> None:
    compact = compact_observation_sequence(_rows())
    activity_prompt = build_open_activity_segmentation_prompt(compact)
    assert "Candidate phases" not in activity_prompt
    assert "activity_segments MUST NOT be empty" in activity_prompt
    activities = normalize_open_activity_response(_activity_payload(), compact)
    mapping_prompt = build_strict_phase_mapping_prompt(
        activities, ["Preparation", "Perfusion Needle Spacer Suturing"]
    )
    assert "Generic suturing cues are insufficient" in mapping_prompt
    assert "Perfusion Needle Spacer Suturing" in mapping_prompt


def test_strict_mapping_rejects_generic_or_low_confidence_labels() -> None:
    compact = compact_observation_sequence(_rows())
    activities = normalize_open_activity_response(_activity_payload(), compact)
    payload = {
        "phase_mappings": [
            {
                "segment_id": "open_activity:00000",
                "label": "Preparation",
                "decision": "supported",
                "confidence": "medium",
                "distinctive_cues": [],
                "missing_evidence": ["no distinctive setup cue"],
                "basis": "generic manipulation only",
            },
            {
                "segment_id": "open_activity:00001",
                "label": "Perfusion Needle Spacer Suturing",
                "decision": "supported",
                "confidence": "high",
                "distinctive_cues": ["mesh-like object secured with thread"],
                "missing_evidence": [],
                "basis": "mesh-like object distinguishes this activity",
            },
        ]
    }
    segments = normalize_strict_phase_mapping_response(
        payload,
        activities,
        ["Preparation", "Perfusion Needle Spacer Suturing"],
    )
    assert segments[0]["label"] == "unknown"
    assert segments[0]["mapping_accepted"] is False
    assert segments[1]["label"] == "Perfusion Needle Spacer Suturing"
    assert segments[1]["mapping_accepted"] is True


def test_open_activity_basis_is_uniformly_reduced_to_five_clips() -> None:
    rows = _rows() + [
        {
            **_rows()[0],
            "clip_id": f"case_vgent_{index:05d}",
            "clip_index": index,
            "start_seconds": index * 64.0,
            "end_seconds": (index + 1) * 64.0,
        }
        for index in range(4, 8)
    ]
    compact = compact_observation_sequence(rows)
    payload = {
        "activity_segments": [
            {
                "activity_label": "continued tool manipulation",
                "start_clip_id": "case_vgent_00000",
                "end_clip_id": "case_vgent_00007",
                "confidence": "medium",
                "basis_clip_ids": [item["clip_id"] for item in rows],
                "observed_pattern": "a tool repeatedly contacts visible material",
                "boundary_reason": "video start",
            }
        ]
    }
    activities = normalize_open_activity_response(payload, compact)
    assert activities[0]["basis_clip_ids"] == [
        "case_vgent_00000",
        "case_vgent_00002",
        "case_vgent_00004",
        "case_vgent_00005",
        "case_vgent_00007",
    ]

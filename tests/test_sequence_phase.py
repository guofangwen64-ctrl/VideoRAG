import pytest

from medhorizon_videorag.graph_rag import (
    EvidenceInterval,
    GraphNode,
    VideoEvidenceGraph,
    build_sequence_phase_prompt,
    compact_observation_sequence,
    normalize_sequence_phase_response,
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

import json
from pathlib import Path

from medhorizon_videorag.graph_rag import (
    build_evidence_graph,
    merge_temporal_events,
    normalize_action,
    normalize_description_rows,
    normalize_entity,
    write_evidence_graph_artifacts,
)


def _row(index: int, *, action: str, subject: str, target: str) -> dict:
    return {
        "clip_id": f"case_vgent_{index:05d}",
        "video_id": "case",
        "clip_index": index,
        "start_seconds": index * 64.0,
        "end_seconds": (index + 1) * 64.0,
        "source_frames": 64,
        "input_frames": 64,
        "padding_frames": 0,
        "prompt_version": "medical_clip_observation_first_v10",
        "description": {
            "summary": "A tool interacts with visible material.",
            "observed_facts": {
                "visible_anatomy": ["reddish tissue", "tubular structure"],
                "visible_instruments": ["metal forceps"],
                "visible_objects": ["blue thread-like material"],
                "actions": [{"subject": subject, "action": action, "target": target}],
                "state_changes": ["The material moves."],
                "visual_evidence": ["The tool changes position."],
            },
            "medical_inferences": [
                {
                    "inference": "This may be suturing.",
                    "basis": "Thread is visible.",
                    "confidence": "low",
                }
            ],
            "uncertainties": [{"item": "structure", "reason": "unclear"}],
        },
    }


def test_normalization_is_conservative_and_splits_compound_actions() -> None:
    assert normalize_entity("metal grasper")[:2] == (
        "grasping_instrument",
        "instrument",
    )
    assert normalize_entity("blood vessel")[:2] == (
        "tubular_structure",
        "anatomy",
    )
    assert normalize_entity("mesh-like ring")[:2] == (
        "grid_like_material",
        "material",
    )
    assert normalize_action("grasps and pulls") == ("grasp", "pull")
    assert normalize_action("pushes through tissue") == ("pass_through",)
    assert normalize_action("passes thread-like material through tissue") == (
        "pass_through",
    )
    assert normalize_entity("multiple thin blue thread-like materials")[:2] == (
        "thread_like_material",
        "material",
    )


def test_temporal_merge_requires_action_and_entity_continuity() -> None:
    rows = [
        _row(
            0,
            action="grasps and pulls",
            subject="metal forceps",
            target="tubular structure",
        ),
        _row(
            1,
            action="grasps and pulls",
            subject="metal forceps",
            target="tubular structure",
        ),
        _row(2, action="contacts", subject="probe", target="reddish tissue"),
    ]
    clips = normalize_description_rows(rows)
    events = merge_temporal_events(clips, threshold=0.45, max_merged_clips=5)

    assert [event.supporting_clip_ids for event in events] == [
        ["case_vgent_00000", "case_vgent_00001"],
        ["case_vgent_00002"],
    ]


def test_builder_preserves_raw_evidence_and_excludes_medical_inferences(
    tmp_path: Path,
) -> None:
    rows = [
        _row(0, action="holds", subject="metal forceps", target="reddish tissue"),
        _row(
            1,
            action="passes through",
            subject="needle-like instrument",
            target="reddish tissue",
        ),
    ]
    frame_paths = {}
    for row in rows:
        path = tmp_path / f"{row['clip_id']}.jpg"
        path.write_bytes(b"frame")
        frame_paths[row["clip_id"]] = [str(path)]

    artifacts = build_evidence_graph(rows, frame_paths_by_clip=frame_paths)
    graph = artifacts.graph
    clip_node = next(node for node in graph.nodes if node.id == "clip:case_vgent_00000")

    assert clip_node.metadata["observation"]["medical_inferences"]
    assert graph.metadata["medical_inferences_used"] is False
    assert clip_node.evidence[0].frame_paths == frame_paths["case_vgent_00000"]
    assert any(edge.relation == "observed_in" for edge in graph.edges)
    assert any(edge.relation == "instance_of" for edge in graph.edges)
    assert any(edge.relation == "acts_on" for edge in graph.edges)
    assert all(edge.relation != "same_entity" for edge in graph.edges)
    assert artifacts.report["missing_frame_path_count"] == 0

    output = tmp_path / "graph"
    write_evidence_graph_artifacts(artifacts, output)
    payload = json.loads((output / "evidence_graph.json").read_text())
    normalized = (output / "normalized_observations.jsonl").read_text()
    assert payload["schema_version"] == "medical-video-evidence-graph-v1"
    assert "medical_inferences_excluded_from_graph" in normalized

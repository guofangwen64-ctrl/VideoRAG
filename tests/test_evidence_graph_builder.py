import json
from pathlib import Path

from medhorizon_videorag.graph_rag import (
    ACTION_VOCABULARY,
    ENTITY_VOCABULARY,
    EVENT_SUPPORT_VERSION,
    GRAPH_SCHEMA_VERSION,
    REPRESENTATIVE_EVIDENCE_VERSION,
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
    assert normalize_action("inserts") == ("insert",)
    assert normalize_action("insert into") == ("insert",)
    assert normalize_action("loops around tissue") == ("loop_around",)
    assert normalize_action("forms loops around tissue") == ("loop_around",)
    assert normalize_action("is pulled and tightened") == ("pull", "tighten")
    assert normalize_action("unrecognized activity wording") == ("other_action",)
    assert all(
        predicate in ACTION_VOCABULARY
        for phrase in (
            "inserts",
            "insert into",
            "forms loops around tissue",
            "is pulled and tightened",
            "unrecognized activity wording",
        )
        for predicate in normalize_action(phrase)
    )
    assert normalize_entity("multiple thin blue thread-like materials")[:2] == (
        "thread_like_material",
        "material",
    )


def test_entity_normalization_separates_base_concept_and_attributes() -> None:
    canonical, category, attributes = normalize_entity(
        "yellowish fatty-looking material"
    )
    assert (canonical, category) == ("generic_material", "material")
    assert attributes == {
        "color": ["yellow"],
        "appearance": ["fatty-looking"],
    }
    canonical, category, attributes = normalize_entity("small white square object")
    assert (canonical, category) == ("generic_object", "object")
    assert attributes["color"] == ["white"]
    assert attributes["shape"] == ["square"]
    assert attributes["size"] == ["small"]
    assert canonical in ENTITY_VOCABULARY


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


def test_temporal_merge_supports_compatible_action_sequence() -> None:
    rows = [
        _row(
            0,
            action="passes through tissue",
            subject="needle-like instrument",
            target="reddish tissue",
        ),
        _row(
            1,
            action="pulls thread-like material",
            subject="metal forceps",
            target="thread-like material",
        ),
        _row(
            2,
            action="tightens",
            subject="thread-like material",
            target="tubular structure",
        ),
    ]
    events = merge_temporal_events(normalize_description_rows(rows))

    assert len(events) == 1
    assert events[0].supporting_clip_ids == [
        "case_vgent_00000",
        "case_vgent_00001",
        "case_vgent_00002",
    ]
    assert [detail["action_relation"] for detail in events[0].merge_details] == [
        "transition",
        "transition",
    ]
    assert events[0].merge_details[0]["compatible_transitions"] == [
        ["pass_through", "pull"]
    ]
    assert events[0].support_mode == "merged_event"
    assert 0.0 < events[0].structural_support_score < 1.0
    assert events[0].support_components["minimum_transition_score"] is not None
    representative_ids = {item["clip_id"] for item in events[0].representative_evidence}
    assert len(representative_ids) == 3
    assert "case_vgent_00002" in representative_ids
    assert events[0].representative_action_coverage == 1.0
    assert all(
        item["clip_id"] in events[0].supporting_clip_ids
        for item in events[0].representative_evidence
    )


def test_temporal_merge_rejects_transition_supported_only_by_generic_entities() -> None:
    rows = [
        _row(
            0,
            action="passes through",
            subject="instrument",
            target="tissue",
        ),
        _row(1, action="pulls", subject="instrument", target="tissue"),
    ]
    for row in rows:
        facts = row["description"]["observed_facts"]
        facts["visible_anatomy"] = ["tissue"]
        facts["visible_instruments"] = ["instrument"]
        facts["visible_objects"] = []

    events = merge_temporal_events(normalize_description_rows(rows))

    assert len(events) == 2


def test_temporal_merge_binds_exact_action_to_matching_roles() -> None:
    rows = [
        _row(
            0,
            action="holds",
            subject="metal forceps",
            target="tubular structure",
        ),
        _row(
            1,
            action="holds",
            subject="needle-like instrument",
            target="thread-like material",
        ),
    ]

    events = merge_temporal_events(normalize_description_rows(rows))

    assert len(events) == 2
    assert all(event.support_mode == "singleton_evidence" for event in events)
    assert all(
        event.support_components["mean_transition_score"] is None for event in events
    )
    assert all(len(event.representative_evidence) == 1 for event in events)


def test_representative_evidence_respects_requested_budget() -> None:
    rows = [
        _row(
            index,
            action=action,
            subject=subject,
            target=target,
        )
        for index, action, subject, target in (
            (0, "passes through", "needle-like instrument", "reddish tissue"),
            (1, "pulls", "metal forceps", "thread-like material"),
            (2, "tightens", "thread-like material", "tubular structure"),
        )
    ]

    event = merge_temporal_events(
        normalize_description_rows(rows), max_representative_clips=2
    )[0]

    assert len(event.representative_evidence) == 2
    terminal = next(
        item
        for item in event.representative_evidence
        if item["clip_id"] == "case_vgent_00002"
    )
    assert set(terminal["reasons"]) & {
        "terminal_action_coverage",
        "covers_terminal_action",
    }


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
    thread_concept = next(
        node
        for node in graph.nodes
        if node.id == "concept:material:thread_like_material"
    )
    event_node = next(
        node for node in graph.nodes if node.node_type == "temporal_event"
    )

    assert clip_node.metadata["observation"]["medical_inferences"]
    assert graph.metadata["medical_inferences_used"] is False
    assert graph.metadata["event_support_version"] == EVENT_SUPPORT_VERSION
    assert (
        graph.metadata["representative_evidence_version"]
        == REPRESENTATIVE_EVIDENCE_VERSION
    )
    assert event_node.confidence == event_node.metadata["structural_support_score"]
    assert event_node.confidence < 1.0
    assert event_node.metadata["representative_evidence"]
    assert thread_concept.metadata["attribute_counts"]["color"]["blue"] == 2
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
    assert payload["schema_version"] == GRAPH_SCHEMA_VERSION
    assert "medical_inferences_excluded_from_graph" in normalized
    assert artifacts.report["other_action_count"] == 0
    assert artifacts.report["event_support_score_summary"]["count"] == 2
    assert artifacts.report["representative_clip_count_summary"]["maximum"] == 1.0

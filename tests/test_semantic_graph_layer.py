from dataclasses import dataclass
from pathlib import Path

from medhorizon_videorag.graph_rag import (
    EvidenceInterval,
    GraphEdge,
    GraphNode,
    VideoEvidenceGraph,
    augment_with_semantic_hypotheses,
    build_video_semantic_ontology,
    extract_phase_name,
    retrieve_phase_boundary_instruments,
)


def _base_graph(tmp_path: Path) -> VideoEvidenceGraph:
    nodes = []
    edges = []
    for index in range(4):
        frame = tmp_path / f"frame_{index}.jpg"
        frame.write_bytes(b"frame")
        interval = EvidenceInterval(
            "case",
            index * 64.0,
            (index + 1) * 64.0,
            [str(frame)],
            metadata={"clip_id": f"case_vgent_{index:05d}"},
        )
        clip = GraphNode(
            f"clip:case_vgent_{index:05d}",
            "case",
            "segment",
            f"clip {index}",
            [interval],
        )
        event = GraphNode(
            f"event:case:{index:05d}",
            "case",
            "temporal_event",
            f"event {index}",
            [interval],
            metadata={"supporting_clip_ids": [f"case_vgent_{index:05d}"]},
        )
        nodes.extend([clip, event])
        edges.append(GraphEdge(event.id, clip.id, "contains", [interval]))
        if index:
            edges.append(
                GraphEdge(f"event:case:{index - 1:05d}", event.id, "temporal_before")
            )
    return VideoEvidenceGraph("case", nodes, edges, "test-v2.1")


def _hypotheses() -> list[dict]:
    return [
        {
            "event_id": "event:case:00000",
            "source": "test-model",
            "phase_hypothesis": {
                "label": "Preparation",
                "confidence": "medium",
                "basis": "setup activity",
            },
            "instrument_hypotheses": [],
        },
        {
            "event_id": "event:case:00001",
            "source": "test-model",
            "phase_hypothesis": {
                "label": "Left Atrium Suturing",
                "confidence": "high",
                "basis": "thread manipulation",
            },
            "instrument_hypotheses": [
                {
                    "label": "Needle Holder",
                    "confidence": "high",
                    "basis": "jaw and needle are visible",
                }
            ],
        },
        {
            "event_id": "event:case:00002",
            "source": "test-model",
            "phase_hypothesis": {
                "label": "Left Atrium Suturing",
                "confidence": "medium",
                "basis": "continued thread manipulation",
            },
            "instrument_hypotheses": [],
        },
        {
            "event_id": "event:case:00003",
            "source": "test-model",
            "phase_hypothesis": {
                "label": "Aortic Clamping",
                "confidence": "low",
                "basis": "clamp-like instrument",
            },
            "instrument_hypotheses": [
                {
                    "label": "Needle Holder",
                    "confidence": "low",
                    "basis": "similar jaw shape",
                }
            ],
        },
    ]


def test_semantic_layer_builds_grounded_phase_boundaries_and_type_tracks(
    tmp_path: Path,
) -> None:
    artifacts = augment_with_semantic_hypotheses(_base_graph(tmp_path), _hypotheses())
    graph = artifacts.graph

    assert graph.schema_version == "medical-video-evidence-graph-v3-pilot"
    assert sum(node.node_type == "phase_hypothesis" for node in graph.nodes) == 3
    assert sum(node.node_type == "phase_boundary" for node in graph.nodes) == 6
    tracks = [node for node in graph.nodes if node.node_type == "instrument_track"]
    assert len(tracks) == 1
    assert tracks[0].metadata["tracking_scope"] == "type_presence_not_physical_identity"
    assert tracks[0].metadata["supporting_event_ids"] == [
        "event:case:00001",
        "event:case:00003",
    ]
    assert graph.metadata["semantic_nodes_are_observed_facts"] is False
    assert any(edge.relation == "has_boundary" for edge in graph.edges)
    assert any(edge.relation == "visible_during" for edge in graph.edges)


def test_phase_boundary_retrieval_returns_onset_and_next_event(tmp_path: Path) -> None:
    graph = augment_with_semantic_hypotheses(_base_graph(tmp_path), _hypotheses()).graph
    result = retrieve_phase_boundary_instruments(
        graph, "Left Atrium Suturing", context_events=1
    )

    assert result["phase_match_score"] == 1.0
    assert result["boundary_seconds"] == 64.0
    assert result["event_ids"] == ["event:case:00001", "event:case:00002"]
    assert [item["label"] for item in result["instruments"]] == ["Needle Holder"]
    assert result["reasoning_path"][1] == "has_boundary"


def test_appearance_tracks_keep_medical_identity_unknown(tmp_path: Path) -> None:
    graph = _base_graph(tmp_path)
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    for index, (label, canonical, attributes) in enumerate(
        (
            (
                "metal surgical instrument",
                "generic_instrument",
                {"material": ["metal"]},
            ),
            ("needle-like instrument", "needle_like_instrument", {}),
            ("needle-like instrument", "needle_like_instrument", {}),
            (
                "curved metal instrument",
                "generic_instrument",
                {"shape": ["curved"]},
            ),
        )
    ):
        event = next(node for node in nodes if node.id == f"event:case:{index:05d}")
        mention = GraphNode(
            f"mention:case:{index:05d}",
            "case",
            "entity_mention",
            label,
            list(event.evidence),
            metadata={
                "canonical": canonical,
                "category": "instrument",
                "source_field": "visible_instruments",
                "attributes": attributes,
                "clip_id": f"case_vgent_{index:05d}",
            },
        )
        nodes.append(mention)
        edges.append(
            GraphEdge(
                mention.id,
                f"clip:case_vgent_{index:05d}",
                "observed_in",
            )
        )
        if index == 1:
            action = GraphNode(
                "action:case:pull",
                "case",
                "action_event",
                "pull",
                list(event.evidence),
            )
            nodes.append(action)
            edges.append(GraphEdge(action.id, mention.id, "has_subject"))
    graph = VideoEvidenceGraph("case", nodes, edges, "test-v2.1")

    artifacts = augment_with_semantic_hypotheses(
        graph,
        _hypotheses(),
        instrument_track_source="appearance_mentions",
    )
    tracks = [
        node for node in artifacts.graph.nodes if node.node_type == "instrument_track"
    ]
    needle_track = next(
        node
        for node in tracks
        if node.metadata["canonical_label"] == "needle_like_instrument"
    )

    assert len(tracks) == 3
    assert needle_track.metadata["supporting_event_ids"] == [
        "event:case:00001",
        "event:case:00002",
    ]
    assert needle_track.metadata["canonical_instrument"] == "unknown"
    assert needle_track.metadata["physical_identity_confirmed"] is False
    assert needle_track.metadata["fact_status"] == "derived_observation_track"
    assert needle_track.metadata["detections"][0]["action_roles"] == ["subject:pull"]
    result = retrieve_phase_boundary_instruments(
        artifacts.graph, "Left Atrium Suturing", context_events=1
    )
    retrieved = next(
        item
        for item in result["instruments"]
        if item["canonical_label"] == "needle_like_instrument"
    )
    assert retrieved["canonical_instrument"] == "unknown"
    assert retrieved["physical_identity_confirmed"] is False


def test_semantic_layer_rejects_unknown_track_source(tmp_path: Path) -> None:
    try:
        augment_with_semantic_hypotheses(
            _base_graph(tmp_path),
            _hypotheses(),
            instrument_track_source="invalid",
        )
    except ValueError as error:
        assert "instrument_track_source" in str(error)
    else:
        raise AssertionError("Expected invalid track source to fail")


@dataclass
class _Question:
    uid: int
    video_key: str
    task_name: str
    question: str
    options: list[str]
    answer: str
    metadata: dict


def test_candidate_ontology_and_phase_parser_do_not_require_answers() -> None:
    question = _Question(
        2,
        "087",
        "Phase-Instrument Association",
        "Which instrument is in the field at the transition into the Left Atrium Suturing phase?",
        ["A. Needle Holder", "B. Aspirator"],
        "SECRET-ANSWER",
        {},
    )
    phase_options = _Question(
        0,
        "087",
        "Action Recognition",
        "Which phase is shown?",
        ["A. Preparation", "B. Aortic Clamping"],
        "ANOTHER-SECRET",
        {"natural_rewrite_v1_kind": "surgical_phase"},
    )
    ontology = build_video_semantic_ontology([question, phase_options], "087")

    assert extract_phase_name(question.question) == "Left Atrium Suturing"
    assert ontology["phases"] == [
        "Aortic Clamping",
        "Left Atrium Suturing",
        "Preparation",
    ]
    assert ontology["instruments"] == ["Aspirator", "Needle Holder"]
    assert ontology["answers_used"] is False
    assert "SECRET-ANSWER" not in str(ontology)
    assert "ANOTHER-SECRET" not in str(ontology)

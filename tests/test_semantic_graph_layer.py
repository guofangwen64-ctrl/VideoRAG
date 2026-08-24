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


@dataclass
class _Question:
    uid: int
    video_key: str
    task_name: str
    question: str
    options: list[str]
    answer: str


def test_candidate_ontology_and_phase_parser_do_not_require_answers() -> None:
    question = _Question(
        2,
        "087",
        "Phase-Instrument Association",
        "Which instrument is in the field at the transition into the Left Atrium Suturing phase?",
        ["A. Needle Holder", "B. Aspirator"],
        "SECRET-ANSWER",
    )
    ontology = build_video_semantic_ontology([question], "087")

    assert extract_phase_name(question.question) == "Left Atrium Suturing"
    assert ontology["phases"] == ["Left Atrium Suturing"]
    assert ontology["instruments"] == ["Aspirator", "Needle Holder"]
    assert ontology["answers_used"] is False
    assert "SECRET-ANSWER" not in str(ontology)

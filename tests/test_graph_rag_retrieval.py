import json
from itertools import pairwise
from pathlib import Path

import pytest

from medhorizon_videorag.graph_rag import (
    DeterministicEventGraphRetriever,
    EvidenceInterval,
    GraphEdge,
    GraphNode,
    VideoEvidenceGraph,
    load_evidence_graph,
    normalize_graph_query,
)


def _graph(tmp_path: Path) -> VideoEvidenceGraph:
    clips = []
    events = []
    concepts = []
    mentions = []
    edges = []
    specifications = [
        ("pass_through", ["needle_like_instrument", "thread_like_material", "tissue"]),
        ("cut", ["cutting_instrument", "tissue"]),
        ("tighten", ["thread_like_material", "tubular_structure"]),
    ]
    for index, (predicate, entities) in enumerate(specifications):
        clip_id = f"case_vgent_{index:05d}"
        frame = tmp_path / f"{clip_id}.jpg"
        frame.write_bytes(b"frame")
        interval = EvidenceInterval(
            "case",
            index * 64.0,
            (index + 1) * 64.0,
            [str(frame)],
            metadata={"clip_id": clip_id, "clip_index": index},
        )
        clips.append(
            GraphNode(
                f"clip:{clip_id}",
                "case",
                "segment",
                f"A visible instrument performs {predicate.replace('_', ' ')}.",
                [interval],
                metadata={"clip_id": clip_id, "clip_index": index},
            )
        )
        events.append(
            GraphNode(
                f"event:case:{index:05d}",
                "case",
                "temporal_event",
                f"{predicate} | {', '.join(entities)}",
                [EvidenceInterval("case", index * 64.0, (index + 1) * 64.0)],
                confidence=0.5 + index * 0.1,
                metadata={
                    "supporting_clip_ids": [clip_id],
                    "predicates": [predicate],
                    "concepts": entities,
                    "structural_support_score": 0.5 + index * 0.1,
                    "representative_action_coverage": 1.0,
                    "representative_entity_coverage": 1.0,
                    "representative_evidence": [
                        {
                            "clip_id": clip_id,
                            "clip_index": index,
                            "role": "primary",
                            "selection_score": 0.9,
                            "covered_actions": [predicate],
                            "covered_informative_concepts": entities,
                        }
                    ],
                },
            )
        )
        edges.append(GraphEdge(events[-1].id, clips[-1].id, "contains", [interval]))
        for mention_index, entity in enumerate(entities):
            mention = GraphNode(
                f"mention:{clip_id}:{mention_index}",
                "case",
                "entity_mention",
                entity.replace("_", " "),
                [EvidenceInterval("case", index * 64.0, (index + 1) * 64.0)],
                metadata={
                    "canonical": entity,
                    "clip_id": clip_id,
                    "attributes": {"color": ["blue"]}
                    if entity == "thread_like_material"
                    else {},
                },
            )
            mentions.append(mention)
    for predicate, _ in specifications:
        concepts.append(
            GraphNode(
                f"concept:action:{predicate}",
                "case",
                "concept",
                predicate,
                [EvidenceInterval("case", 0.0, 192.0)],
                metadata={"category": "action"},
            )
        )
    for entity in sorted({item for _, entities in specifications for item in entities}):
        concepts.append(
            GraphNode(
                f"concept:entity:{entity}",
                "case",
                "concept",
                entity,
                [EvidenceInterval("case", 0.0, 192.0)],
                metadata={"category": "anatomy"},
            )
        )
    for previous, current in pairwise(events):
        edges.append(GraphEdge(previous.id, current.id, "temporal_before"))
    return VideoEvidenceGraph(
        "case", [*clips, *mentions, *concepts, *events], edges, "test-v2.1"
    )


def test_query_normalization_reuses_action_and_entity_vocab(tmp_path: Path) -> None:
    query = normalize_graph_query(
        "Where does the blue suture pass through tissue before tightening?",
        _graph(tmp_path),
    )

    assert set(query.predicates) == {"pass_through", "tighten"}
    assert set(query.concepts) == {"thread_like_material", "tissue"}
    assert query.attributes == {"color": ("blue",)}
    assert query.temporal_relation == "before"


def test_joint_action_entity_match_ranks_expected_event(tmp_path: Path) -> None:
    graph = _graph(tmp_path)
    result = DeterministicEventGraphRetriever(max_hops=0).retrieve(
        "q1",
        "Where does a needle-like instrument pass through blue thread-like material?",
        graph,
        top_k=2,
    )

    first = result.metadata["ranked_event_groups"][0]
    assert first["event_ids"] == ["event:case:00000"]
    assert first["matched_predicates"] == ["pass_through"]
    assert set(first["matched_concepts"]) == {
        "needle_like_instrument",
        "thread_like_material",
    }
    assert result.evidence[0].metadata["clip_id"] == "case_vgent_00000"
    assert len(result.evidence[0].frame_paths) == 1


def test_after_query_expands_to_following_event_with_auditable_path(
    tmp_path: Path,
) -> None:
    result = DeterministicEventGraphRetriever(max_hops=1).retrieve(
        "q2",
        "What happens after the needle-like instrument passes through the thread?",
        _graph(tmp_path),
        top_k=1,
    )

    candidate = result.metadata["ranked_event_groups"][0]
    assert candidate["event_ids"] == ["event:case:00000", "event:case:00001"]
    assert candidate["expansion_reason"] == "query_after_context"
    assert candidate["reasoning_path"][-2:] == [
        "temporal_before",
        "event:case:00001",
    ]


def test_before_query_preserves_anchor_and_inverse_path(tmp_path: Path) -> None:
    result = DeterministicEventGraphRetriever(max_hops=1).retrieve(
        "q-before",
        "What happens before the blue thread is tightened?",
        _graph(tmp_path),
        top_k=1,
    )

    candidate = result.metadata["ranked_event_groups"][0]
    assert candidate["event_ids"] == ["event:case:00001", "event:case:00002"]
    assert candidate["anchor_event_id"] == "event:case:00002"
    assert candidate["reasoning_path"][-2:] == [
        "inverse:temporal_before",
        "event:case:00001",
    ]


def test_loader_accepts_directory_and_rejects_empty_event_graph(
    tmp_path: Path,
) -> None:
    graph = _graph(tmp_path)
    output = tmp_path / "graph"
    output.mkdir()
    (output / "evidence_graph.json").write_text(
        json.dumps(graph.to_dict()), encoding="utf-8"
    )

    assert load_evidence_graph(output).video_id == "case"

    payload = graph.to_dict()
    payload["nodes"] = [
        item for item in payload["nodes"] if item["node_type"] != "temporal_event"
    ]
    payload["edges"] = []
    (output / "evidence_graph.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="no temporal_event"):
        load_evidence_graph(output)


def test_unmatched_question_fails_instead_of_returning_support_only(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no lexical"):
        DeterministicEventGraphRetriever().retrieve(
            "q3", "How is the camera calibrated?", _graph(tmp_path)
        )

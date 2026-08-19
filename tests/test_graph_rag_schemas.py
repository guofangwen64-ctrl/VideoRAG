from pathlib import Path

import pytest

from medhorizon_videorag.core.config import ExperimentConfig, load_config
from medhorizon_videorag.graph_rag import (
    EvidenceInterval,
    GraphEdge,
    GraphNode,
    MedicalGraphQAExample,
    VideoEvidenceGraph,
)


def test_video_evidence_graph_is_grounded_and_serializable() -> None:
    first = EvidenceInterval("case-1", 10.0, 20.0)
    second = EvidenceInterval("case-1", 30.0, 42.0)
    phase = GraphNode("phase-1", "case-1", "phase", "暴露阶段", [first])
    action = GraphNode("action-1", "case-1", "action", "夹闭血管", [second])
    graph = VideoEvidenceGraph(
        "case-1",
        [phase, action],
        [GraphEdge("phase-1", "action-1", "temporal_before", [first, second])],
    )

    payload = graph.to_dict()
    assert payload["schema_version"] == "medical-video-graph-v1"
    assert payload["nodes"][1]["evidence"][0]["start_seconds"] == 30.0


def test_graph_rejects_unknown_edge_endpoint() -> None:
    interval = EvidenceInterval("case-1", 0.0, 5.0)
    node = GraphNode("node-1", "case-1", "segment", "片段", [interval])

    with pytest.raises(ValueError, match="unknown node"):
        VideoEvidenceGraph(
            "case-1", [node], [GraphEdge("node-1", "missing", "contains")]
        )


def test_multi_hop_qa_requires_multiple_ground_truth_intervals() -> None:
    interval = EvidenceInterval("case-1", 0.0, 5.0)

    with pytest.raises(ValueError, match="at least two"):
        MedicalGraphQAExample(
            "q1",
            "case-1",
            "之后发生了什么？",
            "夹闭",
            [interval],
            "multi_hop",
        )


def test_config_loader_preserves_graph_rag_sections(tmp_path: Path) -> None:
    path = tmp_path / "graph.yaml"
    path.write_text(
        "pipeline:\n  name: medical_graph_rag\ngraph:\n  schema_version: medical-video-graph-v1\n",
        encoding="utf-8",
    )

    config = load_config(path)
    assert config.pipeline["name"] == "medical_graph_rag"
    assert config.graph["schema_version"] == "medical-video-graph-v1"
    assert config.retrieval == {}


def test_graph_fields_do_not_change_baseline_positional_config_order() -> None:
    sections = [{"slot": number} for number in range(7)]
    config = ExperimentConfig(*sections)

    assert [
        config.project,
        config.data,
        config.chunking,
        config.vision,
        config.retrieval,
        config.llm,
        config.evaluation,
    ] == sections
    assert config.pipeline == {}
    assert config.graph == {}
    assert config.vgent == {}

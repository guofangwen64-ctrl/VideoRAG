from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

NODE_TYPES = frozenset(
    {
        "segment",
        "phase",
        "action",
        "instrument",
        "anatomy",
        "finding",
        "entity_mention",
        "concept",
        "action_event",
        "temporal_event",
    }
)
EDGE_TYPES = frozenset(
    {
        "temporal_before",
        "contains",
        "uses",
        "acts_on",
        "causes",
        "same_entity",
        "co_occurs",
        "observed_in",
        "instance_of",
        "has_subject",
        "part_of",
        "possible_continuation",
    }
)
REASONING_TYPES = frozenset(
    {"single_hop", "multi_hop", "comparison", "causal", "temporal_order"}
)


@dataclass(frozen=True)
class EvidenceInterval:
    """A traceable interval in the source video, never just a generated description."""

    video_id: str
    start_seconds: float
    end_seconds: float
    frame_paths: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("EvidenceInterval.video_id must not be empty")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError(
                "EvidenceInterval must have 0 <= start_seconds < end_seconds"
            )
        _validate_confidence(self.confidence, "EvidenceInterval.confidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphNode:
    """A typed medical-video concept grounded in one or more source intervals."""

    id: str
    video_id: str
    node_type: str
    label: str
    evidence: list[EvidenceInterval]
    description: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.video_id or not self.label:
            raise ValueError("GraphNode id, video_id, and label must not be empty")
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"Unsupported graph node type: {self.node_type}")
        if not self.evidence:
            raise ValueError(
                "GraphNode must be grounded in at least one evidence interval"
            )
        if any(item.video_id != self.video_id for item in self.evidence):
            raise ValueError("GraphNode evidence must belong to the same video")
        _validate_confidence(self.confidence, "GraphNode.confidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    """A typed relation whose endpoints are node IDs in the same video graph."""

    source: str
    target: str
    relation: str
    evidence: list[EvidenceInterval] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("GraphEdge source and target must not be empty")
        if self.relation not in EDGE_TYPES:
            raise ValueError(f"Unsupported graph edge type: {self.relation}")
        _validate_confidence(self.confidence, "GraphEdge.confidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoEvidenceGraph:
    """Per-video evidence graph; cross-video knowledge graphs are out of scope."""

    video_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    schema_version: str = "medical-video-graph-v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("VideoEvidenceGraph.video_id must not be empty")
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("VideoEvidenceGraph node IDs must be unique")
        if any(node.video_id != self.video_id for node in self.nodes):
            raise ValueError(
                "All graph nodes must belong to VideoEvidenceGraph.video_id"
            )
        known = set(node_ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(
                    f"Graph edge references an unknown node: {edge.source} -> {edge.target}"
                )
            if any(item.video_id != self.video_id for item in edge.evidence):
                raise ValueError(
                    "GraphEdge evidence must belong to VideoEvidenceGraph.video_id"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphRetrievalResult:
    """Auditable output of graph planning, expansion, and evidence consolidation."""

    question_id: str
    video_id: str
    evidence: list[EvidenceInterval]
    node_ids: list[str] = field(default_factory=list)
    reasoning_path: list[str] = field(default_factory=list)
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question_id or not self.video_id:
            raise ValueError(
                "GraphRetrievalResult question_id and video_id must not be empty"
            )
        if any(item.video_id != self.video_id for item in self.evidence):
            raise ValueError(
                "Retrieved evidence must belong to GraphRetrievalResult.video_id"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MedicalGraphQAExample:
    """QA contract for questions that do not reveal their evidence timestamps."""

    id: str
    video_id: str
    question: str
    answer: str
    evidence: list[EvidenceInterval]
    reasoning_type: str
    choices: list[str] = field(default_factory=list)
    required_evidence_count: int = 1
    evidence_relation: str | None = None
    hard_negative_evidence: list[EvidenceInterval] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.video_id or not self.question or not self.answer:
            raise ValueError(
                "MedicalGraphQAExample identity, question, and answer must not be empty"
            )
        if self.reasoning_type not in REASONING_TYPES:
            raise ValueError(f"Unsupported reasoning type: {self.reasoning_type}")
        if self.required_evidence_count < 1:
            raise ValueError("required_evidence_count must be at least 1")
        if len(self.evidence) < self.required_evidence_count:
            raise ValueError(
                "Ground-truth evidence does not satisfy required_evidence_count"
            )
        all_intervals = [*self.evidence, *self.hard_negative_evidence]
        if any(item.video_id != self.video_id for item in all_intervals):
            raise ValueError(
                "QA evidence must belong to MedicalGraphQAExample.video_id"
            )
        if self.reasoning_type == "multi_hop" and self.required_evidence_count < 2:
            raise ValueError(
                "multi_hop questions must require at least two evidence intervals"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_confidence(value: float, field_name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")

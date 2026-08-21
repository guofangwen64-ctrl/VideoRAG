"""Contracts for the medical Graph-RAG research pipeline.

The production baseline remains under :mod:`medhorizon_videorag.pipelines.baseline`.
The first runnable builder creates a conservative, per-video evidence graph
from observation-first clip descriptions. A deterministic event retriever provides
the first auditable retrieval baseline without changing the production pipeline.
"""

from .evidence_builder import (
    ACTION_VOCABULARY,
    BUILDER_VERSION,
    ENTITY_VOCABULARY,
    EVENT_SUPPORT_VERSION,
    GRAPH_SCHEMA_VERSION,
    REPRESENTATIVE_EVIDENCE_VERSION,
    EvidenceGraphArtifacts,
    NormalizedAction,
    NormalizedClip,
    NormalizedMention,
    TemporalEvent,
    build_evidence_graph,
    load_description_rows,
    load_manifest_frame_paths,
    merge_temporal_events,
    normalize_action,
    normalize_description_rows,
    normalize_entity,
    write_evidence_graph_artifacts,
)
from .ports import EvidenceVerifier, GraphBuilder, GraphRetriever, GraphStore
from .retrieval import (
    RETRIEVER_VERSION,
    DeterministicEventGraphRetriever,
    NormalizedGraphQuery,
    load_evidence_graph,
    normalize_graph_query,
)
from .schemas import (
    EvidenceInterval,
    GraphEdge,
    GraphNode,
    GraphRetrievalResult,
    MedicalGraphQAExample,
    VideoEvidenceGraph,
)

__all__ = [
    "ACTION_VOCABULARY",
    "BUILDER_VERSION",
    "ENTITY_VOCABULARY",
    "EVENT_SUPPORT_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "REPRESENTATIVE_EVIDENCE_VERSION",
    "RETRIEVER_VERSION",
    "DeterministicEventGraphRetriever",
    "EvidenceGraphArtifacts",
    "EvidenceInterval",
    "EvidenceVerifier",
    "GraphBuilder",
    "GraphEdge",
    "GraphNode",
    "GraphRetrievalResult",
    "GraphRetriever",
    "GraphStore",
    "MedicalGraphQAExample",
    "NormalizedAction",
    "NormalizedClip",
    "NormalizedGraphQuery",
    "NormalizedMention",
    "TemporalEvent",
    "VideoEvidenceGraph",
    "build_evidence_graph",
    "load_description_rows",
    "load_evidence_graph",
    "load_manifest_frame_paths",
    "merge_temporal_events",
    "normalize_action",
    "normalize_description_rows",
    "normalize_entity",
    "normalize_graph_query",
    "write_evidence_graph_artifacts",
]

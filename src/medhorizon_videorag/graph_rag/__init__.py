"""Contracts for the medical Graph-RAG research pipeline.

The production baseline remains under :mod:`medhorizon_videorag.pipelines.baseline`.
The first runnable builder creates a conservative, per-video evidence graph
from observation-first clip descriptions. Retrieval remains a research stage.
"""

from .evidence_builder import (
    BUILDER_VERSION,
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
from .schemas import (
    EvidenceInterval,
    GraphEdge,
    GraphNode,
    GraphRetrievalResult,
    MedicalGraphQAExample,
    VideoEvidenceGraph,
)

__all__ = [
    "BUILDER_VERSION",
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
    "NormalizedMention",
    "TemporalEvent",
    "VideoEvidenceGraph",
    "build_evidence_graph",
    "load_description_rows",
    "load_manifest_frame_paths",
    "merge_temporal_events",
    "normalize_action",
    "normalize_description_rows",
    "normalize_entity",
    "write_evidence_graph_artifacts",
]

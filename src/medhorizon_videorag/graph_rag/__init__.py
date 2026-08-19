"""Contracts for the medical Graph-RAG research pipeline.

The production baseline remains under :mod:`medhorizon_videorag.pipelines.baseline`.
This package intentionally contains only validated data contracts until graph
construction and retrieval have reproducible implementations.
"""

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
    "EvidenceInterval",
    "EvidenceVerifier",
    "GraphBuilder",
    "GraphEdge",
    "GraphNode",
    "GraphRetrievalResult",
    "GraphRetriever",
    "GraphStore",
    "MedicalGraphQAExample",
    "VideoEvidenceGraph",
]

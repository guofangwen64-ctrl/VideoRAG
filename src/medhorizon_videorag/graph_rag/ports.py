from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from medhorizon_videorag.core.schemas import Chunk

from .schemas import GraphRetrievalResult, VideoEvidenceGraph


class GraphBuilder(Protocol):
    def build(
        self, video_id: str, video_path: str, segments: Sequence[Chunk]
    ) -> VideoEvidenceGraph: ...


class GraphStore(Protocol):
    def save(self, graph: VideoEvidenceGraph, path: Path) -> None: ...
    def load(self, path: Path) -> VideoEvidenceGraph: ...


class GraphRetriever(Protocol):
    def retrieve(
        self,
        question_id: str,
        question: str,
        graph: VideoEvidenceGraph,
        top_k: int,
    ) -> GraphRetrievalResult: ...


class EvidenceVerifier(Protocol):
    def verify(
        self, question: str, result: GraphRetrievalResult
    ) -> GraphRetrievalResult: ...

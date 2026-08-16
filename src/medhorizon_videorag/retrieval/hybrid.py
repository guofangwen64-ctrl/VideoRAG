from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from medhorizon_videorag.core.schemas import RetrievalResult
from medhorizon_videorag.datasets import TemporalQuery

from .retriever import VisualRetriever
from .temporal import TemporalRetriever


@dataclass(frozen=True)
class RetrievalResponse:
    route: str
    results: list[RetrievalResult]
    temporal_query: TemporalQuery | None = None


@dataclass
class HybridRetriever:
    """Route explicit time questions to metadata retrieval, all others to visual search."""
    temporal: TemporalRetriever
    visual_factory: Callable[[], VisualRetriever]
    _visual: VisualRetriever | None = field(default=None, init=False, repr=False)

    @property
    def visual(self) -> VisualRetriever:
        if self._visual is None:
            self._visual = self.visual_factory()
        return self._visual

    def retrieve(self, question: str, video_id: str, top_k: int) -> RetrievalResponse:
        temporal_query, results = self.temporal.retrieve(question, video_id, top_k)
        if temporal_query is not None:
            return RetrievalResponse("temporal", results, temporal_query)
        return RetrievalResponse("visual", self.visual.retrieve(question, top_k, video_id=video_id))

from __future__ import annotations

from dataclasses import dataclass

from medhorizon_videorag.core.ports import VisualEmbedder
from medhorizon_videorag.core.schemas import RetrievalResult
from .numpy_index import NumpyVectorIndex


@dataclass
class VisualRetriever:
    index: NumpyVectorIndex
    embedder: VisualEmbedder

    def retrieve(self, question: str, top_k: int, video_id: str | None = None) -> list[RetrievalResult]:
        return [
            RetrievalResult(item.chunk, item.score, source="visual")
            for item in self.index.search(self.embedder.embed_text([question])[0], top_k, video_id=video_id)
        ]


# Compatibility alias for earlier pipeline imports.
VideoRetriever = VisualRetriever

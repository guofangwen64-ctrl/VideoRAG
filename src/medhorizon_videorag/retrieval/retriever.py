from __future__ import annotations

from dataclasses import dataclass

from medhorizon_videorag.core.ports import VisualEmbedder
from medhorizon_videorag.core.schemas import RetrievalResult
from .numpy_index import NumpyVectorIndex


@dataclass
class VideoRetriever:
    index: NumpyVectorIndex
    embedder: VisualEmbedder

    def retrieve(self, question: str, top_k: int) -> list[RetrievalResult]:
        return self.index.search(self.embedder.embed_text([question])[0], top_k)
